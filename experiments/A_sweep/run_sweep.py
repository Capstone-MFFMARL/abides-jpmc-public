"""ABIDES agent-count sweep: num_mm x seeds -> CSV of runtime, messages, spread, depth, volatility."""
import argparse
import csv
import os
import sys
import time

import numpy as np
import pandas as pd

from abides_core.utils import str_to_ns

GRID_NS = 10 * 10**9  # 10 s grid
TOP_K = 5  # levels used for depth
HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUT = os.path.join(HERE, "sweep_results.csv")
SNAP_DIR = os.path.join(HERE, "snapshots")
FIELDS = [
    "run_id",
    "num_mm",
    "seed",
    "runtime_s",
    "messages",
    "spread",
    "depth",
    "vol_bps",
    "n_grid",
    "n_valid",
    "warmup",
]


def book_metrics(L2, mkt_close_ns, warmup_ns=None, grid_ns=GRID_NS, top_k=TOP_K):
    """L2 = order_book.get_L2_snapshots(nlevels=10). Prices in cents."""
    t = np.asarray(L2["times"], dtype=np.int64)
    bids, asks = np.asarray(L2["bids"]), np.asarray(L2["asks"])
    valid = (bids[:, 0, 1] > 0) & (asks[:, 0, 1] > 0)  # both sides non-empty
    n_valid = int(valid.sum())
    if n_valid < 3:
        return dict(spread=np.nan, depth=np.nan, vol_bps=np.nan, n_grid=0, n_valid=n_valid)
    t, bids, asks = t[valid], bids[valid], asks[valid]
    spread = asks[:, 0, 0] - bids[:, 0, 0]
    mid = (asks[:, 0, 0] + bids[:, 0, 0]) / 2.0
    depth = bids[:, :top_k, 1].sum(1) + asks[:, :top_k, 1].sum(1)

    # Clock-aligned grid starting at ceil(first valid quote time) and ending at market close
    grid_start = int(np.ceil(t[0] / grid_ns)) * grid_ns
    grid = np.arange(grid_start, mkt_close_ns + 1, grid_ns)
    idx = np.searchsorted(t, grid, side="right") - 1
    ok = idx >= 0
    grid, idx = grid[ok], idx[ok]

    # Warm-up filtering
    if warmup_ns is not None:
        mask = grid >= warmup_ns
        grid, idx = grid[mask], idx[mask]

    if len(grid) == 0:
        return dict(spread=np.nan, depth=np.nan, vol_bps=np.nan, n_grid=0, n_valid=n_valid)

    spread_g, mid_g, depth_g = spread[idx], mid[idx], depth[idx]
    rets = np.diff(np.log(mid_g))
    vol_bps = float(rets.std(ddof=1) * 1e4) if len(rets) > 1 else np.nan
    return dict(
        spread=float(spread_g.mean()),
        depth=float(depth_g.mean()),
        vol_bps=vol_bps,
        n_grid=int(len(grid)),
        n_valid=n_valid,
    )


def grid_snapshots(L2, mkt_close_ns, grid_ns=GRID_NS):
    """Order-book snapshots on a fixed grid (last known value). bids/asks: (T, 10, 2) = [price in cents, size]."""
    t = np.asarray(L2["times"], dtype=np.int64)
    bids, asks = np.asarray(L2["bids"]), np.asarray(L2["asks"])
    valid = (bids[:, 0, 1] > 0) & (asks[:, 0, 1] > 0)
    t, bids, asks = t[valid], bids[valid], asks[valid]
    if len(t) < 3:
        return t, bids, asks
    grid_start = int(np.ceil(t[0] / grid_ns)) * grid_ns
    grid = np.arange(grid_start, mkt_close_ns + 1, grid_ns)
    idx = np.searchsorted(t, grid, side="right") - 1
    ok = idx >= 0
    return grid[ok], bids[idx[ok]], asks[idx[ok]]


def run_one(num_mm, seed, end_time="10:00:00", warmup="09:31:00", snap_dir=SNAP_DIR, save_snap=True):
    from abides_core import Kernel
    from abides_core.utils import subdict
    from abides_markets.configs import rmsc_density

    config = rmsc_density.build_config(seed=seed, num_mm=num_mm, end_time=end_time, stdout_log_level="CRITICAL")
    kernel = Kernel(
        random_state=config["random_state_kernel"],
        log_dir="",
        **subdict(config, ["start_time", "stop_time", "agents", "agent_latency_model",
                           "default_computation_delay", "custom_properties"]),
    )
    t0 = time.time()
    end_state = kernel.run()
    runtime = time.time() - t0
    exchange = end_state["agents"][0]
    ob = exchange.order_books["ABM"]
    mkt_close_ns = exchange.mkt_close

    date_ns = pd.Timestamp(mkt_close_ns, unit="ns").floor("D").value
    warmup_ns = date_ns + str_to_ns(warmup) if warmup else None

    L2 = ob.get_L2_snapshots(nlevels=10)
    m = book_metrics(L2, mkt_close_ns=mkt_close_ns, warmup_ns=warmup_ns)
    run_id = "mm%d_s%d" % (num_mm, seed)

    if save_snap and snap_dir:
        os.makedirs(snap_dir, exist_ok=True)
        g_t, g_b, g_a = grid_snapshots(L2, mkt_close_ns=mkt_close_ns)
        np.savez_compressed(os.path.join(snap_dir, run_id + ".npz"), times_ns=g_t, bids=g_b, asks=g_a)

    return dict(
        run_id=run_id,
        num_mm=num_mm,
        seed=seed,
        runtime_s=round(runtime, 2),
        messages=kernel.ttl_messages,
        warmup=warmup if warmup is not None else "",
        **m,
    )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-mm", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--end-time", default="10:00:00")
    p.add_argument("--warmup", default="09:31:00")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--snap-dir", default=SNAP_DIR)
    p.add_argument("--no-snapshots", action="store_true")
    args = p.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    done = set()
    if not new:
        with open(args.out) as f:
            done = {
                (int(r["num_mm"]), int(r["seed"]))
                for r in csv.DictReader(f)
                if "num_mm" in r and "seed" in r and r["num_mm"] and r["seed"]
            }
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for seed in args.seeds:
            for mm in args.num_mm:
                if (mm, seed) in done:
                    continue
                row = run_one(
                    mm,
                    seed,
                    end_time=args.end_time,
                    warmup=args.warmup,
                    snap_dir=args.snap_dir,
                    save_snap=not args.no_snapshots,
                )
                w.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
