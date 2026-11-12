"""ABIDES agent-count sweep: num_mm x seeds -> CSV of runtime, messages, spread, depth, volatility."""
import csv
import os
import sys
import time

import numpy as np

NUM_MM = [1, 2, 5, 10, 20]
SEEDS = [1, 2, 3, 4, 5]
GRID_NS = 10 * 10**9  # 10 s grid
TOP_K = 5  # levels used for depth
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "sweep_results.csv")
FIELDS = ["run_id", "num_mm", "seed", "runtime_s", "messages", "spread", "depth", "vol_bps", "n_grid", "n_valid"]


def book_metrics(L2, grid_ns=GRID_NS, top_k=TOP_K):
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
    grid = np.arange(t[0], t[-1], grid_ns)
    idx = np.searchsorted(t, grid, side="right") - 1
    ok = idx >= 0
    grid, idx = grid[ok], idx[ok]
    spread_g, mid_g, depth_g = spread[idx], mid[idx], depth[idx]
    rets = np.diff(np.log(mid_g))
    return dict(
        spread=float(spread_g.mean()),
        depth=float(depth_g.mean()),
        vol_bps=float(rets.std() * 1e4),
        n_grid=int(len(grid)),
        n_valid=n_valid,
    )


SNAP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "snapshots")


def grid_snapshots(L2, grid_ns=GRID_NS):
    """Order-book snapshots on a fixed grid (last known value). bids/asks: (T, 10, 2) = [price in cents, size]."""
    t = np.asarray(L2["times"], dtype=np.int64)
    bids, asks = np.asarray(L2["bids"]), np.asarray(L2["asks"])
    valid = (bids[:, 0, 1] > 0) & (asks[:, 0, 1] > 0)
    t, bids, asks = t[valid], bids[valid], asks[valid]
    if len(t) < 3:
        return t, bids, asks
    grid = np.arange(t[0], t[-1], grid_ns)
    idx = np.searchsorted(t, grid, side="right") - 1
    ok = idx >= 0
    return grid[ok], bids[idx[ok]], asks[idx[ok]]


def run_one(num_mm, seed):
    from abides_core import Kernel
    from abides_core.utils import subdict
    from abides_markets.configs import rmsc_density

    config = rmsc_density.build_config(seed=seed, num_mm=num_mm, stdout_log_level="CRITICAL")
    kernel = Kernel(
        random_state=config["random_state_kernel"],
        log_dir="",
        **subdict(config, ["start_time", "stop_time", "agents", "agent_latency_model",
                           "default_computation_delay", "custom_properties"]),
    )
    t0 = time.time()
    end_state = kernel.run()
    runtime = time.time() - t0
    ob = end_state["agents"][0].order_books["ABM"]
    L2 = ob.get_L2_snapshots(nlevels=10)
    m = book_metrics(L2)
    run_id = "mm%d_s%d" % (num_mm, seed)
    os.makedirs(SNAP_DIR, exist_ok=True)
    g_t, g_b, g_a = grid_snapshots(L2)
    np.savez_compressed(os.path.join(SNAP_DIR, run_id + ".npz"), times_ns=g_t, bids=g_b, asks=g_a)
    return dict(run_id=run_id, num_mm=num_mm, seed=seed, runtime_s=round(runtime, 2),
                messages=kernel.ttl_messages, **m)


def main():
    new = not os.path.exists(OUT)
    done = set()
    if not new:
        with open(OUT) as f:
            done = {(int(r["num_mm"]), int(r["seed"])) for r in csv.DictReader(f)}
    with open(OUT, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for seed in SEEDS:
            for mm in NUM_MM:
                if (mm, seed) in done:
                    continue
                row = run_one(mm, seed)
                w.writerow(row)
                f.flush()
                print(row, flush=True)


if __name__ == "__main__":
    main()
