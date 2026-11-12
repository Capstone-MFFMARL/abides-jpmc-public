"""Experiment A: ABIDES market-maker count sweep.

Loops num_mm over {1, 2, 5, 10, 20} x 5 seeds on the rmsc_density config and writes
one CSV row per run:

    num_mm, seed, runtime_s, n_messages, mean_spread, mean_depth, mid_volatility

Metric definitions (all from the ExchangeAgent's order-book log, `book_log2`):

* The log is event-driven (one snapshot per book change, irregular timestamps), so it is
  resampled onto a regular grid (`--freq`, default 10s) with last-observation-carried-
  forward. Every metric is therefore a time-weighted average, not an event-weighted one.
* mean_spread     : mean(best_ask - best_bid) over grid points where both sides exist.
                    Units: cents (ABIDES prices are integer cents).
* mean_depth      : mean over the grid of (shares on top-k bid levels + shares on top-k
                    ask levels), k = `--depth-levels` (default 5, max = book_log_depth = 10).
* mid_volatility  : std of log returns of the mid-price on the grid. This is per grid
                    interval (not annualised), so it depends on `--freq`; compare only
                    across runs that use the same `--freq`.
* runtime_s       : wall-clock seconds of `kernel.run()` only (config build excluded).
* n_messages      : `kernel.ttl_messages`, total messages the kernel processed.

Usage:
    python run_sweep.py                      # full sweep: 25 runs -> results/sweep.csv + figures
    python run_sweep.py --plot-only          # redraw figures from an existing CSV
    python run_sweep.py --resume             # skip (num_mm, seed) pairs already in the CSV
    python run_sweep.py --num-mm 1 2 --seeds 1 2 --out results/smoke.csv   # quick smoke test
"""

import argparse
import csv
import gc
import os
import shutil
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
TICKER = "ABM"
COLUMNS = [
    "num_mm",
    "seed",
    "runtime_s",
    "n_messages",
    "mean_spread",
    "mean_depth",
    "mid_volatility",
]
METRICS = {
    "runtime_s": "Runtime (s)",
    "n_messages": "Messages processed",
    "mean_spread": "Mean bid-ask spread (cents)",
    "mean_depth": "Mean depth, top-k levels (shares)",
    "mid_volatility": "Mid-price volatility (std of log returns)",
}


# --------------------------------------------------------------------------------------
# Order-book metrics
# --------------------------------------------------------------------------------------
def book_log_to_frame(book_log, k):
    """Turn exchange `book_log2` into a DataFrame indexed by time.

    Columns: best_bid, best_ask, depth (bid volume + ask volume over top-k levels).
    A side with no resting orders gives NaN for its best price and 0 for its volume.
    """
    times, best_bid, best_ask, depth = [], [], [], []
    for row in book_log:
        bids, asks = row["bids"], row["asks"]  # arrays of [price, volume], best first
        times.append(row["QuoteTime"])
        best_bid.append(bids[0][0] if len(bids) else np.nan)
        best_ask.append(asks[0][0] if len(asks) else np.nan)
        depth.append(
            (bids[:k, 1].sum() if len(bids) else 0)
            + (asks[:k, 1].sum() if len(asks) else 0)
        )
    df = pd.DataFrame(
        {"best_bid": best_bid, "best_ask": best_ask, "depth": depth},
        index=pd.to_datetime(times, unit="ns"),
    )
    # Several book changes can share a timestamp; keep the last (the settled state).
    return df[~df.index.duplicated(keep="last")]


def book_metrics(book_log, mkt_close_ns, k=5, freq="10s"):
    df = book_log_to_frame(book_log, k)
    # Start the grid once both sides of the book exist for the first time.
    both = df.dropna(subset=["best_bid", "best_ask"])
    if both.empty:
        return dict(mean_spread=np.nan, mean_depth=np.nan, mid_volatility=np.nan)
    grid = pd.date_range(
        both.index[0].ceil(freq), pd.to_datetime(mkt_close_ns, unit="ns"), freq=freq
    )
    # last-observation-carried-forward onto the grid
    g = df.reindex(df.index.union(grid)).ffill().loc[grid]
    spread = (g["best_ask"] - g["best_bid"]).dropna()
    mid = ((g["best_ask"] + g["best_bid"]) / 2).dropna()
    rets = np.log(mid).diff().dropna()
    return dict(
        mean_spread=float(spread.mean()),
        mean_depth=float(g["depth"].mean()),
        mid_volatility=float(rets.std(ddof=1)),
    )


# --------------------------------------------------------------------------------------
# One simulation
# --------------------------------------------------------------------------------------
def run_one(num_mm, seed, end_time, k, freq):
    # imported lazily so `--plot-only` works without the ABIDES stack
    from abides_core import Kernel
    from abides_core.utils import subdict
    from abides_markets.configs.rmsc_density import build_config

    config = build_config(
        seed=seed,
        num_mm=num_mm,
        end_time=end_time,
        book_logging=True,  # required: metrics come from the book log
        log_orders=False,
        exchange_log_orders=False,
    )
    log_dir = f"__sweep_{num_mm}_{seed}"
    kernel = Kernel(
        log_dir=log_dir,
        random_state=np.random.RandomState(seed=seed),
        **subdict(
            config,
            [
                "start_time",
                "stop_time",
                "agents",
                "agent_latency_model",
                "default_computation_delay",
                "custom_properties",
            ],
        ),
        skip_log=True,
    )
    t0 = time.perf_counter()
    kernel.run()
    runtime = time.perf_counter() - t0

    exchange = kernel.agents[0]
    metrics = book_metrics(
        exchange.order_books[TICKER].book_log2, exchange.mkt_close, k=k, freq=freq
    )
    out = dict(
        num_mm=num_mm,
        seed=seed,
        runtime_s=runtime,
        n_messages=kernel.ttl_messages,
        **metrics,
    )
    del kernel, config
    gc.collect()
    shutil.rmtree(os.path.join("log", log_dir), ignore_errors=True)
    return out


# --------------------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------------------
def make_figures(csv_path, fig_dir):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    df = pd.read_csv(csv_path)
    os.makedirs(fig_dir, exist_ok=True)
    n_seeds = df["seed"].nunique()
    for col, label in METRICS.items():
        g = df.groupby("num_mm")[col].agg(["mean", "std"]).reset_index()
        fig, ax = plt.subplots(figsize=(5.5, 4))
        ax.scatter(df["num_mm"], df[col], s=14, alpha=0.35, color="gray", label="single seeds")
        ax.errorbar(
            g["num_mm"], g["mean"], yerr=g["std"], fmt="o-", capsize=4, lw=1.8,
            label=f"mean ± std ({n_seeds} seeds)",
        )
        ax.set_xscale("log")
        ax.set_xticks(g["num_mm"])
        ax.set_xticklabels(g["num_mm"].astype(int))
        ax.set_xlabel("Number of market makers (num_mm)")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(fig_dir, f"{col}.png"), dpi=150)
        plt.close(fig)
    print(f"figures written to {fig_dir}")


# --------------------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("--num-mm", type=int, nargs="+", default=[1, 2, 5, 10, 20])
    p.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    p.add_argument("--end-time", default="10:00:00", help="sim end (market opens 09:30:00)")
    p.add_argument("--depth-levels", type=int, default=5, help="k for mean_depth (<= 10)")
    p.add_argument("--freq", default="10s", help="resampling grid for book metrics")
    p.add_argument("--out", default=os.path.join(HERE, "results", "sweep.csv"))
    p.add_argument("--fig-dir", default=os.path.join(HERE, "figures"))
    p.add_argument("--resume", action="store_true")
    p.add_argument("--plot-only", action="store_true")
    p.add_argument("--no-warmup", action="store_true")
    args = p.parse_args()

    if args.plot_only:
        make_figures(args.out, args.fig_dir)
        return

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    done = set()
    if args.resume and os.path.exists(args.out):
        prev = pd.read_csv(args.out)
        done = set(zip(prev["num_mm"], prev["seed"]))
    else:
        with open(args.out, "w", newline="") as f:
            csv.DictWriter(f, fieldnames=COLUMNS).writeheader()

    if not args.no_warmup:
        # first sim in a process pays one-off import / JIT costs; keep them out of the timings
        print("warm-up run (discarded)...")
        run_one(1, 0, "09:35:00", args.depth_levels, args.freq)

    total = len(args.num_mm) * len(args.seeds)
    i = 0
    for num_mm in args.num_mm:
        for seed in args.seeds:
            i += 1
            if (num_mm, seed) in done:
                continue
            row = run_one(num_mm, seed, args.end_time, args.depth_levels, args.freq)
            with open(args.out, "a", newline="") as f:  # append per run: crash-safe
                csv.DictWriter(f, fieldnames=COLUMNS).writerow(row)
            print(
                f"[{i}/{total}] num_mm={num_mm:>2} seed={seed} "
                f"runtime={row['runtime_s']:.1f}s msgs={row['n_messages']:,} "
                f"spread={row['mean_spread']:.2f} depth={row['mean_depth']:.0f} "
                f"vol={row['mid_volatility']:.2e}",
                flush=True,
            )
    make_figures(args.out, args.fig_dir)


if __name__ == "__main__":
    main()
