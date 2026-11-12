"""
npz_analysis.py -- run cvar and depth_to_spread on Part A's real sweep data.

Usage:
    python npz_analysis.py                        # auto-discovers ../A_sweep/snapshots/
    python npz_analysis.py path/to/snapshots/     # explicit directory

Output:
    - Console table: per-file CVaR and mean depth-to-spread
    - Aggregated table by num_mm (mean ± std across seeds)
    - Sanity check: depth/spread should rise and CVaR should not rise with num_mm
    - Saves results to npz_analysis_results.csv
"""

from __future__ import annotations

import glob
import os
import sys
import warnings

import numpy as np

# Allow running from D_cbf directly or from anywhere
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)

from fragility_index import (
    cvar,
    depth_to_spread_series,
    log_returns,
    mid_prices,
)

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
ALPHA = 0.05          # CVaR tail probability
K_LEVELS = 5          # top-k levels for depth
PRICE_SCALE = 100.0   # cents → dollars  (depth reported in shares per dollar)
MIN_FINITE = 20       # minimum finite returns to attempt CVaR


def _parse_name(basename: str) -> tuple[int, int]:
    """Parse 'mm5_s1.npz' → (num_mm=5, seed=1). Returns (-1, -1) on failure."""
    try:
        name = os.path.splitext(basename)[0]  # strip .npz
        mm_part, s_part = name.split("_")
        return int(mm_part[2:]), int(s_part[1:])
    except Exception:
        return -1, -1


def analyse_file(path: str) -> dict:
    """Load one .npz snapshot and compute fragility metrics."""
    d = np.load(path)
    bids = d["bids"].astype(float)
    asks = d["asks"].astype(float)

    mid = mid_prices(bids, asks)                     # (T,) cents
    ret = log_returns(mid)                           # (T-1,) dimensionless
    ret_finite = ret[np.isfinite(ret)]

    if ret_finite.size >= MIN_FINITE:
        cv = cvar(ret_finite, alpha=ALPHA)
    else:
        warnings.warn(
            f"{os.path.basename(path)}: only {ret_finite.size} finite returns "
            f"(< {MIN_FINITE}), CVaR set to NaN"
        )
        cv = float("nan")

    dts = depth_to_spread_series(bids, asks, k=K_LEVELS, price_scale=PRICE_SCALE)
    mean_dts = float(np.nanmean(dts))
    std_dts  = float(np.nanstd(dts))

    num_mm, seed = _parse_name(os.path.basename(path))

    return {
        "path":     path,
        "num_mm":   num_mm,
        "seed":     seed,
        "T":        int(bids.shape[0]),
        "n_finite": int(ret_finite.size),
        "cvar_5pct": cv,
        "mean_dts": mean_dts,
        "std_dts":  std_dts,
    }


def run(snapshot_dir: str | None = None) -> list[dict]:
    if snapshot_dir is None:
        snapshot_dir = os.path.join(_HERE, "..", "A_sweep", "snapshots")
    snapshot_dir = os.path.abspath(snapshot_dir)
    paths = sorted(glob.glob(os.path.join(snapshot_dir, "*.npz")))
    if not paths:
        raise FileNotFoundError(f"No .npz files found in: {snapshot_dir}")

    print(f"\nAnalysing {len(paths)} snapshots from: {snapshot_dir}")
    print(f"Settings: alpha={ALPHA}, k_levels={K_LEVELS}, price_scale={PRICE_SCALE} (shares per dollar)\n")

    rows = [analyse_file(p) for p in paths]

    # --- per-file table ---
    print(f"{'File':<18} {'mm':>4} {'seed':>5} {'T':>5} {'CVaR(5%)':>12} {'mean D/S':>12} {'std D/S':>10}")
    print("-" * 72)
    for r in rows:
        fname = os.path.basename(r["path"])
        print(
            f"{fname:<18} {r['num_mm']:>4} {r['seed']:>5} {r['T']:>5} "
            f"{r['cvar_5pct']:>12.6f} {r['mean_dts']:>12.4f} {r['std_dts']:>10.4f}"
        )

    # --- aggregate by num_mm ---
    all_mm = sorted(set(r["num_mm"] for r in rows if r["num_mm"] > 0))
    print(f"\n{'num_mm':>7} {'CVaR mean':>12} {'CVaR std':>10} {'D/S mean':>12} {'D/S std':>10} {'n':>4}")
    print("-" * 60)
    agg = []
    for mm in all_mm:
        sub = [r for r in rows if r["num_mm"] == mm]
        cv_vals  = np.array([r["cvar_5pct"] for r in sub], dtype=float)
        dts_vals = np.array([r["mean_dts"]  for r in sub], dtype=float)
        agg.append({
            "num_mm":    mm,
            "cvar_mean": float(np.nanmean(cv_vals)),
            "cvar_std":  float(np.nanstd(cv_vals)),
            "dts_mean":  float(np.nanmean(dts_vals)),
            "dts_std":   float(np.nanstd(dts_vals)),
            "n":         len(sub),
        })
        print(
            f"{mm:>7} {agg[-1]['cvar_mean']:>12.6f} {agg[-1]['cvar_std']:>10.6f} "
            f"{agg[-1]['dts_mean']:>12.4f} {agg[-1]['dts_std']:>10.4f} {len(sub):>4}"
        )

    # --- sanity check ---
    print("\n--- Sanity checks ---")
    if len(agg) >= 2:
        dts_monotone = all(
            agg[i]["dts_mean"] <= agg[i + 1]["dts_mean"] for i in range(len(agg) - 1)
        )
        cv_not_rising = all(
            agg[i]["cvar_mean"] >= agg[i + 1]["cvar_mean"] - 1e-6
            for i in range(len(agg) - 1)
        )
        print(f"D/S rises with num_mm (less fragile as MMs added): {'PASS ✓' if dts_monotone else 'WARN — not strictly monotone (check data)'}")
        print(f"CVaR does not rise with num_mm:                     {'PASS ✓' if cv_not_rising else 'WARN — CVaR rises somewhere (investigate)'}")
    else:
        print("Not enough distinct num_mm values for monotonicity check.")

    # --- save CSV ---
    csv_path = os.path.join(_HERE, "npz_analysis_results.csv")
    header = "path,num_mm,seed,T,n_finite,cvar_5pct,mean_dts,std_dts"
    lines = [header] + [
        f"{r['path']},{r['num_mm']},{r['seed']},{r['T']},{r['n_finite']},"
        f"{r['cvar_5pct']:.8f},{r['mean_dts']:.6f},{r['std_dts']:.6f}"
        for r in rows
    ]
    with open(csv_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\nResults saved to: {csv_path}")

    return rows


if __name__ == "__main__":
    snap_dir = sys.argv[1] if len(sys.argv) > 1 else None
    run(snap_dir)

