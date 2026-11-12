"""Stability index for learning curves, after Yamaguchi (2025), arXiv:2511.23315.

    S = 1 - v / v_max

v      variance of the TD error inside one window (or one epoch) of training
v_max  the largest v among everything being compared (windows and/or runs)

S near 1 means calm learning; S near 0 means the most unstable window seen.

Two things to know before using it:

1. S is RELATIVE. Inside a single run the worst window always gets S = 0, even
   if the run is perfectly healthy. To compare runs (e.g. different agent
   counts), give them a shared v_max -- `compare_runs` does this, which is
   also how Yamaguchi uses it (v_max is taken across all conditions).
   To judge one run on its own, use `variance_growth` / `flag_unstable`.

2. For the PPO Kyle log (experiments/B_kyle/ppo_training_log.csv) use the
   `td_var` column: it is the variance of the TD errors
   r_t + gamma*V(s_{t+1}) - V(s_t) across each batch, i.e. Yamaguchi's v.
   Divide it by `return_var` first (`--normalize-by return_var`): profit
   grows from 0 to ~16k dollars during training, so raw variance rises with
   scale even when learning is calm. `value_loss` is MSE against Monte Carlo
   returns and is NOT the same quantity.

3. PPO has no Q-learning TD error. Its closest equivalent is the critic's
   value error. If the training log only stores a value LOSS per epoch
   (a mean squared value error), that number is already a second moment of
   the error, so `stability_from_losses` uses it directly as v.

Usage
-----
    python stability_index.py --demo
    python stability_index.py --log ppo_training_log.csv --column td_var --normalize-by return_var
    python stability_index.py --log td_errors.npy --raw --window 100
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

__all__ = [
    "window_variance",
    "stability_index",
    "stability_from_errors",
    "stability_from_losses",
    "stability_from_variances",
    "compare_runs",
    "variance_growth",
    "flag_unstable",
    "plot_stability",
]


def _as_1d(x: Sequence[float], name: str) -> np.ndarray:
    arr = np.asarray(x, dtype=float).ravel()
    if arr.size == 0:
        raise ValueError(f"{name} is empty")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or inf")
    return arr


def window_variance(errors: Sequence[float], window: int) -> np.ndarray:
    """Variance of `errors` in consecutive non-overlapping windows.

    A trailing partial window is dropped so every variance uses `window` points.
    """
    arr = _as_1d(errors, "errors")
    if window < 2:
        raise ValueError("window must be at least 2")
    n = arr.size // window
    if n == 0:
        raise ValueError(f"need at least {window} points, got {arr.size}")
    return arr[: n * window].reshape(n, window).var(axis=1)


def stability_index(variances: Sequence[float], v_max: float | None = None) -> np.ndarray:
    """Yamaguchi's S = 1 - v / v_max, one value per variance.

    v_max defaults to the largest variance passed in. Pass a shared v_max to
    put several runs on the same scale. Values are clipped to [0, 1].
    """
    v = _as_1d(variances, "variances")
    if np.any(v < 0):
        raise ValueError("variances must be non-negative")
    vmax = float(v.max()) if v_max is None else float(v_max)
    if vmax < 0:
        raise ValueError("v_max must be non-negative")
    if vmax == 0.0:  # no variance anywhere: perfectly stable
        return np.ones_like(v)
    return np.clip(1.0 - v / vmax, 0.0, 1.0)


def stability_from_errors(
    errors: Sequence[float], window: int, v_max: float | None = None
) -> np.ndarray:
    """S per window, from a raw series of TD errors (or value errors)."""
    return stability_index(window_variance(errors, window), v_max)


def stability_from_losses(
    losses: Sequence[float], v_max: float | None = None, smooth: int = 1
) -> np.ndarray:
    """S per epoch, from a value-loss history (one loss per epoch).

    Each loss is treated as that epoch's v. `smooth` > 1 averages the losses
    over a trailing window of that many epochs first, to tame noisy logs.
    """
    v = _as_1d(losses, "losses")
    if np.any(v < 0):
        raise ValueError("losses must be non-negative")
    if smooth < 1:
        raise ValueError("smooth must be at least 1")
    if smooth > 1:
        csum = np.cumsum(np.insert(v, 0, 0.0))
        idx = np.arange(1, v.size + 1)
        start = np.maximum(idx - smooth, 0)
        v = (csum[idx] - csum[start]) / (idx - start)
    return stability_index(v, v_max)


def stability_from_variances(
    variances: Sequence[float],
    scale: Sequence[float] | None = None,
    v_max: float | None = None,
    smooth: int = 1,
) -> np.ndarray:
    """S per epoch, from a logged variance per epoch (e.g. the `td_var` column).

    If `scale` is given (e.g. the `return_var` column), each variance is divided
    by it first, so S tracks instability rather than the growing size of rewards.
    """
    v = _as_1d(variances, "variances")
    if scale is not None:
        sc = _as_1d(scale, "scale")
        if sc.shape != v.shape:
            raise ValueError("variances and scale must have the same length")
        if np.any(sc <= 0):
            raise ValueError("scale must be positive (a zero return_var cannot normalise)")
        v = v / sc
    return stability_from_losses(v, v_max=v_max, smooth=smooth)


def compare_runs(
    runs: Mapping[str, Sequence[float]], window: int
) -> dict[str, dict[str, object]]:
    """Score several runs on one scale (shared v_max), as Yamaguchi does.

    `runs` maps a run name (e.g. "num_mm=20") to its raw error series.
    Returns, per run: the S series, its mean, and its minimum.
    """
    if not runs:
        raise ValueError("runs is empty")
    variances = {name: window_variance(e, window) for name, e in runs.items()}
    v_max = max(float(v.max()) for v in variances.values())
    out: dict[str, dict[str, object]] = {}
    for name, v in variances.items():
        s = stability_index(v, v_max)
        out[name] = {"S": s, "mean_S": float(s.mean()), "min_S": float(s.min())}
    return out


def variance_growth(errors: Sequence[float], window: int, frac: float = 0.25) -> float:
    """How much the error variance grew over a run: late median / early median.

    Compares the median window-variance of the last `frac` of the run with the
    first `frac`. About 1 means steady; well above 1 means variance is growing.
    Unlike S, this is an absolute test that works on a single run.
    """
    if not 0.0 < frac <= 0.5:
        raise ValueError("frac must be in (0, 0.5]")
    v = window_variance(errors, window)
    k = max(1, int(round(v.size * frac)))
    if v.size < 2 * k:
        raise ValueError("series too short: need at least two windows per segment")
    early, late = float(np.median(v[:k])), float(np.median(v[-k:]))
    if early == 0.0:
        return float("inf") if late > 0.0 else 1.0
    return late / early


def flag_unstable(
    errors: Sequence[float], window: int, growth_threshold: float = 4.0
) -> bool:
    """True if the error variance grew by more than `growth_threshold` times.

    The default of 4 means the typical error size (std) at least doubled.
    """
    return variance_growth(errors, window) > growth_threshold


def plot_stability(
    series: Mapping[str, Sequence[float]],
    out_path: str | Path,
    xlabel: str = "Window",
    title: str = "Stability index S = 1 - v / v_max",
) -> Path:
    """Plot one S curve per entry of `series` and save the figure."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 4.5))
    for name, s in series.items():
        ax.plot(np.arange(1, len(s) + 1), s, label=name, linewidth=1.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("S  (1 = stable, 0 = least stable)")
    ax.set_title(title)
    ax.grid(alpha=0.3)
    if len(series) > 1:
        ax.legend()
    fig.tight_layout()
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


# ---------------------------------------------------------------- synthetic data
def make_stable(n: int = 5000, sigma: float = 1.0, seed: int = 0) -> np.ndarray:
    """Noise with constant variance: what a healthy run looks like."""
    return np.random.default_rng(seed).normal(0.0, sigma, n)


def make_growing(n: int = 5000, sigma: float = 1.0, factor: float = 5.0, seed: int = 0) -> np.ndarray:
    """Noise whose std grows linearly from sigma to factor*sigma: a run blowing up."""
    scale = np.linspace(sigma, factor * sigma, n)
    return np.random.default_rng(seed).normal(0.0, 1.0, n) * scale


# ---------------------------------------------------------------- command line
def _load_log(path: Path, column: str | None, extra: str | None = None):
    if path.suffix == ".npy":
        return np.load(path).astype(float).ravel()
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        cols = reader.fieldnames or []
        if column is None:
            raise SystemExit(f"--column is required for CSV logs. Columns found: {cols}")
        if column not in cols:
            raise SystemExit(f"column '{column}' not in {path.name}. Columns found: {cols}")
        if extra is not None and extra not in cols:
            raise SystemExit(f"column '{extra}' not in {path.name}. Columns found: {cols}")
        rows = [r for r in reader if r[column] not in ("", None)]
        data = np.array([float(r[column]) for r in rows])
        if extra is None:
            return data
        return data, np.array([float(r[extra]) for r in rows])


def main() -> None:
    p = argparse.ArgumentParser(description="Yamaguchi stability index S = 1 - v/v_max")
    p.add_argument("--demo", action="store_true", help="run on synthetic stable vs growing series")
    p.add_argument("--log", type=Path, help="training log: .csv (needs --column) or .npy")
    p.add_argument("--column", help="CSV column holding the value loss or the raw errors")
    p.add_argument("--raw", action="store_true", help="log holds raw errors, not a loss per epoch")
    p.add_argument("--window", type=int, default=100, help="window size for raw errors")
    p.add_argument("--smooth", type=int, default=1, help="trailing average over epochs for losses")
    p.add_argument("--normalize-by", dest="normalize_by",
                   help="CSV column to divide by first, e.g. return_var")
    p.add_argument("--out", type=Path, default=Path("stability.png"))
    args = p.parse_args()

    if args.demo:
        runs = {"stable": make_stable(), "growing variance": make_growing()}
        res = compare_runs(runs, args.window)
        for name, e in runs.items():
            print(f"{name:18s} mean S = {res[name]['mean_S']:.3f}  min S = {res[name]['min_S']:.3f}  "
                  f"variance growth = {variance_growth(e, args.window):5.1f}x  "
                  f"unstable = {flag_unstable(e, args.window)}")
        out = plot_stability({k: r["S"] for k, r in res.items()}, args.out,
                             xlabel=f"Window ({args.window} steps each)")
    elif args.log:
        if args.normalize_by:
            if args.raw:
                p.error("--normalize-by works with a per-epoch variance column, not --raw")
            data, scale = _load_log(args.log, args.column, args.normalize_by)
            s = stability_from_variances(data, scale, smooth=args.smooth)
            xlabel = "PPO epoch"
            label = f"{args.column} / {args.normalize_by}"
        else:
            data = _load_log(args.log, args.column)
            label = args.column or args.log.stem
        if args.normalize_by:
            pass
        elif args.raw:
            s = stability_from_errors(data, args.window)
            print(f"variance growth = {variance_growth(data, args.window):.2f}x  "
                  f"unstable = {flag_unstable(data, args.window)}")
            xlabel = f"Window ({args.window} steps each)"
        else:
            s = stability_from_losses(data, smooth=args.smooth)
            xlabel = "PPO epoch"
        print(f"{len(s)} points  mean S = {s.mean():.3f}  min S at index {int(s.argmin()) + 1}")
        out = plot_stability({label: s}, args.out, xlabel=xlabel)
    else:
        p.error("give --demo or --log")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
