"""
cbf_toy.py -- discrete-time Control Barrier Function safety filter (toy example).

System:   x[k+1] = x[k] + u[k]          (scalar, e.g. fragility index value)
Safe set: h(x) = x_max - x >= 0          (stay below a fragility ceiling)
Nominal:  u_nom = +0.05                   (slowly pushes toward the boundary)
Filter:   clip u so h(x[k+1]) >= (1-gamma) * h(x[k])

The CBF filter solves the scalar QP analytically:
    max  u   s.t.  u <= u_nom
                   h(x + u) >= (1-gamma) * h(x)
                 ⟺  x_max - x - u >= (1-gamma)(x_max - x)
                 ⟺  u <= gamma * h(x)

So the safe control is: u* = min(u_nom, gamma * h(x))

Reference: Agrawal & Sreenath (2017) "Discrete Control Barrier Functions for
           Safety-Critical Control of Discrete Systems with Application to
           Bipedal Robot Navigation" (Robotics: Science and Systems).

Usage:
    python cbf_toy.py          # runs simulation and prints trajectory table
    python cbf_toy.py --plot   # also saves a PNG figure (requires matplotlib)
"""

from __future__ import annotations

import sys
from dataclasses import dataclass

import numpy as np

# --------------------------------------------------------------------------- #
# CBF filter
# --------------------------------------------------------------------------- #
@dataclass
class CBFParams:
    x_max: float = 0.80    # safe set ceiling (e.g. max allowed fragility)
    gamma: float = 0.10    # decay rate in [0, 1]; smaller = more conservative
    u_nom: float = 0.05    # nominal (unfiltered) control increment per step


def barrier(x: float, params: CBFParams) -> float:
    """h(x) = x_max - x.  Positive inside the safe set, zero at the boundary."""
    return params.x_max - x


def cbf_filter(x: float, params: CBFParams) -> tuple[float, float]:
    """Return (u_safe, u_nom). u_safe is the CBF-clipped control."""
    h = barrier(x, params)
    u_safe = min(params.u_nom, params.gamma * h)
    return u_safe, params.u_nom


def simulate(
    x0: float,
    n_steps: int,
    params: CBFParams,
    use_filter: bool,
) -> np.ndarray:
    """Simulate the scalar system for n_steps, returning state trajectory (n_steps+1,)."""
    xs = np.empty(n_steps + 1)
    xs[0] = x0
    for k in range(n_steps):
        u_safe, u_nom = cbf_filter(xs[k], params)
        u = u_safe if use_filter else u_nom
        xs[k + 1] = xs[k] + u
    return xs


# --------------------------------------------------------------------------- #
# Main: run, print table, optionally plot
# --------------------------------------------------------------------------- #
def main(plot: bool = False) -> None:
    params  = CBFParams(x_max=0.80, gamma=0.10, u_nom=0.05)
    x0      = 0.20
    n_steps = 40

    xs_filtered   = simulate(x0, n_steps, params, use_filter=True)
    xs_unfiltered = simulate(x0, n_steps, params, use_filter=False)

    print(f"\nCBF toy simulation")
    print(f"  System:   x[k+1] = x[k] + u[k]")
    print(f"  Safe set: h(x) = {params.x_max} - x >= 0")
    print(f"  Nominal:  u_nom = {params.u_nom}  (pushes toward boundary)")
    print(f"  gamma   = {params.gamma}  (decay rate)")
    print(f"  x0      = {x0},  steps = {n_steps}\n")

    print(f"{'k':>4}  {'x (filtered)':>14}  {'h (filtered)':>14}  {'x (unfiltered)':>16}  {'safe?':>6}")
    print("-" * 65)
    for k in range(n_steps + 1):
        xf  = xs_filtered[k]
        xuf = xs_unfiltered[k]
        hf  = barrier(xf, params)
        safe = "YES" if hf >= -1e-9 else "NO ✗"
        marker = " <-- violated" if hf < -1e-9 else ""
        print(f"{k:>4}  {xf:>14.6f}  {hf:>14.6f}  {xuf:>16.6f}  {safe:>6}{marker}")

    # Verify filtered trajectory never crosses x_max
    max_filtered = float(np.max(xs_filtered))
    violated = max_filtered > params.x_max + 1e-9
    print(f"\nFiltered max state: {max_filtered:.6f}  (x_max = {params.x_max})")
    print(f"Safety constraint satisfied: {'YES ✓' if not violated else 'NO ✗'}")

    # Verify unfiltered trajectory crosses x_max
    max_unfiltered = float(np.max(xs_unfiltered))
    print(f"Unfiltered max state: {max_unfiltered:.6f}  (crosses boundary: {'YES' if max_unfiltered > params.x_max else 'NO'})")

    if plot:
        _plot(xs_filtered, xs_unfiltered, params, n_steps)


def _plot(
    xs_filtered: np.ndarray,
    xs_unfiltered: np.ndarray,
    params: CBFParams,
    n_steps: int,
) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("\nmatplotlib not installed — skipping plot.")
        return

    import os
    steps = np.arange(n_steps + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.plot(steps, xs_unfiltered, "r--", linewidth=1.8, label="Unfiltered (u_nom constant)")
    ax.plot(steps, xs_filtered,   "b-",  linewidth=2.0, label=f"CBF-filtered (γ={params.gamma})")
    ax.axhline(params.x_max, color="k", linewidth=1.2, linestyle=":", label=f"Safe boundary x_max={params.x_max}")

    ax.fill_between(steps, params.x_max, ax.get_ylim()[1] if ax.get_ylim()[1] > params.x_max else params.x_max + 0.1,
                    alpha=0.08, color="red", label="Unsafe region")

    ax.set_xlabel("Step k", fontsize=12)
    ax.set_ylabel("State x (fragility)", fontsize=12)
    ax.set_title("Discrete-Time CBF Safety Filter — Toy Scalar Example", fontsize=13)
    ax.legend(fontsize=10)
    ax.set_ylim(bottom=min(0.0, xs_filtered.min() - 0.05))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cbf_toy_plot.png")
    fig.savefig(out, dpi=150)
    print(f"\nPlot saved to: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main(plot="--plot" in sys.argv)

