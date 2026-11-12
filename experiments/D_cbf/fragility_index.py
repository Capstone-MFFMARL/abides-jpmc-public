"""
fragility_index.py -- building blocks of the fragility index (Part D).

Three pure functions, no I/O, no global state:

    cvar(returns, alpha)        tail loss of a return series
    depth_to_spread(book, k)    resting depth divided by bid-ask spread
    spectral_radius(W)          largest eigenvalue magnitude of an interaction matrix

Direction convention for the whole index: HIGH = MORE FRAGILE for cvar and
spectral_radius, and LOW = MORE FRAGILE for depth_to_spread. Part C's stability
index S is the opposite (high = stable). Flip signs only when combining terms.

Book snapshot convention (matches Part A's exchange log and Madhu's .npz):
    book = {"bids": array (L, 2), "asks": array (L, 2)}
    rows are [price, size], best level first, price in integer cents, size in shares.
    A missing level is encoded as size <= 0, price <= 0, or NaN, and is ignored.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

__all__ = [
    "cvar",
    "depth_to_spread",
    "spectral_radius",
    "mid_prices",
    "log_returns",
    "depth_to_spread_series",
]


# --------------------------------------------------------------------------- #
# 1. CVaR
# --------------------------------------------------------------------------- #
def cvar(returns, alpha: float = 0.05, min_samples: int | None = None) -> float:
    """Conditional value at risk of a return series, reported as a positive loss.

    Empirical definition: take the k = ceil(alpha * n) worst returns and return
    minus their mean.

    Parameters
    ----------
    returns : array-like of float
        Log returns on a fixed grid (dimensionless, e.g. -0.002 = -0.2%).
        Non-finite values are dropped.
    alpha : float in (0, 1)
        Tail probability. Default 0.05 (the 5% tail).
    min_samples : int, optional
        Minimum number of finite returns required. Default ceil(1 / alpha),
        which guarantees at least one observation per tail slot. Raises
        ValueError below that, instead of returning a meaningless number.

    Returns
    -------
    float
        Units: return units (dimensionless). Range: any real number, but
        normally >= 0. Larger = fatter left tail = more fragile. A negative
        value means even the worst alpha-tail was a gain.

    Notes
    -----
    With T about 179 grid points per run, alpha = 0.05 rests on about 9
    returns; pool returns across seeds before calling if you need stability.
    """
    if not (0.0 < alpha < 1.0):
        raise ValueError(f"alpha must be in (0, 1), got {alpha}")
    r = np.asarray(returns, dtype=float).ravel()
    r = r[np.isfinite(r)]
    needed = math.ceil(1.0 / alpha) if min_samples is None else int(min_samples)
    if r.size < max(needed, 1):
        raise ValueError(
            f"need at least {needed} finite returns for alpha={alpha}, got {r.size}"
        )
    k = max(1, math.ceil(alpha * r.size))
    worst = np.partition(r, k - 1)[:k]  # the k smallest returns
    return float(-worst.mean())


# --------------------------------------------------------------------------- #
# 2. Depth to spread
# --------------------------------------------------------------------------- #
def _clean_side(side, descending: bool) -> np.ndarray:
    """Return valid [price, size] rows, best first, dropping empty levels."""
    a = np.asarray(side, dtype=float)
    if a.size == 0:
        return np.empty((0, 2))
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"each book side must have shape (L, 2), got {a.shape}")
    ok = np.isfinite(a).all(axis=1) & (a[:, 0] > 0) & (a[:, 1] > 0)
    a = a[ok]
    order = np.argsort(-a[:, 0] if descending else a[:, 0], kind="stable")
    return a[order]


def depth_to_spread(book: Mapping, k: int = 5, price_scale: float = 1.0) -> float:
    """Resting depth near the touch divided by the bid-ask spread.

    depth  = total size over the best k levels of BOTH sides (shares)
    spread = best_ask - best_bid, divided by price_scale

    Parameters
    ----------
    book : mapping with keys "bids" and "asks", each (L, 2) [price, size]
        Prices in integer cents, sizes in shares. Levels with size <= 0,
        price <= 0 or NaN are treated as missing. Input order is not trusted:
        levels are re-sorted best first.
    k : int
        Number of levels per side counted in depth. Default 5 (A's "top k").
    price_scale : float
        Price units per reporting unit. 1.0 reports per cent; 100.0 reports
        per dollar (since prices are in cents).

    Returns
    -------
    float
        Units: shares per price unit (shares per cent by default).
        Range: (0, inf) when defined. Lower = thinner book relative to its
        spread = more fragile. NaN when either side is empty or the book is
        locked or crossed (spread <= 0), because the ratio is undefined.
    """
    if k < 1:
        raise ValueError("k must be >= 1")
    if price_scale <= 0:
        raise ValueError("price_scale must be positive")
    bids = _clean_side(book["bids"], descending=True)
    asks = _clean_side(book["asks"], descending=False)
    if len(bids) == 0 or len(asks) == 0:
        return float("nan")
    spread = (asks[0, 0] - bids[0, 0]) / price_scale
    if spread <= 0:
        return float("nan")
    depth = bids[:k, 1].sum() + asks[:k, 1].sum()
    return float(depth / spread)


# --------------------------------------------------------------------------- #
# 3. Spectral radius
# --------------------------------------------------------------------------- #
def spectral_radius(W) -> float:
    """Largest eigenvalue magnitude of an interaction matrix W.

    W[i, j] is the coupling of agent j's activity onto agent i (the exact
    estimator for W is defined in the Index spec). For comparing with May's
    stability criterion, W should be normalised so a radius of 1 is the
    stability threshold (self-regulation scaled to 1).

    Parameters
    ----------
    W : array-like, shape (n, n)
        Real square matrix, finite entries. Need not be symmetric.

    Returns
    -------
    float
        Units: dimensionless if W is dimensionless. Range: [0, inf).
        0 for the zero matrix. Larger = stronger mutual reaction = more
        fragile; values at or above 1 (after normalisation) indicate that
        interaction strength outpaces self-regulation.
    """
    M = np.asarray(W, dtype=float)
    if M.ndim != 2 or M.shape[0] != M.shape[1]:
        raise ValueError(f"W must be a square 2-D matrix, got shape {M.shape}")
    if M.shape[0] == 0:
        raise ValueError("W must be non-empty")
    if not np.isfinite(M).all():
        raise ValueError("W contains NaN or inf")
    return float(np.max(np.abs(np.linalg.eigvals(M))))


# --------------------------------------------------------------------------- #
# Helpers for Madhu's fixed-grid .npz layout: bids/asks shaped (T, L, 2)
# --------------------------------------------------------------------------- #
def mid_prices(bids, asks) -> np.ndarray:
    """Mid price per snapshot, in the same units as the input prices (cents).

    bids, asks : arrays of shape (T, L, 2). NaN where a side is empty.
    """
    bids = np.asarray(bids, dtype=float)
    asks = np.asarray(asks, dtype=float)
    if bids.ndim != 3 or asks.ndim != 3 or bids.shape[0] != asks.shape[0]:
        raise ValueError("bids and asks must be (T, L, 2) arrays with the same T")
    out = np.full(bids.shape[0], np.nan)
    for t in range(bids.shape[0]):
        b = _clean_side(bids[t], descending=True)
        a = _clean_side(asks[t], descending=False)
        if len(b) and len(a):
            out[t] = 0.5 * (b[0, 0] + a[0, 0])
    return out


def log_returns(mid: np.ndarray) -> np.ndarray:
    """Log differences of a mid-price series; NaN where either price is missing."""
    mid = np.asarray(mid, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.diff(np.log(mid))


def depth_to_spread_series(bids, asks, k: int = 5, price_scale: float = 1.0) -> np.ndarray:
    """depth_to_spread for every snapshot of a (T, L, 2) fixed-grid book."""
    bids = np.asarray(bids, dtype=float)
    asks = np.asarray(asks, dtype=float)
    if bids.ndim != 3 or asks.ndim != 3 or bids.shape[0] != asks.shape[0]:
        raise ValueError("bids and asks must be (T, L, 2) arrays with the same T")
    return np.array(
        [
            depth_to_spread({"bids": bids[t], "asks": asks[t]}, k=k, price_scale=price_scale)
            for t in range(bids.shape[0])
        ]
    )
