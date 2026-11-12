"""Unit tests for fragility_index.py. Run with:  pytest test_fragility_index.py -v"""

import numpy as np
import pytest

from fragility_index import (
    cvar,
    depth_to_spread,
    depth_to_spread_series,
    log_returns,
    mid_prices,
    spectral_radius,
)


# ------------------------------- cvar ------------------------------------- #
def test_cvar_known_value_k1():
    r = np.arange(-10, 10) / 100.0          # 20 values, worst = -0.10
    assert cvar(r, alpha=0.05) == pytest.approx(0.10)


def test_cvar_known_value_k2():
    r = np.arange(-10, 10) / 100.0          # worst two: -0.10, -0.09
    assert cvar(r, alpha=0.10) == pytest.approx(0.095)


def test_cvar_order_invariant():
    rng = np.random.default_rng(0)
    r = rng.normal(0, 0.01, 500)
    assert cvar(r) == pytest.approx(cvar(rng.permutation(r)))


def test_cvar_drops_nan():
    r = np.arange(-10, 10) / 100.0
    r_nan = np.concatenate([r, [np.nan, np.inf]])
    assert cvar(r_nan, alpha=0.05) == pytest.approx(0.10)


def test_cvar_fat_tails_exceed_thin_tails():
    rng = np.random.default_rng(1)
    thin = rng.normal(0, 0.01, 5000)
    fat = rng.standard_t(df=3, size=5000) * 0.01 / np.sqrt(3)  # same variance
    assert cvar(fat) > cvar(thin)


def test_cvar_monotone_in_alpha():
    rng = np.random.default_rng(2)
    r = rng.normal(0, 0.01, 5000)
    assert cvar(r, 0.01) >= cvar(r, 0.05) >= cvar(r, 0.20)


def test_cvar_flags_stressed_series():
    rng = np.random.default_rng(3)
    calm = rng.normal(0, 0.001, 179)
    stressed = calm.copy()
    stressed[-10:] -= 0.02                  # crash at the end
    assert cvar(stressed) > 5 * cvar(calm)


def test_cvar_too_few_samples_raises():
    with pytest.raises(ValueError):
        cvar(np.zeros(5), alpha=0.05)


@pytest.mark.parametrize("alpha", [0.0, 1.0, -0.1, 1.5])
def test_cvar_bad_alpha_raises(alpha):
    with pytest.raises(ValueError):
        cvar(np.zeros(100), alpha=alpha)


# --------------------------- depth_to_spread ------------------------------- #
def _book():
    return {
        "bids": [[10000, 100], [9999, 200]],
        "asks": [[10002, 50], [10003, 150]],
    }


def test_depth_to_spread_known_value():
    # depth = 100+200+50+150 = 500, spread = 2 cents
    assert depth_to_spread(_book(), k=2) == pytest.approx(250.0)


def test_depth_to_spread_top_k_only():
    # k=1: depth = 100+50 = 150, spread = 2
    assert depth_to_spread(_book(), k=1) == pytest.approx(75.0)


def test_depth_to_spread_price_scale_dollars():
    # spread = 0.02 dollars -> 500 / 0.02
    assert depth_to_spread(_book(), k=2, price_scale=100.0) == pytest.approx(25000.0)


def test_depth_to_spread_wider_spread_lowers_ratio():
    wide = {"bids": [[10000, 100]], "asks": [[10010, 100]]}
    tight = {"bids": [[10000, 100]], "asks": [[10001, 100]]}
    assert depth_to_spread(tight) > depth_to_spread(wide)


def test_depth_to_spread_thinner_depth_lowers_ratio():
    thick = {"bids": [[10000, 1000]], "asks": [[10002, 1000]]}
    thin = {"bids": [[10000, 10]], "asks": [[10002, 10]]}
    assert depth_to_spread(thick) > depth_to_spread(thin)


def test_depth_to_spread_ignores_zero_padded_levels():
    bids = np.zeros((10, 2), dtype=np.int64)
    asks = np.zeros((10, 2), dtype=np.int64)
    bids[:2] = [[10000, 100], [9999, 200]]
    asks[:2] = [[10002, 50], [10003, 150]]
    assert depth_to_spread({"bids": bids, "asks": asks}, k=10) == pytest.approx(250.0)


def test_depth_to_spread_unsorted_input():
    shuffled = {"bids": [[9999, 200], [10000, 100]], "asks": [[10003, 150], [10002, 50]]}
    assert depth_to_spread(shuffled, k=2) == pytest.approx(250.0)


def test_depth_to_spread_missing_side_is_nan():
    assert np.isnan(depth_to_spread({"bids": [[10000, 100]], "asks": []}))
    assert np.isnan(depth_to_spread({"bids": np.zeros((10, 2)), "asks": [[10002, 5]]}))


def test_depth_to_spread_crossed_or_locked_is_nan():
    locked = {"bids": [[10000, 100]], "asks": [[10000, 100]]}
    crossed = {"bids": [[10005, 100]], "asks": [[10000, 100]]}
    assert np.isnan(depth_to_spread(locked))
    assert np.isnan(depth_to_spread(crossed))


def test_depth_to_spread_bad_shape_raises():
    with pytest.raises(ValueError):
        depth_to_spread({"bids": [[1, 2, 3]], "asks": [[4, 5]]})


# ------------------------------ spectral_radius ---------------------------- #
def test_spectral_radius_identity():
    assert spectral_radius(np.eye(4)) == pytest.approx(1.0)


def test_spectral_radius_zero_matrix():
    assert spectral_radius(np.zeros((3, 3))) == 0.0


def test_spectral_radius_diagonal_uses_magnitude():
    assert spectral_radius(np.diag([0.5, -2.0, 1.0])) == pytest.approx(2.0)


def test_spectral_radius_symmetric_swap():
    assert spectral_radius([[0, 1], [1, 0]]) == pytest.approx(1.0)


def test_spectral_radius_complex_eigenvalues():
    r, th = 0.7, 0.9
    R = r * np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    assert spectral_radius(R) == pytest.approx(r)


def test_spectral_radius_scaling():
    rng = np.random.default_rng(4)
    A = rng.normal(size=(6, 6))
    assert spectral_radius(-3.0 * A) == pytest.approx(3.0 * spectral_radius(A))


def test_spectral_radius_identical_agents_control():
    # n identical agents, equal coupling a between every pair: radius = a*(n-1)
    n, a = 5, 0.2
    W = a * (np.ones((n, n)) - np.eye(n))
    assert spectral_radius(W) == pytest.approx(a * (n - 1))


def test_spectral_radius_grows_with_coupling():
    W0 = 0.1 * (np.ones((4, 4)) - np.eye(4))
    assert spectral_radius(3 * W0) > spectral_radius(W0)


@pytest.mark.parametrize("bad", [np.ones((2, 3)), np.ones(4), np.array([[1.0, np.nan], [0, 1]])])
def test_spectral_radius_bad_input_raises(bad):
    with pytest.raises(ValueError):
        spectral_radius(bad)


# ------------------------------ helpers (npz layout) ----------------------- #
def _grid(T=30, spread=2, size=100):
    bids = np.zeros((T, 10, 2), dtype=np.int64)
    asks = np.zeros((T, 10, 2), dtype=np.int64)
    for lvl in range(10):
        bids[:, lvl] = [10000 - lvl, size]
        asks[:, lvl] = [10000 + spread + lvl, size]
    return bids, asks


def test_mid_prices_and_returns():
    bids, asks = _grid()
    mid = mid_prices(bids, asks)
    assert mid[0] == pytest.approx(10001.0)
    assert np.allclose(log_returns(mid), 0.0)


def test_mid_prices_missing_side_gives_nan():
    bids, asks = _grid()
    asks[3] = 0
    mid = mid_prices(bids, asks)
    assert np.isnan(mid[3])
    assert np.isnan(log_returns(mid)[2]) and np.isnan(log_returns(mid)[3])


def test_depth_to_spread_series_flags_stress():
    calm_b, calm_a = _grid(spread=2, size=100)
    stress_b, stress_a = _grid(spread=20, size=5)
    calm = depth_to_spread_series(calm_b, calm_a, k=5)
    stress = depth_to_spread_series(stress_b, stress_a, k=5)
    assert np.all(calm > 10 * stress)


def test_depth_to_spread_series_shape_mismatch_raises():
    bids, asks = _grid()
    with pytest.raises(ValueError):
        depth_to_spread_series(bids, asks[:-1])
