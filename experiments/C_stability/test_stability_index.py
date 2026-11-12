import numpy as np
import pytest

from stability_index import (
    compare_runs,
    flag_unstable,
    make_growing,
    make_stable,
    stability_from_errors,
    stability_from_variances,
    stability_from_losses,
    stability_index,
    variance_growth,
    window_variance,
)

W = 100


def test_formula_matches_definition():
    s = stability_index([1.0, 2.0, 4.0])
    assert np.allclose(s, [0.75, 0.5, 0.0])


def test_shared_vmax_and_clipping():
    assert np.allclose(stability_index([1.0, 2.0], v_max=8.0), [0.875, 0.75])
    assert np.allclose(stability_index([10.0], v_max=5.0), [0.0])  # clipped, not negative


def test_zero_variance_is_fully_stable():
    assert np.allclose(stability_index([0.0, 0.0]), [1.0, 1.0])


def test_window_variance_known_values():
    v = window_variance([0, 2, 0, 2, 5, 5, 5, 5, 9], window=4)  # last point dropped
    assert np.allclose(v, [1.0, 0.0])


def test_stable_series_is_not_flagged():
    for seed in range(10):
        e = make_stable(seed=seed)
        assert 0.6 < variance_growth(e, W) < 1.6
        assert not flag_unstable(e, W)


def test_growing_series_is_flagged():
    for seed in range(10):
        e = make_growing(seed=seed)  # quarter medians sit at std 1.5 and 4.5: about 9x
        assert variance_growth(e, W) > 6
        assert flag_unstable(e, W)


def test_growing_series_S_falls_over_time():
    s = stability_from_errors(make_growing(), W)
    k = len(s) // 4
    assert s[:k].mean() > 0.8
    assert s[-k:].mean() < 0.5


def test_compare_runs_ranks_stable_above_growing():
    res = compare_runs({"stable": make_stable(), "growing": make_growing()}, W)
    assert res["stable"]["mean_S"] > 0.9
    assert res["stable"]["min_S"] > 0.85
    assert res["growing"]["mean_S"] < res["stable"]["mean_S"] - 0.2
    assert res["growing"]["min_S"] == 0.0


def test_losses_used_directly_as_variance():
    assert np.allclose(stability_from_losses([4.0, 2.0, 1.0]), [0.0, 0.5, 0.75])


def test_loss_smoothing_is_trailing_mean():
    s = stability_from_losses([4.0, 2.0, 0.0], smooth=2)  # v -> [4, 3, 1]
    assert np.allclose(s, [0.0, 0.25, 0.75])


@pytest.mark.parametrize("bad", [[], [1.0, float("nan")], [1.0, -1.0]])
def test_bad_variances_rejected(bad):
    with pytest.raises(ValueError):
        stability_index(bad)


def test_bad_window_rejected():
    with pytest.raises(ValueError):
        window_variance([1.0, 2.0, 3.0], window=1)
    with pytest.raises(ValueError):
        window_variance([1.0, 2.0, 3.0], window=10)


def test_normalised_variance_removes_reward_scale():
    # td_var grows only because rewards grow; td_var / return_var is flat.
    return_var = np.linspace(1.0, 100.0, 50)
    td_var = 0.1 * return_var
    s = stability_from_variances(td_var, return_var)
    assert np.allclose(s, 0.0) or np.allclose(s, 1.0)  # constant ratio -> no change in S
    raw = stability_from_variances(td_var)
    assert raw[0] > 0.9 and raw[-1] == 0.0  # raw variance would wrongly look unstable


def test_normalised_variance_detects_real_instability():
    return_var = np.full(50, 10.0)
    td_var = np.concatenate([np.full(40, 1.0), np.linspace(1.0, 8.0, 10)])
    s = stability_from_variances(td_var, return_var)
    assert s[:40].min() > 0.8 and s[-1] == 0.0


def test_normalisation_rejects_bad_scale():
    with pytest.raises(ValueError):
        stability_from_variances([1.0, 2.0], [1.0, 0.0])
    with pytest.raises(ValueError):
        stability_from_variances([1.0, 2.0], [1.0])
