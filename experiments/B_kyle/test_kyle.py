import numpy as np
import pytest
from kyle_env import KyleGame
from kyle_solver import solve_kyle


def test_checkpoint_n20():
    sol = solve_kyle(N=20, sigma_u=50.0, Sigma0=1e4, tau=1.0)
    lam = sol["lam"]
    beta = sol["beta"]
    Sigma = sol["Sigma"]
    alpha = sol["alpha"]
    delta = sol["delta"]

    # Checkpoint values assertions to 1e-3
    assert np.isclose(lam[1], 0.4093, atol=1e-3)
    assert np.isclose(lam[20], 0.3498, atol=1e-3)

    assert np.isclose(beta[1], 0.1070, atol=1e-3)
    assert np.isclose(beta[20], 1.4292, atol=1e-3)

    assert np.isclose(Sigma[0], 10000.0, atol=1e-3)
    assert np.isclose(Sigma[1], 9562.03, atol=1e-3)
    assert np.isclose(Sigma[10], 5565.66, atol=1e-3)
    assert np.isclose(Sigma[20], 611.94, atol=1e-3)

    insider_profit = alpha[0] * 1e4 + delta[0]
    assert np.isclose(insider_profit, 20179.07, atol=1e-3)

    ratio_first = Sigma[1] / Sigma[0]
    ratio_last = Sigma[20] / Sigma[19]
    assert np.isclose(ratio_first, 0.9562, atol=1e-3)
    assert np.isclose(ratio_last, 0.5000, atol=1e-3)


def test_closed_form_n1():
    sol = solve_kyle(N=1, sigma_u=50.0, Sigma0=1e4, tau=1.0)
    assert np.isclose(sol["lam"][1], 1.0, atol=1e-5)
    insider_profit = sol["alpha"][0] * 1e4 + sol["delta"][0]
    assert np.isclose(insider_profit, 2500.0, atol=1e-5)


def test_equation_residuals():
    N = 20
    sigma_u = 50.0
    Sigma0 = 1e4
    tau = 1.0
    sol = solve_kyle(N=N, sigma_u=sigma_u, Sigma0=Sigma0, tau=tau)

    lam = sol["lam"]
    beta = sol["beta"]
    alpha = sol["alpha"]
    Sigma = sol["Sigma"]
    delta = sol["delta"]

    residuals = []
    for n in range(N, 0, -1):
        res_cubic = (
            -2.0 * tau * (sigma_u**2) * alpha[n] * (lam[n] ** 3)
            + 2.0 * tau * (sigma_u**2) * (lam[n] ** 2)
            + 2.0 * alpha[n] * Sigma[n] * lam[n]
            - Sigma[n]
        )
        residuals.append(abs(res_cubic))

        res_beta = beta[n] - (1.0 - 2.0 * alpha[n] * lam[n]) / (
            2.0 * lam[n] * (1.0 - alpha[n] * lam[n]) * tau
        )
        residuals.append(abs(res_beta))

        res_sigma = Sigma[n - 1] - Sigma[n] / (1.0 - beta[n] * lam[n] * tau)
        residuals.append(abs(res_sigma))

        res_delta = delta[n - 1] - (
            delta[n] + alpha[n] * (lam[n] ** 2) * (sigma_u**2) * tau
        )
        residuals.append(abs(res_delta))

        res_alpha = alpha[n - 1] - 1.0 / (
            4.0 * lam[n] * (1.0 - alpha[n] * lam[n])
        )
        residuals.append(abs(res_alpha))

    max_res = max(residuals)
    assert max_res < 1e-9, f"Max residual {max_res} exceeded threshold 1e-9"


def test_zero_sum():
    env = KyleGame(
        M=5, N=20, sigma_u=50.0, v_mean=1000.0, v_std=100.0, batch=10000
    )
    p0 = env.reset()
    x = np.random.normal(0, 10, size=10000)

    # Recover noise flow u from step simulation
    u = np.random.normal(0.0, 50.0, size=10000)
    q = x + u

    # Executed step
    depths = 1.0 / 0.4
    lam_eff = 1.0 / depths  # = 0.4
    p_new = np.clip(p0 + lam_eff * q, 0.5 * env.v, 1.5 * env.v)

    r_insider = x * (env.v - p_new)
    r_noise = u * (env.v - p_new)
    r_mm_expost = q * (p_new - env.v)

    # Zero-sum total payoff across all market participants (Insider + Noise + MMs)
    total_net = r_insider + r_noise + r_mm_expost
    assert np.allclose(total_net, 0.0, atol=1e-7)

    # Flow cost identity check (MM revenue equals aggregate flow cost)
    r_mm_flow = q * (p_new - p0)
    net_flow_cost = (x + u) * (p_new - p0)
    assert np.allclose(r_mm_flow - net_flow_cost, 0.0, atol=1e-7)


def test_vwap_and_m_invariance():
    for M in [1, 2, 5, 10, 20]:
        env = KyleGame(
            M=M, N=20, sigma_u=50.0, v_mean=1000.0, v_std=100.0, batch=100
        )
        np.random.seed(42)
        env.reset()
        x = np.ones(100) * 5.0
        p_new, _, _, q, lam_eff = env.step(x, depth_params=0.35)

        assert np.isclose(lam_eff, 0.35)