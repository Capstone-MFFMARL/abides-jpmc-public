import os
import matplotlib.pyplot as plt
import numpy as np
from kyle_env import KyleGame
from kyle_solver import solve_kyle


def main():
    np.random.seed(42)
    N = 20
    sigma_u = 50.0
    Sigma0 = 1e4
    tau = 1.0
    batch_size = 400000

    # Solve analytical Kyle model
    sol = solve_kyle(N=N, sigma_u=sigma_u, Sigma0=Sigma0, tau=tau)
    lam = sol["lam"]
    beta = sol["beta"]
    Sigma = sol["Sigma"]
    alpha = sol["alpha"]
    delta = sol["delta"]

    # 1. Fixed policies MC Run (M=1)
    env = KyleGame(
        M=1,
        N=N,
        sigma_u=sigma_u,
        v_mean=1000.0,
        v_std=100.0,
        p0=1000.0,
        batch=batch_size,
    )
    env.reset()

    mc_var_v_minus_p = np.zeros(N + 1)
    mc_var_v_minus_p[0] = np.mean((env.v - env.p) ** 2)

    total_insider_profit = np.zeros(batch_size)

    for n in range(1, N + 1):
        x_n = beta[n] * (env.v - env.p)
        p_new, r_insider, r_mm, q, lam_eff = env.step(x_n, depth_params=lam[n])
        total_insider_profit += r_insider
        mc_var_v_minus_p[n] = np.mean((env.v - env.p) ** 2)

    mean_insider_profit = np.mean(total_insider_profit)

    # Print Headline MC vs Theory Review Table
    print("=" * 60)
    print("MC vs Theory Comparison (400k Episodes):")
    print("-" * 60)
    print(
        f"mean (v - p_1)^2 : MC = {mc_var_v_minus_p[1]:.1f} | Theory = {Sigma[1]:.1f}"
    )
    print(
        f"mean (v - p_10)^2: MC = {mc_var_v_minus_p[10]:.1f} | Theory = {Sigma[10]:.1f}"
    )
    print(
        f"mean (v - p_20)^2: MC = {mc_var_v_minus_p[20]:.1f} | Theory = {Sigma[20]:.1f}"
    )
    theory_profit = alpha[0] * Sigma0 + delta[0]
    print(
        f"Insider Profit   : MC = {mean_insider_profit:.1f} | Theory = {theory_profit:.1f}"
    )
    print("=" * 60)

    # 2. M-sweep check (M = 1, 2, 5, 10, 20)
    print("\nM-Sweep Invariance Check:")
    for M_val in [1, 2, 5, 10, 20]:
        env_m = KyleGame(
            M=M_val,
            N=N,
            sigma_u=sigma_u,
            v_mean=1000.0,
            v_std=100.0,
            p0=1000.0,
            batch=10000,
        )
        np.random.seed(42)
        env_m.reset()
        profit_m = np.zeros(10000)
        for n in range(1, N + 1):
            x_n = beta[n] * (env_m.v - env_m.p)
            p_new, r_insider, _, _, _ = env_m.step(x_n, depth_params=lam[n])
            profit_m += r_insider
        print(f"M = {M_val:2d} | Mean Insider Profit = {np.mean(profit_m):.1f}")

    # 3. Table-1-style stats evaluation
    print("\nTable-1-style Stats Evaluation (Fixed v=1000, p0=700 & 1300):")
    for start_p0 in [700.0, 1300.0]:
        num_repeats = 2000
        episodes_per_repeat = 30

        phis, half_lives, lambdas = [], [], []

        for _ in range(num_repeats):
            env_stat = KyleGame(
                M=1,
                N=N,
                sigma_u=sigma_u,
                v_mean=1000.0,
                v_std=100.0,
                p0=start_p0,
                batch=episodes_per_repeat,
            )
            env_stat.reset(v_fixed=1000.0, p0_fixed=start_p0)

            dev_history = np.zeros((episodes_per_repeat, N + 1))
            dp_history = np.zeros((episodes_per_repeat, N))
            q_history = np.zeros((episodes_per_repeat, N))

            dev_history[:, 0] = env_stat.v - env_stat.p

            for n in range(1, N + 1):
                p_prev = env_stat.p.copy()
                x_n = beta[n] * (env_stat.v - env_stat.p)
                p_new, _, _, q, _ = env_stat.step(x_n, depth_params=lam[n])

                dev_history[:, n] = env_stat.v - p_new
                dp_history[:, n - 1] = p_new - p_prev
                q_history[:, n - 1] = q

            dev_t = dev_history[:, :-1].flatten()
            dev_t1 = dev_history[:, 1:].flatten()

            num_phi = np.sum(dev_t * dev_t1)
            den_phi = np.sum(dev_t**2)
            phi = num_phi / den_phi if den_phi != 0 else 0.0

            hl = -np.log(2.0) / np.log(abs(phi)) if 0 < abs(phi) < 1 else np.nan

            q_flat = q_history.flatten()
            dp_flat = dp_history.flatten()
            reg_lam = (
                np.sum(q_flat * dp_flat) / np.sum(q_flat**2)
                if np.sum(q_flat**2) != 0
                else 0.0
            )

            phis.append(phi)
            half_lives.append(hl)
            lambdas.append(reg_lam)

        phi_mean = np.mean(phis)
        hl_mean = np.nanmean(half_lives)
        lam_mean = np.mean(lambdas)
        print(
            f"p0 = {start_p0:.0f} -> phi: {phi_mean:.3f} (5-95%: {np.percentile(phis, 5):.3f}-{np.percentile(phis, 95):.3f}) | half-life: {hl_mean:.2f} | lambda: {lam_mean:.2f}"
        )

    # 4. Generate & Save Figures (Path-independent resolution)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    steps = np.arange(1, N + 1)

    axes[0].plot(steps, lam[1:], "o-", color="navy", label=r"$\lambda_n$")
    axes[0].set_title(r"Kyle Price Impact ($\lambda_n$)")
    axes[0].set_xlabel("Step n")
    axes[0].set_ylabel(r"$\lambda_n$")
    axes[0].grid(True, linestyle="--", alpha=0.6)

    axes[1].plot(steps, beta[1:], "s-", color="darkgreen", label=r"$\beta_n$")
    axes[1].set_title(r"Insider Trading Intensity ($\beta_n$)")
    axes[1].set_xlabel("Step n")
    axes[1].set_ylabel(r"$\beta_n$")
    axes[1].grid(True, linestyle="--", alpha=0.6)

    all_steps = np.arange(0, N + 1)
    axes[2].plot(
        all_steps,
        Sigma,
        "d-",
        color="crimson",
        label=r"Theory $\Sigma_n$",
        linewidth=2,
    )
    axes[2].plot(
        all_steps,
        mc_var_v_minus_p,
        "x--",
        color="black",
        label="MC Mean $(v - p_n)^2$",
        markersize=7,
    )
    axes[2].set_title(r"Uncertainty Decay ($\Sigma_n$) Theory vs MC")
    axes[2].set_xlabel("Step n")
    axes[2].set_ylabel(r"$\Sigma_n$")
    axes[2].legend()
    axes[2].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    output_path = os.path.join(os.path.dirname(__file__), "kyle_theory_vs_mc.png")
    plt.savefig(output_path, dpi=300)
    print(f"\nSaved figure to {output_path}")


if __name__ == "__main__":
    main()