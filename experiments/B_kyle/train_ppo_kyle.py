import os
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import csv

from kyle_env import KyleGame
from kyle_solver import solve_kyle

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)


class InsiderActorCritic(nn.Module):
    """Actor-Critic network for Insider agent.

    Input: [n / N (normalized step), (v - p) / v_std (normalized price
    mispricing)] Output: mean and std for Gaussian policy x_n ~ N(mean, std)
    """

    def __init__(self, hidden_dim=64):
        super().__init__()
        self.actor_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        self.log_std = nn.Parameter(torch.zeros(1))

        self.critic_net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, state):
        mean = self.actor_net(state)
        std = torch.exp(self.log_std).expand_as(mean)
        value = self.critic_net(state)
        return mean, std, value


def train_ppo():
    N = 20
    sigma_u = 50.0
    Sigma0 = 1e4
    tau = 1.0
    v_mean = 1000.0
    v_std = 100.0
    batch_size = 4096
    epochs = 250
    lr = 3e-4
    clip_eps = 0.2
    gamma = 0.99

    # Analytical Ground Truth
    sol = solve_kyle(N=N, sigma_u=sigma_u, Sigma0=Sigma0, tau=tau)
    beta_theory = sol["beta"]
    lam_theory = sol["lam"]

    env = KyleGame(
        M=1,
        N=N,
        sigma_u=sigma_u,
        v_mean=v_mean,
        v_std=v_std,
        p0=v_mean,
        batch=batch_size,
    )
    model = InsiderActorCritic(hidden_dim=64)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    print("=" * 60)
    print("Training PPO Insider Policy on KyleGame...")
    print("=" * 60)
    log_rows = []

    for epoch in range(epochs):
        env.reset()

        states_list = []
        actions_list = []
        log_probs_list = []
        rewards_list = []
        values_list = []

        # Collect 1 full multi-step trajectory across batch episodes
        for n in range(1, N + 1):
            mispricing = (env.v - env.p) / v_std
            norm_n = np.full(batch_size, n / N, dtype=np.float32)
            obs = np.column_stack([norm_n, mispricing]).astype(np.float32)

            obs_tensor = torch.tensor(obs, dtype=torch.float32)
            with torch.no_grad():
                mean, std, value = model(obs_tensor)
                dist = torch.distributions.Normal(mean, std)
                action = dist.sample()
                log_prob = dist.log_prob(action).sum(dim=-1)

            x_n = action.numpy().squeeze()
            p_new, r_insider, _, _, _ = env.step(
                x_n, depth_params=lam_theory[n]
            )

            states_list.append(obs_tensor)
            actions_list.append(action)
            log_probs_list.append(log_prob)
            rewards_list.append(torch.tensor(r_insider, dtype=torch.float32))
            values_list.append(value.squeeze())

        # Compute Monte Carlo Returns & Advantages
        returns_list = []
        g = torch.zeros(batch_size)
        for t in reversed(range(N)):
            g = rewards_list[t] + gamma * g
            returns_list.insert(0, g)

        states_b = torch.cat(states_list, dim=0)
        actions_b = torch.cat(actions_list, dim=0)
        old_log_probs_b = torch.cat(log_probs_list, dim=0)
        returns_b = torch.cat(returns_list, dim=0)
        values_b = torch.cat(values_list, dim=0)

        advantages_b = returns_b - values_b
        advantages_b = (advantages_b - advantages_b.mean()) / (
            advantages_b.std() + 1e-8
        )

        # PPO Update Loop
        for _ in range(4):
            mean, std, values_pred = model(states_b)
            dist = torch.distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions_b).sum(dim=-1)

            ratios = torch.exp(new_log_probs - old_log_probs_b)
            surr1 = ratios * advantages_b
            surr2 = (
                torch.clamp(ratios, 1.0 - clip_eps, 1.0 + clip_eps)
                * advantages_b
            )

            policy_loss = -torch.min(surr1, surr2).mean()
            value_loss = nn.MSELoss()(values_pred.squeeze(), returns_b)
            entropy_loss = -dist.entropy().mean()

            loss = policy_loss + 0.5 * value_loss + 0.01 * entropy_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            td = []
            for t in range(N):
                v_next = values_list[t + 1] if t + 1 < N else torch.zeros(batch_size)
                td.append(rewards_list[t] + gamma * v_next - values_list[t])
            td_var = torch.stack(td).var().item()
        log_rows.append((
            epoch + 1,
            torch.stack(rewards_list).sum(dim=0).mean().item(),
            value_loss.item(),
            td_var,
            returns_b.var().item(),
        ))
        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            mean_profit = (
                torch.stack(rewards_list).sum(dim=0).mean().item()
            )
            print(
                f"Epoch {epoch + 1:3d}/{epochs} | Loss: {loss.item():.4f} | Mean Insider Profit: {mean_profit:.1f}"
            )

    log_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ppo_training_log.csv")
    with open(log_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "mean_profit", "value_loss", "td_var", "return_var"])
        w.writerows(log_rows)
    print(f"Saved training log to {log_path}")
    # Evaluate Learned Policy vs Analytical Beta
    print("\nEvaluating Learned Policy vs Analytical Beta_n...")
    model.eval()
    learned_beta = []

    for n in range(1, N + 1):
        test_mispricing = 1.0  # (v - p) = v_std
        norm_n = n / N
        obs = torch.tensor([[norm_n, test_mispricing]], dtype=torch.float32)
        with torch.no_grad():
            mean, _, _ = model(obs)
            x_pred = mean.item()
            # x_n = beta_n * (v - p) => beta_n = x_n / v_std
            b_n = x_pred / v_std
            learned_beta.append(b_n)

    # Plot Comparison
    steps = np.arange(1, N + 1)
    plt.figure(figsize=(7, 4.5))
    plt.plot(
        steps,
        beta_theory[1:],
        "s-",
        color="darkgreen",
        label=r"Analytical $\beta_n$",
    )
    plt.plot(
        steps,
        learned_beta,
        "o--",
        color="darkorange",
        label=r"PPO Learned $\beta_n$",
    )
    plt.title("Insider Trading Intensity: PPO vs Theory")
    plt.xlabel("Step n")
    plt.ylabel(r"$\beta_n$")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.6)
    plt.tight_layout()

    out_file = os.path.join(os.path.dirname(__file__), "ppo_vs_kyle_beta.png")
    plt.savefig(out_file, dpi=300)
    print(f"Saved PPO evaluation plot to {out_file}")


if __name__ == "__main__":
    train_ppo()