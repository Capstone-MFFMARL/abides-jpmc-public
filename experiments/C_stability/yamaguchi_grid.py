"""
Reduced replication of Yamaguchi (2025), "Emergent Coordination and Phase
Structure in Independent Multi-Agent Reinforcement Learning" (arXiv 2511.23315).

Part C of the capstone: density sweep on a small grid, stability index S per
condition.

What follows the paper
----------------------
* Square L x L grid, N agents and one goal, placed without overlap.
  Density rho = N / L^2, swept over {1/32, 1/16, 1/8, 1/4, 1/2}.
* 5 actions: stay, up, down, left, right.
* Reward: -0.005 per step, +1 for reaching the goal. An agent that reaches
  the goal "holds" there and no longer affects the dynamics (here it is
  removed from the grid).
* Episode ends when accumulated reward >= 0.8 N, or after 8 L steps.
* Learner: "independent" agents sharing one Double DQN, with a one-hot
  agent ID appended to each observation ([o || ID_i]).
  2x128 ReLU MLP, Huber loss, grad-norm clip 1.0, Adam lr 1.5e-4,
  gamma 0.95, Polyak tau 1e-3, replay 1e5, warm-up 1500, batch 64.
  epsilon: 1.0, x0.98 per episode, floor 0.01.
* v(L, rho) = variance of the TD error delta = y - Q(s, a) per training
  episode, averaged over the last 25% of episodes.
  S = 1 - v / v_max, v_max taken over ALL conditions in the sweep.
* CSR = fraction of greedy (epsilon = 0) evaluation episodes in which every
  agent reaches the goal.

What is reduced / assumed (the paper does not specify these)
------------------------------------------------------------
* L in {8, 12} instead of {8, 16, 24, 32}; 400 episodes instead of 1500;
  3 seeds instead of 50. Use --episodes / --seeds / --L to scale up.
* Base observation (not described in the paper): own (x, y), goal (x, y),
  offset to goal, and 4 bits for "neighbour cell blocked", all scaled to
  [0, 1] or [-1, 1].
* Collisions: agents move in a random order each step; a move into a wall or
  an occupied cell leaves the agent where it is. No collision penalty.
* One gradient update per environment step.

Run
---
    python yamaguchi_grid.py                       # full reduced sweep
    python yamaguchi_grid.py --L 8 --rho 0.0625 --episodes 50 --seeds 1   # smoke test
    python yamaguchi_grid.py --plot-only           # re-plot from saved CSVs
"""
from __future__ import annotations

import argparse
import os
import time
from dataclasses import dataclass
from multiprocessing import Pool

import numpy as np

from stability_index import stability_index

DENSITIES = [0.03125, 0.0625, 0.125, 0.25, 0.5]
MOVES = np.array([[0, 0], [0, 1], [0, -1], [-1, 0], [1, 0]])  # stay, up, down, left, right


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------
class GridGoalEnv:
    """N agents on an L x L grid race to a single goal cell."""

    def __init__(self, L: int, N: int, rng: np.random.Generator):
        self.L, self.N, self.rng = L, N, rng
        self.max_steps = 8 * L
        self.obs_dim = 10 + N

    def reset(self):
        cells = self.rng.choice(self.L * self.L, size=self.N + 1, replace=False)
        xy = np.stack([cells // self.L, cells % self.L], axis=1)
        self.goal = xy[0]
        self.pos = xy[1:].copy()
        self.active = np.ones(self.N, dtype=bool)
        self.t = 0
        self.total_reward = 0.0
        return self._obs()

    def _occupancy(self):
        occ = np.zeros((self.L, self.L), dtype=bool)
        p = self.pos[self.active]
        occ[p[:, 0], p[:, 1]] = True
        return occ

    def _obs(self):
        L, N = self.L, self.N
        occ = self._occupancy()
        o = np.zeros((N, self.obs_dim), dtype=np.float32)
        s = 1.0 / (L - 1)
        o[:, 0:2] = self.pos * s
        o[:, 2:4] = self.goal * s
        o[:, 4:6] = (self.goal - self.pos) * s
        for k, (dx, dy) in enumerate(MOVES[1:]):
            nx, ny = self.pos[:, 0] + dx, self.pos[:, 1] + dy
            out = (nx < 0) | (nx >= L) | (ny < 0) | (ny >= L)
            blocked = out.copy()
            inside = ~out
            blocked[inside] = occ[nx[inside], ny[inside]]
            o[:, 6 + k] = blocked
        o[np.arange(N), 10 + np.arange(N)] = 1.0  # one-hot agent ID
        return o

    def step(self, actions):
        """actions: (N,) ints. Returns obs, rewards, done_per_agent, episode_over."""
        self.t += 1
        was_active = self.active.copy()
        rewards = np.zeros(self.N, dtype=np.float32)
        rewards[was_active] = -0.005
        occ = self._occupancy()
        for i in self.rng.permutation(np.flatnonzero(was_active)):
            nxt = self.pos[i] + MOVES[actions[i]]
            if not (0 <= nxt[0] < self.L and 0 <= nxt[1] < self.L):
                continue
            if occ[nxt[0], nxt[1]]:
                continue
            occ[self.pos[i][0], self.pos[i][1]] = False
            self.pos[i] = nxt
            if (nxt == self.goal).all():
                rewards[i] += 1.0
                self.active[i] = False  # holds at goal, leaves the dynamics
            else:
                occ[nxt[0], nxt[1]] = True
        self.total_reward += rewards.sum()
        reached = was_active & ~self.active
        over = (self.total_reward >= 0.8 * self.N) or (self.t >= self.max_steps) or not self.active.any()
        return self._obs(), rewards, reached, over, was_active


# --------------------------------------------------------------------------
# Shared Double DQN
# --------------------------------------------------------------------------
@dataclass
class Hyper:
    lr: float = 1.5e-4
    gamma: float = 0.95
    tau: float = 1e-3
    buffer: int = 100_000
    warmup: int = 1_500
    batch: int = 64
    eps0: float = 1.0
    eps_decay: float = 0.98
    eps_min: float = 0.01
    hidden: int = 128


def run_condition(args):
    """Train one (L, rho, seed) condition. Returns (key, per-episode log, CSR)."""
    L, rho, seed, episodes, eval_eps, out_dir = args
    import torch
    import torch.nn as nn

    torch.set_num_threads(1)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    hp = Hyper()
    N = max(2, int(round(rho * L * L)))
    env = GridGoalEnv(L, N, rng)
    d = env.obs_dim

    def mlp():
        return nn.Sequential(nn.Linear(d, hp.hidden), nn.ReLU(),
                             nn.Linear(hp.hidden, hp.hidden), nn.ReLU(),
                             nn.Linear(hp.hidden, 5))

    q, q_tgt = mlp(), mlp()
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=hp.lr)
    huber = nn.SmoothL1Loss()

    B = hp.buffer
    S_ = np.zeros((B, d), np.float32); S2 = np.zeros((B, d), np.float32)
    A_ = np.zeros(B, np.int64); R_ = np.zeros(B, np.float32); D_ = np.zeros(B, np.float32)
    n_stored, ptr = 0, 0

    log = []
    eps = hp.eps0
    for ep in range(episodes):
        obs = env.reset()
        deltas, ep_ret, steps = [], 0.0, 0
        over = False
        while not over:
            with torch.no_grad():
                greedy = q(torch.from_numpy(obs)).argmax(1).numpy()
            explore = rng.random(N) < eps
            acts = np.where(explore, rng.integers(0, 5, N), greedy)
            obs2, r, reached, over, was_active = env.step(acts)
            ep_ret += r.sum()
            steps += 1
            idx = np.flatnonzero(was_active)
            for i in idx:  # store one transition per acting agent
                S_[ptr] = obs[i]; S2[ptr] = obs2[i]; A_[ptr] = acts[i]
                R_[ptr] = r[i]; D_[ptr] = float(reached[i])
                ptr = (ptr + 1) % B; n_stored = min(n_stored + 1, B)
            obs = obs2

            if n_stored >= hp.warmup:
                j = rng.integers(0, n_stored, hp.batch)
                s = torch.from_numpy(S_[j]); s2 = torch.from_numpy(S2[j])
                a = torch.from_numpy(A_[j]); rr = torch.from_numpy(R_[j]); dd = torch.from_numpy(D_[j])
                with torch.no_grad():
                    a2 = q(s2).argmax(1, keepdim=True)              # Double DQN: online picks
                    y = rr + hp.gamma * (1 - dd) * q_tgt(s2).gather(1, a2).squeeze(1)  # target evaluates
                qsa = q(s).gather(1, a.unsqueeze(1)).squeeze(1)
                loss = huber(qsa, y)
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(q.parameters(), 1.0)
                opt.step()
                with torch.no_grad():
                    for p, pt in zip(q.parameters(), q_tgt.parameters()):
                        pt.mul_(1 - hp.tau).add_(hp.tau * p)
                deltas.append((y - qsa).detach().numpy())
        td_var = float(np.var(np.concatenate(deltas))) if deltas else np.nan
        log.append((ep + 1, td_var, ep_ret / N, steps, int((~env.active).sum()), eps))
        eps = max(hp.eps_min, eps * hp.eps_decay)

    # greedy evaluation: success = every agent reaches the goal within 8L steps
    succ = 0
    for _ in range(eval_eps):
        obs = env.reset()
        for _t in range(env.max_steps):
            with torch.no_grad():
                acts = q(torch.from_numpy(obs)).argmax(1).numpy()
            obs, _, _, _, _ = env.step(acts)
            if not env.active.any():
                break
        succ += int(not env.active.any())
    csr = succ / eval_eps

    key = f"L{L}_rho{rho:g}_s{seed}"
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        np.savetxt(os.path.join(out_dir, key + ".csv"), np.array(log, dtype=float), delimiter=",",
                   header="episode,td_var,return_per_agent,steps,reached,epsilon", comments="", fmt="%.6g")
    return dict(L=L, rho=rho, N=N, seed=seed, log=np.array(log, dtype=float), csr=csr)


# --------------------------------------------------------------------------
# Sweep, S, and plots
# --------------------------------------------------------------------------
def summarise(results, tail=0.25):
    """v per run = mean td_var over the last `tail` of episodes; S uses a shared v_max."""
    import pandas as pd
    rows = []
    for r in results:
        tv = r["log"][:, 1]
        tv = tv[~np.isnan(tv)]
        k = max(1, int(len(tv) * tail))
        rows.append(dict(L=r["L"], rho=r["rho"], N=r["N"], seed=r["seed"],
                         v=float(np.mean(tv[-k:])), csr=r["csr"],
                         final_reached_frac=float(np.mean(r["log"][-k:, 4]) / r["N"])))
    df = pd.DataFrame(rows)
    df["S"] = stability_index(df["v"].values)  # v_max over every run in the sweep
    return df


def plot(df, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    g = df.groupby(["L", "rho"]).agg(S=("S", "mean"), S_sd=("S", "std"),
                                      csr=("csr", "mean"), csr_sd=("csr", "std"),
                                      reach=("final_reached_frac", "mean")).reset_index()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for L, sub in g.groupby("L"):
        ax[0].errorbar(sub.rho, sub.S, yerr=sub.S_sd.fillna(0), marker="o", capsize=3, label=f"L = {L}")
        ax[1].errorbar(sub.rho, sub.csr, yerr=sub.csr_sd.fillna(0), marker="o", capsize=3, label=f"L = {L}")
    for a, t in zip(ax, ["Stability index S = 1 - v / v_max", "Coordination success rate (greedy)"]):
        a.set_xscale("log", base=2)
        a.set_xticks(DENSITIES); a.set_xticklabels(["1/32", "1/16", "1/8", "1/4", "1/2"])
        a.set_xlabel("agent density  rho = N / L^2"); a.set_title(t)
        a.set_ylim(-0.05, 1.05); a.grid(alpha=0.3); a.legend()
    ax[0].set_ylabel("S (shared v_max across the sweep)")
    ax[1].set_ylabel("CSR")
    fig.suptitle("Reduced Yamaguchi replication: shared Double DQN, density sweep", y=1.02)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    return g


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--L", type=int, nargs="+", default=[8, 12])
    p.add_argument("--rho", type=float, nargs="+", default=DENSITIES)
    p.add_argument("--seeds", type=int, default=3)
    p.add_argument("--episodes", type=int, default=400)
    p.add_argument("--eval-episodes", type=int, default=20)
    p.add_argument("--workers", type=int, default=os.cpu_count() or 1)
    p.add_argument("--out", default="yamaguchi_runs")
    p.add_argument("--plot-only", action="store_true")
    a = p.parse_args()

    import pandas as pd
    summary_csv = os.path.join(a.out, "summary.csv")
    if a.plot_only:
        df = pd.read_csv(summary_csv)
    else:
        jobs = [(L, rho, s, a.episodes, a.eval_episodes, a.out)
                for L in a.L for rho in a.rho for s in range(1, a.seeds + 1)]
        t0 = time.time()
        with Pool(a.workers) as pool:
            results = []
            for r in pool.imap_unordered(run_condition, jobs):
                results.append(r)
                print(f"[{len(results)}/{len(jobs)}] L={r['L']} rho={r['rho']:g} N={r['N']} seed={r['seed']} "
                      f"CSR={r['csr']:.2f}  ({time.time() - t0:.0f}s)", flush=True)
        df = summarise(results)
        os.makedirs(a.out, exist_ok=True)
        df.to_csv(summary_csv, index=False)
    g = plot(df, os.path.join(a.out, "yamaguchi_density_sweep.png"))
    pd.set_option("display.width", 120)
    print(g.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
