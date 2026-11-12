# C: Stability index (Yamaguchi S)

Part C of the Phase 0 / review milestone.

| File | What it does |
|---|---|
| `stability_index.py` | S = 1 − v / v_max (Yamaguchi, arXiv 2511.23315), plus a single-run instability flag (`variance_growth`, `flag_unstable`). It is a library and also has a CLI. |
| `test_stability_index.py` | 17 tests covering the synthetic stable and growing-variance series, shared v_max, normalisation, and bad inputs. Run with `python -m pytest -q`. |
| `ppo_training_log_B.csv` | Copy of `experiments/B_kyle/ppo_training_log.csv` (250 PPO epochs). |
| `ppo_stability.png` | S over PPO epochs, using `td_var / return_var`. |
| `ppo_stability_raw.png` | The same S with raw `td_var`, for contrast. |
| `yamaguchi_grid.py` | Reduced Yamaguchi replication: grid world, density sweep, S per condition. |
| `yamaguchi_runs/` | Per-run training logs, `summary.csv`, `yamaguchi_density_sweep.png`. |

## 1. S on B's PPO log

```
python stability_index.py --log ppo_training_log_B.csv --column td_var --normalize-by return_var --out ppo_stability.png
```

| Epochs | Mean S | Mean profit |
|---|---|---|
| 1–10 | 0.50 | 1,505 |
| 11–50 | 0.79 | 5,071 |
| 51–100 | 0.81 | 9,423 |
| 101–250 | 0.79–0.80 | 12k–15k |

- Raw `td_var` grows about 7× from the first quarter of training to the last, which would wrongly flag the run as unstable.
- Divided by `return_var`, the growth is 1.03×. The raw growth is reward scale, not instability.
- PPO on the Kyle environment is therefore stable after about epoch 20.
- S is relative: within a single run the worst window is always 0, so the 0 at epoch 1 means "worst point of this run", not "crashed". Use `variance_growth` / `flag_unstable` for a single-run verdict.

## 2. Reduced Yamaguchi replication

```
python yamaguchi_grid.py --L 8 12 16 --episodes 600 --seeds 5     # about 30–60 min on 2 cores
python yamaguchi_grid.py --plot-only                               # re-plot from yamaguchi_runs/summary.csv
```

**Same as the paper:**
- L × L grid, N agents and one goal; density ρ = N/L² ∈ {1/32, 1/16, 1/8, 1/4, 1/2}.
- 5 actions. Reward −0.005 per step, +1 at the goal.
- Episode ends at 0.8N reward or after 8L steps.
- One parameter-shared Double DQN with a one-hot agent ID, using the paper's hyperparameters.
- v = per-episode TD-error variance over the last 25% of training. S uses v_max across the whole sweep.
- CSR = share of greedy episodes in which every agent reaches the goal.

**Reduced or assumed** (the paper doesn't specify these):
- L ∈ {8, 12, 16} instead of up to 32; 600 episodes instead of 1,500; 5 seeds instead of 50.
- Observation: own position, goal position, offset to the goal, and 4 blocked-neighbour bits, plus the ID.
- Moves resolve in a random order, and blocked moves stay put.
- One gradient update per environment step.

### Results (75 runs: L ∈ {8, 12, 16} × 5 densities × 5 seeds, 600 episodes each)

Mean over 5 seeds. S uses one v_max across all 75 runs.

| ρ | S, L=8 | S, L=12 | S, L=16 | CSR, L=8 | CSR, L=12 | CSR, L=16 |
|---|---|---|---|---|---|---|
| 1/32 | 0.81 | 0.89 | 0.89 | 0.97 | 0.96 | 0.92 |
| 1/16 | 0.78 | 0.82 | 0.88 | 0.96 | 0.88 | 0.78 |
| 1/8  | 0.62 | 0.74 | 0.77 | 0.99 | 0.90 | 0.81 |
| 1/4  | 0.36 | 0.41 | 0.55 | 0.97 | 0.81 | 0.78 |
| 1/2  | 0.04 | 0.41 | 0.61 | 0.94 | 0.70 | 0.60 |

- **S falls with density on every grid size.** Learning is least stable at high ρ, as Yamaguchi reports. Seed-to-seed spread is small (sd ≤ 0.04).
- **CSR falls with density on the larger grids.** On L=16 it drops from 0.92 to 0.60. On L=8 it stays high, because 8L steps is enough time even for a crowd.
- **On L=12 and L=16, S levels off or rises at ρ = 1/2 while CSR keeps falling.** This is the jammed regime: blocked agents stop learning much, so the TD variance is low and the run looks "stable" even though coordination fails. S alone would miss this; you need S and CSR together, which is how Yamaguchi classifies phases.
- **Not reproduced at this scale:** the double "instability ridge" (a non-monotone S peak around L=24, ρ≈1/16). That needs L ≥ 24 and around 1,500 episodes. The code supports it (`--L 24 --episodes 1500`), but it would need several hours of CPU.

Rewards here are bounded (−0.005 / +1), so `td_var` needs no normalisation, unlike the Kyle PPO log.
