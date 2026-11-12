# Experiment A: ABIDES market-maker count sweep

Sweeps `num_mm ∈ {1, 2, 5, 10, 20}` × seeds `{1..5}` (25 runs) on `rmsc_density`
(30 simulated minutes, 09:30–10:00) and records runtime, message count and three order-book metrics.

## Reproduce

```bash
# 1. environment (Python 3.9 / 3.10 - the pinned numpy 1.22 + pomegranate 0.14.5 do not build on 3.12)
git clone https://github.com/Capstone-MFFMARL/abides-jpmc-public.git
cd abides-jpmc-public
git checkout density-work              # rmsc_density.py and experiments/ live on this branch, not main
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest matplotlib
pip install -e abides-core -e abides-markets     # editable, from THIS checkout

# 2. sanity check (imports rmsc_density, runs a short sim, runs the Kyle tests: expect "5 passed")
bash experiments/A_sweep/check_install.sh

# 3. full sweep: ~2.5 min on a laptop-class CPU (num_mm=20 dominates), writes CSV + 5 figures
cd experiments/A_sweep
python run_sweep.py

# optional
python run_sweep.py --plot-only        # redraw figures from results/sweep.csv
python run_sweep.py --resume           # continue an interrupted sweep
python run_sweep.py --end-time 16:00:00   # full trading day (much slower)
```

## Outputs

| file | content |
|---|---|
| `results/sweep.csv` | 25 rows: `num_mm, seed, runtime_s, n_messages, mean_spread, mean_depth, mid_volatility` |
| `figures/{runtime_s,n_messages,mean_spread,mean_depth,mid_volatility}.png` | mean ± std across seeds vs `num_mm`, individual seeds in grey |

## Metric definitions

All three book metrics come from the ExchangeAgent's `book_log2` (one snapshot per book change).
Because those timestamps are irregular, the log is resampled to a 10 s grid with
last-observation-carried-forward, so each metric is time-weighted.

- **mean_spread**: mean of best ask − best bid, in cents.
- **mean_depth**: mean shares resting in the top 5 levels of each side, bids + asks (`--depth-levels`).
- **mid_volatility**: std of log mid-price returns on the 10 s grid. Per interval, not annualised; depends on `--freq`.
- **runtime_s**: wall-clock of `kernel.run()` only. One discarded warm-up run precedes the sweep.
- **n_messages**: `kernel.ttl_messages`.

Seeds drive both the config (`build_config(seed=...)`) and the kernel RNG, so a given
`(num_mm, seed)` reproduces exactly (message counts match run to run); only `runtime_s` varies with the machine.
