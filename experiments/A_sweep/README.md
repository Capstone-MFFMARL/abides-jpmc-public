# A_sweep: ABIDES agent-count sweep

Sweeps `num_mm` (market makers) in {1, 2, 5, 10, 20} x seeds 1-5 on `rmsc_density.py` (25 runs, 09:30-10:00 sim time).

## Reproduce

```bash
# 1. Environment setup (Python 3.9 / 3.10)
git clone https://github.com/Capstone-MFFMARL/abides-jpmc-public.git
cd abides-jpmc-public
git checkout sweep-a-unified
python3.9 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt pytest matplotlib
pip install -e abides-core -e abides-markets

# 2. Sanity check (rmsc_density import, short sim, 5 Kyle tests)
bash experiments/A_sweep/check_install.sh

# 3. Full sweep & plots
cd experiments/A_sweep
python run_sweep.py
python plot_sweep.py
```

## Run
Environment (no gym or ray needed):
    uv python install 3.9
    uv venv --python-preference only-managed --python 3.9 .venv-abides
    source .venv-abides/bin/activate
    uv pip install "cython<3" wheel "setuptools<66"
    uv pip install "numpy==1.23.5" "pandas==1.5.3" "scipy==1.10.1" psutil tqdm "coloredlogs==15.0.1" matplotlib
    uv pip install --no-build-isolation "pomegranate==0.14.8"
    uv pip install --no-deps -e abides-core -e abides-markets   (from the repo root)
Then, in experiments/A_sweep: `python run_sweep.py` then `python plot_sweep.py`.

## Outputs
- `sweep_results.csv`: one row per run: `run_id` (mm{num_mm}_s{seed}), `num_mm`, `seed`, `runtime_s`, `messages`, `spread`, `depth`, `vol_bps`, `n_grid`, `n_valid`, `warmup`
- `sweep_summary.csv` and `abides_sweep.png`: mean and std over seeds
- `snapshots/<run_id>.npz`: order-book snapshots on a fixed 10 s grid. Arrays: `times_ns` (T,), `bids` (T,10,2), `asks` (T,10,2). Each level is [price in cents, size]. Levels with size 0 are padding. Last-known-value resampling. Only snapshots with both sides non-empty are used.

## Metric definitions
All metrics are derived from the order book L2 snapshots resampled onto a fixed regular grid:
- **Grid**: Clock-aligned 10 s steps (`grid_ns = 10 * 10**9` ns), starting at `ceil(first valid quote time to grid)` and ending at market close (`10:00:00` or `--end-time`).
- **Warm-up**: By default `--warmup 09:31:00` (sim time). Spread, depth, and volatility are computed exclusively on grid points at or after this timestamp. Full grid snapshots (from market open) are preserved in the `.npz` files. `n_grid` reflects the count of post-warmup grid points used.
- **Spread**: Best ask minus best bid in cents, time-weighted average across post-warmup grid points.
- **Depth**: Total shares resting in the top 5 levels (`top_k = 5`) of both sides (bids + asks), averaged across post-warmup grid points.
- **Volatility (`vol_bps`)**: Sample standard deviation with `ddof=1` of log returns of the mid-price on the 10 s grid, scaled by `1e4` (basis points per 10 s interval).

## Definitions
- Prices are in cents.
- spread = best ask - best bid, averaged over the 10 s grid.
- depth = total size over the top 5 levels of both sides, averaged over the grid.
- vol_bps = std of log mid-price returns on the 10 s grid, times 1e4.
- Interval: fixed 10 s grid (GRID_NS). The index should be computed on the same grid.

## Convention for all experiments
Every output table carries `run_id` and `seed` columns so S (C) and the index (D) can be overlaid per run.

## Baseline finding (fixed-rule market makers)
More market makers: lower spread, lower volatility, higher depth (sharply at 20). The fragility index should not rise with market-maker count in this regime.

## Not available yet
Per-agent signed order flow. ABIDES order logs give ~19% of executed-order rows a timestamp of 0 (most of them from noise agents), so they cannot be time-binned until that is handled (Phase 1).
