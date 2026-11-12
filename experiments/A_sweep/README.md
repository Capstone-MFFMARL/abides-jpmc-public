# A_sweep: ABIDES agent-count sweep

Sweeps `num_mm` (market makers) in {1, 2, 5, 10, 20} x seeds 1-5 on `rmsc_density.py` (25 runs, 09:30-10:00 sim time).

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
- `sweep_results.csv`: one row per run: run_id (mm{num_mm}_s{seed}), num_mm, seed, runtime_s, messages, spread, depth, vol_bps, n_grid, n_valid
- `sweep_summary.csv` and `abides_sweep.png`: mean and std over seeds
- `snapshots/<run_id>.npz`: order-book snapshots on a fixed 10 s grid. Arrays: `times_ns` (T,), `bids` (T,10,2), `asks` (T,10,2). Each level is [price in cents, size]. Levels with size 0 are padding. Last-known-value resampling. Only snapshots with both sides non-empty are used.

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
