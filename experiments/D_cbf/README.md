# Part D — Fragility Index & Toy CBF

**Capstone:** Endogenous Market Stability & System Risk in MARL Systems  
**Part D author:** Anirudh  
**Guide:** Dr. Bhaskarjyoti Das, PES University B.Tech CSE (pw26_bjd_07)

---

## Overview

This folder implements **Part D** of the capstone: the building blocks of a **Fragility Index** for a multi-agent limit-order-book market, plus an optional **discrete-time Control Barrier Function (CBF)** safety filter.

The market is simulated via [ABIDES-JPMC](https://github.com/jpmorganchase/abides-jpmc-public) with MAPPO learning agents. Part D's job is to quantify *how fragile* the market is at any point, using three complementary signals:

| Term | What it measures | High = |
|---|---|---|
| **CVaR(5%)** | Tail loss of mid-price log-returns | More fragile |
| **Depth-to-Spread (D/S)** | Resting book depth ÷ bid-ask spread | Less fragile (invert to composite) |
| **Spectral Radius ρ(W)** | Largest eigenvalue of agent-interaction matrix | More fragile |

Together these form the **Composite Liquidity & Contagion Index (CLCI)** — the early-warning signal for liquidity collapse. See [`index_spec.md`](index_spec.md) for the full formal specification.

---

## File Structure

```
D_cbf/
├── fragility_index.py          # Core library — three pure functions + .npz helpers
├── test_fragility_index.py     # 37 unit tests (pytest)
├── npz_analysis.py             # Runs CVaR + D/S on Part A's real sweep data
├── cbf_toy.py                  # Toy scalar CBF safety filter demonstration
├── index_spec.md               # Formal spec: units, ranges, W definition
├── npz_analysis_results.csv    # (generated) per-file results table
└── cbf_toy_plot.png            # (generated) CBF trajectory plot
```

---

## Dependencies

```
numpy
pytest
matplotlib   # only needed for cbf_toy.py --plot
```

Install with:
```powershell
pip install numpy pytest matplotlib
```

---

## Scripts

### 1. `fragility_index.py` — Core Library

The main building block. **Not run directly** — imported by the other scripts and by downstream phases.

**Functions:**

| Function | Signature | Description |
|---|---|---|
| `cvar` | `(returns, alpha=0.05)` | CVaR at the α% tail — positive = loss |
| `depth_to_spread` | `(book, k=5, price_scale=1.0)` | Depth ÷ spread for one snapshot |
| `spectral_radius` | `(W)` | Largest eigenvalue magnitude of interaction matrix W |
| `mid_prices` | `(bids, asks)` | Mid-price series from a `(T, L, 2)` book array |
| `log_returns` | `(mid)` | Log differences of a mid-price series |
| `depth_to_spread_series` | `(bids, asks, k, price_scale)` | D/S for every snapshot in a `(T, L, 2)` book |

**Book format** (matches Part A's Madhu `.npz` layout):
```
book = {"bids": array (L, 2), "asks": array (L, 2)}
# rows: [price_in_cents, size_in_shares], best level first
# zero-padded levels (price=0 or size=0) are ignored automatically
```

---

### 2. `test_fragility_index.py` — Unit Tests

**Run:**
```powershell
cd experiments\D_cbf
python -m pytest test_fragility_index.py -v
```

**Expected output:**
```
collected 37 items

test_fragility_index.py::test_cvar_known_value_k1              PASSED
test_fragility_index.py::test_cvar_known_value_k2              PASSED
...
test_fragility_index.py::test_depth_to_spread_series_flags_stress PASSED

======================== 37 passed in 0.76s ========================
```

**What is tested:**
- `cvar`: known values, order-invariance, NaN dropping, fat vs thin tails, monotonicity in α, stressed vs calm series, error on bad input
- `depth_to_spread`: known value, top-k logic, price scaling, zero-padded levels, unsorted input, missing side → NaN, locked/crossed book → NaN
- `spectral_radius`: identity, zero matrix, diagonal, complex eigenvalues, scaling, degenerate identical-agent control case
- Helpers: mid-price from real `.npz` layout, NaN propagation, series shape mismatch

---

### 3. `npz_analysis.py` — Real Data Analysis

Loads all 25 `.npz` snapshot files from Part A's sweep (`A_sweep/snapshots/`), runs CVaR and D/S on each, and checks the expected monotonicity with market-maker count.

**Run:**
```powershell
cd experiments\D_cbf
python npz_analysis.py                        # auto-discovers ../A_sweep/snapshots/
python npz_analysis.py path\to\snapshots\     # explicit path
```

**Expected output:**
```
Analysing 25 snapshots from: ..\A_sweep\snapshots
Settings: alpha=0.05, k_levels=5, price_scale=100.0 (shares per dollar)

File               mm  seed    T    CVaR(5%)    mean D/S   std D/S
------------------------------------------------------------------------
mm1_s1.npz          1     1  179    0.000129  134079.4492  77995.6921
...
mm20_s5.npz        20     5  180    0.000032  202220.2381  70078.4460

num_mm    CVaR mean   CVaR std     D/S mean    D/S std    n
------------------------------------------------------------
     1     0.000098   0.000024  128609.9229 13716.8889    5
     2     0.000080   0.000020  133304.3624 14676.0410    5
     5     0.000053   0.000016  144682.3517 15613.8074    5
    10     0.000040   0.000007  163316.1949 14808.4781    5
    20     0.000031   0.000003  216251.0702 13273.8196    5

--- Sanity checks ---
D/S rises with num_mm (less fragile as MMs added): PASS ✓
CVaR does not rise with num_mm:                     PASS ✓

Results saved to: npz_analysis_results.csv
```

**How to interpret:**
- As `num_mm` grows from 1 → 20, **D/S rises** (deeper book relative to spread = more liquid = less fragile) and **CVaR falls** (smaller tail losses). This is the expected result from Phase 0 and confirms the functions are directionally correct on real data.
- Results are saved to `npz_analysis_results.csv` for use in the report or Phase 3 overlays.

> **Note:** `spectral_radius(W)` is not run here — it requires per-agent signed flow data, which needs `--save-raw` from Part A (not yet available).

---

### 4. `cbf_toy.py` — Toy Scalar CBF Safety Filter

Demonstrates a **discrete-time Control Barrier Function** on a minimal 1D system. This motivates Phase 4, where the CLCI drives the CBF to constrain agent actions before a liquidity collapse.

**System:**
```
x[k+1] = x[k] + u[k]          state = fragility level
h(x)   = x_max - x ≥ 0        safe set: stay below ceiling x_max = 0.8
u_nom  = +0.05                 nominal control pushes toward the boundary
```

**CBF filter** (Agrawal & Sreenath 2017, discrete-time):
```
u* = min(u_nom, γ · h(x))     γ = 0.1
```
This clips the control so the barrier decays by at most `γ` per step, keeping the state inside the safe set forever.

**Run:**
```powershell
cd experiments\D_cbf
python cbf_toy.py              # console table only
python cbf_toy.py --plot       # console table + saves cbf_toy_plot.png
```

**Expected output (truncated):**
```
CBF toy simulation
  System:   x[k+1] = x[k] + u[k]
  Safe set: h(x) = 0.8 - x >= 0
  Nominal:  u_nom = 0.05
  gamma   = 0.1
  x0      = 0.2,  steps = 40

   k    x (filtered)    h (filtered)    x (unfiltered)    safe?
-----------------------------------------------------------------
   0        0.200000        0.600000          0.200000      YES
   4        0.395000        0.405000          0.400000      YES
  12        0.625661        0.174339          0.800000      YES   ← unfiltered hits x_max here
  ...
  40        0.790876        0.009124          2.200000      YES

Filtered max state:   0.790876  (x_max = 0.8)
Safety constraint satisfied: YES ✓
Unfiltered max state: 2.200000  (crosses boundary: YES)
```

**The plot** (`cbf_toy_plot.png`) shows two trajectories: the unfiltered one crossing `x_max` at step 12 and the filtered one asymptotically approaching but never crossing it.

---

## Index Specification

See [`index_spec.md`](index_spec.md) for:
- Formal units and ranges for all three terms
- W matrix definition (primary OLS estimator + correlation fallback)
- What W cannot capture (market-maker withdrawal → use D/S instead)
- Direction convention vs Part C (Part C: high = stable; Part D: high = fragile)
- Composite CLCI formula (future work, Phase 3)

---

## Integration with Other Parts

| Part | Interface |
|---|---|
| **A** | Provides `A_sweep/snapshots/*.npz` — consumed directly by `npz_analysis.py` and `depth_to_spread_series` / `mid_prices` helpers. Per-agent flow pending (`--save-raw`). |
| **B** | `kyle_env.py` can generate synthetic data for `spectral_radius` testing (heterogeneous λ needed for W to be informative). |
| **C** | Shares the same run grid. Part C's S is high = stable; Part D's index is high = fragile. Overlay with `1 − S` or note the inversion. |
| **Phase 3** | `npz_analysis_results.csv` feeds the CLCI vs VPIN comparison. |
| **Phase 4** | `cbf_toy.py` demonstrates the filter structure. The real CBF reads CLCI as `h(x)` and constrains order size/timing. |

---

## Reference

- Agrawal & Sreenath (2017). *Discrete Control Barrier Functions for Safety-Critical Control of Discrete Systems.* Robotics: Science and Systems. — **discrete-time CBF** (the correct reference for this work; Ames et al. 2017 is continuous-time).
- Ames et al. (2017). *Control Barrier Functions: Theory and Applications.* IEEE TAC 62(8).
- ABIDES-MARL: Cheridito, Dupret & Wu (arXiv:2511.02016).

