# Fragility Index Specification — Part D

**Project:** Endogenous Market Stability & System Risk in MARL Systems  
**Part D author:** Anirudh  
**Last updated:** 2026-11-13  

---

## 1. Purpose

This document defines the three building-block terms of the **Composite Liquidity & Contagion Index (CLCI)**, including their units, ranges, sign conventions, and estimation procedures. It is the normative reference for anyone calling `fragility_index.py`.

---

## 2. Direction Convention

> **HIGH = MORE FRAGILE** for all three terms (or their inverted form).  
> This is the **opposite** of Part C's stability index S (which is high = stable).  
> When overlaying plots, use `1 − S` from Part C or add an explicit axis note.

---

## 3. Term Definitions

### 3.1 F₁ — Tail Loss (CVaR)

| Attribute | Value |
|---|---|
| **Formula** | CVaR(α) = −E[r \| r ≤ q_α], where q_α is the α-quantile of log-returns |
| **Estimator** | Empirical: sort returns, average the worst ⌈α·n⌉ of them, negate |
| **Input** | Log mid-price returns on a fixed time grid (10 s default) |
| **α (default)** | 0.05 (5% left tail) |
| **Units** | Dimensionless (log-return units; e.g. 0.003 ≈ 0.3% tail loss) |
| **Range** | [0, ∞); normally < 0.02 in calm markets |
| **High = ?** | More fragile (fatter left tail) |
| **Sign convention** | Loss reported as **positive**. A negative value means even the worst tail was a gain. |
| **Minimum n** | ⌈1/α⌉ = 20 finite returns (raises ValueError below this). With T ≈ 179 grid points per run, the 5% tail rests on ≈ 9 points — pool across seeds before interpreting. |
| **Implementation** | `fragility_index.cvar(returns, alpha=0.05)` |

---

### 3.2 F₂ — Liquidity Ratio (Depth-to-Spread)

| Attribute | Value |
|---|---|
| **Formula** | D/S = (Σ size over top-k bid levels + Σ size over top-k ask levels) / (best ask − best bid) |
| **Input** | Order-book snapshot: `{"bids": (L, 2), "asks": (L, 2)}` with rows `[price_cents, size_shares]` |
| **k (default)** | 5 levels per side |
| **Units** | Shares per price unit. With `price_scale=100` (cents → dollars): **shares per dollar** |
| **Range** | (0, ∞) when defined; NaN when either side is empty or spread ≤ 0 |
| **High = ?** | **Less** fragile (deep book relative to spread). To align with the "high = fragile" convention, use **1/F₂** when compositing |
| **Empty side encoding** | Levels with price ≤ 0 or size ≤ 0 are ignored. Madhu's int32 `.npz` uses zero-padding; this is handled automatically. |
| **Locked/crossed book** | Spread ≤ 0 → returns NaN (not an error) |
| **Implementation** | `fragility_index.depth_to_spread(book, k=5, price_scale=100.0)` |
| **Series helper** | `fragility_index.depth_to_spread_series(bids, asks, k, price_scale)` for `(T, L, 2)` arrays |

---

### 3.3 F₃ — Interaction Strength (Spectral Radius of W)

| Attribute | Value |
|---|---|
| **Formula** | ρ(W) = max\|λᵢ(W)\|, where W is the agent-interaction matrix |
| **Units** | Dimensionless (when W is normalized; see §4) |
| **Range** | [0, ∞); values ≥ 1 indicate interaction strength outpaces self-regulation |
| **High = ?** | More fragile; ρ ≥ 1 is the instability threshold (May's criterion) |
| **Implementation** | `fragility_index.spectral_radius(W)` |

---

## 4. W Matrix — Definition and Estimation

### 4.1 What W represents

W[i, j] is the coupling of agent j's signed order flow onto agent i's behaviour (price impact pathway). W is estimated from executed fills only; it **cannot** capture market-maker quote withdrawal, which is reflected in F₂ instead.

### 4.2 Primary estimator (use when per-agent flow is available)

1. Obtain per-step signed flow for each agent group (or individual market maker):  
   `q_i[t]` = Σ(+quantity for BID fills) + Σ(−quantity for ASK fills) in bin t.
2. For each pair (i, j), estimate W[i, j] by OLS:  
   `Δp[t] ≈ Σⱼ W[i,j] · q_j[t]` (regress mid-price changes on binned flows).
3. **Normalize:** divide each row i by W[i, i] (self-impact), so the diagonal is 1 and ρ = 1 is the stability threshold matching May's criterion.
4. With `num_mm = 1`, W is nearly trivial — do not read much into ρ in that case.

**Data requirement:** `exchange_log_orders=True` and the `--save-raw` flag in Part A's sweep. *(Not yet available as of 2026-11-13.)*

### 4.3 Fallback estimator (use until `--save-raw` is available)

Compute the Pearson correlation matrix of per-type signed flows:
- Group agents by type (noise, value, momentum, market-maker).
- `W[i, j] = corr(q_i, q_j)` over the run.

**Limitation:** correlation is symmetric and does not distinguish buyer-initiated from seller-initiated causality. State this explicitly when reporting ρ from the fallback.

### 4.4 Degenerate control case

With identical market makers (same λ, same learned policy), W has equal off-diagonal entries `a` and zero diagonal. Then ρ(W) = a·(n−1). In the Kyle linear control, λ_eff = λ for any M (ABIDES-MARL eq. 13), so W is degenerate and ρ should stay flat as M grows. The index must **not** rise with M in this regime.

---

## 5. Data Sources (from Part A)

| Input | Source | Format |
|---|---|---|
| Order-book snapshots | `A_sweep/snapshots/*.npz` | `bids`, `asks` each `(T, 10, 2)` int32, prices in cents, sizes in shares |
| Mid-price returns | Derived via `mid_prices()` + `log_returns()` | `(T−1,)` float64 |
| Per-agent signed flow | `--save-raw` output (pending) | Per-fill `OrderExecutedMsg` logs |

Fixed grid: T ≈ 179 snapshots per 30-minute run at 10 s intervals.

---

## 6. Composite Index (Future Work)

A scalar CLCI combining all three terms:

```
CLCI = w₁·F₁  +  w₂·(1/F₂, normalized)  +  w₃·F₃
```

Weights w₁, w₂, w₃ and normalization factors are **unspecified** and left for Phase 3. The composite is not part of the current Part D scope.

---

## 7. Summary Table

| Term | Symbol | High = fragile? | Units | Range | Implementation |
|---|---|---|---|---|---|
| Tail loss | F₁ | ✅ Yes | dimensionless (log-return) | [0, ∞) | `cvar()` |
| Liquidity ratio | F₂ | ❌ No (use 1/F₂) | shares per dollar | (0, ∞) | `depth_to_spread()` |
| Interaction strength | F₃ | ✅ Yes | dimensionless | [0, ∞) | `spectral_radius()` |

