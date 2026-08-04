# Stress Scenario Risk Measure (SES) — Simplified Methodology

**Capitalisation of Non-Modellable Risk Factors under the FRTB Internal Models Approach**

| | |
|---|---|
| **Status** | Draft for review |
| **Scope** | SES calculation for all NMRFs across IR, IR Vol, CSR, Equity, FX and cross-currency basis |
| **Primary source** | Aichele, Crotti & Rehle, *A Universal Stress Scenario Approach for Capitalising Non-Modellable Risk Factors under the FRTB*, EBA Staff Paper Series N.14, July 2021 (rev. Oct 2024) |
| **Binding regulation** | BCBS d457 MAR 33.16–33.17; Commission Delegated Regulation (EU) 2024/397; US interagency NPR (March 2026 re-proposal) |
| **Worked example** | `SES_NMRF_Worked_Example.xlsx` — live, formula-driven, end-to-end for a single NMRF |

---

## 1. Purpose and regulatory status

SES capitalises the risk factors that fail the Risk Factor Eligibility Test. It sits **outside and incremental to** the IMCC expected shortfall: modellable risk factors are capitalised in an integrated, diversifying ES measure; non-modellable ones are capitalised individually and then aggregated under a prescribed formula.

BCBS is deliberately thin here. MAR 33.16 says only that SES must be **at least as prudent as a 97.5% expected shortfall calibrated to a period of stress**, determined per risk factor or — subject to supervisory approval — per regulatory bucket. It specifies no method.

The EBA filled that gap under a CRR2 mandate, and the result is now binding EU law via Reg. 2024/397. **It is not binding in the US.** For a US IMA application it is a well-evidenced reference methodology — field-tested by large European banks, calibrated against ~50,000 real risk factors — that can be adopted, adapted or benchmarked against. That cuts both ways: greater design freedom, but the burden of justifying every constant falls on the bank rather than on a published RTS. **Recommendation: adopt the EBA methodology substantially unchanged and cite its calibration evidence, rather than construct a bespoke approach that must be defended from first principles.**

---

## 2. The core idea in one line

The honest calculation — the "direct method" — is: take every 10-day return in the stress period, revalue the SSRM portfolio under each, take the 97.5% ES of the resulting losses. That is ~250 revaluations per risk factor, per reference date, for potentially thousands of NMRFs. It is computationally infeasible and, for sparse risk factors, statistically impossible.

The methodology substitutes:

> **ES of the losses ≈ loss at the ES of the returns, times a curvature correction.**
>
> `ES( l(X) ) ≈ l( ES(X) ) · K`

`l(ES(X))` is a single revaluation. `K` costs two more. Total: **five revaluations instead of ~250 — a ~50× reduction.**

Everything else in the methodology exists to make that substitution defensible: to estimate `ES(X)` robustly when data is sparse, to compensate for the resulting estimation error, and to bound the damage when the approximation fails.

---

## 3. The seven steps

### Step 1 — Build a 10-business-day return series

For each observation date `D_t` in the stress period, find the later observation `D_t'` minimising

```
| 10 / (D_t' − D_t) − 1 |
```

then rescale the realised return by `√(10 / Δ)` where `Δ = D_t' − D_t`.

```
X_t = √(10/Δ) · return( D_t , D_t' )
```

Three design points that matter operationally:

- The criterion is a **relative** deviation, not absolute. It systematically prefers gaps **longer** than 10 days over shorter ones (a 30-day gap and a 6-day gap tie; the tie is broken toward 30). Rationale: NMRF liquidity horizons are floored at 20 days anyway, and rescaling a short return up amplifies short-term noise, whereas rescaling a long return down damps it.
- The stress period is **extended by 20 business days** for end-date purposes only. Start dates must sit inside the stress period. Without this, the final observation gets over-used as an end date and distorts the tail of the series.
- The construction always yields `N = M − 1` returns from `M` observations, however sparse. **This is the mechanism that makes the methodology work at all for illiquid risk factors.**

Returns are **not de-meaned**. Drift in a stress period is a feature of the shock, not a nuisance to be removed — and for risk factors like SABR parameters or curve nodes a zero-mean assumption is simply wrong.

### Step 2 — Calibrate an up shock and a down shock

Two shocks, `CS_up` and `CS_down`, each estimating the corresponding one-sided 97.5% ES of the return distribution. The estimator depends on how much data survived Step 1:

| N | Branch | Estimator |
|---|---|---|
| **≥ 200** | Historical | Interpolated empirical ES: `(Σ worst ⌊αN⌋ + (αN−⌊αN⌋)·next) / αN`. The interpolation avoids cliff effects as ⌊αN⌋ steps between integers. |
| **12 – 199** | **Asigma** | Split the sample at the median. On each half compute a mean and a standard deviation. Shock = `∓µ_half + 3·σ_half`. |
| **< 12** | Fallback | Shocks taken from the SA prescribed shocks, or calibrated on a "similar" risk factor with ≥12 returns; the rest of the methodology then proceeds unchanged. |

The asigma estimator (Eqs. 11–12):

```
AS_down = −µ̂_down + 3 · σ̂_down          σ̂ = √( Σ(X_i − µ̂)² / (N_half − 3/2) )
AS_up   = +µ̂_up   + 3 · σ̂_up
```

Why this works: a standard deviation is estimable from six points; a 2.5% tail quantile is not. Splitting at the median lets `σ_half` absorb the asymmetry, which is why a **single** multiplier of 3 holds across almost the entire distributional family rather than needing to vary by risk factor. Deliberately no third or fourth moments — they are not robust at these sample sizes.

**The asigma shock will routinely exceed the largest return actually observed.** In the worked example, `CS_up` is 1.63× the sample maximum. That is the intended behaviour, not a bug, and it needs stating explicitly in model documentation before a reviewer raises it as one.

### Step 3 — Apply the uncertainty compensation factor

```
UCF(N_eff) = 0.95 + 1 / √(N_eff − 1.5)

N_eff = N        under the historical method
N_eff = N/2      under the asigma method (each half is estimated separately)
```

| N_eff | 6 | 10 | 20 | 50 | 100 | 255 |
|---|---|---|---|---|---|---|
| **UCF** | 1.42 | 1.29 | 1.18 | 1.09 | 1.05 | 1.01 |

This is the "less data, more capital" lever (design goal G7), and it is **material**: the spread from a fully observed risk factor to one at the 12-return floor is roughly 40% of capital on that factor, before any change in underlying risk. It compensates for three things at once — sampling error in the ES estimate, the general lower observability of NMRFs, and residual approximation error elsewhere in the methodology.

Final calibrated shocks: `CS_up = AS_up · UCF(N_up)`, `CS_down = AS_down · UCF(N_down)`.

### Step 4 — Scan the loss profile on four points

```
Θ = { −CS_down , −0.8·CS_down , +0.8·CS_up , +CS_up }
```

Revalue the SSRM portfolio at each. The member of Θ producing the **largest loss** is `FS`, the extreme scenario of future shock.

The two inner points exist because 0.8 is empirically the ratio `VaR(97.5%) / ES(97.5%)` across the fitted distributional family — so the inner point sits approximately at the VaR, which is the lower bound of the tail integral. They also give partial protection where deep tail hedges make the loss profile non-monotonic.

Four points cannot find a maximum sitting strictly inside the range. The methodology's defence is empirical: EBA's data collection across standardised and real bank portfolios showed loss profiles are overwhelmingly directional and monotonic at large shocks. **That defence is portfolio-specific and should be re-evidenced on our own book, not assumed.**

### Step 5 — Non-linearity correction

Applied **only where FS is a boundary point** (`±CS`). Where FS is an inner point (`±0.8·CS`), skip this step entirely and set `SS_10d = l(FS)`.

```
K̃ = 1 + 12.5 · [ l(0.8·FS) − 2·l(FS) + l(1.2·FS) ] / l(FS) · (φ − 1)

K  = max( 0.9 , min( K̃ , 5 ) )
```

The bracketed term is a central second difference with step `h = 0.2·FS` — a deliberately wide step, so that `K` reflects the **global curvature of the loss profile in the tail regime**, not a local wobble at FS. Equivalently: rather than fitting a tangent parabola, the method fits the single parabola through the three evaluated points.

`φ` measures tail heaviness of the return distribution. Estimated from the data when `N ≥ 200`; otherwise the flat value 1.04. At φ = 1.04 the whole prefactor `12.5·(φ−1)` collapses to exactly **0.5** — which is not a coincidence, the constant was chosen so that `(φ−1)/h² = 1`.

Note that `l(0.8·FS)` was already computed in Step 4. **Only one additional revaluation, at `1.2·FS`, is required.**

The floor at 0.9 matters. A negative second difference means a flattening loss profile, which genuinely justifies `K < 1` — but the parabolic approximation does not merely flatten, it bends over and eventually implies *gains* at large shocks, which is not credible. 0.9 is the benefit obtainable under an idealised profile that rises linearly to `l(ES(X))` and then stays flat.

### Step 6 — Scale to the liquidity horizon

```
SS_10d = K · l(FS)                          (boundary FS)
SS_10d = l(FS)                              (inner FS)
SS_10d = 0                                  (all four points are gains)

SS = SS_10d · √( max(LH, 20) / 10 )
```

The 20-day floor is MAR 33.16(1) and applies to every NMRF regardless of its modellable-world liquidity horizon. At LH = 60 the scaling factor is 2.45× — the liquidity horizon is frequently the single largest multiplier in the whole chain, and worth checking before optimising anything upstream.

### Step 7 — Aggregate (MAR 33.17)

```
SES = √( Σ_{CSR idio} SS² )
    + √( Σ_{EQ idio}  SS² )
    + √( ( ρ · Σ_{other} SS )² + (1 − ρ²) · Σ_{other} SS² )        ρ = 0.6
```

Three points that are frequently got wrong:

- **There is no cascading bucket structure.** Unlike IMCC — with its reduced-set ratio adjustment `ES_F,C / ES_R,C` and nested aggregation — each NMRF is capitalised individually with a flat `√(LH/10)` scalar, and cross-factor diversification enters exactly once, at the end, through the fixed ρ = 0.6 formula.
- The two zero-correlation terms are **only** idiosyncratic credit spread and idiosyncratic equity risk, and only where the bank can demonstrate that zero correlation is appropriate. Everything else — including systematic credit, IR vol, FX and basis — sits in the ρ term.
- Because ρ is fixed and the formula is a sum of three square roots, **SES diversification is weak by construction**. In the worked example the benefit against a naive undiversified sum is only ~18%.

---

## 4. Bucket-level application

Where the **regulatory bucketing approach** is used for RFET, a single stress scenario may be determined for the whole bucket. The extension is mechanical:

- Calibrate `CS_up(j)`, `CS_down(j)` for every risk factor `j` in the bucket individually. The branch (historical vs asigma) is set by `N_B = min(N_1, …, N_M)` — **the least-observed factor in the bucket drives the estimator for the whole bucket.**
- Shock the bucket along **contoured shifts**: scale every factor's own calibrated shock by a common `β`, so the shape of the shift respects the fact that (for example) the short end of a curve moves more than the long end.
- Evaluate at `β ∈ {0.8, 1}` in each direction — four vector scenarios instead of four scalar shocks.
- `K` is computed identically; the bucket-level `φ` is the **median** of the constituent factors' `φ`.

**Implication for RFET design.** Bucket-level SES is cheaper to compute but couples the whole bucket to its worst-observed constituent. This interacts directly with the atomisation principle: composed risk factors should be decomposed before RFET assessment to preserve partial modellability — but once decomposed, the resulting granular factors must still be aggregated coherently for SES. The RPO-to-standard-bucket and standard-bucket-to-granular-RF mappings both need to be traceable for this reason.

---

## 5. Parameter register

Every constant, with the reason it takes the value it does. This table is the core of what a supervisor will interrogate.

| Constant | Value | Justification |
|---|---|---|
| `α` | 2.5% | MAR 33.16 — parity with the ES confidence level for modellable risk |
| `N_hist` | 200 | Gives ≥5 observations in the 2.5% tail; also ≈255 business days less ~55 days of holiday-period illiquidity |
| `N_asigma` | 12 | Below this, each half-sample has <6 points and a mean/stdev cease to be estimable |
| `C_ES` (asigma multiplier) | **3** | `C_ES ≈ 3` across the SGT family fitted to ~50,000 real risk factors; conservative for near-normal cases. Well below the Hürlimann bound of 6.245 |
| `φ_asigma` | **1.04** | Typical moderately heavy-tailed value; chosen so `(φ−1)/h² = 1` at `h = 0.2` |
| `C_UC_A` | 0.95 | Set so `UCF(255) ≈ 1.01` — ~1 for a full year of daily data, with a small add-on for NMRF observability |
| `C_UC_B` | 1.0 | One prefactor for both branches (simplification); acknowledged as slightly light for the historical branch |
| UCF offset | 3/2 | Aligned to the approximately unbiased small-sample stdev estimator used in the asigma formula |
| Inner grid factor | **0.8** | Empirically `VaR(97.5%) ≈ 0.8 · ES(97.5%)` across the fitted distributional family |
| Step width `h` | **0.2** | Same empirical fact — places the inner evaluation point at the lower bound of the tail integral |
| `K_min` | 0.9 | Maximum benefit under an idealised linear-then-flat loss profile; prevents concave fits implying gains |
| `K_max` | 5 | Numerical guard against implausible second-derivative estimates |
| `ρ` | 0.6 | MAR 33.17, prescribed |
| LH floor | 20 bd | MAR 33.16(1) |
| Max-loss replacement | `VaR(99.95%)` | Where the theoretical maximum loss is unbounded. Equal to **1.4×–2.4×** the ES-calibrated measure depending on tail thickness |

---

## 6. Failure modes and disposition

Applying the **Prove / Remediate / Build** triage.

| # | Failure mode | Consequence | Disposition |
|---|---|---|---|
| 1 | **Interior maximum** — worst loss sits strictly inside Θ (deep tail hedges, exotic optionality) | Four-point grid misses it. Supervisor may impose maximum loss, replaced by `VaR(99.95%)` where unbounded → **1.4×–2.4× capital step** | **Prove** — evidence loss-profile monotonicity on our actual book across a representative NMRF sample. Reuse the EBA finding as corroboration, not as the primary argument |
| 2 | **FS lands on an inner point** — K skipped, error unbounded in both directions | Silent over- or under-statement, acknowledged in the source methodology | **Build** — instrument the engine to flag and report inner-point FS by risk factor; if the frequency is non-trivial, extend the grid |
| 3 | **Overlapping-return bias** — the nearest-to-10-day construction autocorrelates returns; standard estimators are biased low by 2–7% (σ), 10–50% (skew), 40–90% (excess kurtosis) | UCF compensates only partly. Realised probability of underestimating the true ES runs up to ~70% even after UCF | **Prove** — document the residual bias, cite the source analysis, and state it is accepted as calibrated. Do not silently inherit it |
| 4 | **Pricer failure at extreme shocks** — shocking one NMRF in isolation creates arbitrage conditions some pricers reject | Missing loss evaluation, no SES for that factor | **Build** — sensitivity-based pricing fallback, applied only to the affected instruments, per Reg. 2024/397 |
| 5 | **Fewer than 12 returns in the stress period** | Falls to SA shocks or a proxy risk factor; proxy choice becomes a supervisory conversation | **Remediate** — fix observation capture. Where the proxy route is unavoidable, the proxy hierarchy must be documented **before** it is used and must survive PLA scrutiny |
| 6 | **Stress period unavailable for the risk class** (Assumption A3 unmet) | The entire calibration has no basis | **Remediate** — data sourcing. This is the binding constraint for IR Vol: without a reconstructable stress-period return vector there is no quantitative tail average, only a judgement-based shock |
| 7 | **Bucket coupling** — one poorly observed factor drags the whole regulatory bucket onto the asigma branch | Higher UCF applied bucket-wide | **Prove** — quantify per bucket before committing to the regulatory bucketing approach for RFET |

---

## 7. Data and system requirements

Working backwards from the calculation, SES requires:

**Per NMRF, per reference date**
- A time series of raw observations over the stress period **plus 20 business days**, with event-time dates preserved. Dates matter as much as levels — Step 1 is entirely date-driven.
- The correct return convention (log / relative / absolute) held as versioned metadata, not inferred at runtime.
- A liquidity horizon mapping.
- Five SSRM-portfolio revaluations under single-factor shocks.

**Cross-cutting**
- One 12-month stress period **per risk class**, common to all NMRFs in that class (MAR 33.16).
- Aggregation-set classification per risk factor (CSR idio / EQ idio / other), with evidence supporting any zero-correlation claim.
- Full lineage from raw observation → paired return → calibrated shock → revaluation → SS, reproducible at any past reference date.

**The point that generalises beyond SES:** results cannot be reused across reference dates, because any change in portfolio composition changes every loss profile. Every NMRF is recomputed every time. That makes revaluation throughput, not statistics, the binding engineering constraint — and it reinforces the append-only, dual-timestamp, late-deduplication design already adopted for the RPO platform.

---

## 8. Worked example — headline results

Full calculation in `SES_NMRF_Worked_Example.xlsx`. NMRF: an illustrative EUR 5Y×10Y ATM swaption normal volatility point, 41 observations in the stress period, portfolio net short vega with a convex loss profile.

| Step | Quantity | Value |
|---|---|---|
| 1 | Returns constructed / average period | N = 40 / 12.8 bd |
| 2 | Branch | Asigma (12 ≤ N < 200) |
| 2 | Median split — N_down / N_up | 20 / 20 |
| 2 | `AS_down` / `AS_up` | 0.2514 / 0.2679 |
| 3 | `UCF(N_eff = 20)` | 1.1825 |
| 3 | `CS_down` / `CS_up` | 0.2973 / 0.3168 |
| 3 | `CS_up` ÷ largest observed return | **1.63×** |
| 4 | `FS` (boundary, upward) | +0.3168 |
| 4 | `l(FS)` | $5,484,903 |
| 5 | `K̃` → `K` | 1.0250 → 1.0250 |
| 6 | `SS_10d` | $5,622,124 |
| 6 | LH scaling √(60/10) | 2.4495 |
| 6 | **`SS` for this NMRF** | **$13,771,335** |
| 7 | Total SES across six NMRFs | $22,958,632 |
| 7 | Diversification vs undiversified sum | 17.6% |

**Sensitivity to data availability**, holding the underlying asigma estimate fixed:

| N | 12 | 20 | 40 | 100 | 200 | 255 |
|---|---|---|---|---|---|---|
| **SS ($m)** | 18.2 | 16.2 | 13.8 | 12.4 | 11.7 | 11.4 |
| **Index** | 1.60× | 1.42× | 1.20× | 1.09× | 1.02× | 1.00× |

Moving one risk factor from the 12-return floor to a full year of daily observations removes ~37% of its SES charge with no change in underlying risk. **That is the capital argument for fixing observation capture, and it is the reason the IR Vol historical rebuild is a capital question rather than a documentation exercise.**

**On the direct-method comparison.** Sheet 08 computes the direct method as a diagnostic and it returns roughly half the methodology's answer. That is expected and should not be read as a discrepancy: at N = 40 the 2.5% tail contains a single observation, so the "direct" figure is essentially the single worst simulated loss and is not a credible ES estimate — precisely why Reg. 2024/397 restricts the direct method to N ≥ 200. The reconciliation is worth performing properly on any NMRF that *does* have ≥200 returns; agreement there is the single strongest piece of evidence that the implementation is correct. The EBA field test found agreement within ~1% for one participating bank across both linear and structured interest rate strategies.

---

## 9. Open items

| # | Item | Owner | Status |
|---|---|---|---|
| 1 | Confirm adoption of the EU calibration as-is for US IMA submission, vs. a bespoke justification | Model governance | Open |
| 2 | Evidence loss-profile monotonicity on our own book (failure mode 1) | Quant | Open |
| 3 | Frequency of inner-point FS by asset class — instrument and measure | Tech | Open |
| 4 | IR Vol: whether a stress-period return vector is reconstructable at all, determining quantitative vs judgement-based shock | IR Vol remediation | Open — blocking |
| 5 | Regulatory vs own bucketing decision for curve/surface RFET, including the bucket-coupling cost quantified in failure mode 7 | RFET programme | Open |
| 6 | φ estimator implementation for the N ≥ 200 branch (currently flat 1.04 assumed throughout) | Quant | TBC |
| 7 | Sensitivity-based pricing fallback design (failure mode 4) | Tech | Not started |

---

*Prepared as a working methodology reference. Figures throughout are illustrative and drawn from the accompanying workbook; they are not calibrated to any live portfolio.*
