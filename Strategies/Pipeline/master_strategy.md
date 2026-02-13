# DSU-2026: Master Forecasting Strategy v2

**Status:** APPROVED — REVISED  
**Date:** 2026-02-10  
**Supersedes:** `master_strategy.md` (v1)  
**Objective:** Forecast daily ED encounters (Total & Admitted) for Sept-Oct 2025 across 4 sites × 4 blocks (6-hour).

---

## 0. Validation of v1 Strategy — What Changed and Why

### What v1 Got Right ✅
- **Global GBDT as workhorse** — confirmed by M5 1st place (single LightGBM trained across all series dominates per-series models)
- **Hierarchical reconciliation** — validated by M5 and Store Sales winners (daily → block allocation reduces noise, improves coherence)
- **Admit_rate decomposition** — correct trick: model `total_enc` + `admit_rate` separately, then derive `admitted = total × rate`; enforces `admitted ≤ total` by construction
- **4-period forward validation** — matches competition structure perfectly
- **WAPE as primary metric** — aligned with `eval.md` contract (`primary_admitted_wape`)

### What v1 Was Missing or Suboptimal ⚠️

| Gap | Evidence | Fix in v2 |
|-----|----------|-----------|
| **No COVID-era handling** | 2020-2021 data shows 20-40% drops; training on it without adjustment biases the model toward lower volumes | Add COVID training-window policy (§3.0) |
| **No horizon-specific modeling** | M5 1st place gained +0.4 accuracy from training separate models per horizon day vs. single model | Add Pipeline B-alt: Direct Multi-Step (§3.2) |
| **Neural net (Pipeline C) adds complexity but questionable ROI** for 16-series dataset | Embeddings shine with 1000+ series (Rossmann=1,115 stores); with only 4 sites × 4 blocks = 16 series, GBDT already captures interactions | Downgrade NN to optional; replace with GLM/GAM (§3.4) — low-variance complement |
| **No case-mix latent factor pipeline** | Repo already has `cmix_*_share` features; PCA/NMF on these captures respiratory season / trauma surges as structured features — literature confirms composition shifts drive ED volume | Add Pipeline E: Reason-Mix Factor Model (§3.5) |
| **Ensemble uses only simple weighted blend** | M5 top solutions used stacking (meta-learner on OOF predictions + context features) for final 1-2% improvement | Add Stacking Ensemble (§4.2) |
| **Lag set too shallow** | v1 uses lags >60 days only; M5 1st place also used day-matched lags (lag_364 for same-day-of-year) + trend deltas (`roll_mean_7 − roll_mean_28`) | Expanded lag/rolling feature spec (§3.1) |
| **No sample weighting** | WAPE weights errors by volume — high-volume rows (Site B, Block 2-3) matter more; training without sample weights misaligns optimization | Add sample weight = `total_enc` (§3.1) |
| **Weather features not available** | `sites.json` has `lat/lon = null` for all 4 sites | **SOLVED**: Site B ≈ Sioux Falls (Sanford USD Medical Center), Site A/C ≈ Fargo (Sanford Medical Center Fargo). Use NOAA data for KFSD/KFAR. (§6) |
| **Events config is sparse** | Only Sturgis 2025 + school start 2025; missing historical events for backtesting | Expand events for all years in training history; add **School Calendar** features (§3.1) |

---

## 1. Executive Summary

An **Ensemble of 4-5 diverse pipelines** optimized for **Admitted WAPE**, with a **stacking meta-learner** for final blending:

| # | Pipeline | Role | Diversity Source |
|---|----------|------|-----------------|
| A | Global GBDT (LightGBM) | Primary workhorse | Non-linear interactions, lags/rolling |
| B | Direct Multi-Step GBDT | Horizon-aware complement | Controls accuracy decay at long horizons |
| C | Hierarchical Reconciliation | Structural coherence | Daily → block decomposition reduces noise |
| D | GLM/GAM with Fourier | Low-variance regularizer | Smooth extrapolation, handles trend |
| E | Reason-Mix Factor Model | Composition-aware | Captures seasonal shifts in visit types |

**Hard Constraints (all pipelines must enforce):**
- `Admitted ≤ Total` → model `admit_rate ∈ [0,1]`, derive `admitted = total × rate`
- Non-negativity → `clip(0)`
- Integer outputs → largest-remainder rounding
- Granularity: `(Site, Date, Block)` → 976 rows for Sept-Oct 2025

---

## 2. Validation Framework

### 2.1 Four-Period Forward Validation

Identical to v1 — this is correct and must not change:

| Period | Train ≤ | Validate |
|--------|---------|----------|
| 1 | 2024-12-31 | Jan 1 – Feb 28, 2025 |
| 2 | 2025-02-28 | Mar 1 – Apr 30, 2025 |
| 3 | 2025-04-30 | May 1 – Jun 30, 2025 |
| 4 | 2025-06-30 | Jul 1 – Aug 31, 2025 |

**Final pipeline score** = mean Admitted WAPE across all 4 periods.

### 2.2 Selection Metric

- **Primary**: `avg_admitted_enc_wape` (matches `eval.md`)
- **Diagnostics**: by-site WAPE, by-block WAPE, RMSE, R² (for debugging; do not override WAPE decisions)

### 2.3 Existing Code Mismatch (MUST FIX)

The current `scripts/train_models.py` uses a **28-day rolling window** validation (`_time_splits` with `val_days=28`), NOT the competition-aligned 2-month forward validation. This means:
- Current backtest metrics are **not comparable** to the eval.md contract
- The 28-day windows test much shorter horizons than the actual 61-day forecast
- **Action**: Replace the training script's validation with the 4-period forward validation from §2.1 before trusting any backtest numbers

Similarly, the current lag features (`lags=[1, 7, 14, 28]` with `shift(1)` rolling) **leak future data** when evaluated on the 2-month windows. The lag/rolling overhaul described in §3.1 is a prerequisite, not optional.

### 2.5 Ground Truth & Forecast Grid Construction

**Data reality:** The competition provides **only one raw file**, `DSU-Dataset.csv`, at **hourly grain** with columns:
- `Site, Date, Hour, REASON_VISIT_NAME, ED Enc, ED Enc Admitted`

There is **no separate `train.csv` / `test.csv` split** and no pre-built block-level truth. All ground truth used for validation and for defining the Sept–Oct 2025 “test” grid is derived from this single file.

**Ground truth construction (all folds):**
- **Step 1 — Filter by date window:** For a given validation fold, take all rows from `DSU-Dataset.csv` whose `Date` lies inside the train+validation span (2018‑01‑01 through the fold’s `test_end`).
- **Step 2 — Map hours to blocks:** Compute `Block = Hour // 6` with the canonical mapping:
  - 00:00–05:59 → Block 0  
  - 06:00–11:59 → Block 1  
  - 12:00–17:59 → Block 2  
  - 18:00–23:59 → Block 3
- **Step 3 — Aggregate to competition grain:** Group by `(Site, Date, Block)` and sum:
  - `ED Enc` → block-level total encounters  
  - `ED Enc Admitted` → block-level admitted encounters
- **Step 4 — Normalize dates:** Convert `Date` to `YYYY-MM-DD` strings so joins with prediction CSVs are stable.

This produces the **canonical truth table** at the same grain as the required predictions, exactly matching the evaluation contract in `Strategies/eval.md`.

**Forecast grid for Sept–Oct 2025 (synthetic “test” set):**
- Because there is no separate `test.csv`, the Sept–Oct 2025 “test set” is defined purely by the calendar and site list:
  - `Site ∈ {A,B,C,D}`
  - `Date ∈ 2025-09-01..2025-10-31` (61 days)
  - `Block ∈ {0,1,2,3}`
- This yields **4 × 61 × 4 = 976 rows**. Every pipeline’s final forecast must:
  - Produce **exactly this grid** (no missing or duplicate `(Site, Date, Block)` combinations)
  - Use the column names `(Site, Date, Block, ED Enc, ED Enc Admitted)`
  - Respect hard constraints: `ED Enc ≥ 0`, `ED Enc Admitted ≥ 0`, `ED Enc Admitted ≤ ED Enc`, and integer-valued outputs (after rounding).

**Conclusion:** Ground truth and the “test” grid are **both derived from `DSU-Dataset.csv`** — the hourly raw file is simultaneously:
- The **only source of truth** used by the evaluator (after aggregation to blocks), and
- The basis for defining which future dates/blocks must be forecast (Sept–Oct 2025 grid).

### 2.4 Sanity Checks
- No single site's WAPE should exceed 2× the best site's WAPE
- Block-level errors should be stable (common failure: overnight Block 0 drifts)
- Track per-period variance — if one fold dominates, investigate temporal drift

---

## 3. Pipeline Specifications

### 3.0 COVID-Aware Training Policy (Applies to ALL Pipelines)

**Problem**: 2020 data contains a ~30% drop in ED volumes (national average for non-emergent cases). 2021 saw partial recovery. These anomalous patterns can bias models.

**Policy (choose one per pipeline and test via CV):**
1. **Exclude COVID window**: Train only on data from Jan 2018 – Feb 2020 + Jul 2021 – train_end.
2. **COVID indicator feature**: Binary `is_covid_era` flag (Mar 2020 – Jun 2021) + set to 0 for forecast horizon.
3. **Downweight COVID era**: Sample weight = 0.1 for COVID period rows, 1.0 otherwise (can combine with WAPE volume weighting).

**Recommendation**: Option 3 (downweight) is safest — preserves data for seasonality estimation while reducing COVID bias. Test Option 1 as ablation.

---

### 3.1 Pipeline A: Global GBDT (The Workhorse)
*Based on: M5 1st place, Recruit Restaurant Visitor 1st place*

**Logic:** Single LightGBM/XGBoost model trained across all `(Site, Block)` series. This shares statistical strength across the 16 series.

**Targets (two separate models):**
- **Model A1**: `total_enc` — Objective: `tweedie` (variance_power ≈ 1.5) or `poisson`
- **Model A2**: `admit_rate` — Objective: `reg:squarederror`, bounded [0,1] post-prediction

**Feature Engineering (expanded from v1):**

| Category | Features | Notes |
|----------|----------|-------|
| **Identifiers** | `Site` (cat), `Block` (int) | One-hot or target encoding |
| **Lagged Targets** | `lag_63, lag_70, lag_77` (weekly-aligned >60d), `lag_91, lag_182, lag_364` | All >60d to avoid leakage at 61-day horizon |
| **Rolling Stats** | mean/std/min/max over 7,14,28,56,91-day windows, all shifted by 63 days | Shifted by ≥ horizon length |
| **Trend Deltas** | `roll_mean_7 − roll_mean_28`, `roll_mean_28 − roll_mean_91`, `lag_63 − lag_70` | M5 winning trick: captures momentum |
| **Calendar** | `dow`, `month`, `day`, `doy_sin`, `doy_cos`, `is_weekend`, `week_of_year`, `quarter` | Already in `calendar.py` + extensions |
| **Cyclical** | sin/cos encoding for `dow` (period=7), `month` (period=12) | Captures continuity at boundaries |
| **Holidays** | `is_us_holiday` + **Specific Holiday Lags**: `days_since_xmas`, `days_until_thanksgiving`, `days_since_july4` | **CRITICAL**: General "is_holiday" misses specific impacts (e.g., July 4th trauma surge vs. Xmas lull). |
| **School Calendar** | `school_in_session` (binary), `days_until_school_start`, `days_since_school_start` | **NEW**: Proxies from Sioux Falls/Fargo school districts. Explains late-August pediatric shifts. |
| **External** | `event_intensity`, `event_count`, weather (Tmin, Tmax, Precip) | **UPDATED**: Use NOAA data for KFSD (Sioux Falls) and KFAR (Fargo). |
| **Case-Mix Shares** | `cmix_*_share` (top 5-8 categories) | From `case_mix.py`; climatology-filled for future |
| **Lagged Embedding Summaries** | `lag_emb_entropy_{k}`, `lag_emb_norm_{k}`, `roll_emb_entropy_{w}`, `roll_emb_norm_{w}` | **NEW (§10)**: Lagged visit-reason summaries. Raw `reason_emb_*` features are EXCLUDED (target leakage). |
| **Aggregate Features** | Mean `total_enc` by (Site, Month, Block), by (DOW) — lagged to avoid leakage | M5 trick: hierarchical mean encodings |
| **Trend** | `days_since_epoch` (days since 2018-01-01), `year_frac` (continuous year) | GBDT cannot extrapolate; explicit trend captures secular volume growth (aging population, facility expansion). Pipeline D has this; Pipeline A must too. |
| **Interactions** | `is_holiday × Block`, `is_weekend × Block`, **`Site × DOW`**, **`Site × Month`** | **CRITICAL**: Site-specific rhythms (e.g., Level I Trauma center B vs. others) must be captured via interactions. |

**Rolling Feature Shift Warning:**
- All rolling statistics MUST be shifted by ≥ `max_horizon` (63 days for Pipeline A) to prevent leakage
- The current codebase uses `shift(1)` in `src/dsu_forecast/modeling/features.py` — **this is a leakage bug for multi-step forecasting** and must be fixed before any pipeline runs
- Correct: `shift(63).rolling(w, min_periods=1).mean()` for Pipeline A; see §3.2 for Pipeline B's adaptive shifts

**Loss Alignment with WAPE:**
- **Sample weights** = `total_enc` (or `admitted_enc` for Model A2) — forces the model to care about high-volume rows
- Hyperparameter selection by fold WAPE, not MAE

**Hyperparameter Search (Optuna, 100 trials):**
- `n_estimators`: 800-3000, `max_depth`: 4-8, `learning_rate`: 0.01-0.05
- `subsample`: 0.7-0.95, `colsample_bytree`: 0.6-0.9, `reg_lambda`: 1-10
- `min_child_weight`: 1-10

---

### 3.2 Pipeline B: Direct Multi-Step GBDT
*Based on: M5 1st place — "forecast each day separately" trick*

**Logic:** Instead of one model for all future dates, train models conditioned on the horizon distance. This prevents accuracy degradation at longer horizons (day 61 behaves differently than day 1).

**Two implementation options (test both):**

**Option B1 — Horizon-Bucket Models (PREFERRED):**
- Bucket 1: days 1-15, Bucket 2: days 16-30, Bucket 3: days 31-61
- Train a separate GBDT per bucket (using same features as Pipeline A)
- Use the correct bucket model for each forecast row
- **CRITICAL — Horizon-Adaptive Lag Sets per Bucket:**
  - Bucket 1 (days 1-15): minimum lag = 16 → use lags `[16, 21, 28, 56, 91, 182, 364]` + rolling shifted by 16d
  - Bucket 2 (days 16-30): minimum lag = 31 → use lags `[31, 35, 42, 56, 91, 182, 364]` + rolling shifted by 31d
  - Bucket 3 (days 31-61): minimum lag = 62 → use lags `[63, 70, 77, 91, 182, 364]` + rolling shifted by 63d
  - This is the **primary value of Pipeline B over Pipeline A** — shorter-horizon buckets get access to more recent data, which is the strongest predictive signal

**Option B2 — Single Model with `days_ahead` Feature:**
- Add `days_ahead` as a continuous feature (1 to 61)
- Train on supervised examples: features at date t → target at date t+h
- Simpler to maintain; if horizon effect is smooth, works well
- **Computational note**: This multiplies training data by ~61× (one row per horizon per original date). For ~40K original rows → ~2.4M training rows. Budget ~4× memory and training time vs. Pipeline A. Consider sub-sampling horizons (e.g., every 3rd day) if compute-constrained.
- **Lag policy for B2**: Use only lags ≥ `days_ahead + 1` per row. In practice, compute all lags from 1-364 and mask/NaN those where `lag_k < days_ahead + 1`. The model learns to use shorter lags when available and gracefully degrade when they're masked.

**Option B3 — Recursive GBDT (Fallback):**
- Classical recursive strategy: Predict t+1, use prediction as lag for t+2.
- **Pros**: Simpler training (single model).
- **Cons**: Error accumulation.
- **Use only if**: Direct Multi-Step (B1/B2) proves too computationally expensive.

**Key Difference from Pipeline A:**
- Pipeline A treats all future rows identically and uses only lags ≥63 (safe for worst-case horizon)
- Pipeline B explicitly accounts for horizon distance — **short-horizon predictions get access to recent lags that Pipeline A throws away**
- This is the biggest single source of potential improvement: for day-1 predictions, lag_1 (yesterday's actual) is far more informative than lag_63

**⚠️ Reason-Embedding Leakage (see §10):**
Raw `reason_emb_*` features MUST be excluded from Pipeline B's feature set — they are zero for all Sep-Oct 2025 rows and cause ~84% undercount. Replace with lagged embedding summaries (`lag_emb_entropy`, `lag_emb_norm`, `roll_emb_entropy`, `roll_emb_norm`) shifted by `(h + min_lag)` per bucket, matching the existing lag infrastructure.

---

### 3.3 Pipeline C: Hierarchical Reconciliation (Daily → Block)
*Based on: M5 hierarchical winners, already partially in repo*

**Logic:** Decompose into two easier sub-problems:

**Step 1 — Daily Total Forecast (per Site):**
- GBDT model (same features minus block-specific features) predicting daily `total_enc` per site
- This is a smoother signal (~100 visits/site/day) vs. per-block (~25 visits)
- Use existing `scripts/train_daily_models.py` as base (upgrade to LightGBM + Optuna tuning)

**Step 2 — Block Share Prediction:**
- Predict `p_block` (4 probabilities summing to 1) per (Site, Date)
- Methods (test both):
  - **Dirichlet regression** on historical block shares + calendar features
  - **Softmax output from GBDT** (CatBoost MultiClass with 4 classes)
  - **Simple climatology** fallback: historical average block shares by (Site, DOW, Month)

**Step 3 — Allocation + Reconciliation:**
- `Block_Forecast = round(Daily_Forecast × Block_Share)`
- **Largest-remainder rounding** (already in `scripts/reconcile_block_to_daily.py`) to enforce exact daily sums
- Same process for admitted (model daily admitted → allocate by block admit shares)

---

### 3.4 Pipeline D: GLM/GAM with Fourier Seasonality (Low-Variance Complement)
*Replaces v1 Neural Net — better ROI for 16 series*

**Logic:** Parametric count model with explicit seasonal decomposition. Provides a smooth, low-variance baseline that complements GBDT's flexibility.

**Why it replaces the Neural Net:**
- With only 16 series (4 sites × 4 blocks), neural networks have insufficient data to learn useful embeddings
- GLM/GAM provides true complementary diversity (linear vs. tree-based) which matters for ensemble gains
- Better extrapolation for smooth trends (GBDT cannot extrapolate beyond training data range)

**Implementation:**
- **One model per (Site, Block)** — 16 models total
- **Framework**: `statsmodels` GLM (Poisson family + log link) or `pyGAM`
- **Design matrix:**
  - DOW (one-hot, 6 dummies)
  - Fourier terms: sin/cos at periods 7, 365.25 (order 3 and 10 respectively)
  - `is_us_holiday` + holiday proximity indicators
  - Linear trend component (days since start)
  - Optional: COVID indicator or changepoint terms
- **admit_rate**: logistic regression or Beta regression per (Site, Block)
- **Post-processing**: clip ≥ 0, integer round

---

### 3.5 Pipeline E: Reason-Mix Latent Factor Model (NEW)
*Based on: strategies_2.md Pipeline F concept + ED literature on compositional shifts*

**Logic:** ED volume surges are often driven by shifts in visit composition (flu season → respiratory surge, summer → trauma/injury spike). Instead of forecasting by reason (prohibited in output), compress reason categories into latent factors and use them as regressors.

**Implementation:**
1. **Build category share matrix** from existing `cmix_*_share` features (per Site, Date, Block)
2. **Reduce to k latent factors** (k=3-5) using PCA or NMF on historical shares
3. **Forecast factors forward:**
   - Simple: climatology by (Month, DOW, Block)
   - Better: small GBDT/AR model on factor time series + calendar features
4. **Final model**: GBDT predicting `total_enc` using standard features + predicted latent factors as extra regressors

**Why this adds value:**
- Captures "respiratory season is early this year" signals that raw calendar features miss
- The repo already computes `cmix_*_share` — minimal incremental work
- Provides orthogonal information to pure demand-side models

**CRITICAL — Climatology Circularity Warning:**
- The "Simple: climatology by (Month, DOW, Block)" option for forecasting factors is **nearly useless**: if you reduce climatology-filled shares to PCA factors, the result is a deterministic function of (Month, DOW, Block) — information already captured by calendar features. Zero new signal.
- **The value of Pipeline E comes entirely from capturing deviations from typical seasonal patterns** (e.g., early flu onset, unusual trauma surge). Only the AR/GBDT factor forecasting method preserves this momentum signal.
- **Requirement**: You MUST include `factor_momentum = factor_now - factor_lag_7` as a feature in the final model.
- **Recommendation**: Use ONLY the GBDT/AR factor forecasting approach. If that doesn't improve CV WAPE over Pipeline A alone, drop Pipeline E entirely — it means composition shifts aren't predictable beyond what calendar already captures.

---

## 4. Ensembling Strategy (Upgraded)

### 4.1 Level 1: Simple Weighted Blend (Baseline Ensemble)

Same as v1 but expanded to 4-5 pipelines:
1. Generate OOF predictions for each pipeline across all 4 validation periods
2. Optimize weights $w_A + w_B + w_C + w_D + w_E = 1$ via constrained optimization (minimize WAPE on pooled OOF predictions)
3. Use `scipy.optimize.minimize` with simplex constraint
4. **Alternative: Hill Climbing (Forward Selection)** — iteratively add models that maximize WAPE reduction. Often more robust than `minimize` for small ensembles (<10 models). Run both, pick winner.

### 4.2 Level 2: Stacking Meta-Learner (UPGRADE from v1)
*Based on: M5 top-5 solutions, standard Kaggle stacking*

**Why stacking beats simple blending:**
- Different pipelines may be better for different (Site, Block, DOW) combinations
- A meta-learner can learn "use Pipeline C for weekends, Pipeline A for weekdays"

**Implementation:**
1. For each of the 4 validation periods, collect OOF predictions from all pipelines → creates a meta-feature matrix
2. Meta-features per row: `pred_A_total, pred_B_total, pred_C_total, pred_D_total, pred_E_total` (and same for admit_rate)
3. Context features: `Site, Block, DOW, Month, is_weekend`
4. Meta-learner: **Ridge regression** (STRONGLY preferred over GBDT)
   - Quantitative justification: 4 folds × ~976 rows/fold = ~3,900 OOF samples. With ~15 meta-features (5 pipeline preds + 5 context), Ridge has ~15 effective parameters — safe. A GBDT with max_depth=3 and 100 trees has ~800 leaf nodes and can memorize fold-specific artifacts (e.g., "fold 3 is summer, always trust Pipeline D more"). This creates optimistic CV estimates that don't transfer to the Sept-Oct test.
   - **Only use GBDT meta-learner if** you expand OOF data by also including per-site/per-block sub-folds (increasing effective sample size to ~15K+).
5. Train meta-learner on all 4 periods' OOF; predict on Sept-Oct 2025 using full-retrained pipeline predictions as inputs
6. Post-process: clip, enforce constraints, integer round

### 4.3 Post-Ensemble Reconciliation

Regardless of ensemble method:
1. Reconcile block predictions to daily totals per site (use Pipeline C's daily forecast as anchor if available)
2. Enforce `admitted ≤ total`
3. Clip at 0, integer round (largest-remainder)
4. Validate output against `eval.md` contract (976 rows, correct columns)

---

## 5. Implementation Roadmap

### Phase 0: Data Preparation & Configuration (Days 1-2)
- [ ] **Fix `sites.json`**: Research Sanford Health facility locations, populate lat/lon for weather features
- [ ] **Expand `events.yaml`**: Add historical events for 2018-2025 (Sturgis Rally runs annually, school calendars, etc.)
- [ ] **COVID window flagging**: Add `is_covid_era` column to feature table
- [ ] **Verify validation splits**: Run `eval.md` evaluator on a naive baseline (e.g., same-period-last-year) to establish floor performance

### Phase 1: Feature Engineering (Days 2-3)
- [ ] **Extend `build_feature_table.py`**: Add expanded lag set (63,70,77,91,182,364), trend deltas, aggregate encodings
- [ ] **Add COVID downweighting**: Implement sample weight column
- [ ] **PCA/NMF on case-mix shares**: Create latent factor features for Pipeline E
- [ ] **Generate Fourier terms**: For Pipeline D design matrix

### Phase 2: Pipeline Development (Days 3-7)
- [ ] **Pipeline A**: Implement `pipelines/gbdt_global/` — LightGBM with Optuna + sample weighting
- [ ] **Pipeline B**: Implement `pipelines/direct_multistep/` — horizon-specific models
- [ ] **Pipeline C**: Upgrade existing `scripts/train_daily_models.py` → `pipelines/hierarchical/` with block share model
- [ ] **Pipeline D**: Implement `pipelines/glm_fourier/` — Poisson GLM + Fourier terms
- [ ] **Pipeline E**: Implement `pipelines/casemix_factors/` — factor forecast + GBDT

### Phase 3: Validation & Tuning (Days 7-9)
- [ ] Run all pipelines through 4-period validation
- [ ] Collect per-pipeline WAPE metrics (overall + by-site + by-block breakdowns)
- [ ] Ablation testing: COVID policy variants, lag set variants, sample weighting on/off
- [ ] **Drop underperforming pipelines**: If a pipeline's standalone WAPE is >1.5× best, exclude from ensemble

### Phase 4: Ensembling (Days 9-10)
- [ ] Generate stacked OOF prediction matrix
- [ ] Train weighted blend (Level 1) — quick baseline
- [ ] Train stacking meta-learner (Level 2)
- [ ] Compare Level 1 vs Level 2; pick the one with lower CV WAPE
- [ ] Apply post-ensemble reconciliation

### Phase 5: Final Forecast (Day 10)
- [ ] Retrain all selected pipelines on full history (Jan 2018 – Aug 2025, applying COVID policy)
- [ ] Generate Sept-Oct 2025 predictions from each pipeline
- [ ] Run ensemble → final 976-row submission
- [ ] Validate against `eval.md` contract (schema, constraints, row count)

---

## 6. External Data & Configuration Issues

### Weather (High Priority Fix)
- `sites.json` has `lat/lon = null` for all sites — weather features are currently NaN/0
- **Identified Locations**:
    - **Site B (Largest, ~36% vol)**: Matches **Sanford USD Medical Center (Sioux Falls, SD)**.
    - **Site A/C**: Matches **Sanford Medical Center Fargo (ND)**.
    - **Action**: Use NOAA data for stations **KFSD (Sioux Falls)** and **KFAR (Fargo)**. Map Site B -> KFSD, Sites A/C/D -> KFAR/KFSD (test which fits best).
- **Fallback**: If coordinates unknown, weather adds 0 value; rely on calendar + case-mix features

### Events Configuration (Medium Priority)
- Current `events.yaml` only has 2025 entries — no historical events for backtesting
- **Action**: Add Sturgis Rally dates for 2018-2024 (consistent annual event), school year boundaries for all years
- **School Calendars**:
    - **Sioux Falls**: Start date usually late August (~20th-25th).
    - **Fargo**: Similar.
    - **Impact**: Shift from "summer trauma/play" to "school respiratory" patterns.
- **Optional**: Look for CDC flu season peak dates (could correlate with respiratory ED surges)

### Forecast-Period-Specific Considerations (Sept-Oct 2025)

The target window has distinct seasonal signals that features should capture:

| Signal | Timing | Impact on ED | Feature Strategy |
|--------|--------|-------------|-----------------|
| **School return** | Late Aug – early Sept | Pediatric respiratory, injury uptick; changes block distribution (morning dip as kids in school) | `school_in_session` binary + `days_since_school_start` |
| **Fall allergy season** | Sept peak | Respiratory/allergic ED visits increase | Captured by case-mix factors if Pipeline E AR model works; otherwise by seasonal Fourier in Pipeline D |
| **Early flu onset** (some years) | Late Oct | Respiratory surge; historically variable onset | Pipeline E factor momentum is the best hope here; lag_364 captures prior-year pattern |
| **Halloween** | Oct 31 | Pediatric injury spike, especially Block 3 (18-23h) | Add `is_halloween` indicator; small effect but easy win |
| **Daylight saving fallback** | Nov 2, 2025 | Outside forecast window but affects Oct 31-Nov 1 perception | No action needed (Oct 31 is last forecast day) |
| **Secular volume trend** | Ongoing | ~2-4% annual growth typical for ED utilization | `days_since_epoch` feature (added to Pipeline A spec above) |

### External Data (Optional, If Allowed)
- **CDC ILINet data**: Weekly flu surveillance → interpolate to daily; strong signal for respiratory ED visits
- **AQI data**: Air quality index → respiratory exacerbation driver
- **Caveat**: Must be available for all 4 folds consistently (2018-2025) and not leak future info

---

## 7. System Architecture: Data Source → Pipelines → Eval

### 7.1 Three-Layer Design

```
┌──────────────────────────────────────────────────┐
│  DATA SOURCE (data_source.md)                    │
│  • Ingests visits, events, weather, school cal   │
│  • Produces master_block_history.parquet          │
│  • NO imputation, NO feature engineering         │
│  • Fold-agnostic: full grid, all dates           │
└──────────────┬───────────────────────────────────┘
               │ One unified dataset
    ┌──────────┼──────────┬───────────┬────────────┐
    ▼          ▼          ▼           ▼            ▼
 Pipeline A  Pipeline B  Pipeline C  Pipeline D  Pipeline E
 (GBDT)      (Multi-Step)(Hierarchy) (GLM/GAM)   (Factor)
    │          │          │           │            │
    ▼          ▼          ▼           ▼            ▼
 Submission  Submission  Submission  Submission   Submission
 CSVs (×4    CSVs (×4    CSVs (×4   CSVs (×4     CSVs (×4
  folds)      folds)      folds)     folds)       folds)
    │          │          │           │            │
    └──────────┴──────────┴───────────┴────────────┘
               │
    ┌──────────┴───────────────────────────────────┐
    │  COMMON EVAL (eval.md)                       │
    │  • Validates submission contract              │
    │  • Scores via WAPE (primary) + diagnostics    │
    │  • Ranks pipelines on mean admitted WAPE      │
    │  • Pipeline-agnostic: only sees output CSVs   │
    └──────────┬───────────────────────────────────┘
               │
    ┌──────────┴───────────────────────────────────┐
    │  CONVERGENCE ANALYSIS + ENSEMBLE              │
    │  • If pipelines converge → dataset ceiling    │
    │  • Best combo via stacking/blending (§4)      │
    │  • Final submission (976 rows)                │
    └──────────────────────────────────────────────┘
```

**Key contracts:**
- **Ground truth source (single file):** All truth comes from `Dataset/DSU-Dataset.csv` at hourly grain; there is **no separate competition test file**. The evaluator aggregates this to `(Site, Date, Block)` using `Block = Hour // 6` to create fold-specific truth tables.
- **Data Source → Pipelines**: `master_block_history.parquet` with schema defined in `data_source.md` §2. Block numbering is `0-3` (matching `eval.md`). No imputation — NaN preserved.
- **Pipelines → Eval (CV)**: One submission-shaped CSV **per validation window** with columns `(Site, Date, Block, ED Enc, ED Enc Admitted)` covering the full required grid for that fold.
- **Pipelines → Eval (final forecast)**: One submission-shaped CSV for the synthetic Sept–Oct 2025 grid (4 sites × 61 days × 4 blocks = 976 rows) with the same columns and hard constraints.
- **Eval → Selection**: `primary_admitted_wape` (mean across 4 folds) is the single ranking metric.

### 7.2 Pipeline Convergence as Predictive Ceiling

The diversity across 5 pipelines is not just for ensemble gains — it serves as an **empirical bound on the dataset's predictive power**.

**Argument:** If structurally different models (non-linear tree ensembles, parametric GLMs, hierarchical decomposition, compositional factor models) all converge to similar WAPE scores, then:
1. The residual error is dominated by **irreducible noise** (random patient arrivals, unobserved events) rather than model bias.
2. Additional modeling complexity will yield diminishing returns — the signal in the features has been exhausted.
3. The converged WAPE is a defensible estimate of the dataset's **predictive floor**.

**How to assess convergence:**
- After running all pipelines through the 4-fold eval, compute the **coefficient of variation (CV)** of per-pipeline mean WAPE scores:
  - `CV = std(pipeline_wapes) / mean(pipeline_wapes)`
  - If `CV < 0.05` (i.e., all pipelines within ~5% of each other), declare convergence.
- Also check per-fold agreement: if diverse pipelines make similar errors on the same (Site, Date, Block) rows, the errors are data-driven, not model-driven.
- **Diagnostic**: Compute pairwise prediction correlation between pipelines. If `corr > 0.95` across all pairs, ensemble gains will be marginal — focus on post-processing instead.

**Actionable implications:**
- If convergence is reached early (e.g., A, B, C all within 2% WAPE of each other), deprioritize Pipelines D/E and invest time in post-ensemble reconciliation, constraint enforcement, and error analysis.
- If one pipeline is a clear outlier (much better or worse), investigate why — it likely exploits (or misses) a specific data pattern.

---

## 8. Risk Assessment & Fallback Strategy

| Risk | Mitigation |
|------|------------|
| Overfitting (too many pipelines for 16 series) | Keep meta-learner extremely simple (Ridge/small GBDT); use all 4 CV folds for meta-training |
| COVID distortion | Test all 3 COVID policies in §3.0; pick by CV |
| Weather data unavailable (lat/lon unknown) | Model works without weather — calendar + lags account for seasonality; weather is marginal improvement |
| Pipeline D or E underperforms | Drop from ensemble; retain A+B+C core which has highest expected value |
| Stacking overfits on 4 folds | Fall back to simple weighted blend (Level 1) |
| **Reason-embedding target leakage (§10)** | Raw `reason_emb_*` features are unavailable for future dates (all zeros). CV metrics appear good but final forecast underpredicts by ~84%. **Fix**: exclude raw embeddings, use lagged summaries instead. Affects Pipelines A & B. |

**Minimum Viable Submission**: Pipelines A + C with simple weighted blend. This alone should beat the current XGBoost baseline significantly.

---

## 9. Expected Improvement Over v1

| Improvement | Source | Expected WAPE Reduction | Confidence |
|-------------|--------|------------------------|------------|
| **Fix lag leakage** (shift(1)→shift(63)+) | Eliminates data leakage in current code; current metrics are artificially optimistic | Metrics will GET WORSE before they get better — this reveals true baseline | Certain |
| COVID-aware training | Removes volume bias from 2020-2021 anomaly | 2-5% | High |
| Sample weighting (WAPE-aligned) | Better high-volume accuracy | 3-5% | High |
| Expanded lag/rolling features (lag_63+) | More temporal signal from weekly-aligned and yearly lags | 2-4% | High |
| **Horizon-adaptive lags (Pipeline B)** | Short-horizon buckets get access to recent actuals — strongest single modeling improvement | 3-8% | High (this is the biggest lever) |
| Trend feature (`days_since_epoch`) | Captures secular ED volume growth that GBDT can't extrapolate | 1-3% | Medium |
| Holiday × Block interactions | Corrects block distribution shifts on holidays | 0.5-1.5% | Medium |
| GLM/GAM diversity in ensemble | Lower-variance complement | 1-2% | Medium |
| Ridge stacking meta-learner | Context-aware blending (Ridge, not GBDT) | 0.5-1.5% | Medium |
| Pipeline E factor model (if AR-based) | Captures composition momentum shifts | 0-2% | Low-Medium (high if flu onset is predictable) |
| **Cumulative (estimated, post-leakage-fix)** | | **10-20% WAPE reduction over corrected baseline** | |

*Note: The current baseline metrics are inflated by lag leakage. After fixing leakage, the true baseline WAPE will be significantly higher. The improvements above are measured against that corrected baseline. Absolute WAPE reduction vs. the current (leaky) numbers may appear smaller initially.*

---

## 10. Critical Finding: Reason-Embedding Target Leakage (2026-02-12)

### 10.1 The Problem

**Severity:** CRITICAL — causes ~84% undercount in final Sep-Oct 2025 predictions for affected pipelines.

**Affected Pipelines:** A and B (the two GBDT-based pipelines that ingest all columns from `master_block_history`).  
**Unaffected Pipelines:** C, D, E (do not use `reason_emb_*` features).

**Root cause:** The Data Source layer generates **67 reason-embedding features** per `(site, date, block)` row:
- `reason_emb_0` … `reason_emb_63` — 64-dim SapBERT/MiniLM embedding of visit reasons, aggregated by ED-encounter-weighted mean
- `reason_emb_norm` — L2 norm of the embedding vector
- `reason_emb_entropy` — Shannon entropy of the visit-reason distribution
- `reason_emb_cluster` — k-means cluster assignment of the embedding

These features encode **the actual composition of patient visits that occurred on the target date**. They are only computable after the visits happen. For Sep-Oct 2025 (the forecast horizon), no visits exist, so all 67 features are **zero-filled**.

### 10.2 Why CV Metrics Were Misleading

During 4-fold cross-validation, the validation periods (Jan-Aug 2025) have **real visits** in `master_block_history`. The embeddings for those dates are populated with actual values. The model sees real embeddings during both training and CV evaluation, producing excellent metrics:

| Metric | Pipeline B CV (all 4 folds) |
|--------|---------------------------|
| Mean Total WAPE | 14.7% |
| Mean Admitted WAPE | 27.9% |
| Total R² | 0.87 |

But for the **final Sep-Oct forecast**, all embeddings are zero. The model has never seen this all-zeros pattern during training (every historical date has some visits), so predictions collapse:

| Pipeline | Site A Sep 2025 Daily Mean | vs. Actual Aug (131.6/day) |
|----------|---------------------------|---------------------------|
| **B** | **21.6** | **-83.6%** |
| C (unaffected) | 130.6 | -0.8% |
| D (unaffected) | 131.4 | -0.2% |

### 10.3 Feature Importance Confirms Heavy Dependence

Pipeline B Bucket 1 (total_enc model):
- `reason_emb_*` features account for **78.7% of total split importance**
- Top feature: `reason_emb_entropy` (5,868 splits) — 2.8× the next non-embedding feature
- All top 22 features are embedding-derived

Pipeline A:
- `reason_emb_entropy` is the **#3 feature** (mean importance 1,836)
- `reason_emb_norm` is **#4** (526)
- Combined: ~25% of total importance

The models learned to use these features as a high-signal proxy for volume (more visits → higher entropy/norm → higher prediction). When zeroed out at inference, the models extrapolate to a regime they were never trained on.

### 10.4 Recommended Fix: Lagged Embedding Summary Features

**Principle:** The visit-reason composition is temporally stable (week-over-week changes are small). Lagged embedding summaries — computed from dates that are known at prediction time — preserve the signal without leakage.

**Implementation (for Pipelines A and B):**

**Step 1 — Exclude current-date embedding features from the model:**
Add `reason_emb_*` columns (all 67) to each pipeline's `_EXCLUDE_COLS` set. These must never appear as model inputs because they are unavailable at inference.

**Step 2 — Create lagged embedding summary features:**
In each pipeline's feature engineering (e.g., `build_bucket_data` for Pipeline B, `build_features` for Pipeline A), compute lagged versions of the 3 scalar summaries using the same lag/shift logic already used for `lag_total`:

| New Feature | Computation | Rationale |
|-------------|-------------|-----------|
| `lag_emb_entropy_{k}` | `reason_emb_entropy.shift(h + k)` | Visit diversity from k days before as-of date |
| `lag_emb_norm_{k}` | `reason_emb_norm.shift(h + k)` | Embedding magnitude (volume proxy) |
| `lag_emb_cluster_{k}` | `reason_emb_cluster.shift(h + k)` | Visit-profile cluster |
| `roll_emb_entropy_{w}` | Rolling mean of shifted entropy over w-day window | Smoothed recent visit diversity |
| `roll_emb_norm_{w}` | Rolling mean of shifted norm over w-day window | Smoothed recent volume proxy |

Use the **same lag set per bucket** as `lag_total`:
- Pipeline B Bucket 1: lags `[16, 21, 28, 56, 91, 182, 364]` → 7 × 2 scalar lags + rolling = ~21 new features
- Pipeline B Bucket 2: lags `[31, 35, 42, 56, 91, 182, 364]`
- Pipeline B Bucket 3: lags `[63, 70, 77, 91, 182, 364]`
- Pipeline A: lags `[63, 70, 77, 91, 182, 364]` (same minimum shift as existing lag features)

**Why only the 3 summaries, not all 64 dimensions?**
- `reason_emb_entropy` + `reason_emb_norm` account for ~70% of the embedding feature importance — they are the key scalars
- Lagging all 64 dims × 7 lags = 448 features, which causes curse-of-dimensionality for LightGBM with ~40K training rows
- The 3-summary approach adds ~21 features (manageable) and captures the majority of the signal
- If more granularity is needed later, add rolling-mean of the top 5 embedding dimensions as an enhancement

**Step 3 — Retrain and validate:**
After implementing lagged summaries, re-run the full 4-fold CV to confirm:
1. CV WAPE remains comparable (may improve slightly — lagged features are less noisy than same-date features)
2. Final Sep-Oct predictions are now in the plausible range (~125-135 daily for Site A)
3. No new leakage is introduced (verify: all lag shifts ≥ `min_lag` for the bucket/pipeline)

### 10.5 Why Not Simply Drop Embeddings?

Dropping all 67 embedding features is the safest quick fix, but suboptimal:
- Pipeline B loses **78.7%** of its feature importance — the model must redistribute signal to weaker features (rolling means, calendar, weather)
- The visit-reason composition IS genuinely predictive and changes slowly, so lagged versions carry real information
- The 3-summary lag approach recovers the majority of the signal with zero leakage risk
- If time-constrained: drop first (quick sanity restore), then add lagged summaries as enhancement

### 10.6 Lessons & Prevention

1. **Any feature derived from the target date's actual data is potential leakage** when used for future forecasting. This includes: visit counts, reason distributions, embeddings, case-mix shares.
2. **CV metrics alone cannot detect this class of leakage** because CV validation dates have real data populated. Only comparing CV vs. final-forecast behavior reveals the mismatch.
3. **Prevention rule**: Every feature in `master_block_history` must be tagged as either:
   - **Static/deterministic** (calendar, weather forecast, events) — safe for any date
   - **Lagged/historical** (computed from past data only) — safe if shifted correctly
   - **Contemporaneous** (derived from same-date actuals) — **MUST be excluded or lagged**

---

## 11. Site D Isolation Strategy (2026-02-13)

### 11.1 The Problem — Audit Summary

Site D is the hardest site for **every** pipeline (admitted WAPE ~0.47–0.94 vs ~0.23–0.30 for A/B/C). A full data audit reveals this is **not** a data quality issue — there are no Site D-specific closures, missingness gaps, or structural breaks. The Sep-Oct 2025 zero region is identical across all 4 sites. The difficulty is intrinsic to Site D's volume profile:

| Metric | Site A | Site B | Site C | **Site D** |
|--------|--------|--------|--------|------------|
| Daily total_enc | 121.3 | 165.7 | 95.7 | **72.3** |
| Daily admitted_enc | 38.3 | 42.2 | 29.2 | **12.5** |
| Overall admit rate | 31.6% | 25.5% | 30.6% | **17.4%** |
| Block 0 admitted=0 rate | 1.3% | 0.6% | 6.6% | **28.9%** |
| Block 0 admitted CV | 0.513 | 0.484 | 0.632 | **0.926** |
| "Other" reason share | 43.8% | 50.6% | 46.2% | **52.5%** |

**Root causes:**
1. **Low volume** — 12.5 daily admissions means a 3-patient error costs 24% WAPE (vs 8% at Site A)
2. **Extreme zero-inflation** — Block 0 admitted_enc is 0 on 28.9% of nights; the distribution is {0: 29%, 1: 36%, 2: 23%, 3+: 12%} — essentially a zero-inflated Poisson with mode at 0–1
3. **Low acuity** — 17.4% admit rate is structurally different from A/C (~31%), suggesting Site D is a lower-acuity ED (more walk-in/urgent-care volume)
4. **Noisier reason-mix** — 52.5% in the "other" catch-all means the top-20 reason features capture less signal for Site D

### 11.2 Why Vanilla "Partial Pooling" Is Already Failing

The textbook prescription for low-volume series is partial pooling — train a global model, let it borrow strength from higher-volume peers. **Pipelines A, B, and E already do this.** They train a single LightGBM across all 4 sites with `site` as a categorical feature. This IS partial pooling. And Site D still has a WAPE of ~0.47 while A/B/C sit at ~0.23–0.30. The strategy isn't wrong in principle — it's failing in execution due to three specific mechanisms.

#### 11.2.1 The Volume-Bias Trap

In a global GBDT, the loss function (here: MAE or WAPE-weighted MSE) minimizes **total error across the full dataset**. The gradient updates are dominated by whichever rows contribute the most loss:

| Site | Share of Rows | Daily Volume | Loss Contribution (proportional) |
|------|:------------:|:------------:|:-------------------------------:|
| B | ~36% | 162 | **Highest** — high volume × many rows |
| A | ~27% | 119 | High |
| C | ~21% | 94 | Moderate |
| **D** | **~16%** | **71** | **Lowest** — low volume × fewest rows |

Result: The optimizer reduces global loss fastest by fitting Site B's patterns. Site D's gradient signal is drowned out. "Borrowing strength" effectively becomes **"overwriting Site D with Site B's patterns."** Even with `sample_weight = total_enc`, Site D's weights are the smallest, amplifying the imbalance.

#### 11.2.2 GBDT Cannot Do True Partial Pooling

A GLM with random effects achieves genuine partial pooling: shared slopes (β for weather, trend) with site-specific intercepts shrunk toward the population mean. GBDTs have no such mechanism. At each tree split, the algorithm faces a binary choice:

- **Split on `site`** → The left/right child becomes a **local model** for that site (no pooling)
- **Don't split on `site`** → All sites get the same prediction adjustment (**total pooling**)

There is no tree operation that says "apply 70% of the global pattern and 30% of Site D's local deviation." The model either pools completely or isolates completely at each node. With Site D having 16% of the data and the lowest volume, the tree frequently chooses NOT to split on site (the split gain is small) — meaning Site D gets the same adjustment as Site B. When it does split, Site D's leaf has ~2,800 samples with high variance → noisy leaf value.

#### 11.2.3 The Admit-Rate Structural Gap

Site D's 17.4% admit rate is **14 percentage points** below A/C (~31%). This isn't noise — it's a fundamentally different patient population (lower-acuity ED). A global model that predicts `admitted_enc` directly must learn this 14pp gap entirely from the `site` feature. If the tree doesn't split on site early enough, predictions for Site D are biased toward the population mean (~27%), **systematically overpredicting** admissions.

### 11.3 The Upgraded Strategy: Target Encoding + Hybrid-Residual

The solution is not "more pooling" — it's giving the global model the right **numerical context** so it can apply shared patterns to the correct baseline, then cleaning up what the global model misses with a local specialist.

#### 11.3.1 Component 1: Target Encoding (Fix the Baseline Problem)

**Problem:** The categorical feature `site = D` tells the tree "this is site D" but nothing about what that *means* quantitatively. The tree must discover that Site D has ~71 daily visits and a 17.4% admit rate through splits — burning tree depth on something that should be a given.

**Fix:** Replace (or augment) the raw `site` categorical with continuous **target-encoded** features that embed Site D's behavioral profile directly into the feature space:

| Feature | Computation | Rationale |
|---------|-------------|-----------|
| `te_site_mean_total` | Trailing 90-day mean of `total_enc` for this site (lagged by `min_lag`) | Gives the tree the site's current volume level as a number, not a category. A holiday feature can now learn "+10% of baseline" and apply it correctly to Site D's 71 vs Site B's 162 |
| `te_site_mean_admitted` | Trailing 90-day mean of `admitted_enc` for this site (lagged by `min_lag`) | Same for the prediction target — the tree knows Site D's baseline is ~3/block, not ~10/block |
| `te_site_block_mean` | Trailing 90-day mean of target for this `(site, block)` combo (lagged) | Encodes that Site D Block 0 averages 1.24 admitted vs Site A Block 0 at 4.50 — critical for block-level granularity |
| `te_site_dow_mean` | Trailing 90-day mean of target for this `(site, dow)` combo (lagged) | Encodes Site D's DOW pattern: almost flat (weekday/weekend gap = −0.1%) vs Site A's visible weekday drop |
| `te_site_admit_rate` | Trailing 90-day `admitted_enc / total_enc` for this site (lagged) | Directly encodes the 17.4% vs 31.6% structural gap as a feature instead of forcing the tree to discover it |
| `te_site_month_mean` | Trailing mean of target for this `(site, month)` combo across prior years (lagged) | Encodes Site D's seasonal shape (July peak = 80.8/day vs January = 67.8/day) |

**Why this works where raw `site` fails:**
- Trees can now split on `te_site_mean_total < 80` to isolate low-volume sites (D, and sometimes C) without burning a split on the categorical
- Multiplicative effects (holiday = +10%, flu season = +15%) are applied to the *encoded baseline* rather than a global mean — so Site D gets +10% of 71, not +10% of 112
- The features are **continuous and ordered** — the tree naturally handles the gradient from B (162) → A (119) → C (94) → D (71) without needing 3 binary splits

**Leakage prevention:** All target-encoded features use a trailing window with the same `min_lag` shift as existing lag features. Never use current-period actuals. During CV, the trailing window is computed only on the training fold.

**Implementation location:** Add to `build_features()` (Pipeline A), `build_bucket_data()` (Pipeline B), factor feature step (Pipeline E). These are standard lagged aggregation features — same pattern as existing `lag_total_{k}` and `roll_mean_{w}`.

#### 11.3.2 Component 2: Hybrid-Residual Model (Fix the Tail Error)

Target encoding fixes the baseline problem, but the global model will still make systematic errors on Site D because:
- It optimizes for all 4 sites jointly — Site D's loss gradient is 16% of the total
- Site D's unique variance patterns (zero-inflation, low-acuity mix) get averaged out in global tree splits

The hybrid-residual approach decomposes the prediction into two stages:

```
Stage 1 (Global):   ŷ_global = GlobalGBDT(X_all_sites)     ← captures shared patterns
Stage 2 (Local):    ŷ_residual = LocalGBDT_D(X_site_d)     ← captures Site D's deviations
Final:              ŷ_D = ŷ_global_D + ŷ_residual
```

**Stage 1 — Global Model (unchanged):**
The existing Pipelines A/B/E trained on all sites with the new target-encoded features from §11.3.1. This model captures:
- Seasonal patterns (shared across sites)
- Weather effects (shared, since D uses the same Fargo station as A/C)
- DOW patterns (largely shared)
- Holiday effects (shared)
- Trend (shared)

The global model will get Site D's *level* approximately right (thanks to target encoding) but will have systematic residual patterns — e.g., consistently overpredicting Block 0, missing the zero-inflation spikes, under-responding to Site D's weaker weekend effect.

**Stage 2 — Local Residual Model (NEW):**

| Aspect | Specification |
|--------|--------------|
| **Training target** | `residual_D = actual_D − ŷ_global_D` (out-of-fold predictions from Stage 1 CV) |
| **Training data** | Site D rows only (~11,200 blocks, ~2,800 per block) |
| **Features** | Subset of the global feature set + Site D-specific additions (see below) |
| **Model** | LightGBM with **aggressive regularization**: `num_leaves=15`, `min_child_samples=50`, `lambda_l2=5.0`, `max_depth=4` |
| **Why heavy regularization?** | Only ~11K training rows. Overfitting the residuals is worse than underfitting them — a bad residual model will *increase* error. The model should only correct strong, consistent biases, not chase noise |

**Local model features (Site D only):**

| Feature | Why |
|---------|-----|
| `block` | Block 0 has structurally different residuals (overprediction bias due to zero-inflation) |
| `dow`, `month` | Site D's DOW/seasonal patterns may deviate from the global model's assumptions |
| `te_site_block_mean` | The target-encoded baseline — residuals likely correlate with volume level |
| `lag_admitted_{k}` (Site D) | Site D's own recent history — captures local autocorrelation the global model missed |
| `lag_total_{k}` (Site D) | Total volume context |
| `ŷ_global_D` | The global model's own prediction — residuals may be heteroscedastic (larger when prediction is high) |
| `is_covid_era`, `is_holiday` | Regime indicators where the global model may systematically err for D |

**What NOT to include:** Weather, reason-mix features, cross-site lags. These are already captured by the global model — re-including them gives the local model a path to overfit by re-learning the global signal instead of correcting residuals.

**Stage 2 training procedure:**
1. Run Stage 1 (global model) 4-fold CV → collect out-of-fold predictions for Site D
2. Compute residuals: `r = actual_D − oof_pred_D`
3. Train Stage 2 on `(X_D, r)` using nested 3-fold CV within the training portion to tune `shrinkage_weight`
4. Final prediction: `ŷ_D = global_pred_D + shrinkage_weight × local_pred_residual_D`
5. `shrinkage_weight ∈ [0.3, 0.8]`, tuned on the inner CV — this controls how much the local correction is trusted

**Why shrinkage_weight < 1.0?** The local model is trained on ~2,800 rows per block. It WILL have estimation noise. A shrinkage weight of 0.5 means "trust the residual correction at 50%" — this is the bias-variance tradeoff. If the local model is well-calibrated, the weight will converge toward 0.7–0.8; if it's noisy, toward 0.3–0.4.

#### 11.3.3 Component 3: Zero-Inflation Post-Hoc Correction (Block 0 Only)

The hybrid-residual model corrects the *mean* prediction but still outputs continuous values. Site D Block 0 has 29% zero-nights for `admitted_enc` — the final ensemble will predict ~1.0–1.5 for every overnight block when the correct answer is often exactly 0.

**After** the hybrid-residual produces its final prediction, apply a targeted zero-inflation adjustment:

**Step 1 — Train a lightweight zero-classifier:**
- Target: `admitted_enc == 0` (binary)
- Features: `block`, `dow`, `month`, `total_enc_predicted` (from the ensemble's total prediction), `lag_admitted_{k}` from Site D
- Model: Logistic regression or small LightGBM classifier (`max_depth=3`, few trees)
- Training data: Site D blocks where `admitted_enc == 0` rate > 5% (Blocks 0, 1, 3)

**Step 2 — Apply correction:**
```python
p_zero = zero_classifier.predict_proba(X_site_d)[:, 1]
# Shrink prediction toward zero when classifier is confident
corrected = hybrid_pred * (1 - p_zero * shrinkage_factor)
# shrinkage_factor ∈ [0.3, 0.7], tuned on CV folds
```

This is conservative — it nudges predictions downward on nights the classifier thinks will be zero. The `shrinkage_factor` prevents overcorrection. It's easy to ablate (set `shrinkage_factor=0`) and runs after all other models.

### 11.4 Pipeline D — Mixed-Effects GLM (Separate Track)

Pipeline D (GLM/GAM) is architecturally different from the GBDT pipelines and can implement **true partial pooling** natively. The hybrid-residual approach above applies to Pipelines A/B/E; Pipeline D gets its own fix.

**Change:** Replace the fixed `site` dummy with a **random intercept** per site using a mixed-effects formulation:

```
log(admitted_enc + 1) ~ Fourier(day_of_year) + DOW + is_holiday + trend
                        + (1 | site)           ← random intercept
                        + (1 | site:block)     ← random intercept per site×block
```

This shrinks Site D's intercept toward the population mean proportional to its variance — low-volume Site D Block 0 gets pulled toward the population pattern more than high-volume Site B Block 2. Unlike GBDT, this is genuine partial pooling with a principled shrinkage estimator (REML).

**Implementation:** Use `statsmodels.formula.api.mixedlm()` or `pymer4`.

### 11.5 Admit-Rate Isolation

Site D's 17.4% admit rate is **14pp below** A/C (~31%). Pipelines that model `admit_rate` as a derived feature (total × rate = admitted) should:

1. **Separate admit-rate models per site group:** {A, C} share similar rates (~30%), B is mid (25.5%), D is isolated (17.4%). A single admit-rate model across all sites forces the model to explain a 14pp structural gap via features — fitting separate models per group (or a global model with strong target encoding per §11.3.1) is more efficient.

2. **Admit-rate guardrails for Site D Block 0:** Empirical admit rate at Site D Block 0 is 16.7% with high variance. Clamp predicted admit rates to the historical [5th, 95th] percentile range (~8%–30%) as a sanity floor/ceiling.

### 11.6 Implementation Sequence & Ablation Protocol

Each component is **independent and additive** — if any phase hurts, disable it without affecting the others.

| Phase | Component | Pipelines | Depends On |
|:-----:|-----------|-----------|:----------:|
| 1 | Target-encoded features (§11.3.1) | A, B, E | — |
| 2 | Mixed-effects GLM (§11.4) | D only | — |
| 3 | Hybrid-residual model (§11.3.2) | A, B, E | Phase 1 (needs global model OOF preds) |
| 4 | Zero-inflation correction (§11.3.3) | Ensemble | Phase 3 (runs on final preds) |
| 5 | Admit-rate isolation (§11.5) | Any using rate decomposition | — |

**Ablation at each phase:** Re-run 4-fold CV. Check:

| Check | Pass Criterion |
|-------|---------------|
| Site D admitted WAPE improves | Δ ≤ −0.01 (any improvement counts) |
| Sites A/B/C admitted WAPE not degraded | Δ ≤ +0.005 (tight — no collateral damage) |
| Overall mean admitted WAPE improves | Net positive vs previous phase |

**Critical:** Phase 3 (hybrid-residual) must be validated using **nested CV** — the residual model trains on inner folds only. Using the same OOF residuals for both training and evaluation would overestimate the residual model's contribution.

### 11.7 Expected Impact

| Component | Site D WAPE Impact (est.) | Confidence | Risk |
|-----------|:------------------------:|:----------:|:----:|
| Target encoding | −0.03 to −0.06 | High — fixes a known mechanism (baseline bias) | Low |
| Hybrid-residual | −0.04 to −0.08 | Medium — depends on regularization tuning | Medium (overfit risk) |
| Zero-inflation correction | −0.01 to −0.03 | Medium — narrow scope (Block 0 only) | Low |
| Pipeline D mixed-effects | −0.02 to −0.04 | Medium — Pipeline D is weakest overall | Low |
| **Combined (Pipelines A/B/E)** | **−0.07 to −0.14** | **Medium** | **Medium** |

Reducing Site D admitted WAPE from ~0.47 to ~0.35 would bring the overall mean admitted WAPE from ~0.277 to ~0.255 — a ~2pp improvement from a single-site fix. If the residual model is well-calibrated, this could push Site D below 0.40, which would be competitive.

### 11.8 What This Won't Fix

- **Site D Block 0 admitted CV of 0.926** — near-irreducible noise. The residual model corrects *bias* (systematic over/under-prediction), not *variance* (per-night randomness). The best we can do is get the average right.
- **The WAPE metric's penalty structure** — A 3-patient miss at Site D costs 24% WAPE vs 8% at Site A. This is a property of the metric, not the model. No modeling trick changes this arithmetic.
- **Site D's reason-mix opacity** — 52.5% in "other" limits reason-based features. This is a data limitation, not addressable by the residual model.

### 11.9 Why This Beats Vanilla Partial Pooling

| Aspect | Vanilla (current) | Upgraded (§11.3) |
|--------|-------------------|-------------------|
| How Site D baseline is communicated | Categorical `site=D` → tree must discover volume level | Continuous `te_site_mean=71` → tree knows immediately |
| Global model's loss allocation | Site D = 16% of gradient → drowned out | Same, but target encoding means the global model's errors on D are smaller → residual model cleans up the rest |
| Local deviation handling | None — global model or nothing | Dedicated residual model with heavy regularization |
| Zero-inflation | Ignored — continuous prediction for a zero-inflated target | Explicitly modeled post-hoc |
| GBDT "partial pooling" | Binary: split on site (local) or don't (total) | Target encoding gives continuous interpolation; residual model adds local correction |
| Failure mode | Silently overpredicts Site D by borrowing from B's patterns | Residual model has `shrinkage_weight` safety valve — worst case reverts to global-only |
