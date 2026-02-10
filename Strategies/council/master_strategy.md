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
- **Data Source → Pipelines**: `master_block_history.parquet` with schema defined in `data_source.md` §2. Block numbering is `0-3` (matching `eval.md`). No imputation — NaN preserved.
- **Pipelines → Eval**: Submission-shaped CSV per fold, with columns `(Site, Date, Block, ED Enc, ED Enc Admitted)` and all hard constraints from `eval.md`.
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
