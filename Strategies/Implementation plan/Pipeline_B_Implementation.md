# Pipeline B: Direct Multi-Step GBDT — Implementation Plan

**Status:** READY FOR IMPLEMENTATION  
**Source:** `master_strategy.md` §3.2  
**Core Idea:** Train horizon-aware GBDT models so short-horizon predictions exploit recent lags that Pipeline A throws away.  
**Location:** `Pipelines/pipeline_B/`

---

## File Structure

```
Pipelines/pipeline_B/
├── config.py          # Constants, bucket definitions, hyperparameter search spaces
├── data_loader.py     # Load master_block_history, COVID policy, admit_rate derivation
├── features.py        # Horizon-adaptive feature engineering (lags, rolling, calendar, interactions)
├── training.py        # LightGBM training + Optuna tuning per bucket
├── predict.py         # Prediction, constraint enforcement, post-processing
├── evaluate.py        # 4-fold forward validation wrapper using eval.md contract
├── run_pipeline.py    # Full orchestrator: data → features → train → predict → eval
└── __init__.py
```

---

## Step 0: Configuration (`config.py`)

### What It Does
Centralizes all pipeline constants so nothing is hardcoded in logic files.

### Key Definitions
- **Horizon Buckets** (Option B1 — PREFERRED per master strategy):
  - Bucket 1: days 1–15 → `min_lag = 16`
  - Bucket 2: days 16–30 → `min_lag = 31`
  - Bucket 3: days 31–61 → `min_lag = 62`
- **Lag sets per bucket:**
  - Bucket 1: `[16, 21, 28, 56, 91, 182, 364]`
  - Bucket 2: `[31, 35, 42, 56, 91, 182, 364]`
  - Bucket 3: `[63, 70, 77, 91, 182, 364]`
- **Rolling windows:** `[7, 14, 28, 56, 91]` — each shifted by bucket's `min_lag`
- **COVID era:** 2020-03-01 to 2021-06-30
- **COVID policy:** Downweight (weight=0.1) — Option 3 from §3.0
- **Validation folds:** 4-period forward validation from eval.md §2.1
- **Targets:** `total_enc` (tweedie/poisson), `admit_rate` (squared error, bounded [0,1])
- **Optuna trials:** 100 per bucket-model pair
- **LightGBM search space ranges** from §3.1

### Eval Check for Step 0
- [ ] All bucket lag sets satisfy `min(lags) >= bucket_min_lag` — no leakage possible
- [ ] Fold definitions match eval.md exactly (dates, train_end boundaries)
- [ ] Print config summary on import for manual review

---

## Step 1: Data Loading & Preprocessing (`data_loader.py`)

### What It Does
Loads the unified `master_block_history.parquet` from the Data Source layer and prepares it for Pipeline B's feature engineering.

### Sub-Steps

#### 1.1 Load Master Dataset
- Read `Pipelines/Data Source/Data/master_block_history.parquet`
- Validate schema: assert required columns exist (`site`, `date`, `block`, `total_enc`, `admitted_enc`, calendar columns, weather columns, event columns)
- Sort by `(site, block, date)` — critical for lag/rolling computation

#### 1.2 Derive Admit Rate
- `admit_rate = admitted_enc / total_enc`
- Handle division by zero: where `total_enc == 0`, set `admit_rate = 0.0`
- Clip to `[0, 1]` as safety net

#### 1.3 Apply COVID Sample Weights
- Create `sample_weight` column:
  - Default: `total_enc` (WAPE-aligned, per §3.1)
  - COVID era rows (2020-03-01 to 2021-06-30): multiply by 0.1
  - Minimum weight floor: 1.0 (avoid zero-weight rows)
- Formula: `sample_weight = max(total_enc, 1) * (0.1 if is_covid_era else 1.0)`

#### 1.4 Weather Imputation (Pipeline-Specific)
- Forward-fill weather columns (`temp_min`, `temp_max`, `precip`, `snowfall`) within each site
- Then backward-fill remaining NaNs
- Remaining NaNs (start of series): fill with site-level climatology (monthly mean)

#### 1.5 Filter Training Window
- Provide a `get_fold_data(df, train_end)` function that returns:
  - `train_df`: rows where `date <= train_end`
  - Full df preserved for lag computation (lags need history before train_end)

### Eval Check for Step 1
- [ ] Row count matches expected: `4 sites × 4 blocks × N_days`
- [ ] No NaN in `total_enc`, `admitted_enc` (should be 0-filled by Data Source)
- [ ] `admit_rate` is in `[0, 1]` for all rows
- [ ] Sample weights: COVID-era rows have weights ~10× smaller than equivalent non-COVID rows
- [ ] Weather columns have no remaining NaN after imputation
- [ ] Print: date range, row count, target distributions (mean/std/min/max per site)

---

## Step 2: Feature Engineering (`features.py`)

### What It Does
Builds horizon-adaptive features per bucket. This is the **core differentiator** of Pipeline B over Pipeline A — shorter-horizon buckets get access to more recent lag data.

### Sub-Steps

#### 2.1 Create Supervised Training Examples
For each row `(site, date, block)` with target at date `t`, create training examples for each horizon `h` in `[1..61]`:
- Features computed at date `t - h` (the "as-of" date)
- Target: `total_enc` (or `admit_rate`) at date `t`
- Add `days_ahead = h` as a feature
- Assign to bucket based on `h`: Bucket 1 (1-15), Bucket 2 (16-30), Bucket 3 (31-61)

**Memory Note (from §3.2):** This multiplies training data by ~61×. For ~40K original rows → ~2.4M rows. Mitigate by:
- Sub-sampling horizons (every 3rd day within each bucket)
- Or building features only for bucket-representative horizons: `h ∈ {1,4,7,10,13}` for Bucket 1, `{16,20,24,28}` for Bucket 2, `{31,40,50,61}` for Bucket 3

#### 2.2 Horizon-Adaptive Lag Features
Per bucket, compute lags shifted from the "as-of" date:

| Bucket | Lags (days from as-of date) | Rolling Shift |
|--------|----------------------------|---------------|
| 1 (d1–15) | `[16, 21, 28, 56, 91, 182, 364]` | shift by 16 |
| 2 (d16–30) | `[31, 35, 42, 56, 91, 182, 364]` | shift by 31 |
| 3 (d31–61) | `[63, 70, 77, 91, 182, 364]` | shift by 63 |

Implementation: for a given `(site, block)` series sorted by date:
```
lag_k(t) = target[t - k]   where k >= bucket_min_lag
```

**Critical:** Lags are computed on the TARGET series (total_enc / admit_rate), grouped by `(site, block)`.

#### 2.3 Rolling Statistics (Horizon-Shifted)
For each bucket's `min_lag`, compute rolling stats shifted by that amount:
- `roll_mean_{w}` = `series.shift(min_lag).rolling(w).mean()`
- `roll_std_{w}` = `series.shift(min_lag).rolling(w).std()`
- `roll_min_{w}` = `series.shift(min_lag).rolling(w).min()`
- `roll_max_{w}` = `series.shift(min_lag).rolling(w).max()`

Windows: `[7, 14, 28, 56, 91]`

#### 2.4 Trend Deltas
Capture momentum from shifted rolling features:
- `delta_7_28 = roll_mean_7 - roll_mean_28`
- `delta_28_91 = roll_mean_28 - roll_mean_91`
- `lag_diff = lag[min] - lag[min+7]` (week-over-week change at safe horizon)

#### 2.5 Calendar & Cyclical Features
Same for all buckets (deterministic, no leakage):
- `dow`, `month`, `day`, `week_of_year`, `quarter`, `is_weekend`
- Cyclical: `dow_sin`, `dow_cos` (period=7), `month_sin`, `month_cos` (period=12), `doy_sin`, `doy_cos` (period=365.25)
- `days_since_epoch` (trend)
- `year_frac` (continuous year for trend)

#### 2.6 Holiday & Event Features
- `is_holiday`, `is_halloween`, `event_count`
- Holiday proximity: `days_since_xmas`, `days_until_thanksgiving`, `days_since_july4`
  - Compute from the nearest occurrence relative to row date
- School: `school_in_session`, `days_since_school_start`, `days_until_school_start`

#### 2.7 Interaction Features
- `is_holiday × block`
- `is_weekend × block`
- `site × dow` (label-encoded or target-encoded)
- `site × month`

#### 2.8 Aggregate / Mean-Encoding Features (Lagged)
- Historical mean `total_enc` by `(site, month, block)` — computed only from data before the as-of date
- Historical mean `total_enc` by `(dow)` — same constraint

#### 2.9 Weather Features
- `temp_min`, `temp_max`, `precip`, `snowfall` (already imputed in Step 1)
- Optional: `temp_range = temp_max - temp_min`

#### 2.10 `days_ahead` Feature
- Continuous integer feature: the horizon distance for this training example
- This is Pipeline B's signature feature — lets the model learn horizon-dependent patterns

### Eval Check for Step 2
- [ ] **Leakage audit**: For bucket `k`, verify NO lag feature uses data closer than `bucket_min_lag` days to the target date. Test by asserting `min_lag_used >= min_lag_required` for each bucket.
- [ ] **NaN budget**: After feature engineering, count NaN per column. Lags at the start of history will be NaN — verify they're dropped from training (not filled with 0, which would be misleading).
- [ ] **Feature count**: Print total features per bucket (expected: ~80-120).
- [ ] **Spot-check**: For 5 random training examples, manually verify lag values against raw data.
- [ ] **Shape check**: Training data per bucket should be: `~N_train_rows × subsample_horizons × 4_sites × 4_blocks`. Print actual vs expected.

---

## Step 3: Model Training (`training.py`)

### What It Does
Trains 6 LightGBM models: 2 targets (total_enc, admit_rate) × 3 buckets.

### Sub-Steps

#### 3.1 Train/Validation Split Within Fold
For each of the 4 forward-validation folds:
- **Train set**: All supervised examples where the TARGET date ≤ `train_end`
- **Internal validation**: Last 30 days of the train set (for early stopping only — NOT for fold scoring)
- **Held-out**: Fold validation window (for WAPE scoring via eval.md contract)

#### 3.2 LightGBM Configuration Per Target

**Model B_total (total_enc):**
- `objective`: `tweedie` (variance_power ≈ 1.5) OR `poisson`
- `metric`: `mae` (proxy for WAPE during training)
- `sample_weight`: from Step 1.3
- Categorical features: `site`, `block` (use LightGBM native categorical)

**Model B_rate (admit_rate):**
- `objective`: `regression` (MSE)
- `metric`: `mae`
- `sample_weight`: `admitted_enc` (from original data, pre-rate) × COVID factor
- Post-prediction: clip to `[0, 1]`

#### 3.3 Optuna Hyperparameter Tuning
Per bucket × per target (6 Optuna studies total):

**Search Space:**
```
n_estimators:      [800, 3000]
max_depth:         [4, 8]
learning_rate:     [0.01, 0.05]
subsample:         [0.7, 0.95]
colsample_bytree:  [0.6, 0.9]
reg_lambda:        [1, 10]
min_child_weight:  [1, 10]
num_leaves:        [31, 255]
```

**Objective function:** Mean admitted WAPE across internal validation rows (for rate model) or mean WAPE for total model.

**Early stopping:** 50 rounds patience on internal validation set.

**Trials:** 100 per study (600 total). If compute-constrained, reduce to 50.

#### 3.4 Model Serialization
- Save each trained model as `model_{bucket}_{target}.pkl`
- Save Optuna best params as `best_params_{bucket}_{target}.json`
- Save feature importance as `feature_importance_{bucket}_{target}.csv`

### Eval Check for Step 3
- [ ] **No leakage in folds**: Verify that for fold k, NO training example has a target date after `train_end[k]`. Print max target date in training set vs train_end.
- [ ] **Training convergence**: For each model, print final train vs validation metric. Flag if validation metric > 2× train metric (overfitting).
- [ ] **Feature importance sanity**: Top 5 features should include lags and/or rolling stats. If `days_ahead` dominates, it suggests the model is just learning horizon-dependent baselines — not enough signal from lags.
- [ ] **Hyperparameter convergence**: Check Optuna study — if best trial is one of the last 10, might need more trials.
- [ ] **Per-bucket comparison**: Print WAPE per bucket. Bucket 1 should have lowest WAPE (most recent lags); Bucket 3 highest. If inverted, something is wrong.
- [ ] **Model file sizes**: Sanity check that .pkl files are reasonable (1-50 MB each).

---

## Step 4: Prediction & Post-Processing (`predict.py`)

### What It Does
Generates forecasts for a given date range, assembles submission-shaped output, and enforces all hard constraints.

### Sub-Steps

#### 4.1 Assign Forecast Rows to Buckets
For the forecast window `[test_start, test_end]`:
- Compute `days_ahead` for each row relative to the last training date (`train_end`)
- Assign each row to its bucket: Bucket 1 (d1–15), Bucket 2 (d16–30), Bucket 3 (d31–61)

#### 4.2 Build Features for Forecast Rows
- Use the same `features.py` functions, but now the "as-of" date for features is computed from `train_end` offset by `days_ahead`
- Lag features pull from ACTUAL historical data (no recursive predictions needed — this is the Direct approach)
- Calendar/event features are deterministic for future dates
- Weather for forecast dates: if unavailable, use climatology (monthly avg) or last-known value

#### 4.3 Generate Raw Predictions
For each bucket:
1. Load the bucket's trained `model_total` and `model_rate`
2. Predict on the bucket's forecast rows
3. `pred_total = model_total.predict(X_bucket)`
4. `pred_rate = model_rate.predict(X_bucket)` → clip to `[0, 1]`
5. `pred_admitted = pred_total × pred_rate`

#### 4.4 Enforce Hard Constraints
Sequentially apply:
1. **Non-negativity**: `pred_total = max(pred_total, 0)`, same for admitted
2. **admitted ≤ total**: already enforced by construction (`total × rate` where `rate ∈ [0,1]`), but clip again as safety
3. **Integer rounding**: Apply **largest-remainder rounding** per (site, date) to preserve daily totals:
   - Round down all block predictions
   - Distribute remainders to blocks with largest fractional parts
4. **Final admitted ≤ total check**: after rounding, re-enforce

#### 4.5 Assemble Submission DataFrame
Output columns (exact names from eval.md):
- `Site`, `Date`, `Block`, `ED Enc`, `ED Enc Admitted`
- Verify row count = `4 sites × N_days × 4 blocks`
- Save as CSV per fold

### Eval Check for Step 4
- [ ] **Bucket assignment**: Verify no row falls outside [1, 61] days_ahead range. Print distribution of rows per bucket.
- [ ] **Feature parity**: Assert forecast features have same columns as training features (no extra, no missing).
- [ ] **Prediction distribution**: Print mean/std/min/max of pred_total and pred_admitted per site. Compare against historical averages — should be in the same ballpark.
- [ ] **Constraint satisfaction**: Assert `ED Enc Admitted <= ED Enc` for ALL rows. Assert all values >= 0.
- [ ] **Integer check**: Assert all output values are integers.
- [ ] **Row count**: Assert output has exactly `4 × N_days × 4` rows.
- [ ] **Schema check**: Assert column names match eval.md exactly.

---

## Step 5: Evaluation (`evaluate.py`)

### What It Does
Runs the full 4-fold forward validation and produces scored results per the eval.md contract.

### Sub-Steps

#### 5.1 Fold Loop
For each fold in the 4-period validation:

| Fold | Train ≤ | Validate |
|------|---------|----------|
| 1 | 2024-12-31 | 2025-01-01 – 2025-02-28 |
| 2 | 2025-02-28 | 2025-03-01 – 2025-04-30 |
| 3 | 2025-04-30 | 2025-05-01 – 2025-06-30 |
| 4 | 2025-06-30 | 2025-07-01 – 2025-08-31 |

1. Load master data → filter to `date <= train_end` for training
2. Build features (Step 2) for training data
3. Train models (Step 3) on this fold's training data
4. Predict (Step 4) on the validation window
5. Score using eval.md's `score_window()` function

#### 5.2 Metrics Collection
Per fold, collect:
- `primary_admitted_wape` (THE ranking metric)
- `total_wape`, `admitted_wape`
- `total_rmse`, `admitted_rmse`
- `total_r2`, `admitted_r2`
- By-site breakdown (WAPE per site)
- By-block breakdown (WAPE per block)

#### 5.3 Summary Statistics
- **Mean admitted WAPE across 4 folds** = Pipeline B's final score
- Per-fold variance: if one fold dominates, flag temporal drift
- Per-bucket WAPE: verify Bucket 1 < Bucket 2 < Bucket 3

#### 5.4 Save OOF Predictions
- Save out-of-fold predictions for ALL 4 folds as `oof_predictions.csv`
- These are used downstream for ensemble stacking (§4.2 of master strategy)

### Eval Check for Step 5
- [ ] **No leakage across folds**: For fold k, assert max training date ≤ train_end[k].
- [ ] **Submission contract**: Each fold's predictions pass `validate_prediction_df()` from eval.md evaluator.
- [ ] **Sanity thresholds** (from master strategy §2.4):
  - No single site WAPE > 2× best site WAPE
  - Block-level WAPE stable (Block 0 overnight doesn't drift)
  - Per-fold variance is reasonable (std/mean < 0.3)
- [ ] **Comparison baseline**: Report how Pipeline B compares to a naive baseline (same-period-last-year).
- [ ] **OOF file**: oof_predictions.csv has correct shape (sum of all 4 folds' validation rows).

---

## Step 6: Orchestrator (`run_pipeline.py`)

### What It Does
Single entry point that runs the entire Pipeline B end-to-end.

### Execution Modes

#### Mode 1: Full CV Evaluation (default)
```bash
python Pipelines/pipeline_B/run_pipeline.py --mode cv
```
Runs all 4 folds, trains, predicts, evaluates, saves results.

#### Mode 2: Final Submission
```bash
python Pipelines/pipeline_B/run_pipeline.py --mode submit
```
Trains on ALL data through Aug 2025, predicts Sept-Oct 2025 (976 rows).

#### Mode 3: Single Fold (for debugging)
```bash
python Pipelines/pipeline_B/run_pipeline.py --mode fold --fold-id 1
```

### Pipeline Flow
```
1. Load config (Step 0)
2. Load & preprocess data (Step 1)
   → CHECKPOINT: validate data shape, print summary
3. Build features per bucket (Step 2)
   → CHECKPOINT: leakage audit, NaN count, feature list
4. Train models (Step 3)
   → CHECKPOINT: convergence metrics, feature importance
5. Predict on validation/test window (Step 4)
   → CHECKPOINT: constraint checks, distribution comparison
6. Evaluate (Step 5)
   → CHECKPOINT: WAPE scores, sanity thresholds
7. Save all artifacts to Pipelines/pipeline_B/output/
```

### Output Artifacts
```
Pipelines/pipeline_B/output/
├── models/
│   ├── fold_{k}/
│   │   ├── model_bucket{1,2,3}_total.pkl
│   │   ├── model_bucket{1,2,3}_rate.pkl
│   │   └── best_params_bucket{1,2,3}_{total,rate}.json
├── predictions/
│   ├── fold_{k}_predictions.csv      (submission-shaped per fold)
│   └── oof_predictions.csv           (all folds combined for stacking)
├── evaluation/
│   ├── cv_results.csv                (fold-level metrics)
│   ├── cv_summary.json               (mean WAPE + diagnostics)
│   ├── by_site_wape.csv
│   └── by_block_wape.csv
├── feature_importance/
│   └── importance_bucket{1,2,3}_{total,rate}.csv
└── logs/
    └── pipeline_b_run_{timestamp}.log
```

### Eval Check for Step 6
- [ ] **End-to-end smoke test**: Run on 1 fold with reduced Optuna trials (5) to verify the full pipeline executes without errors.
- [ ] **Reproducibility**: Set random seeds in config. Re-running should produce identical results.
- [ ] **Runtime**: Log wall-clock time per step. Full CV should complete in <2 hours on a modern laptop.
- [ ] **Artifact completeness**: After a full run, verify all expected output files exist and are non-empty.

---

## Appendix A: Option B2 — Single Model with `days_ahead` Feature (Alternative)

If Option B1 (bucket models) proves too expensive or underperforms:

### Differences from B1
- **One model** instead of 3 per target (2 total instead of 6)
- `days_ahead` is a continuous feature (1-61)
- Lags: compute ALL lags from 1-364, but **mask** those where `lag_k < days_ahead + 1` (set to NaN; LightGBM handles natively)
- Training data: ~61× original size (subsample every 3rd horizon if memory-constrained)

### When to Use B2 Over B1
- Bucket boundaries feel arbitrary and Bucket 3 has too few distinct lag features
- The model is already large and 6 models is compute-heavy
- B2's mean WAPE is within 0.5% of B1 in preliminary testing

---

## Appendix B: Option B3 — Recursive GBDT (Fallback Only)

Per master strategy: use ONLY if B1/B2 are too computationally expensive.

- Single model predicting t+1
- Use own predictions as lag inputs for t+2, t+3, ...
- **Known issue**: Error accumulation over 61 steps
- Not recommended; included for completeness

---

## Appendix C: Key Differences from Pipeline A

| Aspect | Pipeline A | Pipeline B (this) |
|--------|-----------|-------------------|
| Horizon handling | All future rows identical | Explicit horizon awareness via buckets |
| Minimum lag | 63 days (worst-case safe) | 16/31/62 per bucket (adaptive) |
| Number of models | 2 (total + rate) | 6 (2 targets × 3 buckets) |
| Key advantage | Simple, robust | Short-horizon gets recent lags (biggest accuracy lever) |
| Training data | ~40K rows | ~200K-2.4M rows (horizon expansion) |
| Expected WAPE improvement | Baseline | 3-8% over Pipeline A (per master strategy §9) |
