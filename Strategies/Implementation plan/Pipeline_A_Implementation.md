# Pipeline A: Global GBDT — Step-by-Step Implementation Plan

**Pipeline Role:** Primary workhorse — single LightGBM model trained across all (Site, Block) series.  
**Source:** `master_strategy.md` §3.1  
**Code Location:** `Pipelines/pipeline_a/`  
**Data Dependency:** `master_block_history.parquet` produced by the Data Source step (see `data_source.md`). STRICTLY REQUIRED. Pipeline A consumes this unified dataset.

---

## File Map

```
Pipelines/
  pipeline_a/
    config.py                 # Constants, paths, validation folds, feature lists
    step_01_data_loading.py   # Load raw data → block-level aggregation + skeleton grid
    step_02_feature_eng.py    # All feature engineering (lags, rolling, calendar, etc.)
    step_03_train.py          # Train Model A1 (total_enc) + A2 (admit_rate) with default HP
    step_04_tune.py           # Optuna hyperparameter search for A1 + A2
    step_05_predict.py        # Generate forecast + post-processing (clip, round, constraints)
    step_06_evaluate.py       # Score against eval.md contract across 4 folds
    run_pipeline.py           # End-to-end orchestrator (calls steps 1-6 sequentially)
```

---

## Step 0: Configuration (`config.py`)

### Purpose
Single source of truth for all constants, paths, feature lists, and validation fold definitions.

### Contents

```python
# --- Paths ---
RAW_DATA_PATH = "Dataset/DSU-Dataset.csv"
MASTER_PARQUET_PATH = "Pipelines/Data Source/Data/master_block_history.parquet"
OUTPUT_DIR = "Pipelines/pipeline_a/output/"          # fold CSVs go here
MODEL_DIR = "Pipelines/pipeline_a/models/"           # saved .pkl / .txt models

# --- Sites & Blocks ---
SITES = ["A", "B", "C", "D"]
BLOCKS = [0, 1, 2, 3]

# --- Validation Folds (from eval.md) ---
FOLDS = [
    {"id": 1, "train_end": "2024-12-31", "val_start": "2025-01-01", "val_end": "2025-02-28"},
    {"id": 2, "train_end": "2025-02-28", "val_start": "2025-03-01", "val_end": "2025-04-30"},
    {"id": 3, "train_end": "2025-04-30", "val_start": "2025-05-01", "val_end": "2025-06-30"},
    {"id": 4, "train_end": "2025-06-30", "val_start": "2025-07-01", "val_end": "2025-08-31"},
]

# --- COVID Era ---
COVID_START = "2020-03-01"
COVID_END = "2021-06-30"
COVID_SAMPLE_WEIGHT = 0.1          # Downweight factor for COVID rows (Policy 3)

# --- Feature Engineering ---
MAX_HORIZON = 63                   # Max forecast horizon in days (61d + 2d buffer)
LAG_DAYS = [63, 70, 77, 91, 182, 364]
ROLLING_WINDOWS = [7, 14, 28, 56, 91]
ROLLING_SHIFT = 63                 # Shift all rolling by ≥ max horizon

# --- Hyperparameter Defaults (before Optuna) ---
LGBM_DEFAULT_A1 = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 1500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 5,
    "verbosity": -1,
}
LGBM_DEFAULT_A2 = {
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 5,
    "verbosity": -1,
}

# --- Optuna ---
OPTUNA_N_TRIALS = 100
```

### Eval Notes — Step 0
- [ ] **Check**: All fold date ranges match `eval.md` exactly.
- [ ] **Check**: `MAX_HORIZON = 63` is conservative enough (longest val window is 62 days; 63 provides buffer).
- [ ] **Check**: Lag values are all ≥ 63 — no leakage possible.
- [ ] **Check**: `ROLLING_SHIFT = 63` ensures rolling stats don't peek into the forecast horizon.

---

## Step 1: Data Loading & Block Aggregation (`step_01_data_loading.py`)

### Purpose
### Purpose
Load the unified `master_block_history.parquet` produced by the Data Source step. This file already contains the full block grid, core targets (`total_enc`, `admitted_enc`), reason counts, calendar features, and event/weather data.

### Logic

1. **Load Parquet**: Read `MASTER_PARQUET_PATH`.
2. **Validate Schema**:
   - Check critical columns exist: `Site`, `Date`, `Block`, `total_enc`, `admitted_enc`.
   - Check row count: ~45,776 rows (4 sites × 2861 days × 4 blocks).
   - Check no NaNs in targets.
3. **Derive `admit_rate`**:
   - `admit_rate = admitted_enc / total_enc` (fill 0/0 → 0.0).
   - This is a derived feature specific to modeling, not storage.
4. **Basic Pre-processing**:
   - Ensure `Date` is datetime type.
   - Sort by `(Site, Date, Block)`.

**Note:** The Data Source pipeline handles all aggregation from raw CSVs, grid creation, zero-filling, and external data joins. Pipeline A strictly consumes this output.

### Fallback Mode (Dev Only)
If `master_block_history.parquet` is missing (e.g., during isolated testing), this step may temporarily regenerate it from `RAW_DATA_PATH` using the logic defined in `data_source.md`. However, production runs MUST use the shared Data Source artifact.

### Key Implementation Detail: Reason Categories

```python
# Top 20 reason categories by total volume across all sites/dates
top_reasons = (
    df.groupby("REASON_VISIT_NAME")["ED Enc"]
    .sum()
    .nlargest(20)
    .index.tolist()
)
# Pivot + "other" bucket
```

### Eval Notes — Step 1
- [ ] **Row count check**: Skeleton should have `4 sites × num_days × 4 blocks` rows. For 2018-01-01 to 2025-10-31 = 2,861 days → 4 × 2861 × 4 = **45,776 rows**.
- [ ] **Zero-fill check**: After merge, assert `total_enc >= 0` everywhere, no NaN in targets.
- [ ] **Block mapping check**: Verify `Hour // 6` maps to eval.md blocks (0=00-05, 1=06-11, 2=12-17, 3=18-23).
- [ ] **Admitted ≤ Total**: Assert `admitted_enc <= total_enc` for all rows. If violated, there's a data bug upstream.
- [ ] **Print summary stats**: Per-site mean/std/min/max of `total_enc` and `admitted_enc`. Compare visually — Site B should be largest (~36% of total volume per master strategy).
- [ ] **Spot check**: Pick 5 random (Site, Date, Block) tuples, manually verify aggregation against raw CSV.
- [ ] **Date range check**: Min date = 2018-01-01, max date should cover up to 2025-08-31 (last validation end).
- [ ] **Case-mix check**: Sum of `count_reason_*` columns should equal `total_enc` for each row.
- [ ] **`admit_rate` range**: Assert all values in [0.0, 1.0].

---

## Step 2: Feature Engineering (`step_02_feature_eng.py`)

### Purpose
Transform the raw block-level dataset into a model-ready feature matrix. This is the most complex step. All features are computed **per (Site, Block) group** to avoid cross-series contamination.

### Feature Groups

#### 2A. Lagged Target Features

For each `(Site, Block)` group, create:

| Feature | Formula | Notes |
|---------|---------|-------|
| `lag_63` | `total_enc.shift(63)` | Closest safe lag (>= MAX_HORIZON) |
| `lag_70` | `total_enc.shift(70)` | 10 weeks ago (weekly-aligned) |
| `lag_77` | `total_enc.shift(77)` | 11 weeks ago |
| `lag_91` | `total_enc.shift(91)` | ~3 months ago |
| `lag_182` | `total_enc.shift(182)` | ~6 months ago |
| `lag_364` | `total_enc.shift(364)` | Same day last year |

Same lags for `admit_rate`.

#### 2B. Rolling Statistics

For windows `[7, 14, 28, 56, 91]`, all shifted by 63 days:

```python
# Per (Site, Block) group:
for w in ROLLING_WINDOWS:
    shifted = group["total_enc"].shift(ROLLING_SHIFT)
    df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
    df[f"roll_std_{w}"]  = shifted.rolling(w, min_periods=1).std()
    df[f"roll_min_{w}"]  = shifted.rolling(w, min_periods=1).min()
    df[f"roll_max_{w}"]  = shifted.rolling(w, min_periods=1).max()
```

#### 2C. Trend Deltas (M5 winning trick)

| Feature | Formula | Captures |
|---------|---------|----------|
| `delta_7_28` | `roll_mean_7 - roll_mean_28` | Short-term momentum |
| `delta_28_91` | `roll_mean_28 - roll_mean_91` | Medium-term trend |
| `delta_lag_63_70` | `lag_63 - lag_70` | Week-over-week change at horizon boundary |

#### 2D. Calendar & Cyclical Encoding

Already have from Data Source (Step 1): `dow, day, week_of_year, month, quarter, day_of_year, year, is_weekend, days_since_epoch, is_covid_era, is_halloween`

Add cyclical (sin/cos) encodings:
```python
df["dow_sin"]  = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"]  = np.cos(2 * np.pi * df["dow"] / 7)
df["doy_sin"]  = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"]  = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
```

Add continuous year fraction:
```python
df["year_frac"] = df["year"] + (df["day_of_year"] - 1) / 365.25
```

#### 2E. Holiday Features

Using US Federal Holidays (from `holidays` library or manual list):

| Feature | Description |
|---------|-------------|
| `is_us_holiday` | Binary — is this date a US federal holiday? |
| `days_since_xmas` | Days since most recent Dec 25 |
| `days_until_thanksgiving` | Days until next Thanksgiving |
| `days_since_july4` | Days since most recent July 4 |
| `days_to_nearest_holiday` | Min distance to any holiday |

#### 2F. School Calendar Features

| Feature | Description |
|---------|-------------|
| `school_in_session` | Binary — estimated from Sioux Falls/Fargo school calendars |
| `days_since_school_start` | Days since most recent school start date |
| `days_until_school_start` | Days until next school start |

**Implementation note**: If external school calendar data is not yet available, derive approximate dates:
- School starts: ~Aug 20-25 each year
- School ends: ~May 25-30 each year
- Mark as approximate; refine later.

#### 2G. Event Features

Load from `config/events.yaml`:
- `event_count` (already from Step 1 if events are integrated)
- `event_intensity` (if defined in events.yaml, else = `event_count`)

**TODO**: If `events.yaml` only has 2025 events, we need to expand it for 2018-2024 before this step.

#### 2H. Case-Mix Share Features

```python
# Top 5-8 categories as shares (from raw counts in Step 1)
reason_cols = [c for c in df.columns if c.startswith("count_reason_")]
total_reasons = df[reason_cols].sum(axis=1).clip(lower=1)  # avoid div/0
for col in top_5_reason_cols:
    df[col.replace("count_", "share_")] = df[col] / total_reasons
```

Shift by 63 days (use lagged shares, not current — would be leakage):
```python
for col in share_cols:
    df[f"{col}_lag63"] = group[col].shift(63)
```

#### 2I. Aggregate Mean Encodings (M5 trick)

Computed on **training data only** (must be recalculated per fold to avoid leakage):

| Feature | Grouping | Notes |
|---------|----------|-------|
| `site_month_block_mean` | (Site, Month, Block) | Historical mean total_enc |
| `site_dow_mean` | (Site, DOW) | Historical mean total_enc |
| `site_month_mean` | (Site, Month) | Historical mean total_enc |

**CRITICAL**: These must be computed inside the training loop (Step 3), not here, to prevent data leakage from validation periods. Step 2 only creates the infrastructure/function; Step 3 calls it per fold.

#### 2J. Interaction Features

```python
df["holiday_x_block"] = df["is_us_holiday"].astype(int) * df["block"]
df["weekend_x_block"] = df["is_weekend"].astype(int) * df["block"]
df["site_x_dow"] = df["site"].astype(str) + "_" + df["dow"].astype(str)   # categorical
df["site_x_month"] = df["site"].astype(str) + "_" + df["month"].astype(str)  # categorical
```

#### 2K. Weather Features (if available)

If weather data has been integrated in Step 1 (or via Data Source):
- `temp_min`, `temp_max`, `precip`, `snowfall`
- **Imputation**: Forward-fill NaN within each site, then fill remaining with monthly climatology mean.
- Create derived: `temp_range = temp_max - temp_min`

If weather data is NOT yet available, skip these features (model works without them per master strategy §6).

### 2L. Sample Weights

Two separate weight columns:

```python
# COVID downweighting (Policy 3 from §3.0)
df["covid_weight"] = np.where(df["is_covid_era"], COVID_SAMPLE_WEIGHT, 1.0)

# WAPE-aligned volume weighting
df["volume_weight"] = df["total_enc"].clip(lower=1)  # avoid 0-weight

# Combined
df["sample_weight_a1"] = df["covid_weight"] * df["volume_weight"]
df["sample_weight_a2"] = df["covid_weight"] * df["admitted_enc"].clip(lower=1)
```

### Final Output

Save feature-enriched DataFrame to `Pipelines/pipeline_a/data/features.parquet`.

### Feature List Summary

The final model feature columns (excluding targets and weights):

```python
FEATURE_COLS_A1 = [
    # Identifiers (categorical)
    "site", "block",
    # Lags
    "lag_63", "lag_70", "lag_77", "lag_91", "lag_182", "lag_364",
    # Rolling stats (×5 windows × 4 stats = 20 features)
    "roll_mean_7", "roll_std_7", "roll_min_7", "roll_max_7",
    "roll_mean_14", "roll_std_14", "roll_min_14", "roll_max_14",
    "roll_mean_28", "roll_std_28", "roll_min_28", "roll_max_28",
    "roll_mean_56", "roll_std_56", "roll_min_56", "roll_max_56",
    "roll_mean_91", "roll_std_91", "roll_min_91", "roll_max_91",
    # Trend deltas
    "delta_7_28", "delta_28_91", "delta_lag_63_70",
    # Calendar
    "dow", "day", "week_of_year", "month", "quarter", "day_of_year",
    "is_weekend", "is_halloween",
    # Cyclical
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    # Trend
    "days_since_epoch", "year_frac",
    # Holidays
    "is_us_holiday", "days_since_xmas", "days_until_thanksgiving",
    "days_since_july4", "days_to_nearest_holiday",
    # School
    "school_in_session", "days_since_school_start", "days_until_school_start",
    # Events
    "event_count",
    # Case-mix shares (lagged)
    # ... dynamically populated from top 5-8 share_*_lag63 cols
    # Aggregate encodings (computed per fold)
    "site_month_block_mean", "site_dow_mean", "site_month_mean",
    # Interactions
    "holiday_x_block", "weekend_x_block",
    # site_x_dow, site_x_month → handled as categoricals or target-encoded
    # Weather (if available)
    # "temp_min", "temp_max", "precip", "snowfall", "temp_range",
    # COVID indicator
    "is_covid_era",
]
```

### Eval Notes — Step 2
- [ ] **No future leakage**: For each feature, verify it only uses data from ≥63 days ago. Print min shift used.
- [ ] **NaN audit**: Print NaN count per feature column. Lags will naturally have NaN for the first ~364 rows — this is expected. Decide: drop NaN rows from training or let LightGBM handle them natively (recommended — LGBM handles NaN).
- [ ] **Feature count**: Print total number of features. Expected: ~60-80 columns.
- [ ] **Distribution check**: For each lag/rolling feature, print mean/std/min/max. Flag any features with >90% zero values (likely useless).
- [ ] **Correlation check**: Compute pairwise correlation between lag features. If any pair has `|corr| > 0.98`, consider dropping one to reduce noise.
- [ ] **Sample weight check**: Verify `sample_weight_a1 > 0` for all rows. COVID-era rows should have `weight ≈ 0.1 × total_enc`.
- [ ] **Interaction features**: Spot-check that `holiday_x_block` is non-zero only on holidays and encodes the correct block.
- [ ] **Temporal integrity**: Plot `lag_63` vs `total_enc` shifted by 63 days — should be identical (sanity).

---

## Step 3: Model Training (`step_03_train.py`)

### Purpose
Train two LightGBM models per fold using default hyperparameters:
- **Model A1**: Predicts `total_enc` (Tweedie objective)
- **Model A2**: Predicts `admit_rate` (Regression objective)

### Logic (per fold)

```
For each fold in FOLDS:
    1. Split data: train = rows where date <= fold.train_end
                   val   = rows where date in [fold.val_start, fold.val_end]
    
    2. Drop rows where lag features are NaN from training
       (first ~364 days of history will have NaN for lag_364)
    
    3. Compute fold-specific aggregate encodings on TRAIN ONLY:
       - site_month_block_mean = train.groupby([site, month, block]).total_enc.mean()
       - site_dow_mean         = train.groupby([site, dow]).total_enc.mean()
       - site_month_mean       = train.groupby([site, month]).total_enc.mean()
       Map these onto both train and val rows.
    
    4. Encode categoricals:
       - site → LabelEncoder (A=0, B=1, C=2, D=3)
       - site_x_dow, site_x_month → LabelEncoder or let LightGBM handle as categorical
    
    5. Train Model A1 (total_enc):
       - X = train[FEATURE_COLS_A1]
       - y = train["total_enc"]
       - sample_weight = train["sample_weight_a1"]
       - lgb.Dataset with categorical_feature=[site, block, ...]
       - Train with LGBM_DEFAULT_A1 params
       - Use early stopping on val set (eval_metric = "mae" as proxy;
         actual selection is by WAPE computed post-training)
    
    6. Train Model A2 (admit_rate):
       - X = train[FEATURE_COLS_A1]  (same features, or a subset)
       - y = train["admit_rate"]
       - sample_weight = train["sample_weight_a2"]
       - Train with LGBM_DEFAULT_A2 params
    
    7. Predict on val:
       - pred_total = model_a1.predict(val[FEATURE_COLS_A1]).clip(0)
       - pred_rate  = model_a2.predict(val[FEATURE_COLS_A1]).clip(0, 1)
       - pred_admitted = (pred_total * pred_rate).round().astype(int)
       - pred_total = pred_total.round().astype(int)
       - Enforce: pred_admitted = min(pred_admitted, pred_total)
    
    8. Save fold predictions as submission CSV:
       columns: Site, Date, Block, ED Enc, ED Enc Admitted
       → OUTPUT_DIR/fold_{id}_predictions.csv
    
    9. Save models:
       → MODEL_DIR/fold_{id}_model_a1.txt
       → MODEL_DIR/fold_{id}_model_a2.txt
```

### Post-Processing Detail (Integer Rounding with Largest-Remainder)

```python
def largest_remainder_round(values):
    """Round array to integers preserving the sum."""
    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = int(round(values.sum())) - floored.sum()
    # Give +1 to the `deficit` items with largest remainders
    indices = np.argsort(-remainders)[:deficit]
    floored[indices] += 1
    return floored
```

Apply this per (Site, Date) group to ensure block-level predictions sum consistently.

### Eval Notes — Step 3
- [ ] **Training set size**: Print rows per fold after dropping NaN lags. Expected: ~35K-40K rows for early folds, growing for later folds.
- [ ] **Early stopping**: Verify that early stopping triggers (model doesn't use all `n_estimators`). If it does, increase `n_estimators`.
- [ ] **Feature importance**: Print top 20 features by `gain` from LightGBM. Expect lags and rolling means to dominate. If calendar features dominate, something may be leaking.
- [ ] **Prediction range check**:
  - `pred_total`: Should be non-negative integers. Print min/max/mean. Compare to train distribution.
  - `pred_rate`: Should be in [0, 1]. Print any values outside this range before clipping.
  - `pred_admitted`: Should be ≤ `pred_total` for every row after enforcement.
- [ ] **Per-fold WAPE (quick)**: Compute WAPE on val set immediately after prediction. Print for both total and admitted. This gives early signal before the full evaluator runs.
  ```
  Fold 1: total_wape=X.XX, admitted_wape=X.XX
  Fold 2: total_wape=X.XX, admitted_wape=X.XX
  ...
  Mean:   total_wape=X.XX, admitted_wape=X.XX
  ```
- [ ] **Residual analysis**: For each fold, compute residuals `(actual - predicted)`. Check:
  - Mean residual ≈ 0 (no systematic bias)
  - Plot residual vs. predicted — should show no pattern
  - By-site residual means — flag if any site has consistent over/under-prediction
  - By-block residual means — flag if Block 0 (overnight) has high error (common failure mode from master strategy §2.4)
- [ ] **Constraint satisfaction**: Assert `ED Enc Admitted <= ED Enc` for every row in output CSV.
- [ ] **Row count**: Assert each fold CSV has exactly `4 × num_val_days × 4` rows.
- [ ] **No data leakage sanity**: Verify the earliest training date used for lag_364 in fold 1 is ≥ 2018-01-01 + 364 days = 2018-12-31.

---

## Step 4: Hyperparameter Tuning (`step_04_tune.py`)

### Purpose
Use Optuna to find optimal LightGBM hyperparameters, selecting by mean admitted WAPE across all 4 validation folds.

### Search Space

```python
def objective(trial):
    params = {
        "objective": trial.suggest_categorical("objective_a1", ["tweedie", "poisson"]),
        "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    if params["objective"] == "tweedie":
        params["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.1, 1.9
        )
    
    # Train on all 4 folds, compute mean admitted WAPE
    wapes = []
    for fold in FOLDS:
        model, predictions = train_fold(fold, params)
        fold_wape = compute_admitted_wape(predictions, actuals)
        wapes.append(fold_wape)
    
    return np.mean(wapes)  # Optuna minimizes this
```

### Tuning Strategy

1. **Model A1 first** (total_enc): Run 100 Optuna trials. Select best params.
2. **Model A2 second** (admit_rate): Run 50 Optuna trials (smaller search — admit_rate is simpler).
3. **Save** best params to `MODEL_DIR/best_params_a1.json` and `best_params_a2.json`.

### COVID Policy Ablation (During Tuning)

Add COVID policy as a tuning dimension:
```python
covid_policy = trial.suggest_categorical("covid_policy", ["downweight", "exclude"])
# "downweight": use sample_weight = 0.1 for COVID rows
# "exclude": drop rows where is_covid_era == True from training
```

### Eval Notes — Step 4
- [ ] **Optuna convergence**: Plot `trial.value` (WAPE) over trial number. Should plateau after ~50-70 trials. If still decreasing at trial 100, consider extending.
- [ ] **Best vs Default**: Compare best Optuna params' mean WAPE against Step 3's default-param WAPE. Expected improvement: 3-8%.
- [ ] **Param stability**: Check if top-5 trials have similar hyperparams. If wildly different, the landscape is noisy — consider more trials or simpler search.
- [ ] **Overfitting check**: Compare train WAPE vs. val WAPE for the best trial. If train WAPE << val WAPE (e.g., 5x gap), the model is overfitting. Consider:
  - Increase `reg_lambda`
  - Decrease `max_depth`
  - Increase `min_child_weight`
- [ ] **COVID policy result**: Report which COVID policy won (downweight vs. exclude). Log the WAPE difference.
- [ ] **Objective result**: Report whether `tweedie` or `poisson` won for Model A1. Tweedie is expected to win.
- [ ] **Per-fold variance**: Compute std across fold WAPEs for best trial. If one fold is a large outlier, investigate (temporal drift per master strategy §2.4).
- [ ] **Feature importance shift**: Compare top features between default-param model and tuned model. Major shifts may indicate the tuner is exploiting noise.

---

## Step 5: Prediction & Post-Processing (`step_05_predict.py`)

### Purpose
Using the tuned models, generate final submission-format CSVs for each validation fold (and optionally for the Sept-Oct 2025 test period).

### Logic

```
For each fold (or test period):
    1. Load best params from Step 4
    2. Retrain on full training data (≤ train_end) with best params
    3. Build features for prediction window
    4. Predict:
       pred_total = model_a1.predict(X_pred).clip(0)
       pred_rate  = model_a2.predict(X_pred).clip(0, 1)
    5. Derive admitted:
       pred_admitted = pred_total * pred_rate
    6. Post-process:
       a. Clip negatives to 0
       b. Round total_enc via largest-remainder per (Site, Date)
       c. Round admitted_enc via largest-remainder per (Site, Date)
       d. Enforce admitted <= total row-by-row: admitted = min(admitted, total)
    7. Format as submission CSV:
       Site, Date, Block, ED Enc, ED Enc Admitted
    8. Save to OUTPUT_DIR/
```

### Sept-Oct 2025 Final Forecast

For the competition submission:
```
Train on ALL data ≤ 2025-08-31 (applying COVID policy)
Predict 2025-09-01 to 2025-10-31
→ 4 × 61 × 4 = 976 rows
```

### Eval Notes — Step 5
- [ ] **Row count**: Each fold CSV must have exactly `4 × num_val_days × 4` rows. Sept-Oct: 976 rows.
- [ ] **Schema check**: Columns are exactly `["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]`.
- [ ] **Constraint check**:
  - All `ED Enc >= 0` ✓
  - All `ED Enc Admitted >= 0` ✓
  - All `ED Enc Admitted <= ED Enc` ✓
  - All values are integers ✓
- [ ] **No missing combos**: Assert full grid coverage (all sites × all dates × all blocks).
- [ ] **No duplicates**: Assert unique on `(Site, Date, Block)`.
- [ ] **Distribution sanity**: Compare prediction distribution (mean, std, percentiles) to historical training distribution. Flag if mean prediction is >20% off from recent months' average.
- [ ] **Per-site volume share**: Check that Site B still gets ~36% of total volume (it's the largest facility). If predictions give Site B << 30%, investigate.
- [ ] **Block distribution**: Check that block shares roughly match historical patterns (Block 2-3 typically highest for ED).
- [ ] **Largest-remainder rounding**: Verify that per-(Site, Date) block sums are integers and consistent.

---

## Step 6: Evaluation (`step_06_evaluate.py`)

### Purpose
Run the `eval.md` evaluator on Pipeline A's fold CSVs. Produces the official pipeline score.

### Logic

```python
# 1. Load ground truth
df_hourly = pd.read_csv(RAW_DATA_PATH)
truth_blocks = hourly_to_blocks_truth(df_hourly)

# 2. Score each fold
results = []
for fold in FOLDS:
    pred_df = pd.read_csv(f"{OUTPUT_DIR}/fold_{fold['id']}_predictions.csv")
    scored = score_window(truth_blocks, pred_df, fold["val_start"], fold["val_end"])
    results.append(scored)

# 3. Aggregate
mean_admitted_wape = np.mean([r["overall"]["primary_admitted_wape"] for r in results])
print(f"Pipeline A — Mean Admitted WAPE: {mean_admitted_wape:.4f}")
```

### Output Report

```
═══════════════════════════════════════════════════════════
 PIPELINE A: GLOBAL GBDT — EVALUATION REPORT
═══════════════════════════════════════════════════════════

 OVERALL (4-fold mean):
   Primary Admitted WAPE: X.XXXX
   Total WAPE:           X.XXXX
   Admitted RMSE:        X.XX
   Total RMSE:           X.XX

 PER-FOLD:
   Fold 1 (Jan-Feb 2025): admitted_wape=X.XXXX  total_wape=X.XXXX
   Fold 2 (Mar-Apr 2025): admitted_wape=X.XXXX  total_wape=X.XXXX
   Fold 3 (May-Jun 2025): admitted_wape=X.XXXX  total_wape=X.XXXX
   Fold 4 (Jul-Aug 2025): admitted_wape=X.XXXX  total_wape=X.XXXX

 BY-SITE:
   Site A: admitted_wape=X.XXXX  total_wape=X.XXXX
   Site B: admitted_wape=X.XXXX  total_wape=X.XXXX
   Site C: admitted_wape=X.XXXX  total_wape=X.XXXX
   Site D: admitted_wape=X.XXXX  total_wape=X.XXXX

 BY-BLOCK:
   Block 0: admitted_wape=X.XXXX  total_wape=X.XXXX
   Block 1: admitted_wape=X.XXXX  total_wape=X.XXXX
   Block 2: admitted_wape=X.XXXX  total_wape=X.XXXX
   Block 3: admitted_wape=X.XXXX  total_wape=X.XXXX

═══════════════════════════════════════════════════════════
```

### Eval Notes — Step 6
- [ ] **Primary metric**: Mean Admitted WAPE — this is THE number that ranks Pipeline A against other pipelines.
- [ ] **Sanity check (master strategy §2.4)**: No single site's WAPE should exceed 2× the best site's WAPE.
- [ ] **Block stability**: Block-level errors should be roughly uniform. If Block 0 (overnight) has WAPE >2× others, investigate (common failure mode).
- [ ] **Per-fold variance**: If one fold's WAPE is >2× others, investigate temporal drift (e.g., fold 4 = summer might behave differently).
- [ ] **Compare to naive baseline**: Before accepting Pipeline A's score, compute a naive baseline (e.g., same-period-last-year) and verify Pipeline A beats it. If not, something is fundamentally broken.
- [ ] **Save report**: Write evaluation results to `OUTPUT_DIR/evaluation_report.json` for later cross-pipeline comparison.

---

## Step 7: End-to-End Orchestrator (`run_pipeline.py`)

### Purpose
Run the full Pipeline A from data loading to evaluation in one command.

### Usage

```bash
# Run full pipeline with default params
python Pipelines/pipeline_a/run_pipeline.py

# Run only training + eval (skip tuning) — for quick iteration
python Pipelines/pipeline_a/run_pipeline.py --skip-tune

# Run tuning only
python Pipelines/pipeline_a/run_pipeline.py --tune-only

# Generate final Sept-Oct forecast
python Pipelines/pipeline_a/run_pipeline.py --final-forecast
```

### Logic

```python
def main(args):
    print("=" * 60)
    print("PIPELINE A: GLOBAL GBDT — Starting")
    print("=" * 60)
    
    # Step 1: Data Loading
    print("\n[Step 1/6] Loading and aggregating data...")
    master_df = run_data_loading()
    validate_step_1(master_df)
    
    # Step 2: Feature Engineering
    print("\n[Step 2/6] Engineering features...")
    features_df = run_feature_engineering(master_df)
    validate_step_2(features_df)
    
    if not args.tune_only:
        # Step 3: Train with default params
        print("\n[Step 3/6] Training models (default hyperparameters)...")
        fold_results = run_training(features_df, params="default")
        validate_step_3(fold_results)
    
    if not args.skip_tune:
        # Step 4: Hyperparameter Tuning
        print("\n[Step 4/6] Tuning hyperparameters (Optuna)...")
        best_params = run_tuning(features_df)
        validate_step_4(best_params)
        
        # Re-train with best params
        print("\n[Step 3b/6] Re-training with tuned hyperparameters...")
        fold_results = run_training(features_df, params=best_params)
    
    # Step 5: Post-processing & Submission CSVs
    print("\n[Step 5/6] Generating submission CSVs...")
    run_prediction(fold_results)
    validate_step_5()
    
    # Step 6: Evaluation
    print("\n[Step 6/6] Evaluating against eval.md contract...")
    report = run_evaluation()
    print_report(report)
    
    if args.final_forecast:
        print("\n[FINAL] Generating Sept-Oct 2025 forecast...")
        run_final_forecast(features_df, best_params)
    
    print("\n" + "=" * 60)
    print("PIPELINE A: COMPLETE")
    print("=" * 60)
```

---

## Site D Isolation Enhancement (master_strategy.md §11)

Site D has a WAPE of ~0.47 vs ~0.24 for A/B/C. The global model already trains on all sites (vanilla partial pooling), but this fails because: (1) Site B dominates the loss gradient (36% of rows, highest volume), (2) GBDT can't do true partial pooling — it either splits on `site` (local) or doesn't (total pooling), (3) Site D's 17.4% admit rate is structurally different from A/C (~31%). See §11.2 for full diagnosis.

The fix has two components: **Target Encoding** (fix the baseline problem) and **Hybrid-Residual** (fix the tail error).

### Enhancement A: Target Encoding Features (add to `step_02_feature_eng.py`)

Add these to feature engineering, computed **per (Site, Block) group** with the same lag/shift rules as existing features. All use trailing windows with `ROLLING_SHIFT = 63` to avoid leakage.

#### 2L. Site-Level Target Encodings

```python
# --- Computed per (site, block) group, shifted by ROLLING_SHIFT ---

# 1. Site baseline volume (trailing 90-day mean, lagged)
for target_col in ["total_enc", "admitted_enc"]:
    shifted = group[target_col].shift(ROLLING_SHIFT)
    df[f"te_site_mean_{target_col}"] = shifted.rolling(90, min_periods=30).mean()

# 2. Site × Block baseline (same as above, but already per-group since we group by site,block)
#    This IS te_site_block_mean — the group-level rolling mean
df["te_site_block_mean_total"] = group["total_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).mean()
df["te_site_block_mean_admitted"] = group["admitted_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).mean()

# 3. Site admit rate (trailing 90-day, lagged) — encodes the 17.4% vs 31.6% gap directly
shifted_total = group["total_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).sum()
shifted_admitted = group["admitted_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).sum()
df["te_site_admit_rate"] = (shifted_admitted / shifted_total.clip(lower=1))

# 4. Site × DOW mean (trailing, lagged) — encodes Site D's flat weekday/weekend pattern
#    Must be computed at the (site, dow) level, then merged back
site_dow_means = (
    df.groupby(["site", "dow"])
    .apply(lambda g: g["total_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).mean())
)
df["te_site_dow_mean"] = site_dow_means.values

# 5. Site × Month mean (historical same-month average from prior years, lagged)
#    Computed per fold inside training loop (like existing mean encodings in 2I)
#    Infrastructure function defined here, called in Step 3
def compute_te_site_month(train_df, target_col="total_enc"):
    """Trailing same-month mean per site from training data only."""
    return train_df.groupby(["site", "month"])[target_col].transform("mean")
```

**Why these work where raw `site` fails:** Trees can now split on `te_site_mean_total < 80` to isolate low-volume sites numerically. Multiplicative effects (holiday = +10%) apply to the encoded baseline (71 for Site D) instead of the population mean (112). The admit-rate gap (17.4% vs 31.6%) becomes a continuous feature instead of forcing the tree to discover it from a categorical.

**Feature count increase:** ~8 new features. Minimal impact on training time.

#### Eval Notes — Enhancement A
- [ ] **Leakage check**: All `te_*` features use `shift(ROLLING_SHIFT)` — verify no same-period data leaks
- [ ] **NaN budget**: First ~90 days of each group will be NaN due to rolling window — confirm these are excluded from training
- [ ] **Value range**: `te_site_admit_rate` should be in [0, 1]; `te_site_mean_total` should be ~70–170 (matching known site ranges)
- [ ] **Ablation**: Train with and without `te_*` features; check Site D WAPE improvement and A/B/C stability

### Enhancement B: Hybrid-Residual Model for Site D (add to `step_03_train.py`)

After the global model (A1 + A2) is trained and produces out-of-fold predictions, train a small local residual model on Site D's errors.

#### Step 3B: Residual Model Training

```python
# --- Inside the per-fold training loop, AFTER global model training ---

# 1. Collect OOF predictions for Site D from the global model
oof_preds_d = global_model.predict(X_val[X_val["site"] == "D"])
actuals_d = y_val[X_val["site"] == "D"]
residuals_d = actuals_d - oof_preds_d

# 2. Build residual training set (Site D only, from training portion)
#    Use 3-fold inner CV to avoid overfitting the residual model to itself
X_train_d = X_train[X_train["site"] == "D"]
y_train_d_residual = y_train_d_actual - global_model.predict(X_train_d)

# 3. Residual model features (SUBSET — do NOT re-include weather/reason-mix/cross-site)
RESIDUAL_FEATURES = [
    "block", "dow", "month", "day_of_year",
    "te_site_block_mean_total", "te_site_block_mean_admitted",
    "te_site_admit_rate",
    "lag_63", "lag_91", "lag_182", "lag_364",   # Site D's own lags
    "roll_mean_7", "roll_mean_28",               # Site D's own rolling stats
    "global_pred",                                # The global model's prediction (heteroscedasticity)
    "is_covid_era", "is_holiday", "is_weekend",
]

# 4. Train with AGGRESSIVE regularization (only ~11K rows)
RESIDUAL_LGBM_PARAMS = {
    "objective": "regression",     # Predict continuous residual
    "n_estimators": 300,           # Low — don't overfit
    "num_leaves": 15,              # Very constrained tree
    "max_depth": 4,
    "min_child_samples": 50,       # Large — need stable leaves
    "learning_rate": 0.03,
    "reg_lambda": 5.0,             # Strong L2
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "verbosity": -1,
}

residual_model = lgb.LGBMRegressor(**RESIDUAL_LGBM_PARAMS)
residual_model.fit(
    X_train_d[RESIDUAL_FEATURES], y_train_d_residual,
    eval_set=[(X_val_d_inner[RESIDUAL_FEATURES], y_val_d_inner_residual)],
    callbacks=[lgb.early_stopping(30)],
)

# 5. Final prediction for Site D
SHRINKAGE_WEIGHT = 0.5  # Tune via inner CV in [0.3, 0.8]
pred_d = global_pred_d + SHRINKAGE_WEIGHT * residual_model.predict(X_d[RESIDUAL_FEATURES])

# 6. For sites A/B/C — use global predictions unchanged
pred_abc = global_model.predict(X_abc)
```

**Key implementation notes:**
- `SHRINKAGE_WEIGHT` controls how much to trust the residual correction. Start at 0.5. Tune via 3-fold inner CV within the training fold — try `[0.3, 0.4, 0.5, 0.6, 0.7, 0.8]`, pick the value that minimizes Site D WAPE without degrading A/B/C.
- The residual model trains on `y_actual - y_global_pred` (the global model's systematic errors on Site D). It should learn patterns like "Block 0 is overpredicted" or "Mondays are underpredicted."
- **Do NOT include** weather, reason-mix, or cross-site features — these are already captured by the global model. Re-including them gives the residual model a path to overfit by re-learning global signal.
- The residual model is saved alongside the global model: `fold_{k}_residual_d.txt`.

#### Step 3C: Zero-Inflation Post-Hoc (Site D Block 0/1/3)

```python
# --- After hybrid-residual produces final predictions ---

# 1. Train zero-classifier on Site D sparse blocks
from sklearn.linear_model import LogisticRegression

sparse_blocks = [0, 1, 3]  # Blocks where admitted_enc == 0 rate > 3%
mask_sparse = (df_train["site"] == "D") & (df_train["block"].isin(sparse_blocks))

zero_target = (df_train.loc[mask_sparse, "admitted_enc"] == 0).astype(int)
zero_features = df_train.loc[mask_sparse, ["block", "dow", "month", "te_site_block_mean_admitted"]]

zero_clf = LogisticRegression(C=1.0, max_iter=500)
zero_clf.fit(zero_features, zero_target)

# 2. Apply correction to Site D admitted predictions
ZERO_SHRINKAGE = 0.5  # Tune via CV in [0.3, 0.7]
p_zero = zero_clf.predict_proba(X_pred_d_sparse[["block", "dow", "month", "te_site_block_mean_admitted"]])[:, 1]
pred_admitted_d_corrected = pred_admitted_d * (1 - p_zero * ZERO_SHRINKAGE)

# Ensure non-negative after correction
pred_admitted_d_corrected = pred_admitted_d_corrected.clip(lower=0)
```

#### Eval Notes — Enhancement B
- [ ] **Nested CV**: Residual model MUST use inner CV folds — never train and evaluate on the same OOF residuals
- [ ] **Shrinkage tuning**: Log Site D WAPE for each `SHRINKAGE_WEIGHT` value; verify optimum is not at 0.0 (would mean residual model adds nothing) or 1.0 (would mean it's overfitting)
- [ ] **Collateral damage check**: Sites A/B/C predictions are UNCHANGED — verify identical to pre-enhancement
- [ ] **Feature importance**: Top features in residual model should be `block`, `global_pred`, `te_site_block_mean` — if it's dominated by lags, it may be overfitting to autocorrelation
- [ ] **Residual distribution**: Plot residuals before and after correction — should see reduced bias for Block 0/1/3

### Enhancement C: Admit-Rate Guardrails (add to `step_05_predict.py`)

```python
# After computing admitted = total × admit_rate for Site D:
# Clamp admit_rate to historical [5th, 95th] percentile per (site, block)
ADMIT_RATE_BOUNDS = {
    ("D", 0): (0.08, 0.30),
    ("D", 1): (0.10, 0.35),
    ("D", 2): (0.10, 0.30),
    ("D", 3): (0.08, 0.28),
}
for (site, block), (lo, hi) in ADMIT_RATE_BOUNDS.items():
    mask = (df["site"] == site) & (df["block"] == block)
    df.loc[mask, "admit_rate_pred"] = df.loc[mask, "admit_rate_pred"].clip(lo, hi)
```

---

## Dependencies

```
lightgbm>=4.0
optuna>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3   # LabelEncoder, train_test_split, LogisticRegression (zero-inflation)
holidays>=0.40      # US holiday calendar
pyarrow>=14.0       # Parquet I/O
```

---

## Execution Order & Time Estimates

| Step | File | Estimated Time | Can Skip? |
|------|------|---------------|-----------|
| 0 | `config.py` | — (static) | No |
| 1 | `step_01_data_loading.py` | ~30s | No |
| 2 | `step_02_feature_eng.py` | ~2-5 min | No |
| 3 | `step_03_train.py` | ~5-10 min (4 folds × 2 models) | No |
| 4 | `step_04_tune.py` | ~2-4 hours (100 trials × 4 folds) | Yes (use defaults) |
| 5 | `step_05_predict.py` | ~1 min | No |
| 6 | `step_06_evaluate.py` | ~10s | No |

**Total (with tuning):** ~3-5 hours  
**Total (skip tuning):** ~15-20 minutes

---

## Key Risk Mitigations

| Risk | Mitigation |
|------|------------|
| **Lag leakage** (most critical) | All lags ≥ 63; all rolling stats shifted by 63; validated in Step 2 eval notes |
| **Aggregate encoding leakage** | Computed on training fold only; mapped to val via left join (Step 3) |
| **COVID bias** | Downweight policy (0.1× sample weight); tested against exclusion in Step 4 |
| **Overfitting to high-volume Site B** | Volume-based sample weights already favor Site B proportionally; by-site WAPE breakdown in Step 6 catches imbalance |
| **Block 0 drift** | By-block WAPE check in Step 6; if Block 0 WAPE >> others, add block-specific features or separate Block 0 model |
| **Tweedie vs Poisson mismatch** | Both tested in Optuna search space (Step 4) |
| **Missing weather data** | Weather features are optional; model runs without them; add as ablation |

---

## Output Artifacts

After a complete Pipeline A run:

```
Pipelines/pipeline_a/
  data/
    master_block_history.parquet      # Step 1 output
    features.parquet                  # Step 2 output
  models/
    fold_1_model_a1.txt               # LightGBM model (total_enc)
    fold_1_model_a2.txt               # LightGBM model (admit_rate)
    fold_2_model_a1.txt
    fold_2_model_a2.txt
    fold_3_model_a1.txt
    fold_3_model_a2.txt
    fold_4_model_a1.txt
    fold_4_model_a2.txt
    best_params_a1.json               # Tuned hyperparameters
    best_params_a2.json
  output/
    fold_1_predictions.csv            # Submission-format (Site, Date, Block, ED Enc, ED Enc Admitted)
    fold_2_predictions.csv
    fold_3_predictions.csv
    fold_4_predictions.csv
    final_sept_oct_2025.csv           # Final competition submission (976 rows)
    evaluation_report.json            # Full eval results
    feature_importance.csv            # Top features by gain
```
