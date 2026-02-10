# Pipeline C: Hierarchical Reconciliation (Daily → Block) — Implementation Plan

**Status:** READY FOR IMPLEMENTATION  
**Source:** `master_strategy.md` §3.3  
**Core Idea:** Decompose forecasting into two easier sub-problems: (1) predict smooth daily totals per site, (2) predict block shares per (site, date), then allocate. Reduces noise by modeling at a higher aggregation level first.  
**Location:** `Pipelines/pipeline_C/`  
**Data Dependency:** `master_block_history.parquet` produced by the Data Source step (see `data_source.md`). Pipeline C aggregates block→daily internally for the daily model, and uses block-level data for share estimation.

---

## Why Pipeline C Adds Value

- **Noise Reduction:** Daily site-level signal (~100 visits/site/day) is much smoother than block-level (~25 visits). GBDT trains on cleaner targets.
- **Structural Coherence:** Block predictions are forced to sum exactly to the daily total — no incoherent forecasts where blocks don't add up.
- **Diverse Error Structure:** Errors come from daily-level prediction + share allocation, fundamentally different from Pipeline A/B's direct block prediction. This diversity is what gives ensemble gains.
- **Simpler Daily Model:** Fewer series (4 sites vs. 16 site×block combos), fewer NaN issues, faster iteration.

---

## File Structure

```
Pipelines/
  pipeline_C/
    __init__.py
    config.py                       # Constants, paths, fold definitions, feature lists
    step_01_data_loading.py         # Load master parquet, build daily aggregates + block shares
    step_02_feature_eng_daily.py    # Feature engineering for daily-level GBDT model
    step_03_feature_eng_shares.py   # Feature engineering for block-share prediction model
    step_04_train_daily.py          # Train daily total_enc + daily admit_rate models (LightGBM)
    step_05_train_shares.py         # Train block share models (Softmax GBDT / Dirichlet / Climatology)
    step_06_tune.py                 # Optuna tuning for daily models + share models
    step_07_predict.py              # Allocate daily forecasts to blocks + post-processing
    step_08_evaluate.py             # Score against eval.md contract across 4 folds
    run_pipeline.py                 # End-to-end orchestrator
```

---

## Step 0: Configuration (`config.py`)

### Purpose
Single source of truth for all constants, paths, feature lists, fold definitions, and share-model configuration.

### Contents

```python
# --- Paths ---
MASTER_PARQUET_PATH = "Pipelines/Data Source/Data/master_block_history.parquet"
RAW_DATA_PATH = "Dataset/DSU-Dataset.csv"
OUTPUT_DIR = "Pipelines/pipeline_C/output/"
MODEL_DIR = "Pipelines/pipeline_C/models/"
DATA_DIR = "Pipelines/pipeline_C/data/"

# --- Sites & Blocks ---
SITES = ["A", "B", "C", "D"]
BLOCKS = [0, 1, 2, 3]
N_BLOCKS = 4

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
COVID_SAMPLE_WEIGHT = 0.1

# --- Daily Model Feature Engineering ---
MAX_HORIZON = 63
LAG_DAYS_DAILY = [63, 70, 77, 91, 182, 364]
ROLLING_WINDOWS_DAILY = [7, 14, 28, 56, 91]
ROLLING_SHIFT_DAILY = 63

# --- Share Model Configuration ---
SHARE_MODEL_TYPE = "softmax_gbdt"  # Options: "softmax_gbdt", "dirichlet", "climatology"
# Climatology grouping keys for fallback
CLIMATOLOGY_KEYS = ["site", "dow", "month"]
# Share model lags (block-level historical shares)
LAG_DAYS_SHARES = [63, 70, 77, 91, 182, 364]
ROLLING_SHIFT_SHARES = 63

# --- LightGBM Defaults (Daily Total Model) ---
LGBM_DAILY_TOTAL = {
    "objective": "tweedie",
    "tweedie_variance_power": 1.5,
    "n_estimators": 1500,
    "max_depth": 6,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 10,   # Higher than Pipeline A — fewer rows (daily, not block)
    "verbosity": -1,
}

# --- LightGBM Defaults (Daily Admit Rate Model) ---
LGBM_DAILY_RATE = {
    "objective": "regression",
    "n_estimators": 1000,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 5.0,
    "min_child_weight": 10,
    "verbosity": -1,
}

# --- LightGBM Defaults (Softmax Block Share Model) ---
LGBM_SHARE = {
    "objective": "multiclass",
    "num_class": 4,
    "n_estimators": 800,
    "max_depth": 5,
    "learning_rate": 0.03,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "min_child_weight": 5,
    "verbosity": -1,
}

# --- Optuna ---
OPTUNA_N_TRIALS_DAILY = 100
OPTUNA_N_TRIALS_SHARE = 50
```

### Eval Notes — Step 0
- [ ] **Check**: All fold date ranges match `eval.md` exactly.
- [ ] **Check**: `MAX_HORIZON = 63` is conservative enough for the longest validation window (62 days + 1 buffer).
- [ ] **Check**: All daily lags ≥ 63 — no leakage possible.
- [ ] **Check**: `ROLLING_SHIFT_DAILY = 63` prevents rolling stats from peeking into forecast horizon.
- [ ] **Check**: `SHARE_MODEL_TYPE` is set to `"softmax_gbdt"` (preferred). Other options available as fallbacks.
- [ ] **Check**: `num_class = 4` in `LGBM_SHARE` matches `N_BLOCKS`.

---

## Step 1: Data Loading & Daily Aggregation (`step_01_data_loading.py`)

### Purpose
Load the unified `master_block_history.parquet`, then construct two derived datasets:
1. **Daily-level dataset** — aggregated across blocks per (site, date) — for the daily model.
2. **Block-share dataset** — historical block-level proportions per (site, date) — for the share model.

### Sub-Steps

#### 1.1 Load Master Dataset
- Read `MASTER_PARQUET_PATH`.
- Validate schema: assert required columns exist (`site`, `date`, `block`, `total_enc`, `admitted_enc`, calendar cols, etc.).
- Ensure `date` is datetime, sort by `(site, date, block)`.

#### 1.2 Build Daily Aggregates
```python
daily_df = (
    block_df
    .groupby(["site", "date"], as_index=False)
    .agg(
        total_enc=("total_enc", "sum"),
        admitted_enc=("admitted_enc", "sum"),
        # Carry forward calendar features (same for all blocks on a date)
        dow=("dow", "first"),
        day=("day", "first"),
        week_of_year=("week_of_year", "first"),
        month=("month", "first"),
        quarter=("quarter", "first"),
        day_of_year=("day_of_year", "first"),
        year=("year", "first"),
        is_weekend=("is_weekend", "first"),
        is_covid_era=("is_covid_era", "first"),
        is_holiday=("is_holiday", "first"),
        is_halloween=("is_halloween", "first"),
        event_count=("event_count", "first"),
        days_since_epoch=("days_since_epoch", "first"),
        # Weather (same across blocks for a site-date)
        temp_min=("temp_min", "first"),
        temp_max=("temp_max", "first"),
        precip=("precip", "first"),
        snowfall=("snowfall", "first"),
        # School
        school_in_session=("school_in_session", "first"),
    )
    .sort_values(["site", "date"])
    .reset_index(drop=True)
)
```

#### 1.3 Derive Daily `admit_rate`
```python
daily_df["admit_rate"] = (
    daily_df["admitted_enc"] / daily_df["total_enc"].clip(lower=1)
).clip(0, 1)
# Where total_enc == 0, admit_rate = 0.0
daily_df.loc[daily_df["total_enc"] == 0, "admit_rate"] = 0.0
```

#### 1.4 Build Block Share Dataset
For each (site, date), compute historical block shares:
```python
block_df["block_share"] = (
    block_df["total_enc"]
    / block_df.groupby(["site", "date"])["total_enc"].transform("sum").clip(lower=1)
)
# Where daily total is 0, distribute equally: 0.25 per block
zero_daily_mask = block_df.groupby(["site", "date"])["total_enc"].transform("sum") == 0
block_df.loc[zero_daily_mask, "block_share"] = 1.0 / N_BLOCKS
```

Same for admitted block shares:
```python
block_df["admit_block_share"] = (
    block_df["admitted_enc"]
    / block_df.groupby(["site", "date"])["admitted_enc"].transform("sum").clip(lower=1)
)
zero_admit_mask = block_df.groupby(["site", "date"])["admitted_enc"].transform("sum") == 0
block_df.loc[zero_admit_mask, "admit_block_share"] = 1.0 / N_BLOCKS
```

#### 1.5 Build Share Pivot (Wide Format for Softmax)
For the softmax GBDT share model, pivot block shares into wide format per (site, date):
```python
share_wide = block_df.pivot_table(
    index=["site", "date"],
    columns="block",
    values="block_share",
    aggfunc="first",
).rename(columns={0: "share_b0", 1: "share_b1", 2: "share_b2", 3: "share_b3"})
```

This produces one row per (site, date) with 4 share columns summing to 1.

**For softmax GBDT**: The target is the block index (0-3), and each training row represents one (site, date, block) tuple — the model learns `P(block | site, date features)`. Alternatively, we can use the wide format and train a Dirichlet model.

**Decision**: For the softmax approach, keep the long format with `block` as the target class label.

#### 1.6 Save Intermediate Data
- `DATA_DIR/daily_df.parquet` — daily aggregated dataset
- `DATA_DIR/block_shares_df.parquet` — block-level data with share columns
- `DATA_DIR/share_wide_df.parquet` — pivoted shares (for Dirichlet if needed)

### Eval Notes — Step 1
- [ ] **Daily row count**: `4 sites × N_days`. For 2018-01-01 to 2025-10-31 = 2,861 days → **11,444 rows**.
- [ ] **Block share sum check**: For each (site, date), assert `sum(block_shares) ≈ 1.0` (tolerance 1e-6). Print any violations.
- [ ] **Admitted ≤ Total (daily)**: Assert `daily_df.admitted_enc <= daily_df.total_enc` for all rows.
- [ ] **admit_rate range**: Assert all values in [0.0, 1.0].
- [ ] **Share range**: Assert all `block_share` values in [0.0, 1.0].
- [ ] **Zero-day handling**: Count days with `total_enc == 0`. Should be very rare (<0.1%). Verify equal share distribution (0.25) is applied.
- [ ] **Print summary stats**: Per-site mean/std/min/max of daily `total_enc`. Site B should be largest (~36% of total volume).
- [ ] **Block share distribution**: Print mean share per block across all (site, date). Expected: Block 2 (12-17h) and Block 3 (18-23h) should be highest for most sites.
- [ ] **Date range check**: Min date = 2018-01-01, max date covers up to 2025-08-31.
- [ ] **No duplicate (site, date) in daily_df**: Assert uniqueness.
- [ ] **No duplicate (site, date, block) in block_shares_df**: Assert uniqueness.

---

## Step 2: Feature Engineering — Daily Model (`step_02_feature_eng_daily.py`)

### Purpose
Build features for the daily-level GBDT that predicts `total_enc` and `admit_rate` per (site, date). This is structurally similar to Pipeline A's feature engineering, but operates on daily site-level aggregates (no block dimension).

### Feature Groups

#### 2A. Lagged Target Features (Daily)
Per (site) group:

| Feature | Formula | Notes |
|---------|---------|-------|
| `lag_63` | `total_enc.shift(63)` | Closest safe lag |
| `lag_70` | `total_enc.shift(70)` | 10 weeks ago |
| `lag_77` | `total_enc.shift(77)` | 11 weeks ago |
| `lag_91` | `total_enc.shift(91)` | ~3 months |
| `lag_182` | `total_enc.shift(182)` | ~6 months |
| `lag_364` | `total_enc.shift(364)` | Same day last year |

Same lags for `admit_rate`.

**Key difference from Pipeline A**: Grouped by `(site)` only (no block), because we're operating on daily aggregates.

#### 2B. Rolling Statistics (Daily)
For windows `[7, 14, 28, 56, 91]`, all shifted by 63 days:
```python
for w in ROLLING_WINDOWS_DAILY:
    shifted = group["total_enc"].shift(ROLLING_SHIFT_DAILY)
    df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
    df[f"roll_std_{w}"]  = shifted.rolling(w, min_periods=1).std()
    df[f"roll_min_{w}"]  = shifted.rolling(w, min_periods=1).min()
    df[f"roll_max_{w}"]  = shifted.rolling(w, min_periods=1).max()
```

#### 2C. Trend Deltas
| Feature | Formula | Captures |
|---------|---------|----------|
| `delta_7_28` | `roll_mean_7 - roll_mean_28` | Short-term momentum |
| `delta_28_91` | `roll_mean_28 - roll_mean_91` | Medium-term trend |
| `delta_lag_63_70` | `lag_63 - lag_70` | Week-over-week change at horizon boundary |

#### 2D. Calendar & Cyclical Encoding
Already available from daily aggregation:
- `dow`, `day`, `week_of_year`, `month`, `quarter`, `day_of_year`, `year`, `is_weekend`, `days_since_epoch`, `is_covid_era`, `is_halloween`

Add cyclical encodings:
```python
df["dow_sin"]   = np.sin(2 * np.pi * df["dow"] / 7)
df["dow_cos"]   = np.cos(2 * np.pi * df["dow"] / 7)
df["doy_sin"]   = np.sin(2 * np.pi * df["day_of_year"] / 365.25)
df["doy_cos"]   = np.cos(2 * np.pi * df["day_of_year"] / 365.25)
df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
df["year_frac"] = df["year"] + (df["day_of_year"] - 1) / 365.25
```

#### 2E. Holiday Features
| Feature | Description |
|---------|-------------|
| `is_us_holiday` | Binary flag from data source `is_holiday` |
| `days_since_xmas` | Days since most recent Dec 25 |
| `days_until_thanksgiving` | Days until next Thanksgiving |
| `days_since_july4` | Days since most recent July 4 |
| `days_to_nearest_holiday` | Min distance to any US federal holiday |

#### 2F. School Calendar Features
| Feature | Description |
|---------|-------------|
| `school_in_session` | Binary from data source |
| `days_since_school_start` | Days since ~Aug 22 of current/recent year |
| `days_until_school_start` | Days until next ~Aug 22 |

#### 2G. Event Features
- `event_count` — number of events on this date (from data source)

#### 2H. Weather Features (if available)
- `temp_min`, `temp_max`, `precip`, `snowfall`
- Forward-fill NaN within each site, then backward-fill
- Derived: `temp_range = temp_max - temp_min`

#### 2I. Aggregate Mean Encodings (computed per fold — NOT here)
| Feature | Grouping | Notes |
|---------|----------|-------|
| `site_month_mean` | (site, month) | Historical mean daily total_enc |
| `site_dow_mean` | (site, dow) | Historical mean daily total_enc |

**CRITICAL**: These must be computed inside the training loop (Step 4) on training data only to prevent leakage. Step 2 only defines the function.

#### 2J. Interaction Features
```python
df["holiday_x_site"] = df["is_us_holiday"].astype(int) * df["site"].cat.codes
df["weekend_x_site"] = df["is_weekend"].astype(int) * df["site"].cat.codes
```

**Note**: No `block` interactions here — the daily model is block-agnostic by design.

#### 2K. Sample Weights
```python
# COVID downweighting
df["covid_weight"] = np.where(df["is_covid_era"], COVID_SAMPLE_WEIGHT, 1.0)

# WAPE-aligned volume weighting (daily total)
df["volume_weight"] = df["total_enc"].clip(lower=1)

# Combined
df["sample_weight_total"] = df["covid_weight"] * df["volume_weight"]
df["sample_weight_rate"] = df["covid_weight"] * df["admitted_enc"].clip(lower=1)
```

### Final Feature List (Daily Model)

```python
FEATURE_COLS_DAILY = [
    # Identifier (categorical)
    "site",
    # Lags
    "lag_63", "lag_70", "lag_77", "lag_91", "lag_182", "lag_364",
    # Rolling stats (5 windows × 4 stats = 20)
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
    # Aggregate encodings (computed per fold in Step 4)
    "site_month_mean", "site_dow_mean",
    # Interactions
    "holiday_x_site", "weekend_x_site",
    # Weather (if available)
    # "temp_min", "temp_max", "precip", "snowfall", "temp_range",
    # COVID indicator
    "is_covid_era",
]
```

### Save Output
- `DATA_DIR/daily_features.parquet`

### Eval Notes — Step 2
- [ ] **No future leakage**: For each feature, verify it only uses data from ≥63 days ago. Print min shift used per lag/rolling feature.
- [ ] **NaN audit**: Print NaN count per feature column. Lags will have NaN for the first ~364 rows — expected. LightGBM handles NaN natively.
- [ ] **Feature count**: Print total. Expected: ~50-60 columns (fewer than Pipeline A since no block-level features).
- [ ] **Distribution check**: For key features (lags, rolling means), print mean/std/min/max. Flag features with >90% zero values.
- [ ] **Correlation check**: Pairwise correlation between lag features. If `|corr| > 0.98`, consider dropping one.
- [ ] **Sample weight check**: `sample_weight_total > 0` for all rows. COVID-era rows should have `weight ≈ 0.1 × total_enc`.
- [ ] **Temporal integrity**: Plot `lag_63` vs `total_enc.shift(63)` — should be identical (sanity).
- [ ] **Compare daily feature matrix to Pipeline A's**: Daily model should have FEWER features (no block, no block interactions). Print both counts.

---

## Step 3: Feature Engineering — Block Share Model (`step_03_feature_eng_shares.py`)

### Purpose
Build features for the block-share prediction model. This model predicts `P(block | site, date features)` — the proportion of daily encounters that fall in each 6-hour block.

### Three Approaches (implement all, select via config)

#### Approach A: Softmax GBDT (PREFERRED)
Uses CatBoost or LightGBM multiclass (`num_class=4`) to predict which block a "representative encounter" falls into. The predicted probabilities directly give block shares.

**Training data format**: One row per (site, date, block), target = block index (0-3). Sample weight = `total_enc` for that block (ensures the model focuses on share accuracy proportional to volume).

#### Approach B: Dirichlet Regression
Predicts a 4-dimensional probability vector constrained to sum to 1. More principled for compositional data.

**Training data format**: One row per (site, date), targets = `[share_b0, share_b1, share_b2, share_b3]`.

**Implementation**: Use `statsmodels` or custom implementation with Dirichlet log-likelihood.

#### Approach C: Climatology Fallback
No model — just historical average shares by `(site, dow, month)`.

```python
climatology = (
    train_df
    .groupby(["site", "dow", "month", "block"])["block_share"]
    .mean()
    .reset_index()
)
```

### Share Model Features (for Approach A and B)

Block shares have a strong **time-of-day × day-of-week** pattern but can shift with:
- Season (longer summer daylight → higher Block 2-3 share)
- Holidays (Block distribution changes — e.g., Thanksgiving overnight Block 0 drops)
- School calendar (school hours pull pediatric visits to Block 1-2)

| Feature | Description | Notes |
|---------|-------------|-------|
| `site` | Categorical | Block patterns are site-specific |
| `dow` | Day of week (0-6) | Strongest single predictor of share pattern |
| `month` | Month (1-12) | Seasonal shift in block distribution |
| `is_weekend` | Binary | Weekend vs weekday share patterns differ |
| `is_us_holiday` | Binary | Holiday block distributions shift |
| `is_halloween` | Binary | Block 3 (evening) surges |
| `quarter` | Quarter (1-4) | Coarser seasonal proxy |
| `dow_sin`, `dow_cos` | Cyclical DOW | Continuity at week boundaries |
| `month_sin`, `month_cos` | Cyclical month | Continuity at year boundaries |
| `school_in_session` | Binary | Affects daytime vs. evening distribution |
| `event_count` | Number of events | Events shift block patterns |
| `share_lag_63_b{0-3}` | Block shares 63 days ago | Recent share pattern momentum |
| `share_lag_364_b{0-3}` | Block shares same day last year | Year-over-year share stability |
| `share_roll_mean_28_b{0-3}` | 28-day rolling mean share (shifted 63d) | Smooth share trend |
| `admit_share_lag_63_b{0-3}` | Admitted block shares 63 days ago | For admitted share model |

**Lag computation for shares**: Computed per (site, block) group on the `block_share` column, shifted by ≥ 63 days (same leakage rule as daily model).

### Admitted Block Shares
Same approach applied separately for admitted encounters:
- Train a second share model predicting `P(admit_block | site, date)`
- Or use the same total-share model for admitted (simpler, test if degradation is acceptable)

### Save Output
- `DATA_DIR/share_features.parquet`

### Eval Notes — Step 3
- [ ] **Share feature NaN audit**: Print NaN per feature. Share lags will have NaN for first ~364 rows — expected.
- [ ] **Share lag leakage check**: Assert all share lags use shifts ≥ 63 days.
- [ ] **Feature count**: Print total for share model. Expected: ~25-35 columns (much smaller than daily model).
- [ ] **Share stability check**: Compute std of `block_share` per (site, block) across all dates. If std < 0.02 for all blocks, climatology might suffice (shares barely change). If std > 0.05 for some, the share model adds real value.
- [ ] **Climatology baseline**: Compute mean block shares by (site, dow, month). This is the lower bound — the share model must beat it.
- [ ] **Admitted share correlation**: Check if `block_share ≈ admit_block_share` for each block. If correlation > 0.95, a single share model may suffice for both total and admitted.

---

## Step 4: Train Daily Models (`step_04_train_daily.py`)

### Purpose
Train two LightGBM models per fold on daily-level data:
- **Model C1_total**: Predicts daily `total_enc` per site (Tweedie objective)
- **Model C1_rate**: Predicts daily `admit_rate` per site (Regression objective)

### Logic (per fold)

```
For each fold in FOLDS:
    1. Split data:
       train = daily_df rows where date <= fold.train_end
       val   = daily_df rows where date in [fold.val_start, fold.val_end]
    
    2. Drop rows where lag features are NaN from training
       (first ~364 days will have NaN for lag_364)
    
    3. Compute fold-specific aggregate encodings on TRAIN ONLY:
       - site_month_mean = train.groupby([site, month]).total_enc.mean()
       - site_dow_mean   = train.groupby([site, dow]).total_enc.mean()
       Map these onto both train and val rows.
    
    4. Encode categoricals:
       - site → LabelEncoder or LightGBM native categorical
    
    5. Train Model C1_total (daily total_enc):
       - X = train[FEATURE_COLS_DAILY]
       - y = train["total_enc"]
       - sample_weight = train["sample_weight_total"]
       - lgb.Dataset with categorical_feature=["site"]
       - Train with LGBM_DAILY_TOTAL params
       - Early stopping on val set (metric: mae)
    
    6. Train Model C1_rate (daily admit_rate):
       - X = train[FEATURE_COLS_DAILY]
       - y = train["admit_rate"]
       - sample_weight = train["sample_weight_rate"]
       - Train with LGBM_DAILY_RATE params
    
    7. Predict on val (DAILY level — NOT yet block-level):
       - pred_daily_total = model_c1_total.predict(val_X).clip(0)
       - pred_daily_rate  = model_c1_rate.predict(val_X).clip(0, 1)
       - pred_daily_admitted = (pred_daily_total * pred_daily_rate).clip(0)
    
    8. Compute DAILY-LEVEL WAPE as intermediate diagnostic:
       - daily_total_wape = sum(|actual - pred|) / sum(|actual|)
       - daily_admitted_wape = same for admitted
       This tells us how well the daily model performs BEFORE block allocation.
    
    9. Save fold daily predictions:
       → DATA_DIR/fold_{id}_daily_predictions.parquet
    
   10. Save models:
       → MODEL_DIR/fold_{id}_model_c1_total.txt
       → MODEL_DIR/fold_{id}_model_c1_rate.txt
```

### Eval Notes — Step 4
- [ ] **Training set size**: Print rows per fold. Expected: ~8,000-10,000 rows (4 sites × ~2,000-2,500 days, minus NaN lag rows).
- [ ] **Early stopping**: Verify it triggers. If model uses all `n_estimators`, increase the cap.
- [ ] **Feature importance**: Print top 20 features by `gain`. Expect lags and rolling means to dominate. If calendar features dominate, suspect leakage.
- [ ] **Daily prediction range check**:
  - `pred_daily_total`: Non-negative. Print min/max/mean per site. Site B should be highest.
  - `pred_daily_rate`: In [0, 1]. Print any values outside range before clipping.
  - `pred_daily_admitted`: Should be ≤ `pred_daily_total`.
- [ ] **Daily-level WAPE**: Print per fold. This is the UPPER BOUND of block-level accuracy — block allocation can only add error, not remove it.
  ```
  Fold 1: daily_total_wape=X.XXXX, daily_admitted_wape=X.XXXX
  Fold 2: daily_total_wape=X.XXXX, daily_admitted_wape=X.XXXX
  ...
  ```
- [ ] **Residual analysis (daily)**:
  - Mean residual ≈ 0 (no systematic bias)
  - By-site residual means — flag consistent over/under-prediction
  - By-DOW residual pattern — should be flat
- [ ] **Compare daily WAPE to Pipeline A block-level WAPE** (if available): Daily model WAPE gives a sense of how much room block allocation has to degrade.
- [ ] **No data leakage**: Verify earliest training date used for lag_364 in fold 1 is ≥ 2019-01-01 (2018-01-01 + 364).

---

## Step 5: Train Block Share Models (`step_05_train_shares.py`)

### Purpose
Train models that predict the block-level distribution of encounters for each (site, date). Three approaches implemented; select via `SHARE_MODEL_TYPE` in config.

### Approach A: Softmax GBDT (PREFERRED)

Uses LightGBM with `objective="multiclass"` and `num_class=4`.

```
For each fold in FOLDS:
    1. Split block-level data:
       train = block_shares_df rows where date <= fold.train_end
       val   = block_shares_df rows where date in [fold.val_start, fold.val_end]
    
    2. Build share features (from Step 3) for train and val.
       Drop rows with NaN share lags.
    
    3. Train Softmax GBDT (total encounter shares):
       - X = train[SHARE_FEATURE_COLS]
       - y = train["block"]    # Target is the block index (0-3)
       - sample_weight = train["total_enc"]  # Volume-weighted
       - lgb.Dataset with categorical_feature=["site"]
       - Train with LGBM_SHARE params
       - Predict probabilities: model.predict(X) → (N, 4) matrix
    
    4. Train Softmax GBDT (admitted encounter shares):
       Same setup but:
       - sample_weight = train["admitted_enc"]
       - Or: reuse total share model if admitted share correlation > 0.95
    
    5. Predict on val:
       pred_shares = model.predict(val_X)  # shape (N_val_dates, 4)
       # Normalize to sum to 1 (softmax output already does, but safety):
       pred_shares = pred_shares / pred_shares.sum(axis=1, keepdims=True)
    
    6. Save fold predictions:
       → DATA_DIR/fold_{id}_share_predictions.parquet
    
    7. Save models:
       → MODEL_DIR/fold_{id}_model_share_total.txt
       → MODEL_DIR/fold_{id}_model_share_admitted.txt
```

### Approach B: Dirichlet Regression (Alternative)

```python
from scipy.optimize import minimize
from scipy.special import digamma, gammaln

def dirichlet_nll(params, X, Y):
    """Negative log-likelihood for Dirichlet regression.
    params: (n_features × n_classes) flattened
    X: (N, n_features) design matrix
    Y: (N, n_classes) observed shares (must be > 0)
    """
    K = Y.shape[1]
    alpha = np.exp(X @ params.reshape(-1, K))  # (N, K), all positive
    nll = -(
        gammaln(alpha.sum(axis=1))
        - gammaln(alpha).sum(axis=1)
        + ((alpha - 1) * np.log(Y.clip(1e-10))).sum(axis=1)
    ).sum()
    return nll
```

**Note**: Dirichlet regression requires shares > 0. Replace exact 0 shares with a small epsilon (1e-6) and renormalize.

### Approach C: Climatology Fallback

```python
def compute_climatology(train_df, keys=["site", "dow", "month"]):
    """Historical mean block shares by grouping keys."""
    clim = (
        train_df
        .groupby(keys + ["block"])["block_share"]
        .mean()
        .reset_index()
        .pivot_table(index=keys, columns="block", values="block_share")
        .reset_index()
    )
    # Renormalize rows to sum to 1 (float precision)
    share_cols = [0, 1, 2, 3]
    clim[share_cols] = clim[share_cols].div(clim[share_cols].sum(axis=1), axis=0)
    return clim
```

### Eval Notes — Step 5
- [ ] **Share prediction sum check**: For every predicted row, assert `sum(pred_shares) ≈ 1.0` (tolerance 1e-6).
- [ ] **Share prediction range**: All predicted shares in [0, 1]. Print any negative values before normalization.
- [ ] **Softmax GBDT accuracy**: Compute multi-class log-loss on validation set. Print per-fold.
- [ ] **Share WAPE**: Compute WAPE on predicted shares vs actual shares per block. Print per block, per site.
  ```
  Fold 1 Share WAPE:
    Block 0: share_wape=X.XXXX
    Block 1: share_wape=X.XXXX
    Block 2: share_wape=X.XXXX
    Block 3: share_wape=X.XXXX
  ```
- [ ] **Climatology comparison**: Compare share model's share WAPE against climatology baseline. If share model doesn't beat climatology by >5%, it's not worth the complexity — fall back to climatology.
- [ ] **Block share feature importance**: Print top 10 features by gain. `dow` and `month` should dominate (block distributions are primarily driven by time patterns).
- [ ] **Stability across folds**: Check if predicted share distributions are stable across folds. Large fold-to-fold variation in share predictions suggests the model is unstable.
- [ ] **Admitted vs Total share correlation**: If using separate models, compare predicted total shares vs admitted shares. If nearly identical, simplify to one model.
- [ ] **Edge case: zero total prediction**: When the daily model predicts total_enc ≈ 0 for a (site, date), block shares become irrelevant (all blocks get 0). Verify this edge case is handled correctly.

---

## Step 6: Hyperparameter Tuning (`step_06_tune.py`)

### Purpose
Use Optuna to find optimal hyperparameters for both the daily models and the share model, selecting by mean admitted WAPE across all 4 validation folds (the full pipeline: daily prediction → share allocation → block-level WAPE).

### Tuning Strategy

**CRITICAL**: The tuning objective for Pipeline C must be the **final block-level admitted WAPE**, not the daily-level WAPE. This means each Optuna trial must run the full pipeline (daily predict → share predict → allocate → evaluate at block level). This is more expensive but ensures we optimize the right thing.

#### Phase 1: Tune Daily Models (100 trials)

```python
def objective_daily(trial):
    params_total = {
        "objective": trial.suggest_categorical("objective", ["tweedie", "poisson"]),
        "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 5, 20),
    }
    if params_total["objective"] == "tweedie":
        params_total["tweedie_variance_power"] = trial.suggest_float(
            "tweedie_variance_power", 1.1, 1.9
        )
    
    covid_policy = trial.suggest_categorical("covid_policy", ["downweight", "exclude"])
    
    # For each fold: train daily → predict daily → allocate with FIXED share model → score block-level
    wapes = []
    for fold in FOLDS:
        daily_preds = train_and_predict_daily(fold, params_total, covid_policy)
        block_preds = allocate_to_blocks(daily_preds, share_preds[fold["id"]])  # Use pre-trained share model
        fold_wape = compute_admitted_wape(block_preds, actuals[fold["id"]])
        wapes.append(fold_wape)
    
    return np.mean(wapes)
```

#### Phase 2: Tune Share Model (50 trials)

```python
def objective_share(trial):
    params_share = {
        "n_estimators": trial.suggest_int("n_estimators", 400, 1500),
        "max_depth": trial.suggest_int("max_depth", 3, 7),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
    }
    
    # Also test share model type
    share_type = trial.suggest_categorical("share_type", ["softmax_gbdt", "climatology"])
    
    # For each fold: use FIXED daily model → predict shares → allocate → score
    wapes = []
    for fold in FOLDS:
        share_preds = train_and_predict_shares(fold, params_share, share_type)
        block_preds = allocate_to_blocks(daily_preds_fixed[fold["id"]], share_preds)
        fold_wape = compute_admitted_wape(block_preds, actuals[fold["id"]])
        wapes.append(fold_wape)
    
    return np.mean(wapes)
```

#### Phase 3: Joint Re-Tune (Optional, 30 trials)
After Phase 1 and 2, optionally run a small joint search where both daily and share params are tuned simultaneously. This captures interactions between the two model components. Only worth doing if Phase 1+2 yields promising results.

### Save Artifacts
- `MODEL_DIR/best_params_daily_total.json`
- `MODEL_DIR/best_params_daily_rate.json`
- `MODEL_DIR/best_params_share.json`
- `MODEL_DIR/optuna_study_daily.pkl`
- `MODEL_DIR/optuna_study_share.pkl`

### Eval Notes — Step 6
- [ ] **Optuna convergence (daily)**: Plot trial WAPE over trial number. Should plateau after ~50-70 trials.
- [ ] **Optuna convergence (share)**: Same check. May converge faster (simpler model).
- [ ] **Best vs Default**: Compare tuned mean WAPE against Step 4+5 default-param WAPE. Expected improvement: 2-5%.
- [ ] **COVID policy result**: Report which policy won. Downweight expected to win.
- [ ] **Objective result**: Tweedie vs Poisson for daily model.
- [ ] **Share model type result**: Does softmax GBDT beat climatology? If not, use climatology (simpler, more stable).
- [ ] **Overfitting check**: Compare train WAPE vs val WAPE for best trial. If large gap, increase regularization.
- [ ] **Per-fold variance**: Std across fold WAPEs for best trial. Flag if one fold is a large outlier.
- [ ] **Param stability**: Check if top-5 trials have similar hyperparams.

---

## Step 7: Prediction & Block Allocation (`step_07_predict.py`)

### Purpose
Combine daily forecasts with predicted block shares to produce final block-level predictions. Apply all hard constraints and produce submission-format CSVs.

### Logic (per fold or test period)

```
For each fold (or test period):
    1. Load best daily model params (from Step 6) and share model params.
    
    2. Retrain daily model on full training data (≤ train_end) with best params.
    
    3. Build daily features for prediction window.
    
    4. Predict daily totals:
       pred_daily_total = model_c1_total.predict(X_daily).clip(0)
       pred_daily_rate  = model_c1_rate.predict(X_daily).clip(0, 1)
       pred_daily_admitted = pred_daily_total * pred_daily_rate
    
    5. Predict block shares (using share model or climatology):
       pred_shares_total    = share_model.predict(X_share)  # (N, 4) probabilities
       pred_shares_admitted = share_model_adm.predict(X_share)
       Normalize each row to sum to 1.
    
    6. Allocate daily to blocks:
       For each (site, date):
         # Total encounters
         block_total_raw = pred_daily_total * pred_shares_total  # (4,) float values
         block_total_int = largest_remainder_round(block_total_raw)
         
         # Admitted encounters
         block_admitted_raw = pred_daily_admitted * pred_shares_admitted
         block_admitted_int = largest_remainder_round(block_admitted_raw)
    
    7. Enforce hard constraints:
       For each (site, date, block):
         a. Non-negativity: block_total = max(block_total, 0), same for admitted
         b. Admitted ≤ Total: block_admitted = min(block_admitted, block_total)
    
    8. Verify daily sum consistency:
       For each (site, date):
         assert sum(block_total[0:4]) == round(pred_daily_total)
       (Largest-remainder rounding guarantees this.)
    
    9. Format as submission CSV:
       columns: Site, Date, Block, ED Enc, ED Enc Admitted
    
   10. Save:
       → OUTPUT_DIR/fold_{id}_predictions.csv
```

### Largest-Remainder Rounding Implementation

```python
def largest_remainder_round(values, target_sum=None):
    """Round array to non-negative integers preserving the sum.
    
    Args:
        values: Array of non-negative floats to round.
        target_sum: Desired integer sum. If None, use round(sum(values)).
    
    Returns:
        Integer array summing to target_sum.
    """
    values = np.maximum(values, 0)  # safety: no negatives
    if target_sum is None:
        target_sum = int(round(np.sum(values)))
    
    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = target_sum - floored.sum()
    
    if deficit > 0:
        # Give +1 to the `deficit` items with largest remainders
        indices = np.argsort(-remainders)[:deficit]
        floored[indices] += 1
    elif deficit < 0:
        # Take -1 from items with smallest remainders (rare edge case)
        indices = np.argsort(remainders)[:abs(deficit)]
        floored[indices] = np.maximum(floored[indices] - 1, 0)
    
    return floored
```

### Sept-Oct 2025 Final Forecast

```
Train on ALL data ≤ 2025-08-31 (applying COVID policy)
Train share model on all historical block shares
Predict daily totals for 2025-09-01 to 2025-10-31
Predict block shares for 2025-09-01 to 2025-10-31
Allocate → 4 sites × 61 days × 4 blocks = 976 rows
```

### Eval Notes — Step 7
- [ ] **Row count**: Each fold CSV must have `4 × num_val_days × 4` rows. Sept-Oct: 976 rows.
- [ ] **Schema check**: Columns are exactly `["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]`.
- [ ] **Constraint check**:
  - All `ED Enc >= 0` ✓
  - All `ED Enc Admitted >= 0` ✓
  - All `ED Enc Admitted <= ED Enc` ✓
  - All values are integers ✓
- [ ] **Daily sum consistency**: For each (site, date), assert `sum(ED Enc across 4 blocks) == round(pred_daily_total)`. This is the core promise of hierarchical reconciliation.
- [ ] **No missing combos**: Assert full grid coverage (all sites × all dates × all blocks).
- [ ] **No duplicates**: Assert unique on `(Site, Date, Block)`.
- [ ] **Distribution sanity**: Compare prediction distribution (mean, std, percentiles) to recent training months' actuals.
- [ ] **Per-site volume share**: Site B should get ~36% of total volume.
- [ ] **Block distribution**: Predicted block shares should roughly match historical averages (Block 2-3 typically highest).
- [ ] **Allocation degradation check**: Compare daily-level WAPE (from Step 4) with block-level WAPE (after allocation). The difference quantifies how much error the share allocation adds.
  ```
  Fold X: daily_wape = 0.XXXX → block_wape = 0.XXXX (Δ = +0.XXXX)
  ```
  If the delta is large (>50% of daily WAPE), the share model is the bottleneck — focus improvement there.

---

## Step 8: Evaluation (`step_08_evaluate.py`)

### Purpose
Run the `eval.md` evaluator on Pipeline C's fold CSVs. Produces the official pipeline score for cross-pipeline comparison.

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
print(f"Pipeline C — Mean Admitted WAPE: {mean_admitted_wape:.4f}")
```

### Output Report

```
═══════════════════════════════════════════════════════════════
 PIPELINE C: HIERARCHICAL RECONCILIATION — EVALUATION REPORT
═══════════════════════════════════════════════════════════════

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

 PIPELINE C DIAGNOSTIC — HIERARCHICAL DECOMPOSITION:
   Daily-Level Total WAPE (before allocation):   X.XXXX
   Block-Level Total WAPE (after allocation):    X.XXXX
   Allocation Error Contribution:                X.XXXX (Δ)
   Share Model Type Used:                        softmax_gbdt / climatology

 SANITY CHECKS:
   Max site WAPE / Min site WAPE ratio:          X.XX (should be < 2.0)
   Block 0 WAPE / Block avg WAPE ratio:          X.XX (flag if > 2.0)
   Daily sum consistency violations:              0 (should always be 0)
   Per-fold WAPE std / mean:                      X.XX (flag if > 0.30)

═══════════════════════════════════════════════════════════════
```

### OOF Predictions for Ensemble
Save out-of-fold predictions for all 4 folds for downstream stacking:
- `OUTPUT_DIR/oof_predictions.csv` — combined OOF predictions (same schema as submission)
- This is consumed by the ensemble step (master strategy §4)

### Eval Notes — Step 8
- [ ] **Primary metric**: Mean Admitted WAPE — this ranks Pipeline C against other pipelines.
- [ ] **Sanity check (master strategy §2.4)**: No single site's WAPE should exceed 2× the best site's WAPE.
- [ ] **Block stability**: Block 0 (overnight) WAPE should not be >2× others. If it is, the share model is failing for overnight hours — investigate.
- [ ] **Per-fold variance**: If one fold's WAPE is >2× others, investigate temporal drift.
- [ ] **Compare to Pipeline A/B**: Is Pipeline C competitive? Master strategy says A+C is the "Minimum Viable Submission" — Pipeline C must add value.
- [ ] **Hierarchical decomposition diagnostic**: The "Allocation Error Contribution" (difference between daily WAPE and block WAPE) tells us:
  - If small (<20% of daily WAPE): Share allocation is good. Daily model is the bottleneck.
  - If large (>50% of daily WAPE): Share model is poor. Try different share approach or fall back to climatology.
- [ ] **Daily sum consistency**: Must be exactly 0 violations. This is Pipeline C's structural guarantee.
- [ ] **Compare to naive baseline**: Same-period-last-year at block level. Pipeline C must beat it.
- [ ] **Save report**: Write to `OUTPUT_DIR/evaluation_report.json`.

---

## Step 9: End-to-End Orchestrator (`run_pipeline.py`)

### Purpose
Run the full Pipeline C from data loading to evaluation in one command.

### Usage

```bash
# Full pipeline with default params
python Pipelines/pipeline_C/run_pipeline.py

# Skip tuning (use defaults — for quick iteration)
python Pipelines/pipeline_C/run_pipeline.py --skip-tune

# Tuning only
python Pipelines/pipeline_C/run_pipeline.py --tune-only

# Single fold (for debugging)
python Pipelines/pipeline_C/run_pipeline.py --fold 1

# Generate final Sept-Oct forecast
python Pipelines/pipeline_C/run_pipeline.py --final-forecast

# Force share model type (override config)
python Pipelines/pipeline_C/run_pipeline.py --share-model climatology
```

### Pipeline Flow

```python
def main(args):
    print("=" * 65)
    print("PIPELINE C: HIERARCHICAL RECONCILIATION — Starting")
    print("=" * 65)
    
    # Step 1: Data Loading & Daily Aggregation
    print("\n[Step 1/8] Loading data + building daily aggregates & block shares...")
    block_df, daily_df, share_df = run_data_loading()
    validate_step_1(block_df, daily_df, share_df)
    
    # Step 2: Feature Engineering (Daily Model)
    print("\n[Step 2/8] Engineering daily-model features...")
    daily_features = run_daily_feature_engineering(daily_df)
    validate_step_2(daily_features)
    
    # Step 3: Feature Engineering (Share Model)
    print("\n[Step 3/8] Engineering share-model features...")
    share_features = run_share_feature_engineering(share_df)
    validate_step_3(share_features)
    
    if not args.tune_only:
        # Step 4: Train Daily Models (default params)
        print("\n[Step 4/8] Training daily models (default HP)...")
        daily_results = run_daily_training(daily_features, params="default")
        validate_step_4(daily_results)
        
        # Step 5: Train Share Models
        print("\n[Step 5/8] Training share models...")
        share_results = run_share_training(share_features, params="default")
        validate_step_5(share_results)
    
    if not args.skip_tune:
        # Step 6: Hyperparameter Tuning
        print("\n[Step 6/8] Tuning hyperparameters (Optuna)...")
        best_params = run_tuning(daily_features, share_features)
        validate_step_6(best_params)
        
        # Re-train with best params
        print("\n[Step 4b/8] Re-training daily models with tuned HP...")
        daily_results = run_daily_training(daily_features, params=best_params["daily"])
        print("\n[Step 5b/8] Re-training share models with tuned HP...")
        share_results = run_share_training(share_features, params=best_params["share"])
    
    # Step 7: Prediction & Block Allocation
    print("\n[Step 7/8] Allocating daily forecasts to blocks...")
    run_prediction(daily_results, share_results)
    validate_step_7()
    
    # Step 8: Evaluation
    print("\n[Step 8/8] Evaluating against eval.md contract...")
    report = run_evaluation()
    print_report(report)
    
    if args.final_forecast:
        print("\n[FINAL] Generating Sept-Oct 2025 forecast...")
        run_final_forecast(daily_features, share_features, best_params)
    
    print("\n" + "=" * 65)
    print("PIPELINE C: COMPLETE")
    print("=" * 65)
```

---

## Dependencies

```
lightgbm>=4.0
optuna>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3      # LabelEncoder, metrics utilities
scipy>=1.10            # Dirichlet regression (if used), optimize
holidays>=0.40         # US holiday calendar
pyarrow>=14.0          # Parquet I/O
```

---

## Execution Order & Time Estimates

| Step | File | Estimated Time | Can Skip? |
|------|------|---------------|-----------|
| 0 | `config.py` | — (static) | No |
| 1 | `step_01_data_loading.py` | ~20s | No |
| 2 | `step_02_feature_eng_daily.py` | ~1-2 min | No |
| 3 | `step_03_feature_eng_shares.py` | ~1-2 min | No |
| 4 | `step_04_train_daily.py` | ~3-5 min (4 folds × 2 models, fewer rows than Pipeline A) | No |
| 5 | `step_05_train_shares.py` | ~2-4 min (4 folds × 1-2 share models) | No |
| 6 | `step_06_tune.py` | ~1.5-3 hours (150 trials, each runs full pipeline) | Yes (use defaults) |
| 7 | `step_07_predict.py` | ~30s | No |
| 8 | `step_08_evaluate.py` | ~10s | No |

**Total (with tuning):** ~2-4 hours  
**Total (skip tuning):** ~10-15 minutes

---

## Key Risk Mitigations

| Risk | Mitigation |
|------|------------|
| **Share model adds more error than it saves** | Compare block-level WAPE (with allocation) vs daily WAPE. If allocation adds >50% error, fall back to climatology shares. |
| **Daily model lag leakage** | All daily lags ≥ 63; all rolling shifted by 63; validated in Step 2. |
| **Aggregate encoding leakage** | Computed on training fold only (Step 4). |
| **COVID bias** | Downweight policy (0.1× sample weight); tested vs exclusion in Step 6. |
| **Block 0 (overnight) drift** | By-block WAPE check in Step 8; if Block 0 >> others, add block-specific share adjustments or site-block climatology correction. |
| **Zero-day allocation** | When daily total ≈ 0, shares are irrelevant. Handled by setting all blocks to 0. |
| **Admitted > Total after rounding** | Re-enforce `admitted = min(admitted, total)` per row after allocation + rounding. |
| **Softmax share model instability** | Climatology fallback always available. Share model variance check in Step 5. |
| **Dirichlet zero-share edge case** | Replace exact 0 shares with epsilon (1e-6) before fitting. Not an issue for softmax GBDT. |

---

## Output Artifacts

After a complete Pipeline C run:

```
Pipelines/pipeline_C/
  data/
    daily_df.parquet                    # Step 1 output (daily aggregates)
    block_shares_df.parquet             # Step 1 output (block shares)
    share_wide_df.parquet               # Step 1 output (pivoted shares)
    daily_features.parquet              # Step 2 output
    share_features.parquet              # Step 3 output
    fold_{1-4}_daily_predictions.parquet  # Step 4 intermediate
    fold_{1-4}_share_predictions.parquet  # Step 5 intermediate
  models/
    fold_{1-4}_model_c1_total.txt       # Daily total LightGBM models
    fold_{1-4}_model_c1_rate.txt        # Daily admit rate LightGBM models
    fold_{1-4}_model_share_total.txt    # Share models (total)
    fold_{1-4}_model_share_admitted.txt # Share models (admitted)
    best_params_daily_total.json        # Tuned daily model HP
    best_params_daily_rate.json
    best_params_share.json              # Tuned share model HP
    optuna_study_daily.pkl
    optuna_study_share.pkl
  output/
    fold_1_predictions.csv              # Submission-format (Site, Date, Block, ED Enc, ED Enc Admitted)
    fold_2_predictions.csv
    fold_3_predictions.csv
    fold_4_predictions.csv
    oof_predictions.csv                 # All folds combined (for stacking ensemble)
    final_sept_oct_2025.csv             # Final competition submission (976 rows)
    evaluation_report.json              # Full eval results
    daily_feature_importance.csv        # Top features by gain (daily model)
    share_feature_importance.csv        # Top features by gain (share model)
  logs/
    pipeline_c_run_{timestamp}.log      # Full execution log
```

---

## Appendix A: Key Differences from Pipeline A and B

| Aspect | Pipeline A | Pipeline B | Pipeline C (this) |
|--------|-----------|------------|-------------------|
| Modeling grain | Block-level directly | Block-level with horizon buckets | **Daily-level → allocate to blocks** |
| Number of target series | 16 (4 sites × 4 blocks) | 16 × 3 buckets | **4 (daily per site) + share model** |
| Training data noise | Higher (block-level ~25/obs) | Same as A | **Lower (daily ~100/obs)** |
| Structural coherence | None (blocks may not sum) | None | **Guaranteed (block sums = daily total)** |
| Share prediction | N/A | N/A | **Softmax GBDT / Dirichlet / Climatology** |
| Key advantage | Simple, direct | Horizon-aware | **Noise reduction + coherent block sums** |
| Key risk | Block-level noise | Computational cost | **Share allocation error** |
| Expected WAPE | Baseline | 3-8% better than A | **Comparable to A; diversifies ensemble** |

---

## Appendix B: When to Use Climatology vs. Softmax GBDT for Shares

Use **Climatology** if:
- Share model's block WAPE is not >5% better than climatology
- Share std per block < 0.02 across all (site, block) combos (shares barely change)
- Tuning shows climatology wins in Optuna (Step 6)
- Simplicity is preferred (fewer failure modes)

Use **Softmax GBDT** if:
- Block shares show meaningful temporal variation (std > 0.05)
- Holiday/event days show distinctly different share patterns
- Share model beats climatology by >5% on block WAPE
- Computational budget allows it

Use **Dirichlet Regression** if:
- You want principled compositional modeling
- Softmax GBDT overfits (small effective sample sizes)
- Want uncertainty estimates on shares (Dirichlet provides a full distribution)

---

## Appendix C: Integration with Ensemble (master_strategy §4)

Pipeline C's OOF predictions (`oof_predictions.csv`) are used as inputs to the stacking meta-learner:
- `pred_C_total` and `pred_C_admitted` are meta-features alongside Pipeline A/B/D/E predictions
- The meta-learner can learn to weight Pipeline C more heavily for specific (site, block, dow) combinations where hierarchical decomposition excels
- Pipeline C's daily forecasts can also serve as a reconciliation anchor for post-ensemble processing (master strategy §4.3: "use Pipeline C's daily forecast as anchor if available")

**Specific ensemble value of Pipeline C:**
- Provides structurally coherent block predictions that sum correctly — other pipelines don't guarantee this
- Errors are correlated at the daily level but distributed differently across blocks → low correlation with Pipeline A/B errors at block level
- Particularly valuable as ensemble anchor when Pipeline A/B produce incoherent block sums
