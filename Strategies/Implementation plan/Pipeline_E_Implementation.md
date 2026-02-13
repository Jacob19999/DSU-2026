# Pipeline E: Reason-Mix Latent Factor Model — Step-by-Step Implementation Plan

**Status:** READY FOR IMPLEMENTATION  
**Source:** `master_strategy.md` §3.5, `strategies_2.md` Pipeline F  
**Core Idea:** Compress visit-reason composition into latent factors (PCA/NMF), forecast factors forward via AR/GBDT to capture momentum deviations (e.g., early flu onset, unexpected trauma surge), then use predicted factors as extra regressors in a final GBDT that predicts `total_enc` and `admit_rate`.  
**Location:** `Pipelines/pipeline_E/`  
**Data Dependency:** `master_block_history.parquet` produced by the Data Source step. Pipeline E specifically needs the `count_reason_*` columns to build the share matrix.

---

## Why Pipeline E Adds Value

- **Composition-Aware:** ED volume surges are often driven by shifts in visit type (flu season → respiratory surge, summer → trauma spike). Calendar features capture *expected* seasonal patterns, but Pipeline E captures **deviations from expected patterns** — an early flu onset or an unusual trauma week.
- **Orthogonal Signal:** Pipelines A/B model demand-side patterns (lags, trends, calendar). Pipeline E adds supply-side / epidemiological composition signals. This structural diversity maximizes ensemble gains.
- **Factor Momentum:** The key feature `factor_momentum = factor_now − factor_lag_7` tells the model "respiratory visits are trending up faster than usual" — a signal no other pipeline captures.
- **Minimal Incremental Work:** The Data Source already produces `count_reason_*` columns. Pipeline E just compresses and forecasts them.

### Critical Warning: Climatology Circularity (from §3.5)

> If you reduce climatology-filled shares to PCA factors, the result is a deterministic function of (Month, DOW, Block) — information already captured by calendar features. **Zero new signal.**

**This pipeline's value comes ENTIRELY from capturing deviations via the AR/GBDT factor forecasting approach.** If that doesn't improve CV WAPE over Pipeline A alone, drop Pipeline E from the ensemble.

---

## File Structure

```
Pipelines/
  pipeline_E/
    __init__.py
    config.py                           # Constants, paths, folds, factor config
    step_01_data_loading.py             # Load master parquet, validate, derive admit_rate
    step_02_share_matrix.py             # Build reason-category share matrix from count_reason_*
    step_03_factor_extraction.py        # PCA/NMF dimensionality reduction → k latent factors
    step_04_factor_forecasting.py       # AR/GBDT models to forecast factors forward
    step_05_feature_eng.py              # Combine standard features + predicted factors + momentum
    step_06_train.py                    # Train final GBDT (total_enc + admit_rate)
    step_07_tune.py                     # Optuna hyperparameter search
    step_08_predict.py                  # Generate forecasts + post-processing
    step_09_evaluate.py                 # 4-fold forward validation scoring
    run_pipeline.py                     # End-to-end orchestrator
```

---

## Step 0: Configuration (`config.py`)

### Purpose
Single source of truth for all constants — paths, fold definitions, factor extraction params, feature lists, hyperparameter defaults.

### Contents

```python
# --- Paths ---
MASTER_PARQUET_PATH = "Pipelines/Data Source/Data/master_block_history.parquet"
RAW_DATA_PATH = "Dataset/DSU-Dataset.csv"
OUTPUT_DIR = "Pipelines/pipeline_E/output/"
MODEL_DIR = "Pipelines/pipeline_E/models/"
DATA_DIR = "Pipelines/pipeline_E/data/"

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
COVID_SAMPLE_WEIGHT = 0.1

# --- Factor Extraction ---
N_FACTORS = 5                     # Number of latent factors (test 3-5 via ablation)
FACTOR_METHOD = "pca"             # "pca" or "nmf" — test both
TOP_N_REASONS = 20                # Number of reason categories to include in share matrix
MIN_CATEGORY_VOLUME = 100         # Drop categories below this total volume threshold

# --- Factor Forecasting ---
FACTOR_LAG_DAYS = [7, 14, 28, 56, 91, 182, 364]
FACTOR_ROLLING_WINDOWS = [7, 14, 28]
FACTOR_ROLLING_SHIFT = 7          # Factor forecasting shift (factors predicted ~weekly)
FACTOR_FORECAST_METHOD = "gbdt"   # "gbdt" or "ar" — GBDT preferred per §3.5

# --- Final Model (Main GBDT on total_enc/admit_rate) ---
MAX_HORIZON = 63                  # Same as Pipeline A — max forecast horizon
LAG_DAYS = [63, 70, 77, 91, 182, 364]
ROLLING_WINDOWS = [7, 14, 28, 56, 91]
ROLLING_SHIFT = 63                # All rolling stats shifted by >= max horizon

# --- LightGBM Defaults (before Optuna) ---
LGBM_DEFAULT_TOTAL = {
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
LGBM_DEFAULT_RATE = {
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
LGBM_FACTOR_FORECAST = {
    "objective": "regression",
    "n_estimators": 500,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 3.0,
    "min_child_weight": 3,
    "verbosity": -1,
}

# --- Optuna ---
OPTUNA_N_TRIALS_FINAL = 100       # For the main total_enc / admit_rate models
OPTUNA_N_TRIALS_FACTOR = 30       # For factor forecast models (simpler)
```

### Eval Notes — Step 0
- [ ] **Fold dates**: Verify all 4 fold date ranges match `eval.md` exactly.
- [ ] **MAX_HORIZON = 63**: Conservative enough for 61-day val windows + 2-day buffer.
- [ ] **LAG_DAYS all >= 63**: No leakage in the final model's target lags.
- [ ] **FACTOR_LAG_DAYS >= 7**: Factor forecasting uses shorter lags because it predicts factor values, not encounter targets — but validate no circular dependency (factor lags must reference actual historical factors, not predicted ones).
- [ ] **N_FACTORS = 5**: Reasonable starting point for ~20 reason categories. Test 3 and 7 as ablation.
- [ ] **TOP_N_REASONS = 20**: Matches data_source.md §2.2 specification.

---

## Step 1: Data Loading & Preprocessing (`step_01_data_loading.py`)

### Purpose
Load the unified `master_block_history.parquet`, validate schema, derive `admit_rate`, and prepare the base DataFrame for Pipeline E's factor extraction.

### Logic

1. **Load Parquet**: Read `MASTER_PARQUET_PATH`.
2. **Validate Schema**:
   - Assert required columns: `site`, `date`, `block`, `total_enc`, `admitted_enc`
   - Assert `count_reason_*` columns exist (Pipeline E's primary input)
   - Check row count: ~45,776 rows (4 sites × 2,861 days × 4 blocks)
3. **Derive `admit_rate`**:
   - `admit_rate = admitted_enc / total_enc` (fill 0/0 → 0.0)
   - Clip to [0, 1]
4. **Sort**: By `(site, block, date)` — critical for lag computation.
5. **Date Typing**: Ensure `date` is `pd.Timestamp`.

### Key Check: Reason Column Discovery

```python
reason_cols = sorted([c for c in df.columns if c.startswith("count_reason_")])
assert len(reason_cols) >= 5, f"Expected >=5 reason columns, got {len(reason_cols)}"
print(f"Found {len(reason_cols)} reason categories: {reason_cols[:5]}...")
```

### Eval Notes — Step 1
- [ ] **Row count**: Verify `4 × num_days × 4` rows. Expected ~45,776.
- [ ] **Reason column count**: Should be ~20 `count_reason_*` columns + 1 `count_reason_other`. Print all found.
- [ ] **No NaN in targets**: Assert `total_enc` and `admitted_enc` have no NaN.
- [ ] **Reason sum check**: `sum(count_reason_*) == total_enc` for each row (or close — may differ if some encounters lack reason codes). Print mismatches.
- [ ] **admit_rate range**: Assert all values in [0.0, 1.0].
- [ ] **Date range**: Min = 2018-01-01, max covers through 2025-08-31.
- [ ] **Site B volume**: Verify Site B accounts for ~36% of total volume (largest facility per master strategy).
- [ ] **Print summary**: Per-site mean/std of `total_enc`, `admitted_enc`, and total reason volume.

---

## Step 2: Build Reason-Category Share Matrix (`step_02_share_matrix.py`)

### Purpose
Transform raw reason counts into a **share matrix** suitable for dimensionality reduction. Each row's reason shares sum to 1.0, capturing the *composition* of visits rather than the volume.

### Logic

#### 2.1 Filter Reason Categories

```python
# Identify top N categories by total volume across all rows
reason_cols = [c for c in df.columns if c.startswith("count_reason_")]
category_volumes = df[reason_cols].sum().sort_values(ascending=False)

# Keep top N that exceed minimum volume threshold
top_categories = category_volumes[
    category_volumes >= MIN_CATEGORY_VOLUME
].head(TOP_N_REASONS).index.tolist()

# Everything else → "other"
other_cols = [c for c in reason_cols if c not in top_categories]
df["count_reason_other_combined"] = df[other_cols].sum(axis=1)
```

#### 2.2 Compute Shares

```python
selected_cols = top_categories + ["count_reason_other_combined"]
row_totals = df[selected_cols].sum(axis=1).clip(lower=1)  # Avoid div/0

share_df = df[["site", "date", "block"]].copy()
for col in selected_cols:
    share_name = col.replace("count_reason_", "share_")
    share_df[share_name] = df[col] / row_totals
```

Each row now has shares summing to ~1.0 (may be slightly off due to clipping, which is fine for PCA/NMF).

#### 2.3 Smooth Shares (Optional but Recommended)

Block-level shares are noisy (small denominators). Apply 7-day rolling mean per `(site, block)` to stabilize:

```python
share_cols = [c for c in share_df.columns if c.startswith("share_")]
for col in share_cols:
    share_df[col] = (
        share_df.groupby(["site", "block"])[col]
        .transform(lambda s: s.rolling(7, min_periods=1).mean())
    )
```

**Rationale**: A single day where only 2 patients visit (both respiratory) gives share_respiratory=1.0, which is noise, not signal. Smoothing stabilizes this.

#### 2.4 Save Share Matrix

Save to `DATA_DIR/share_matrix.parquet`.

### Eval Notes — Step 2
- [ ] **Share sum check**: After computation, verify `sum(share_cols) ≈ 1.0` for each row (tolerance ±0.01). Print max deviation.
- [ ] **No NaN in shares**: After smoothing, shares should have no NaN (rolling with `min_periods=1` ensures this). Assert and print any violations.
- [ ] **Distribution inspection**: Print mean share per category across all rows. Expect a few dominant categories (respiratory, injury, etc.) and a long tail.
- [ ] **Temporal patterns**: Plot mean share of top 3 categories over time (monthly average). Should show clear seasonality (e.g., respiratory peaks in winter, injury peaks in summer). If flat, the factor model won't help.
- [ ] **COVID era check**: Print share distributions during COVID (2020-03 to 2021-06) vs. normal. Expect composition shifts (less non-emergent, higher acuity).
- [ ] **Zero-row check**: Count rows where `row_total == 0` (no visits at all). These should be rare and handled by the clip(lower=1).
- [ ] **Category count**: Print final number of share columns (expected: TOP_N_REASONS + 1 other).

---

## Step 3: Factor Extraction — PCA/NMF (`step_03_factor_extraction.py`)

### Purpose
Reduce the ~20-dimensional share matrix to `k` latent factors (k=3-5) that capture the dominant composition patterns (e.g., "respiratory season factor", "trauma factor", "general acuity factor").

### Logic

#### 3.1 Prepare Input Matrix

```python
share_cols = [c for c in share_df.columns if c.startswith("share_")]
X_shares = share_df[share_cols].values  # Shape: (N_rows, N_categories)
```

#### 3.2 Fit PCA (Primary Method)

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=N_FACTORS)
factors = pca.fit_transform(X_shares)  # Shape: (N_rows, N_FACTORS)

# Name the factors
for i in range(N_FACTORS):
    share_df[f"factor_{i}"] = factors[:, i]

# Save explained variance
explained_var = pca.explained_variance_ratio_
cumulative_var = np.cumsum(explained_var)
print(f"Explained variance per factor: {explained_var}")
print(f"Cumulative: {cumulative_var}")
```

**Target**: k=5 should capture ≥80% of variance. If not, increase k or revisit category selection.

#### 3.3 Fit NMF (Alternative — Test as Ablation)

```python
from sklearn.decomposition import NMF

nmf = NMF(n_components=N_FACTORS, init="nndsvda", max_iter=500, random_state=42)
factors_nmf = nmf.fit_transform(X_shares)

for i in range(N_FACTORS):
    share_df[f"factor_nmf_{i}"] = factors_nmf[:, i]
```

NMF enforces non-negativity in both factors and loadings → more interpretable ("how much of each archetype"). PCA is more general. Test both; pick by CV WAPE.

#### 3.4 Factor Interpretation (Diagnostic, Not Used in Model)

```python
# For PCA: inspect loadings to understand what each factor represents
loadings = pd.DataFrame(
    pca.components_.T,
    index=share_cols,
    columns=[f"PC{i}" for i in range(N_FACTORS)]
)
print("PCA Loadings (top contributors per factor):")
for i in range(N_FACTORS):
    top_pos = loadings[f"PC{i}"].nlargest(3).index.tolist()
    top_neg = loadings[f"PC{i}"].nsmallest(3).index.tolist()
    print(f"  Factor {i}: + {top_pos}, - {top_neg}")
```

Expect something like:
- Factor 0: Respiratory vs. Injury (seasonal axis)
- Factor 1: Trauma acuity shift
- Factor 2: GI / viral outbreak component

#### 3.5 CRITICAL: Fold-Aware Fitting

**PCA/NMF must be fit ONLY on training data for each fold to prevent leakage.** The transform is then applied to both train and validation rows.

```python
def fit_factors(df, train_end, method="pca", n_factors=N_FACTORS):
    """Fit factor model on training data only, transform all rows."""
    train_mask = df["date"] <= train_end
    X_train = df.loc[train_mask, share_cols].values
    X_all = df[share_cols].values
    
    if method == "pca":
        model = PCA(n_components=n_factors)
    else:
        model = NMF(n_components=n_factors, init="nndsvda", max_iter=500, random_state=42)
    
    model.fit(X_train)
    factors_all = model.transform(X_all)
    return factors_all, model
```

#### 3.6 Save Artifacts

- Factor-enriched DataFrame → `DATA_DIR/factors.parquet`
- Fitted PCA/NMF model → `MODEL_DIR/factor_model.pkl`
- Loadings table → `DATA_DIR/factor_loadings.csv`

### Eval Notes — Step 3
- [ ] **Explained variance (PCA)**: Print per-factor and cumulative. Target: ≥80% in k=5 factors. If <60%, the share matrix might be too noisy or categories too granular.
- [ ] **Reconstruction error (NMF)**: Print `||X - W*H||` — should decrease with k. Compare PCA vs NMF reconstruction.
- [ ] **Factor stationarity**: Plot each factor over time (monthly mean). They should show clear seasonal patterns, NOT a random walk. If trending, that's fine (the AR model will capture it).
- [ ] **Factor spread**: Print mean/std of each factor. If any factor has near-zero variance, it's not capturing useful signal — consider reducing k.
- [ ] **COVID impact on factors**: Plot factors during COVID era. Expect visible shifts (composition changed dramatically). Verify the downstream COVID downweighting handles this.
- [ ] **Fold stability check**: Fit PCA separately on fold 1 and fold 4 training data. Compare loadings — they should be qualitatively similar (same dominant categories per factor). If not, factors are unstable and Pipeline E may not add value.
- [ ] **No leakage**: Verify that `fit()` only sees data ≤ `train_end`. Print the max date in the training subset used for fitting.

---

## Step 4: Factor Forecasting (`step_04_factor_forecasting.py`)

### Purpose
Forecast each latent factor forward into the validation/test window. This is where Pipeline E's value lives — capturing **momentum deviations** from seasonal norms. Uses GBDT (preferred) or AR models on the factor time series.

### CRITICAL: Why Not Climatology

Per master strategy §3.5:
> The "Simple: climatology by (Month, DOW, Block)" option is **nearly useless**: if you reduce climatology-filled shares to PCA factors, the result is a deterministic function of (Month, DOW, Block) — information already captured by calendar features. **Zero new signal.**

The value is in the **deviations from climatology** — an early flu onset shows up as factor_respiratory being unusually high for the time of year. Only an AR/GBDT model on factor history can capture this.

### Logic

#### 4.1 Build Factor Time Series

For each `(site, block)` group, extract the factor time series sorted by date:

```python
factor_cols = [f"factor_{i}" for i in range(N_FACTORS)]

# Each (site, block) group has one time series per factor
groups = df.groupby(["site", "block"])
```

#### 4.2 Feature Engineering for Factor Forecasting

For each factor's time series within a `(site, block)` group:

| Feature | Formula | Notes |
|---------|---------|-------|
| `factor_lag_7` | `factor_i.shift(7)` | 1-week ago factor value |
| `factor_lag_14` | `factor_i.shift(14)` | 2-weeks ago |
| `factor_lag_28` | `factor_i.shift(28)` | 4-weeks ago |
| `factor_lag_56` | `factor_i.shift(56)` | 8-weeks ago |
| `factor_lag_91` | `factor_i.shift(91)` | ~3 months ago |
| `factor_lag_182` | `factor_i.shift(182)` | ~6 months ago |
| `factor_lag_364` | `factor_i.shift(364)` | Same time last year |
| `factor_roll_mean_7` | `factor_i.shift(7).rolling(7).mean()` | Recent weekly average |
| `factor_roll_mean_14` | `factor_i.shift(7).rolling(14).mean()` | 2-week average |
| `factor_roll_mean_28` | `factor_i.shift(7).rolling(28).mean()` | Monthly average |
| `factor_momentum_7` | `factor_lag_7 - factor_lag_14` | Week-over-week change **← KEY FEATURE** |
| `factor_momentum_28` | `factor_lag_28 - factor_lag_56` | Month-over-month change |
| `factor_deviation_from_yearly` | `factor_lag_7 - factor_lag_364` | Year-over-year deviation |

Calendar features (deterministic, safe for any horizon):
- `dow`, `month`, `day_of_year`, `is_weekend`, `week_of_year`
- `doy_sin`, `doy_cos`, `month_sin`, `month_cos` (cyclical encodings)

Identifiers:
- `site` (categorical), `block` (int)

#### 4.3 Handling the Forecast Gap

**Problem**: To predict factors at date `t` (in the val/test window), we need factor values at `t-7` etc. But if `t-7` is also in the forecast window, we don't have actual factor values.

**Solution — Two-Stage Approach**:
1. For validation rows where `t - lag` falls within training history: use actual factor values (no issue).
2. For validation rows where `t - lag` falls in the forecast window itself: use **predicted factor values** from earlier predictions (recursive).
3. **Simpler fallback** (recommended for v1): Only use lags ≥ `max_horizon` (63 days) for factor forecasting features. This means the factor forecast uses the same safe lag policy as Pipeline A's target lags. The momentum signal comes from `factor_lag_63 - factor_lag_70` instead of `factor_lag_7 - factor_lag_14`.

**Recommended for v1**: Use the simpler fallback (lags ≥ 63 only). It's safer against error accumulation and still captures the "current composition is unusual for this time of year" signal via year-over-year comparisons.

**For v2 (if v1 shows promise)**: Implement recursive factor prediction with shorter lags. This is the higher-risk, higher-reward approach.

#### 4.4 Train Factor Forecast Models

One GBDT per factor (total: `N_FACTORS` models):

```python
for factor_idx in range(N_FACTORS):
    target = f"factor_{factor_idx}"
    features = factor_lag_cols + factor_rolling_cols + calendar_cols + ["site", "block"]
    
    model = lgb.LGBMRegressor(**LGBM_FACTOR_FORECAST)
    model.fit(
        X_train[features], y_train[target],
        eval_set=[(X_val[features], y_val[target])],
        callbacks=[lgb.early_stopping(50)],
    )
```

#### 4.5 Alternative: AR Model (Simpler)

```python
from statsmodels.tsa.ar_model import AutoReg

# Per (site, block, factor) — 4 × 4 × 5 = 80 small AR models
for (site, block), group in df.groupby(["site", "block"]):
    for factor_idx in range(N_FACTORS):
        series = group[f"factor_{factor_idx}"].dropna()
        ar_model = AutoReg(series, lags=[7, 14, 28, 364], seasonal=True, period=7)
        ar_fit = ar_model.fit()
        # Predict forward...
```

Test GBDT vs. AR; pick by factor forecast MAE on validation folds.

#### 4.6 Generate Predicted Factors

For each fold's validation window (and eventually Sept-Oct 2025):

```python
predicted_factors = {}
for factor_idx in range(N_FACTORS):
    model = load_model(f"factor_model_{factor_idx}")
    X_pred = build_factor_features(df, forecast_dates, factor_idx)
    predicted_factors[f"factor_{factor_idx}_pred"] = model.predict(X_pred)
```

#### 4.7 Compute Factor Momentum Features (THE KEY)

This is the signature feature of Pipeline E per master strategy §3.5:

```python
# From actual (training period) or predicted (forecast period) factors:
for i in range(N_FACTORS):
    # Momentum: how much the factor has changed recently
    df[f"factor_{i}_momentum"] = df[f"factor_{i}"] - df[f"factor_{i}"].shift(7)
    
    # For forecast rows, momentum uses predicted values
    # factor_momentum = predicted_factor_now - actual_factor_7_days_ago
    # (or predicted_factor_now - predicted_factor_7_days_ago for far-horizon rows)
```

**CRITICAL**: `factor_momentum` is the primary value-add of Pipeline E. Without it, factors are just redundant calendar proxies.

#### 4.8 Save Factor Forecast Artifacts

- Predicted factor values for all rows → `DATA_DIR/predicted_factors.parquet`
- Factor forecast models → `MODEL_DIR/factor_forecast_model_{i}.pkl`
- Factor forecast accuracy metrics → `DATA_DIR/factor_forecast_eval.json`

### Eval Notes — Step 4
- [ ] **Factor forecast accuracy**: For each factor, compute MAE and R² on the validation window. Print per-factor. If R² < 0.1 for all factors, factor forecasting isn't working — consider dropping Pipeline E.
- [ ] **Momentum signal check**: Compute correlation between `factor_momentum` and `total_enc` residuals from a baseline (e.g., Pipeline A predictions). If correlation is near zero, momentum doesn't carry signal for encounter forecasting.
- [ ] **Lag leakage audit**: Verify all factor forecast features use lags ≥ chosen minimum. Print minimum shift per feature column.
- [ ] **Temporal stability**: Plot predicted vs. actual factors for each validation fold. Predicted should track the general shape even if it misses exact values. If predictions are flat/constant, the model isn't learning.
- [ ] **COVID-era factors**: Verify predicted factors during COVID-adjacent periods aren't wildly off. If so, COVID downweighting in the factor forecast training may be needed.
- [ ] **Recursive prediction check (v2 only)**: If using recursive approach, plot error accumulation over forecast horizon. Error should grow gradually, not explode. If day-60 factor error is 3× day-1 error, switch to the safe-lag fallback.
- [ ] **Feature importance for factor models**: Print top 5 features per factor model. Expect factor lags and momentum to dominate. If calendar features dominate, the factor forecast is just reproducing seasonality (low value).

---

## Step 5: Feature Engineering for Final Model (`step_05_feature_eng.py`)

### Purpose
Build the full feature matrix for the final GBDT that predicts `total_enc` and `admit_rate`. Combines standard demand-side features (same as Pipeline A) with Pipeline E's unique factor features.

### Feature Groups

#### 5A. Standard Target Lags (Same as Pipeline A)

Per `(site, block)` group, all shifted ≥ 63 days:

| Feature | Formula |
|---------|---------|
| `lag_63` | `total_enc.shift(63)` |
| `lag_70` | `total_enc.shift(70)` |
| `lag_77` | `total_enc.shift(77)` |
| `lag_91` | `total_enc.shift(91)` |
| `lag_182` | `total_enc.shift(182)` |
| `lag_364` | `total_enc.shift(364)` |

Same lags for `admit_rate`.

#### 5B. Rolling Statistics (Shifted by 63)

For windows `[7, 14, 28, 56, 91]`:

```python
for w in ROLLING_WINDOWS:
    shifted = group["total_enc"].shift(ROLLING_SHIFT)
    df[f"roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
    df[f"roll_std_{w}"]  = shifted.rolling(w, min_periods=1).std()
    df[f"roll_min_{w}"]  = shifted.rolling(w, min_periods=1).min()
    df[f"roll_max_{w}"]  = shifted.rolling(w, min_periods=1).max()
```

#### 5C. Trend Deltas

| Feature | Formula |
|---------|---------|
| `delta_7_28` | `roll_mean_7 - roll_mean_28` |
| `delta_28_91` | `roll_mean_28 - roll_mean_91` |
| `delta_lag_63_70` | `lag_63 - lag_70` |

#### 5D. Calendar & Cyclical

- `dow`, `day`, `week_of_year`, `month`, `quarter`, `day_of_year`, `is_weekend`, `is_halloween`
- `dow_sin`, `dow_cos`, `doy_sin`, `doy_cos`, `month_sin`, `month_cos`
- `days_since_epoch`, `year_frac`

#### 5E. Holiday & Event Features

- `is_us_holiday`, `days_since_xmas`, `days_until_thanksgiving`, `days_since_july4`, `days_to_nearest_holiday`
- `event_count`
- `school_in_session`, `days_since_school_start`, `days_until_school_start`

#### 5F. PIPELINE E UNIQUE — Predicted Factor Features

This is what differentiates Pipeline E from Pipeline A:

| Feature | Source | Notes |
|---------|--------|-------|
| `factor_0_pred` .. `factor_k_pred` | Step 4 output | Predicted factor values for each row |
| `factor_0_momentum` .. `factor_k_momentum` | `factor_pred - factor_pred_lag_7` | **REQUIRED** per §3.5 — the key signal |
| `factor_0_deviation_yearly` .. | `factor_pred - factor_lag_364` | Year-over-year composition deviation |
| `factor_0_lag_63` .. `factor_k_lag_63` | Actual factor at t-63 | Safe lagged actual factor (no prediction needed) |
| `factor_0_roll_mean_28` .. | `factor.shift(63).rolling(28).mean()` | Smoothed recent factor level |

Total factor-derived features: `N_FACTORS × ~5 variants = ~25 features`.

#### 5G. Interaction Features

```python
df["holiday_x_block"] = df["is_us_holiday"].astype(int) * df["block"]
df["weekend_x_block"] = df["is_weekend"].astype(int) * df["block"]
# Factor × site interactions (composition shifts may be site-specific)
for i in range(N_FACTORS):
    df[f"factor_{i}_x_site"] = df[f"factor_{i}_pred"] * df["site_encoded"]
```

#### 5H. Sample Weights

```python
df["covid_weight"] = np.where(df["is_covid_era"], COVID_SAMPLE_WEIGHT, 1.0)
df["volume_weight"] = df["total_enc"].clip(lower=1)
df["sample_weight_total"] = df["covid_weight"] * df["volume_weight"]
df["sample_weight_rate"] = df["covid_weight"] * df["admitted_enc"].clip(lower=1)
```

### Final Feature List

```python
FEATURE_COLS = [
    # Identifiers
    "site", "block",
    # Target lags (6)
    "lag_63", "lag_70", "lag_77", "lag_91", "lag_182", "lag_364",
    # Rolling stats (5 windows × 4 stats = 20)
    *[f"roll_{stat}_{w}" for w in [7,14,28,56,91] for stat in ["mean","std","min","max"]],
    # Trend deltas (3)
    "delta_7_28", "delta_28_91", "delta_lag_63_70",
    # Calendar (8)
    "dow", "day", "week_of_year", "month", "quarter", "day_of_year", "is_weekend", "is_halloween",
    # Cyclical (6)
    "dow_sin", "dow_cos", "doy_sin", "doy_cos", "month_sin", "month_cos",
    # Trend (2)
    "days_since_epoch", "year_frac",
    # Holidays (5)
    "is_us_holiday", "days_since_xmas", "days_until_thanksgiving", "days_since_july4", "days_to_nearest_holiday",
    # School (3)
    "school_in_session", "days_since_school_start", "days_until_school_start",
    # Events (1)
    "event_count",
    # COVID (1)
    "is_covid_era",
    # PIPELINE E UNIQUE — Factor features (~25-30)
    *[f"factor_{i}_pred" for i in range(N_FACTORS)],
    *[f"factor_{i}_momentum" for i in range(N_FACTORS)],
    *[f"factor_{i}_deviation_yearly" for i in range(N_FACTORS)],
    *[f"factor_{i}_lag_63" for i in range(N_FACTORS)],
    *[f"factor_{i}_roll_mean_28" for i in range(N_FACTORS)],
    # Interactions
    "holiday_x_block", "weekend_x_block",
    *[f"factor_{i}_x_site" for i in range(N_FACTORS)],
]
```

**Expected total**: ~90-110 features (vs. Pipeline A's ~60-80 — the delta is factor features).

### Eval Notes — Step 5
- [ ] **No future leakage**: For every feature column, verify it only uses data from ≥ 63 days ago (for target-based features) or ≥ 7 days ago (for factor-based features using safe-lag approach). Print min shift per feature.
- [ ] **NaN audit**: Print NaN count per column. Factor features may have more NaN near the start of history (due to PCA needing reason count history). Drop these rows from training (LightGBM can handle NaN, but too many NaN rows degrades quality).
- [ ] **Feature count**: Print total. Expected ~90-110.
- [ ] **Factor feature correlation with calendar**: Compute correlation between `factor_0_pred` and `month`. If |corr| > 0.9, the factor is just recoding seasonality (low incremental value — but momentum/deviation features may still add value).
- [ ] **Factor momentum distribution**: Print mean/std of `factor_i_momentum` features. Should be centered near 0 with non-trivial variance. If variance ≈ 0, momentum isn't varying enough to be useful.
- [ ] **Sample weight check**: COVID-era rows should have weight ≈ 0.1 × total_enc. Verify no zero weights.

---

## Step 6: Model Training (`step_06_train.py`)

### Purpose
Train two LightGBM models per fold:
- **Model E1**: `total_enc` (Tweedie objective)
- **Model E2**: `admit_rate` (Regression objective, bounded [0,1])

### Logic (Per Fold)

```
For each fold in FOLDS:
    1. FACTOR EXTRACTION (fold-specific):
       - Fit PCA/NMF on TRAINING data only (date <= fold.train_end)
       - Transform ALL rows (train + val)
       
    2. FACTOR FORECASTING (fold-specific):
       - Train factor forecast GBDTs on training data only
       - For training rows: use actual factor values
       - For validation rows: use predicted factor values from Step 4
       
    3. BUILD FEATURES:
       - Run Step 5 feature engineering
       - Compute fold-specific aggregate encodings on TRAIN ONLY:
         - site_month_block_mean, site_dow_mean, site_month_mean
         - Map onto both train and val via left join
       
    4. SPLIT:
       - train = rows where date <= fold.train_end (drop NaN lag rows)
       - val = rows where date in [fold.val_start, fold.val_end]
       
    5. TRAIN Model E1 (total_enc):
       - X = train[FEATURE_COLS], y = train["total_enc"]
       - sample_weight = train["sample_weight_total"]
       - lgb.Dataset with categorical_feature=[site, block]
       - Early stopping on val set (metric = "mae")
       
    6. TRAIN Model E2 (admit_rate):
       - X = train[FEATURE_COLS], y = train["admit_rate"]
       - sample_weight = train["sample_weight_rate"]
       
    7. PREDICT on val:
       - pred_total = model_e1.predict(val[FEATURE_COLS]).clip(0)
       - pred_rate = model_e2.predict(val[FEATURE_COLS]).clip(0, 1)
       - pred_admitted = (pred_total * pred_rate).round().astype(int)
       - pred_total = pred_total.round().astype(int)
       - Enforce: pred_admitted = min(pred_admitted, pred_total)
       
    8. SAVE fold predictions as submission CSV:
       → OUTPUT_DIR/fold_{id}_predictions.csv
       
    9. SAVE models:
       → MODEL_DIR/fold_{id}_model_e1.txt
       → MODEL_DIR/fold_{id}_model_e2.txt
       → MODEL_DIR/fold_{id}_factor_model.pkl
       → MODEL_DIR/fold_{id}_factor_forecast_models/
```

### Post-Processing: Largest-Remainder Rounding

```python
def largest_remainder_round(values):
    """Round array to integers preserving the sum."""
    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = int(round(values.sum())) - floored.sum()
    indices = np.argsort(-remainders)[:max(deficit, 0)]
    floored[indices] += 1
    return floored
```

Apply per `(site, date)` group for block-level consistency.

### Eval Notes — Step 6
- [ ] **Training set size**: Print rows per fold after NaN drops. Expected ~35K-40K (early folds smaller due to lag NaN).
- [ ] **Early stopping**: Verify early stopping triggers. If model uses all `n_estimators`, increase the budget.
- [ ] **Feature importance — THE KEY CHECK**: Print top 20 features by `gain`. Look for:
  - If factor features (factor_*_pred, factor_*_momentum) appear in top 20: **Pipeline E is adding value**.
  - If factor features are ALL outside top 30: factors aren't helping — Pipeline E may not beat Pipeline A.
  - If `factor_*_momentum` features rank higher than `factor_*_pred`: momentum is the key signal (confirms master strategy hypothesis).
- [ ] **Per-fold WAPE**: Print immediately:
  ```
  Fold 1: total_wape=X.XX, admitted_wape=X.XX
  Fold 2: total_wape=X.XX, admitted_wape=X.XX
  ...
  Mean:   total_wape=X.XX, admitted_wape=X.XX
  ```
- [ ] **Comparison to Pipeline A**: If Pipeline A's numbers are available, print the delta. Pipeline E should have WAPE within ±5% of Pipeline A. If much worse (>10% worse), something is broken.
- [ ] **Prediction range**:
  - `pred_total`: Non-negative integers. Print min/max/mean.
  - `pred_rate`: In [0, 1]. Print any pre-clip outliers.
  - `pred_admitted ≤ pred_total` for every row.
- [ ] **Residual analysis**: By-site and by-block residual means. Flag if any consistently over/under-predicts.
- [ ] **Constraint check**: `ED Enc Admitted <= ED Enc` for every output row.
- [ ] **Row count**: Each fold CSV has exactly `4 × num_val_days × 4` rows.

---

## Step 7: Hyperparameter Tuning (`step_07_tune.py`)

### Purpose
Use Optuna to optimize both the factor extraction and the final GBDT models.

### Two-Stage Tuning

#### Stage 1: Factor Extraction + Forecasting (30 trials)

```python
def objective_factors(trial):
    n_factors = trial.suggest_int("n_factors", 3, 7)
    method = trial.suggest_categorical("factor_method", ["pca", "nmf"])
    smooth_window = trial.suggest_int("smooth_window", 3, 14)
    
    # Factor forecast GBDT params
    factor_lr = trial.suggest_float("factor_lr", 0.01, 0.1, log=True)
    factor_depth = trial.suggest_int("factor_depth", 3, 6)
    
    # Run factor pipeline with these params across all folds
    # Metric: mean factor forecast MAE (as proxy for downstream value)
    factor_mae = evaluate_factor_forecast(n_factors, method, smooth_window, 
                                           factor_lr, factor_depth)
    return factor_mae
```

#### Stage 2: Final Model GBDT (100 trials, using best factor config)

```python
def objective_final(trial):
    params = {
        "objective": trial.suggest_categorical("objective", ["tweedie", "poisson"]),
        "n_estimators": trial.suggest_int("n_estimators", 800, 3000),
        "max_depth": trial.suggest_int("max_depth", 4, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.05, log=True),
        "subsample": trial.suggest_float("subsample", 0.7, 0.95),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 0.9),
        "reg_lambda": trial.suggest_float("reg_lambda", 1.0, 10.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
    }
    if params["objective"] == "tweedie":
        params["tweedie_variance_power"] = trial.suggest_float("tweedie_vp", 1.1, 1.9)
    
    covid_policy = trial.suggest_categorical("covid_policy", ["downweight", "exclude"])
    
    # Train on all 4 folds, compute mean admitted WAPE
    wapes = []
    for fold in FOLDS:
        model, preds = train_fold(fold, params, covid_policy)
        wapes.append(compute_admitted_wape(preds))
    return np.mean(wapes)
```

### Eval Notes — Step 7
- [ ] **Stage 1 convergence**: Plot factor MAE over trials. Should plateau by trial 20-25.
- [ ] **Best n_factors**: Report optimal k. If k=3 wins, the share matrix is simpler than expected.
- [ ] **PCA vs NMF**: Report which won and by how much. Document for future runs.
- [ ] **Stage 2 convergence**: Plot admitted WAPE over trials. Should plateau by trial 60-80.
- [ ] **Best vs Default**: Compare tuned WAPE against Step 6's default-param WAPE. Expected improvement: 2-5%.
- [ ] **COVID policy result**: Report downweight vs exclude winner.
- [ ] **Overfitting check**: Train WAPE vs val WAPE for best trial. If gap > 5×, model is overfitting.
- [ ] **Save best params**: `MODEL_DIR/best_params_total.json`, `best_params_rate.json`, `best_factor_config.json`.

---

## Step 8: Prediction & Post-Processing (`step_08_predict.py`)

### Purpose
Generate final submission-format CSVs using tuned models.

### Logic

```
For each fold (or final test period):
    1. Load best factor config + best GBDT params from Step 7
    2. Fit factor extraction on full training data
    3. Train factor forecast models on full training data
    4. Predict factors for validation/test window
    5. Compute factor momentum features
    6. Build full feature matrix
    7. Retrain final GBDT on full training data with best params
    8. Predict:
       pred_total = model_e1.predict(X).clip(0)
       pred_rate = model_e2.predict(X).clip(0, 1)
    9. Derive admitted:
       pred_admitted = pred_total * pred_rate
   10. Post-process:
       a. Clip negatives to 0
       b. Largest-remainder round per (site, date) for total
       c. Largest-remainder round per (site, date) for admitted
       d. Enforce admitted <= total row-by-row
   11. Format as submission CSV:
       Site, Date, Block, ED Enc, ED Enc Admitted
   12. Save to OUTPUT_DIR/
```

### Sept-Oct 2025 Final Forecast

```
Train on ALL data <= 2025-08-31 (applying COVID policy)
Predict 2025-09-01 to 2025-10-31
→ 4 sites × 61 days × 4 blocks = 976 rows
```

### Eval Notes — Step 8
- [ ] **Row count**: Each fold CSV has `4 × num_val_days × 4` rows. Sept-Oct: 976.
- [ ] **Schema**: Columns exactly `["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]`.
- [ ] **Constraints**:
  - All `ED Enc >= 0` ✓
  - All `ED Enc Admitted >= 0` ✓
  - All `ED Enc Admitted <= ED Enc` ✓
  - All values are integers ✓
- [ ] **No missing combos**: Full grid (all sites × all dates × all blocks).
- [ ] **No duplicates**: Unique on `(Site, Date, Block)`.
- [ ] **Distribution sanity**: Compare mean predictions to recent months' actuals. Flag if >20% deviation.
- [ ] **Site B share**: ~36% of total predicted volume (largest facility).
- [ ] **Block distribution**: Block 2-3 typically highest for ED. Check.
- [ ] **Largest-remainder**: Verify per-(Site, Date) block sums are consistent integers.

---

## Step 9: Evaluation (`step_09_evaluate.py`)

### Purpose
Run the `eval.md` evaluator on Pipeline E's fold CSVs. Produces the official pipeline score.

### Logic

```python
# 1. Load ground truth from raw data
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
```

### Output Report

```
═══════════════════════════════════════════════════════════════
 PIPELINE E: REASON-MIX LATENT FACTOR MODEL — EVALUATION REPORT
═══════════════════════════════════════════════════════════════

 OVERALL (4-fold mean):
   Primary Admitted WAPE:  X.XXXX
   Total WAPE:             X.XXXX
   Admitted RMSE:          X.XX
   Total RMSE:             X.XX

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

 PIPELINE E-SPECIFIC DIAGNOSTICS:
   Factor config: method=XXX, n_factors=X
   Factor forecast mean MAE: X.XXXX
   Factor features in top-20 importance: X/Y
   Factor momentum features in top-20: X/Y

═══════════════════════════════════════════════════════════════
```

### Eval Notes — Step 9
- [ ] **Primary metric**: Mean Admitted WAPE — THE ranking metric vs. other pipelines.
- [ ] **Sanity (§2.4)**: No single site WAPE > 2× best site WAPE.
- [ ] **Block stability**: Block 0 (overnight) shouldn't have WAPE >> others.
- [ ] **Per-fold variance**: If one fold dominates, investigate temporal drift.
- [ ] **Naive baseline**: Verify Pipeline E beats same-period-last-year.
- [ ] **Pipeline A comparison**: If available, compare Pipeline E WAPE to Pipeline A. Key questions:
  - Is Pipeline E within 5% of Pipeline A? (Good — adds diversity for ensemble)
  - Is Pipeline E >10% worse? (Factor features aren't helping — investigate or drop)
  - Is Pipeline E better than Pipeline A? (Factor momentum is adding real signal!)
- [ ] **Factor value attribution**: Run an ablation: train the same final GBDT *without* factor features. Compare WAPE. The difference = Pipeline E's marginal value from factors. If <0.5% improvement, factors are noise — drop Pipeline E.
- [ ] **Save report**: `OUTPUT_DIR/evaluation_report.json` for cross-pipeline comparison.
- [ ] **Save OOF predictions**: `OUTPUT_DIR/oof_predictions.csv` for ensemble stacking (§4.2 of master strategy).

---

## Step 10: End-to-End Orchestrator (`run_pipeline.py`)

### Purpose
Run the full Pipeline E from data loading through evaluation in one command.

### Usage

```bash
# Full pipeline with default params
python Pipelines/pipeline_E/run_pipeline.py

# Skip tuning (use defaults) — for quick iteration
python Pipelines/pipeline_E/run_pipeline.py --skip-tune

# Tune only (save best params, don't generate final predictions)
python Pipelines/pipeline_E/run_pipeline.py --tune-only

# Final Sept-Oct forecast
python Pipelines/pipeline_E/run_pipeline.py --final-forecast

# Ablation: run without factor features (to measure factor value)
python Pipelines/pipeline_E/run_pipeline.py --ablation-no-factors

# Single fold for debugging
python Pipelines/pipeline_E/run_pipeline.py --fold 1
```

### Pipeline Flow

```python
def main(args):
    print("=" * 65)
    print("PIPELINE E: REASON-MIX LATENT FACTOR MODEL — Starting")
    print("=" * 65)
    
    # Step 1: Data Loading
    print("\n[Step 1/9] Loading master dataset...")
    master_df = run_data_loading()
    validate_step_1(master_df)
    
    # Step 2: Build Share Matrix
    print("\n[Step 2/9] Building reason-category share matrix...")
    share_df = run_share_matrix(master_df)
    validate_step_2(share_df)
    
    # Step 3: Factor Extraction
    print("\n[Step 3/9] Extracting latent factors (PCA/NMF)...")
    factor_df = run_factor_extraction(share_df)
    validate_step_3(factor_df)
    
    # Step 4: Factor Forecasting
    print("\n[Step 4/9] Training factor forecast models...")
    factor_forecast_df = run_factor_forecasting(factor_df)
    validate_step_4(factor_forecast_df)
    
    # Step 5: Feature Engineering
    print("\n[Step 5/9] Building final feature matrix...")
    features_df = run_feature_engineering(master_df, factor_forecast_df)
    validate_step_5(features_df)
    
    if not args.tune_only:
        # Step 6: Train with default params
        print("\n[Step 6/9] Training final models (default hyperparameters)...")
        fold_results = run_training(features_df, params="default")
        validate_step_6(fold_results)
    
    if not args.skip_tune:
        # Step 7: Hyperparameter Tuning
        print("\n[Step 7/9] Tuning hyperparameters (Optuna)...")
        best_params = run_tuning(features_df)
        validate_step_7(best_params)
        
        # Re-train with best params
        print("\n[Step 6b/9] Re-training with tuned hyperparameters...")
        fold_results = run_training(features_df, params=best_params)
    
    # Step 8: Prediction & Post-Processing
    print("\n[Step 8/9] Generating submission CSVs...")
    run_prediction(fold_results)
    validate_step_8()
    
    # Step 9: Evaluation
    print("\n[Step 9/9] Evaluating against eval.md contract...")
    report = run_evaluation()
    print_report(report)
    
    if args.ablation_no_factors:
        print("\n[ABLATION] Running without factor features...")
        ablation_results = run_training(features_df, params=best_params, 
                                         exclude_factor_features=True)
        ablation_report = run_evaluation(ablation_results)
        print_ablation_comparison(report, ablation_report)
    
    if args.final_forecast:
        print("\n[FINAL] Generating Sept-Oct 2025 forecast...")
        run_final_forecast(features_df, best_params)
    
    print("\n" + "=" * 65)
    print("PIPELINE E: COMPLETE")
    print("=" * 65)
```

---

## Dependencies

```
lightgbm>=4.0
optuna>=3.0
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3       # PCA, NMF, LabelEncoder
statsmodels>=0.14       # AR models (if using AR factor forecasting)
holidays>=0.40          # US holiday calendar
pyarrow>=14.0           # Parquet I/O
joblib>=1.3             # Model serialization
```

---

## Execution Order & Time Estimates

| Step | File | Estimated Time | Can Skip? |
|------|------|---------------|-----------|
| 0 | `config.py` | — (static) | No |
| 1 | `step_01_data_loading.py` | ~30s | No |
| 2 | `step_02_share_matrix.py` | ~1 min | No |
| 3 | `step_03_factor_extraction.py` | ~1-2 min (PCA is fast; NMF ~2min) | No |
| 4 | `step_04_factor_forecasting.py` | ~5-10 min (N_FACTORS × 4 folds × train) | No |
| 5 | `step_05_feature_eng.py` | ~3-5 min | No |
| 6 | `step_06_train.py` | ~5-10 min (4 folds × 2 models) | No |
| 7 | `step_07_tune.py` | ~3-5 hours (130 trials: 30 factor + 100 final) | Yes (use defaults) |
| 8 | `step_08_predict.py` | ~1 min | No |
| 9 | `step_09_evaluate.py` | ~10s | No |

**Total (with tuning):** ~4-6 hours  
**Total (skip tuning):** ~20-30 minutes  
**Total (single fold, no tuning):** ~5-8 minutes

---

## Key Risk Mitigations

| Risk | Mitigation |
|------|------------|
| **Factor circularity** (factors = calendar proxy) | ONLY use AR/GBDT factor forecasting, NOT climatology. Ablation test (Step 9 eval notes) proves factor value. |
| **Lag leakage in final model** | All target lags ≥ 63; all rolling shifted by 63. Validated in Step 5 eval notes. |
| **Factor leakage** | PCA/NMF fit on training data ONLY per fold (Step 3.5). Factor lags ≥ 63 in safe-lag approach. |
| **Factor model overfitting** | Small GBDT for factor forecasting (max_depth=4, 500 trees). Early stopping. |
| **Noisy shares (small denominators)** | 7-day rolling smooth on shares (Step 2.3). |
| **COVID distortion of factors** | COVID downweighting on all models (factor forecast + final GBDT). COVID era shares will look different — downweighting prevents the factor model from over-indexing on them. |
| **Factor instability across folds** | Fold stability check in Step 3 eval notes. If unstable, increase smoothing or reduce k. |
| **Pipeline E doesn't beat Pipeline A** | Master strategy §3.5 says: "If that doesn't improve CV WAPE over Pipeline A alone, drop Pipeline E entirely." The ablation in Step 9 explicitly tests this. |
| **Too many features → overfitting** | Factor features add ~25 cols to Pipeline A's ~80. Total ~105 is still manageable for LightGBM. Monitor via train/val WAPE gap. |

---

## Output Artifacts

After a complete Pipeline E run:

```
Pipelines/pipeline_E/
  data/
    share_matrix.parquet                 # Step 2: reason share matrix
    factors.parquet                      # Step 3: factor-enriched data
    factor_loadings.csv                  # Step 3: PCA/NMF loadings
    predicted_factors.parquet            # Step 4: predicted factor values
    factor_forecast_eval.json            # Step 4: factor forecast accuracy
    features.parquet                     # Step 5: full feature matrix
  models/
    fold_1/
      factor_model.pkl                   # PCA/NMF fitted model
      factor_forecast_model_0.pkl        # Factor 0 GBDT forecast model
      factor_forecast_model_1.pkl        # Factor 1 ...
      ...
      model_e1.txt                       # Final LightGBM (total_enc)
      model_e2.txt                       # Final LightGBM (admit_rate)
    fold_2/ ...
    fold_3/ ...
    fold_4/ ...
    best_params_total.json               # Tuned hyperparameters (final model)
    best_params_rate.json
    best_factor_config.json              # Tuned factor config (n_factors, method, etc.)
  output/
    fold_1_predictions.csv               # Submission-format per fold
    fold_2_predictions.csv
    fold_3_predictions.csv
    fold_4_predictions.csv
    oof_predictions.csv                  # All folds combined (for stacking)
    final_sept_oct_2025.csv              # Final submission (976 rows)
    evaluation_report.json               # Full eval results
    feature_importance.csv               # Top features by gain
    ablation_no_factors.json             # Ablation: WAPE without factor features
  logs/
    pipeline_e_run_{timestamp}.log       # Full run log
```

---

## Site D Isolation Enhancement (master_strategy.md §11)

Site D has a WAPE of ~0.47 vs ~0.24 for A/B/C. Pipeline E's global GBDT (Step 6) trains on all sites, but Site D's gradient signal is drowned out by higher-volume sites. Additionally, Pipeline E's reason-mix factors are noisier for Site D because 52.5% of its visits fall in the "other" catch-all (vs ~44-50% for other sites). See §11.2 of master_strategy.md for the full diagnosis.

The fix: **Target Encoding** (fix baseline) + **Hybrid-Residual** (fix tail error).

### Enhancement A: Target Encoding Features (add to `step_05_feature_eng.py`)

Add alongside existing standard features (lags, rolling, calendar) in Step 5. Use the same lag/shift rules as Pipeline A (ROLLING_SHIFT = 63, matching Pipeline E's existing lag structure from §3.1).

#### 5.X Site-Level Target Encodings

```python
# Per (site, block) group, shifted by ROLLING_SHIFT (63 days):

# 1. Site baseline volume
for target_col in ["total_enc", "admitted_enc"]:
    shifted = group[target_col].shift(ROLLING_SHIFT)
    df[f"te_site_mean_{target_col}"] = shifted.rolling(90, min_periods=30).mean()

# 2. Site × Block baseline
df["te_site_block_mean_total"] = group["total_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).mean()
df["te_site_block_mean_admitted"] = group["admitted_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).mean()

# 3. Site admit rate (trailing 90-day, lagged)
shifted_total = group["total_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).sum()
shifted_admitted = group["admitted_enc"].shift(ROLLING_SHIFT).rolling(90, min_periods=30).sum()
df["te_site_admit_rate"] = shifted_admitted / shifted_total.clip(lower=1)

# 4. Site × DOW mean (trailing, lagged)
df["te_site_dow_mean"] = (
    df.groupby(["site", "dow"])["total_enc"]
    .transform(lambda g: g.shift(ROLLING_SHIFT).rolling(90, min_periods=30).mean())
)
```

**Pipeline E-specific note:** Target encoding is especially valuable here because Pipeline E adds ~30 factor features on top of the standard ~60. For Site D, many of these factor features are noisier (due to the 52.5% "other" share reducing factor quality). The target encoding gives the tree a reliable numeric baseline so it doesn't over-rely on noisy factor features for Site D.

**Feature count increase:** ~6–8 new features added to the existing ~90–110 feature set.

### Enhancement B: Hybrid-Residual Model for Site D (add to `step_06_train.py`)

After the final GBDT (E1 for total_enc, E2 for admit_rate) is trained:

```python
# --- Inside per-fold training, AFTER global model E1/E2 training ---

# 1. Compute OOF residuals for Site D
mask_d = X_train["site"] == "D"
global_pred_d = global_model.predict(X_train[mask_d])
residuals_d = y_train[mask_d] - global_pred_d

# 2. Residual feature set (SUBSET — exclude factor features and reason-mix)
#    Factor features are already captured by the global model — re-including
#    them in the residual model risks overfitting to factor noise on Site D
RESIDUAL_FEATURES = [
    "block", "dow", "month", "day_of_year",
    "te_site_block_mean_total", "te_site_block_mean_admitted",
    "te_site_admit_rate",
    "lag_63", "lag_91", "lag_182", "lag_364",
    "roll_mean_7", "roll_mean_28",
    "global_pred",
    "is_covid_era", "is_holiday", "is_weekend",
]

# 3. Heavily regularized LightGBM (same spec as Pipeline A)
residual_model = lgb.LGBMRegressor(
    objective="regression", n_estimators=300, num_leaves=15,
    max_depth=4, min_child_samples=50, learning_rate=0.03,
    reg_lambda=5.0, subsample=0.7, colsample_bytree=0.7, verbosity=-1,
)
residual_model.fit(X_train_d[RESIDUAL_FEATURES], residuals_d, ...)

# 4. Final Site D prediction
SHRINKAGE_WEIGHT = 0.5  # Tune via inner CV
pred_d = global_pred_d + SHRINKAGE_WEIGHT * residual_model.predict(X_d[RESIDUAL_FEATURES])
```

**Why factor features are EXCLUDED from the residual model:** Pipeline E's factors are derived from reason-mix shares. For Site D, 52.5% of visits are "other" — the factors are already noisy. The global model uses them as best it can. The residual model should correct *structural biases* (block-level, DOW-level) using the reliable target-encoded baselines, not try to squeeze more signal from noisy factors.

### Enhancement C: Zero-Inflation Post-Hoc + Admit-Rate Guardrails

Same as Pipeline A — apply to Site D admitted predictions after hybrid-residual:

```python
# Zero-inflation correction (Blocks 0, 1, 3)
# corrected = pred_admitted_d * (1 - p_zero * ZERO_SHRINKAGE)

# Admit-rate guardrails
ADMIT_RATE_BOUNDS_D = {0: (0.08, 0.30), 1: (0.10, 0.35), 2: (0.10, 0.30), 3: (0.08, 0.28)}
```

### Eval Notes — Site D Enhancements
- [ ] **Factor quality by site**: Print per-factor variance for Site D vs A/B/C — confirms D's factors are noisier
- [ ] **Ablation**: Compare (1) base Pipeline E, (2) + target encoding, (3) + residual model. Each should improve Site D WAPE without hurting A/B/C
- [ ] **Residual features**: Verify no factor features leak into the residual model
- [ ] **Shrinkage**: Tune independently from Pipeline A/B — Pipeline E may need different weight since its global model already has factor features

---

## Appendix A: Pipeline E vs Pipeline A — Architectural Delta

| Aspect | Pipeline A | Pipeline E (this) |
|--------|-----------|-------------------|
| Core idea | Pure demand-side GBDT with lags/calendar | Demand + composition signals via latent factors |
| Unique features | None (baseline feature set) | `factor_*_pred`, `factor_*_momentum`, `factor_*_deviation_yearly` |
| Feature count | ~60-80 | ~90-110 |
| Additional models | None | N_FACTORS factor forecast GBDTs + 1 PCA/NMF |
| Total models per fold | 2 (total + rate) | 2 + N_FACTORS + 1 = 8 (with k=5) |
| Value hypothesis | Lags + calendar capture demand patterns | Factor momentum captures composition shifts (early flu, trauma surges) |
| Expected WAPE delta vs A | — | 0-2% improvement (Low-Medium confidence, per §9) |
| Diversity for ensemble | Baseline | Orthogonal composition signal → ensemble gains |
| When to drop | Never (workhorse) | If ablation shows <0.5% WAPE improvement over A |

---

## Appendix B: Decision Tree — When Pipeline E Adds Value

```
Is factor forecast R² > 0.1 for at least 2 factors?
  ├─ NO  → Factors aren't predictable. DROP Pipeline E.
  │
  └─ YES → Do factor features appear in top-20 feature importance?
              ├─ NO  → GBDT isn't using factors. Check feature engineering for bugs.
              │         If no bugs, DROP Pipeline E.
              │
              └─ YES → Does Pipeline E WAPE beat Pipeline A by ≥ 0.5%?
                          ├─ NO  → Factors add diversity but not accuracy.
                          │         KEEP in ensemble (diversity value) but don't invest
                          │         more time tuning.
                          │
                          └─ YES → Pipeline E is adding real composition signal.
                                    Invest in:
                                    - Recursive factor prediction (v2, shorter lags)
                                    - More Optuna trials
                                    - Cross-site factor interactions
```

---

## Appendix C: Sept-Oct 2025 Forecast-Specific Notes

Per master strategy §6, the Sept-Oct window has distinct signals Pipeline E is uniquely positioned to capture:

| Signal | Timing | Pipeline E Relevance |
|--------|--------|---------------------|
| **Fall allergy season** | Sept peak | Factor capturing respiratory share should spike. If AR model sees this momentum, Pipeline E adds real signal. |
| **Early flu onset** (some years) | Late Oct | Factor momentum is the BEST hope for early detection. If flu starts 2-3 weeks early, factor_respiratory will trend up before calendar expects it. |
| **School return** | Late Aug → Sept | Composition shifts (more pediatric, more respiratory). Factors should capture this transition. |
| **Secular composition shift** | Ongoing | If ED case-mix is slowly changing year-over-year (e.g., more behavioral health), year-over-year factor deviation captures this. |

**Bottom line**: Pipeline E's highest-value scenario is when Sept-Oct 2025 has an unusual composition pattern (e.g., early flu). If the season is perfectly average, Pipeline E adds little beyond what calendar features already capture — but it also doesn't hurt (the ensemble can downweight it).
