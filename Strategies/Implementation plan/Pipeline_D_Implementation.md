# Pipeline D: GLM/GAM with Fourier Seasonality — Step-by-Step Implementation Plan

**Pipeline Role:** Low-variance regularizer — smooth parametric baseline that complements GBDT's flexibility.  
**Source:** `master_strategy.md` §3.4  
**Core Idea:** Poisson GLM with explicit Fourier seasonal decomposition. Provides smooth extrapolation for trends that GBDT cannot capture, and adds structural diversity to the ensemble.  
**Code Location:** `Pipelines/pipeline_D/`  
**Data Dependency:** `master_block_history.parquet` produced by the Data Source step (see `data_source.md`). STRICTLY REQUIRED.

---

## Why GLM/GAM Replaces the Neural Net (v1 Pipeline C)

- **16 series is too few** for embedding-based neural nets (Rossmann had 1,115 stores; we have 4 sites × 4 blocks = 16). NN would overfit or collapse to a mean.
- **True ensemble diversity**: Linear (GLM) vs. tree-based (GBDT) is the highest-leverage diversity axis for ensemble gains.
- **Trend extrapolation**: GBDT cannot extrapolate beyond training-data range. A GLM with a linear trend term handles secular ED volume growth natively.
- **Interpretability**: Coefficients are directly readable — useful for debugging and stakeholder trust.

---

## File Structure

```
Pipelines/pipeline_D/
├── config.py              # Constants, Fourier specs, regularization defaults, fold defs
├── data_loader.py         # Load master_block_history, derive admit_rate, COVID weights
├── features.py            # Build Fourier design matrix + calendar + holiday + trend
├── training.py            # Fit 16 Poisson GLMs (total_enc) + 16 Logistic/Beta models (admit_rate)
├── tuning.py              # Grid/Optuna search over Fourier orders, regularization strength
├── predict.py             # Forecast, constraint enforcement, largest-remainder rounding
├── evaluate.py            # 4-fold forward validation, WAPE scoring via eval.md contract
├── run_pipeline.py        # End-to-end orchestrator
└── __init__.py
```

---

## Step 0: Configuration (`config.py`)

### Purpose
Single source of truth for all constants, paths, Fourier specifications, and validation fold definitions.

### Key Definitions

```python
# --- Paths ---
MASTER_PARQUET_PATH = "Pipelines/Data Source/Data/master_block_history.parquet"
OUTPUT_DIR = "Pipelines/pipeline_D/output/"
MODEL_DIR = "Pipelines/pipeline_D/models/"

# --- Sites & Blocks ---
SITES = ["A", "B", "C", "D"]
BLOCKS = [0, 1, 2, 3]

# --- Validation Folds (from eval.md §2.1) ---
FOLDS = [
    {"id": 1, "train_end": "2024-12-31", "val_start": "2025-01-01", "val_end": "2025-02-28"},
    {"id": 2, "train_end": "2025-02-28", "val_start": "2025-03-01", "val_end": "2025-04-30"},
    {"id": 3, "train_end": "2025-04-30", "val_start": "2025-05-01", "val_end": "2025-06-30"},
    {"id": 4, "train_end": "2025-06-30", "val_start": "2025-07-01", "val_end": "2025-08-31"},
]

# --- COVID Era ---
COVID_START = "2020-03-01"
COVID_END   = "2021-06-30"
COVID_SAMPLE_WEIGHT = 0.1   # Downweight factor (Policy 3 from §3.0)

# --- Fourier Specification ---
# Master strategy §3.4: "sin/cos at periods 7, 365.25 (order 3 and 10 respectively)"
FOURIER_TERMS = [
    {"period": 7,      "order": 3},    # Weekly seasonality (6 features: sin/cos × 3 harmonics)
    {"period": 365.25, "order": 10},   # Annual seasonality (20 features: sin/cos × 10 harmonics)
]

# --- Model Configuration ---
# One model per (Site, Block) → 16 total_enc models + 16 admit_rate models = 32 models
GLM_FAMILY = "poisson"          # Poisson family with log link for count data
GLM_ALPHA = 0.1                 # L2 regularization strength (statsmodels alpha param)
ADMIT_MODEL_TYPE = "logistic"   # "logistic" or "beta" — logistic is simpler, beta is more flexible

# --- Tuning Search Space ---
FOURIER_ORDER_SEARCH = {
    "weekly_order":  [1, 2, 3, 4, 5],
    "annual_order":  [3, 5, 7, 10, 12, 15],
}
ALPHA_SEARCH = [0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]

# --- Post-processing ---
CLIP_MIN = 0
ADMIT_RATE_CLIP = (0.0, 1.0)
```

### Eval Check for Step 0
- [ ] Fold date ranges match `eval.md` exactly — cross-reference with Pipeline A/B configs.
- [ ] Fourier periods make physical sense: 7 = weekly cycle, 365.25 = annual cycle.
- [ ] Fourier order 3 for weekly → captures up to 3rd harmonic (Mon-Fri work pattern + weekend dip). Order 10 for annual → captures multi-modal seasonal peaks (flu, summer trauma, holidays).
- [ ] `GLM_ALPHA` default is non-zero — pure unregularized Poisson GLM can diverge on sparse blocks.
- [ ] Print full config on import for manual review.

---

## Step 1: Data Loading & Preprocessing (`data_loader.py`)

### Purpose
Load the unified `master_block_history.parquet` and prepare it for Pipeline D's design matrix construction. Pipeline D does NOT use lagged target features (unlike GBDT pipelines) — it relies entirely on deterministic time-based features. This is both its strength (no leakage risk from lags) and its limitation (no recent-history signal).

### Sub-Steps

#### 1.1 Load Master Dataset
- Read `MASTER_PARQUET_PATH`.
- Validate schema: assert required columns exist (`Site`, `Date`, `Block`, `total_enc`, `admitted_enc`).
- Sort by `(Site, Block, Date)` — this groups each series together for per-model fitting.
- Ensure `Date` is `datetime64`.

#### 1.2 Derive Admit Rate
```python
df["admit_rate"] = df["admitted_enc"] / df["total_enc"].replace(0, np.nan)
df["admit_rate"] = df["admit_rate"].fillna(0.0).clip(0, 1)
```
- Where `total_enc == 0`, admit_rate = 0.0 (no encounters → no admissions).
- Clip to `[0, 1]` as safety net.

#### 1.3 Apply COVID Sample Weights
```python
# Rely on 'is_covid_era' from Data Source (master_block_history.parquet)
# This ensures consistency with the definition in data_source.md
if "is_covid_era" not in df.columns:
    raise ValueError("Column 'is_covid_era' missing from input data. Check Data Source.")

df["sample_weight"] = np.where(df["is_covid_era"], COVID_SAMPLE_WEIGHT, 1.0)
```
- **Note**: `statsmodels` GLM accepts `freq_weights` or `var_weights`. For downweighting, use `var_weights` (inverse variance interpretation) or pass as `exposure` adjustment.
- **Implementation detail**: For Poisson GLM with `freq_weights`, weight=0.1 effectively says "this row counts as 0.1 observations." This is the cleanest approach.

#### 1.4 Provide Fold Splitter
```python
def get_fold_data(df, fold):
    """Split data into train and validation for a given fold."""
    train = df[df["Date"] <= fold["train_end"]].copy()
    val = df[(df["Date"] >= fold["val_start"]) & (df["Date"] <= fold["val_end"])].copy()
    return train, val

def get_site_block_subset(df, site, block):
    """Extract a single (Site, Block) series."""
    mask = (df["Site"] == site) & (df["Block"] == block)
    return df[mask].copy().sort_values("Date").reset_index(drop=True)
```

### Eval Check for Step 1
- [ ] **Row count**: `4 sites × 4 blocks × N_days`. For full history ~45,776 rows.
- [ ] **No NaN** in `total_enc`, `admitted_enc`.
- [ ] **`admit_rate`** is in `[0, 1]` for all rows — `assert (df["admit_rate"] >= 0).all() and (df["admit_rate"] <= 1).all()`.
- [ ] **COVID weights**: Count rows with `is_covid_era == True`. Should be ~16 × 488 days (Mar 2020 – Jun 2021). Print actual count.
- [ ] **Per-(Site, Block) series length**: All 16 series should have the same number of rows. Assert uniform length.
- [ ] **Print summary**: Per-site mean/std of `total_enc`. Site B should be largest (~36% volume).
- [ ] **Date range**: Min date = 2018-01-01, max date covers through at least 2025-08-31.

---

## Step 2: Feature Engineering — Design Matrix (`features.py`)

### Purpose
Build the deterministic design matrix for the Poisson GLM. Unlike GBDT pipelines, Pipeline D uses **no lagged target features** — only time-based, calendar, and Fourier features. This means:
- **Zero leakage risk** from lag computation.
- Features for future dates are fully known at forecast time (no need for recursive prediction or lag shifting).
- The model's power comes from capturing structured seasonality and trend, not recent history.

### Sub-Steps

#### 2.1 Fourier Terms (Core of Pipeline D)

Generate sin/cos pairs for each period and harmonic order:

```python
def make_fourier_features(dates, period, order):
    """Generate Fourier basis for a given period and order.
    
    Returns 2*order columns: sin_1, cos_1, sin_2, cos_2, ..., sin_order, cos_order
    """
    t = np.arange(len(dates))  # OR use day-of-year / days-since-epoch depending on period
    features = {}
    for k in range(1, order + 1):
        features[f"fourier_{period}_sin_{k}"] = np.sin(2 * np.pi * k * t / period)
        features[f"fourier_{period}_cos_{k}"] = np.cos(2 * np.pi * k * t / period)
    return pd.DataFrame(features, index=dates.index)
```

**Critical implementation detail — the time index `t`:**
- For **weekly** Fourier (period=7): use `day_of_week` (0-6) as `t`. This ensures Mondays always map to the same phase regardless of the date range.
  ```python
  t_weekly = df["Date"].dt.dayofweek  # 0=Mon, 6=Sun
  sin_k = np.sin(2 * pi * k * t_weekly / 7)
  ```
- For **annual** Fourier (period=365.25): use `day_of_year` (1-366) as `t`. This ensures Jan 1 always maps to the same phase.
  ```python
  t_annual = df["Date"].dt.dayofyear
  sin_k = np.sin(2 * pi * k * t_annual / 365.25)
  ```

**Feature count from Fourier**: `2 × 3` (weekly) + `2 × 10` (annual) = **26 features**.

#### 2.2 Day-of-Week Dummies

One-hot encode DOW with one category dropped (reference = Sunday or Monday):
```python
dow_dummies = pd.get_dummies(df["Date"].dt.dayofweek, prefix="dow", drop_first=True)
# → 6 features: dow_1, dow_2, ..., dow_6
```

**Why both DOW dummies AND weekly Fourier?**
- DOW dummies capture arbitrary day-specific effects (e.g., Monday ED surge from weekend-deferred patients).
- Weekly Fourier captures the smooth cyclical pattern.
- They're partially redundant, but L2 regularization will shrink the redundant coefficients. Keep both and let the regularizer decide.

#### 2.3 Linear Trend

```python
epoch = pd.Timestamp("2018-01-01")
df["trend"] = (df["Date"] - epoch).dt.days
```

- Captures secular ED volume growth (~2-4% annual per master strategy §6).
- In the Poisson GLM with log link, a positive trend coefficient means exponential growth: `E[y] = exp(β_trend × t + ...)`.
- **Consideration**: If growth is sublinear, add `trend_sqrt = np.sqrt(trend)` or a piecewise linear term. Test during tuning.

#### 2.4 Holiday Features

From the Data Source (`master_block_history.parquet`), we already have `is_holiday` (boolean) and `event_type` derived from `events.yaml`. We trust the Data Source for the binary flag but compute proximity features here.

```python
# Already present in Data Source output
# df["is_holiday"] (bool)

# Specific holiday flags (if not in Data Source, derive from date)
df["is_halloween"] = (df["Date"].dt.month == 10) & (df["Date"].dt.day == 31)

# Proximity indicators (continuous, derived here)
# Note: Ensure holiday dates match strictly with events.yaml logic
df["days_since_xmas"] = ...       # Days since most recent Dec 25
df["days_until_thanksgiving"] = ...  # Days until next Thanksgiving
df["days_since_july4"] = ...      # Days since most recent July 4
df["days_to_nearest_holiday"] = ...  # Min distance to any US holiday (using is_holiday=True dates)
```

#### 2.5 School Calendar Features

Data Source provides `school_in_session` (bool). Use it directly.
Calculate proximity if needed:
```python
df["days_since_school_start"] = ... # Continuous (0 if not in session)
```

#### 2.6 COVID Indicator

Data Source provides `is_covid_era` (bool). Use it directly.
```python
# Ensure it matches config if re-derived, but prefer Data Source column:
# df["is_covid_era"]
pass
```

- In the GLM, this acts as a level-shift indicator for the COVID period.
- For forecast horizon (Sept-Oct 2025), this is 0 → no COVID effect projected.
- Alternative: use a changepoint/step function that allows a permanent level shift post-COVID (test via tuning).

#### 2.7 Weather Features (Optional)

If weather data is available in the master parquet:
```python
weather_cols = ["temp_min", "temp_max", "precip", "snowfall"]
# Forward-fill then backward-fill NaN within each site
# For forecast dates: use climatology (monthly mean)
df["temp_range"] = df["temp_max"] - df["temp_min"]
```

If weather is not available, skip — Pipeline D works fine without it. Calendar + Fourier already capture most weather-correlated seasonality.

#### 2.8 Assemble Final Design Matrix

```python
def build_design_matrix(df, fourier_config=FOURIER_TERMS):
    """Build the full design matrix X for the GLM.
    
    Returns: X (DataFrame), feature_names (list)
    """
    X = pd.DataFrame(index=df.index)
    
    # 2.1 Fourier terms
    for spec in fourier_config:
        X = pd.concat([X, make_fourier_features(df, spec["period"], spec["order"])], axis=1)
    
    # 2.2 DOW dummies
    X = pd.concat([X, pd.get_dummies(df["Date"].dt.dayofweek, prefix="dow", drop_first=True)], axis=1)
    
    # 2.3 Trend
    X["trend"] = (df["Date"] - pd.Timestamp("2018-01-01")).dt.days
    
    # 2.4 Holiday features
    X["is_holiday"] = df["is_holiday"].astype(int)  # From Data Source
    X["is_halloween"] = df["is_halloween"].astype(int)
    X["days_since_xmas"] = df["days_since_xmas"]
    X["days_until_thanksgiving"] = df["days_until_thanksgiving"]
    X["days_since_july4"] = df["days_since_july4"]
    X["days_to_nearest_holiday"] = df["days_to_nearest_holiday"]
    
    # 2.5 School calendar
    X["school_in_session"] = df["school_in_session"].astype(int) # From Data Source
    X["days_since_school_start"] = df["days_since_school_start"]
    
    # 2.6 COVID indicator
    X["is_covid_era"] = df["is_covid_era"].astype(int) # From Data Source
    
    # 2.7 Weather (if available)
    for col in ["temp_min", "temp_max", "precip", "snowfall", "temp_range"]:
        if col in df.columns:
            X[col] = df[col]
    
    # Add intercept (statsmodels GLM needs explicit constant)
    X.insert(0, "const", 1.0)
    
    return X
```

**Expected total feature count**: ~45-55 columns (1 intercept + 26 Fourier + 6 DOW + 1 trend + ~6 holiday + 2 school + 1 COVID + 0-5 weather).

### Eval Check for Step 2
- [ ] **Fourier phase alignment**: Verify that `fourier_7_sin_1` has period exactly 7 days by checking values repeat every 7 rows. Plot one month of weekly Fourier terms — should show clean weekly oscillation.
- [ ] **Fourier annual alignment**: Verify that `fourier_365_sin_1` peaks at roughly the same day-of-year each year. Plot across full history.
- [ ] **No target leakage**: The entire design matrix is deterministic (computable from the date alone + static external data). No lagged targets, no rolling stats of y. Assert no column correlates with future `total_enc`.
- [ ] **No NaN in design matrix**: Fourier, DOW, trend are fully deterministic — should have zero NaN. Weather/holiday features may have NaN if data is incomplete — handle before fitting.
- [ ] **Multicollinearity check**: Compute VIF (Variance Inflation Factor) or correlation matrix for the design matrix. DOW dummies + weekly Fourier will be correlated — this is expected and handled by regularization. Flag if VIF > 20 for any feature.
- [ ] **Feature count**: Print total columns. Should match expected ~45-55.
- [ ] **Spot-check**: Pick a known holiday (e.g., 2024-12-25) and verify `is_us_holiday=1`, `days_since_xmas=0`.
- [ ] **Trend range**: Print min/max of trend column. Min should be 0 (2018-01-01), max should correspond to the last date in the dataset.

---

## Step 3: Model Training (`training.py`)

### Purpose
Fit 16 Poisson GLM models (one per Site × Block) for `total_enc`, and 16 logistic/beta regression models for `admit_rate`. Total: 32 models per fold.

### Why Per-(Site, Block) Models?

Unlike Pipeline A/B (global GBDT across all series), Pipeline D fits **separate models** per series because:
- GLMs have far fewer parameters (~50) than GBDTs (~thousands of leaves). Per-series fitting is computationally trivial.
- Each (Site, Block) has distinct seasonal amplitude, trend slope, and holiday sensitivity. Separate coefficients capture this naturally.
- A global GLM would require interaction terms for every feature × site × block → explosion of parameters that defeats the "low-variance" purpose.

### Sub-Steps

#### 3.1 Model D1: Poisson GLM for `total_enc`

For each `(site, block)` pair:

```python
import statsmodels.api as sm

def train_total_model(X_train, y_train, weights, alpha=GLM_ALPHA):
    """Fit a regularized Poisson GLM for total_enc.
    
    Args:
        X_train: Design matrix (from features.py)
        y_train: total_enc series
        weights: Sample weights (COVID downweighted)
        alpha: L2 regularization strength
    
    Returns:
        Fitted GLM results object
    """
    # Poisson family with log link
    family = sm.families.Poisson(link=sm.families.links.Log())
    
    model = sm.GLM(
        endog=y_train,
        exog=X_train,
        family=family,
        freq_weights=weights,
    )
    
    # Fit with L2 regularization (Ridge-like)
    # statsmodels GLM: fit_regularized(alpha=..., L1_wt=0.0) → pure L2
    result = model.fit_regularized(
        alpha=alpha,
        L1_wt=0.0,    # 0.0 = pure L2 (Ridge); 1.0 = pure L1 (Lasso)
        maxiter=200,
    )
    
    return result
```

**Key modeling decisions:**
- **Poisson + log link**: Natural choice for count data. `E[total_enc] = exp(X @ β)` guarantees non-negative predictions.
- **L2 regularization**: Shrinks coefficients toward zero, preventing overfitting on the ~50 features. Crucial because high-order Fourier terms can capture noise rather than signal.
- **`freq_weights`**: Treats COVID-era rows as fractional observations (0.1 "counts" each). This is the cleanest way to implement downweighting in a GLM context.

**Alternative: pyGAM (if non-linear effects needed)**
```python
from pygam import PoissonGAM, s, f, l

# Spline-based smooth terms for trend and weather
gam = PoissonGAM(
    s(feature_idx_trend, n_splines=10) +    # Smooth trend (non-linear growth)
    l(feature_idx_fourier_terms) +           # Linear Fourier terms
    f(feature_idx_dow) +                     # Factor DOW
    l(feature_idx_holidays)                  # Linear holidays
)
gam.fit(X_train, y_train, weights=weights)
```
- Use pyGAM as an **ablation** — try if the pure linear GLM underperforms. The spline on trend allows non-linear growth rates (e.g., faster growth 2023+ due to facility expansion).

#### 3.2 Model D2: Admit Rate Model

For each `(site, block)` pair:

**Option A — Logistic Regression (PREFERRED for simplicity):**
```python
from sklearn.linear_model import LogisticRegression

# Transform admit_rate to binary events for logistic regression
# OR use it as a continuous target with a quasi-binomial GLM

def train_rate_model_logistic(X_train, y_train_rate, weights, total_enc_train, alpha=GLM_ALPHA):
    """Fit admit_rate using a Binomial GLM (logit link).
    
    We model the number of successes (admitted) given trials (total_enc).
    """
    family = sm.families.Binomial(link=sm.families.links.Logit())
    
    # Proper statsmodels Binomial setup for proportions:
    # endog = rate, var_weights = number of trials
    # freq_weights = sample weight (for COVID downweighting)
    
    # Note: older statsmodels versions might require endog=[success, failure]
    
    model = sm.GLM(
        endog=y_train_rate,
        exog=X_train,
        family=family,
        var_weights=total_enc_train,  # Number of trials
        freq_weights=weights,         # Sample importance/frequency
    )
    result = model.fit_regularized(alpha=alpha, L1_wt=0.0, maxiter=200)
    return result
```

**Option B — Beta Regression (more flexible for proportions):**
```python
# If admit_rate has heavy tails or bimodality, Beta regression is better
# Requires betareg (R-like) or manual implementation via statsmodels
# For simplicity, start with quasi-binomial GLM above — switch to Beta only if residuals show issues
```

**Option C — Simple OLS on logit-transformed rate (Fallback):**
```python
from scipy.special import logit, expit

y_logit = logit(y_train_rate.clip(0.01, 0.99))  # Avoid log(0)
# Fit OLS on logit(rate), back-transform predictions with expit()
```

**Recommendation**: Start with Option A (quasi-binomial GLM). It naturally bounds predictions to [0,1] via the logit link, handles the proportion nature of the target, and is consistent with the Poisson GLM framework.

#### 3.3 Training Loop (Per Fold)

```python
def train_all_models(train_df, fourier_config, alpha):
    """Train 32 models: 16 total_enc + 16 admit_rate.
    
    Returns: dict keyed by (site, block) → {"total_model": ..., "rate_model": ...}
    """
    models = {}
    
    for site in SITES:
        for block in BLOCKS:
            # Extract series
            series = get_site_block_subset(train_df, site, block)
            
            # Build design matrix
            X = build_design_matrix(series, fourier_config)
            y_total = series["total_enc"]
            y_rate = series["admit_rate"]
            weights = series["sample_weight"]
            
            # Drop rows with NaN in X or y (if any)
            valid_mask = X.notna().all(axis=1) & y_total.notna()
            X_clean = X[valid_mask]
            y_total_clean = y_total[valid_mask]
            y_rate_clean = y_rate[valid_mask]
            w_clean = weights[valid_mask]
            
            # Train total_enc model
            total_model = train_total_model(X_clean, y_total_clean, w_clean, alpha)
            
            # Train admit_rate model
            # Pass total_enc (y_total_clean) as var_weights
            rate_model = train_rate_model_logistic(X_clean, y_rate_clean, w_clean, y_total_clean, alpha)
            
            models[(site, block)] = {
                "total_model": total_model,
                "rate_model": rate_model,
            }
            
            print(f"  [{site}, Block {block}] total_enc: "
                  f"deviance={total_model.deviance:.1f}, "
                  f"n_obs={len(X_clean)}, "
                  f"n_params={len(total_model.params)}")
    
    return models
```

#### 3.4 Model Serialization

- Save fitted model objects as `.pkl` per (site, block, target).
- Save coefficient tables as `.csv` for interpretability.
- Save model summary text (from `result.summary()`) as `.txt`.

```
Pipelines/pipeline_D/models/fold_{k}/
├── total_model_A_0.pkl      # Poisson GLM for Site A, Block 0
├── rate_model_A_0.pkl       # Binomial GLM for Site A, Block 0
├── coefficients_A_0.csv     # β coefficients with std errors
├── ...                      # (repeat for all 16 site-block combos)
└── model_summary.txt        # Concatenated summaries
```

### Eval Check for Step 3
- [ ] **Convergence**: All 32 models should converge (`result.converged == True`). If any fail to converge, print warning with (site, block) and try: (a) increase `maxiter`, (b) increase `alpha` (more regularization), (c) remove high-VIF features.
- [ ] **Deviance check**: For each model, compute `deviance / df_resid` (deviance per residual degree of freedom). For a well-fitted Poisson: this ratio ≈ 1.0. If >> 1 (overdispersion), consider switching to Negative Binomial family. If << 1 (underdispersion), Poisson is fine.
  ```
  Expected: deviance/df_resid in [0.5, 3.0] for most models
  Action if > 5.0: Switch to NegativeBinomial(alpha=...) family
  ```
- [ ] **Coefficient sanity**:
  - `trend` coefficient should be **positive** (ED volumes growing over time). If negative for any (site, block), investigate — possible COVID distortion or data issue.
  - Weekly Fourier: `fourier_7_sin_1` / `fourier_7_cos_1` should have non-trivial magnitude (weekly pattern exists). If coefficients ≈ 0 for all weekly harmonics, the model isn't capturing the weekly cycle.
  - Annual Fourier: Higher-order harmonics (order 8-10) should have smaller coefficients than lower-order ones. If high-order terms dominate, they're fitting noise → increase regularization.
  - `is_us_holiday` should be **negative** (holidays typically have lower ED volumes except trauma-related). Check sign makes clinical sense.
- [ ] **Fitted values**: For each model, compute in-sample `ŷ = model.predict(X_train)`. Print mean(ŷ) vs mean(y). Should be close (within 5%).
- [ ] **Residual distribution**: Plot Pearson residuals for 2-3 representative models. Should be roughly symmetric around 0 with no time-dependent pattern. Heavy tails → consider NegBin.
- [ ] **Per-(Site, Block) training size**: Each model trains on `~N_days` rows (e.g., 2,557 days for fold 1). Print actual. Minimum viable is ~365 days (one full year of seasonality). We have ~7 years — plenty.
- [ ] **Weight impact**: Compare model trained with COVID downweighting vs. without on one (site, block). The trend coefficient should be less biased downward with downweighting.

---

## Step 4: Hyperparameter Tuning (`tuning.py`)

### Purpose
Select optimal Fourier orders and regularization strength via cross-validated WAPE. Pipeline D has far fewer hyperparameters than GBDT pipelines, so tuning is fast.

### Hyperparameters to Tune

| Parameter | Search Space | What It Controls |
|-----------|-------------|-----------------|
| `weekly_order` | `[1, 2, 3, 4, 5]` | Number of weekly harmonics (higher = more flexible weekly pattern) |
| `annual_order` | `[3, 5, 7, 10, 12, 15]` | Number of annual harmonics (higher = sharper seasonal peaks) |
| `alpha` | `[0.001, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0]` | L2 regularization strength (higher = smoother, more constrained) |
| `covid_policy` | `["downweight", "exclude", "indicator"]` | How to handle COVID era |
| `trend_type` | `["linear", "sqrt", "piecewise"]` | Shape of the trend component |
| `family` | `["poisson", "negativebinomial"]` | GLM family (Poisson vs. NegBin for overdispersion) |

**Total grid size**: 5 × 6 × 7 × 3 × 3 × 2 = **3,780 combinations** (too large for exhaustive grid).

### Tuning Strategy

**Two-phase approach:**

**Phase 1 — Coarse grid on key parameters (Fourier order + alpha):**
- Fix covid_policy = "downweight", trend_type = "linear", family = "poisson"
- Grid search: `weekly_order × annual_order × alpha` = 5 × 6 × 7 = **210 combinations**
- For each combination, train all 16 models on fold 1 only (fast screening)
- Select top 10 by admitted WAPE

**Phase 2 — Full CV on top candidates:**
- Take top 10 from Phase 1 + vary covid_policy, trend_type, family
- Run full 4-fold CV for each candidate → ~30-50 evaluations
- Select config with best mean admitted WAPE across 4 folds

```python
def tune_pipeline_d(master_df):
    """Two-phase tuning for Pipeline D."""
    
    # Phase 1: Coarse grid on fold 1
    print("Phase 1: Coarse grid search (fold 1 only)...")
    results = []
    for w_order in FOURIER_ORDER_SEARCH["weekly_order"]:
        for a_order in FOURIER_ORDER_SEARCH["annual_order"]:
            for alpha in ALPHA_SEARCH:
                fourier_config = [
                    {"period": 7, "order": w_order},
                    {"period": 365.25, "order": a_order},
                ]
                wape = evaluate_single_fold(master_df, FOLDS[0], fourier_config, alpha)
                results.append({
                    "weekly_order": w_order, "annual_order": a_order,
                    "alpha": alpha, "fold1_wape": wape
                })
    
    # Select top 10
    top_10 = sorted(results, key=lambda x: x["fold1_wape"])[:10]
    
    # Phase 2: Full 4-fold CV on top candidates
    print("Phase 2: Full 4-fold CV on top 10 candidates...")
    best_config = None
    best_wape = float("inf")
    for cfg in top_10:
        mean_wape = evaluate_all_folds(master_df, cfg)
        if mean_wape < best_wape:
            best_wape = mean_wape
            best_config = cfg
    
    # Also test ablations on the winner
    for covid_pol in ["downweight", "exclude", "indicator"]:
        for trend in ["linear", "sqrt"]:
            for fam in ["poisson", "negativebinomial"]:
                cfg_variant = {**best_config, "covid_policy": covid_pol,
                               "trend_type": trend, "family": fam}
                wape = evaluate_all_folds(master_df, cfg_variant)
                if wape < best_wape:
                    best_wape = wape
                    best_config = cfg_variant
    
    return best_config
```

### Why This Is Fast

- Each (site, block) GLM fits in <1 second (50 parameters, ~2,500 rows).
- 16 models × 1 second = ~16 seconds per fold.
- Phase 1: 210 combos × 16 sec = ~56 minutes.
- Phase 2: ~50 combos × 4 folds × 16 sec = ~53 minutes.
- **Total tuning time: ~2 hours** (vs. 4+ hours for GBDT pipelines).

### Eval Check for Step 4
- [ ] **Fourier order selection**: Print winning `weekly_order` and `annual_order`. If max order wins (e.g., `annual_order=15`), the model may be overfitting annual noise → inspect whether extending the search space further improves or consider enforcing a cap.
- [ ] **Regularization**: Print winning `alpha`. If `alpha` = smallest value (0.001), the model wants minimal regularization → data is clean. If `alpha` = largest (5.0), the model is heavily constrained → possible underfitting or too many features.
- [ ] **COVID policy**: Report winning policy. Expected: "downweight" (per master strategy recommendation).
- [ ] **Family**: If NegativeBinomial wins, it confirms overdispersion in the data. Report the dispersion parameter.
- [ ] **Top-10 spread**: If the top 10 candidates all have similar WAPE (within 1%), the model is robust to hyperparameter choices → good sign. If large spread → more tuning may help.
- [ ] **Phase 1 vs Phase 2 rank stability**: Check if the Phase 1 winner (fold 1 only) is still top-3 after full 4-fold CV. If not, Phase 1 was a poor proxy → consider using 2 folds for screening.
- [ ] **Overfitting check**: Compare in-sample deviance vs. out-of-fold WAPE. If in-sample is much better, increase `alpha`.
- [ ] **Save**: Best config to `MODEL_DIR/best_config.json`.

---

## Step 5: Prediction & Post-Processing (`predict.py`)

### Purpose
Generate forecasts for validation windows or the final Sept-Oct 2025 period. Enforce all hard constraints from `eval.md`.

### Sub-Steps

#### 5.1 Build Forecast Design Matrix
Since Pipeline D uses only deterministic features, building the forecast matrix is trivial:
```python
def predict_window(master_df, models, forecast_dates, fourier_config):
    """Generate predictions for all (site, block) on forecast dates.
    
    Args:
        master_df: DataFrame containing full history (including future 2025 grid from Data Source)
        models: dict of fitted models keyed by (site, block)
        forecast_dates: DatetimeIndex for the forecast period
        fourier_config: Fourier specs (from tuning)
    
    Returns:
        DataFrame with columns: Site, Date, Block, ED Enc, ED Enc Admitted
    """
    rows = []
    
    for site in SITES:
        for block in BLOCKS:
            # Build design matrix for forecast dates
            # CRITICAL: Use the future grid from Data Source (master_block_history.parquet)
            # which contains the full grid for 2025 with pre-calculated events/calendar.
            
            # 1. Load subset of master grid for this (site, block) and filter to forecast window
            full_series = get_site_block_subset(master_df, site, block)
            forecast_df = full_series[full_series["Date"].isin(forecast_dates)].copy()
            
            if forecast_df.empty:
                raise ValueError(f"No data found in master_df for {site} Block {block} on forecast dates!")
            
            # 2. Enrich only with calculated features (Fourier, trend, proximity)
            # Holiday proximity can be re-calculated or loaded if saved in master.
            # Assuming basic enrichment is deterministic:
            # forecast_df = enrich_proximity_features(forecast_df)
            
            # 3. Weather Imputation
            # Data Source leaves future weather as NaN. If using weather cols, fill with climatology here.
            for col in ["temp_min", "temp_max"]:
                if col in forecast_df.columns and forecast_df[col].isna().any():
                     forecast_df[col] = forecast_df[col].fillna(
                        master_df.groupby("month")[col].transform("mean")
                     )

            X_forecast = build_design_matrix(forecast_df, fourier_config)
            
            # Predict total_enc
            total_model = models[(site, block)]["total_model"]
            pred_total = total_model.predict(X_forecast)
            
            # Predict admit_rate
            rate_model = models[(site, block)]["rate_model"]
            pred_rate = rate_model.predict(X_forecast)
            
            # Derive admitted
            pred_admitted = pred_total * pred_rate
            
            # Store results
            for i, idx in enumerate(forecast_df.index):
                date = forecast_df.loc[idx, "Date"]
                rows.append({
                    "Site": site,
                    "Date": date.strftime("%Y-%m-%d"),
                    "Block": block,
                    "pred_total": pred_total.iloc[i] if hasattr(pred_total, 'iloc') else pred_total[i],
                    "pred_rate": pred_rate.iloc[i] if hasattr(pred_rate, 'iloc') else pred_rate[i],
                    "pred_admitted": pred_admitted.iloc[i] if hasattr(pred_admitted, 'iloc') else pred_admitted[i],
                })
    
    return pd.DataFrame(rows)
```

#### 5.2 Enforce Hard Constraints

```python
def post_process(pred_df):
    """Apply all hard constraints from eval.md."""
    
    # 1. Non-negativity (Poisson GLM with log link already guarantees this,
    #    but clip as safety net for numerical edge cases)
    pred_df["pred_total"] = pred_df["pred_total"].clip(lower=0)
    pred_df["pred_admitted"] = pred_df["pred_admitted"].clip(lower=0)
    
    # 2. Admitted <= Total (already by construction since rate ∈ [0,1],
    #    but re-enforce after rounding)
    
    # 3. Integer rounding via largest-remainder per (Site, Date)
    pred_df["ED Enc"] = 0
    pred_df["ED Enc Admitted"] = 0
    
    for (site, date), group in pred_df.groupby(["Site", "Date"]):
        idx = group.index
        
        # Round total_enc
        total_rounded = largest_remainder_round(group["pred_total"].values)
        pred_df.loc[idx, "ED Enc"] = total_rounded
        
        # Round admitted_enc
        admitted_rounded = largest_remainder_round(group["pred_admitted"].values)
        pred_df.loc[idx, "ED Enc Admitted"] = admitted_rounded
    
    # 4. Final admitted <= total enforcement (rounding can violate this)
    pred_df["ED Enc Admitted"] = np.minimum(
        pred_df["ED Enc Admitted"], pred_df["ED Enc"]
    )
    
    # 5. Ensure integer types
    pred_df["ED Enc"] = pred_df["ED Enc"].astype(int)
    pred_df["ED Enc Admitted"] = pred_df["ED Enc Admitted"].astype(int)
    
    # 6. Ensure strict date format (YYYY-MM-DD) per eval.md
    pred_df["Date"] = pd.to_datetime(pred_df["Date"]).dt.strftime("%Y-%m-%d")

    return pred_df[["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]]


def largest_remainder_round(values):
    """Round array to integers preserving the sum."""
    floored = np.floor(values).astype(int)
    remainders = values - floored
    deficit = int(round(values.sum())) - floored.sum()
    if deficit > 0:
        indices = np.argsort(-remainders)[:deficit]
        floored[indices] += 1
    elif deficit < 0:
        indices = np.argsort(remainders)[:abs(deficit)]
        floored[indices] -= 1
    return floored
```

#### 5.3 Weather Handling for Forecast Dates

For validation folds, actual weather is available (it's in the past). For the final Sept-Oct 2025 forecast:
- **If weather features are in the model**: Use climatology (monthly averages from training data) as proxy.
- **If weather features are NOT in the model**: No action needed — this is one of Pipeline D's advantages: fully deterministic features.

### Eval Check for Step 5
- [ ] **Prediction range**: Print mean/std/min/max of `pred_total` and `pred_admitted` per site. Compare against historical averages.
  - Expected: `pred_total` per (site, block, day) should be in the range of historical values. Site B ~25-40 per block-day, Site D ~5-15.
  - **Red flag**: If any prediction is > 3× historical max or < 0.3× historical min, the Fourier extrapolation may be diverging.
- [ ] **Constraint satisfaction**:
  - `assert (pred_df["ED Enc"] >= 0).all()`
  - `assert (pred_df["ED Enc Admitted"] >= 0).all()`
  - `assert (pred_df["ED Enc Admitted"] <= pred_df["ED Enc"]).all()`
  - `assert pred_df["ED Enc"].dtype == int` (or int64)
- [ ] **Row count**: `4 sites × N_days × 4 blocks`. For Sept-Oct 2025: 4 × 61 × 4 = **976 rows**. For validation folds: varies (59-62 days × 16).
- [ ] **Full grid coverage**: Assert every (site, date, block) combination is present — no missing rows.
- [ ] **No duplicates**: Assert `pred_df.duplicated(subset=["Site", "Date", "Block"]).sum() == 0`.
- [ ] **Block distribution**: Check that the 4 blocks' share of total volume roughly matches historical patterns. If one block has 0 predictions for many days, the model for that (site, block) may have collapsed.
- [ ] **Largest-remainder audit**: For 5 random (site, date) groups, verify `sum(ED Enc) == round(sum(pred_total))` — the rounding preserves the daily total.
- [ ] **Schema check**: Columns are exactly `["Site", "Date", "Block", "ED Enc", "ED Enc Admitted"]`.
- [ ] **Date format check**: Verify that `Date` column contains strings in "YYYY-MM-DD" format, not timestamps.

---

## Step 6: Evaluation (`evaluate.py`)

### Purpose
Run the full 4-fold forward validation and produce WAPE scores per the eval.md contract.

### Sub-Steps

#### 6.1 Fold Loop

For each fold in the 4-period validation:

| Fold | Train ≤ | Validate |
|------|---------|----------|
| 1 | 2024-12-31 | 2025-01-01 – 2025-02-28 |
| 2 | 2025-02-28 | 2025-03-01 – 2025-04-30 |
| 3 | 2025-04-30 | 2025-05-01 – 2025-06-30 |
| 4 | 2025-06-30 | 2025-07-01 – 2025-08-31 |

```python
def run_cv_evaluation(master_df, best_config):
    """Full 4-fold cross-validation for Pipeline D."""
    
    all_oof = []
    fold_metrics = []
    
    for fold in FOLDS:
        print(f"\n--- Fold {fold['id']} ---")
        
        # 1. Split
        train_df, val_df = get_fold_data(master_df, fold)
        
        # 2. Train 32 models
        models = train_all_models(train_df, best_config["fourier_config"], best_config["alpha"])
        
        # 3. Predict on validation window
        val_dates = pd.date_range(fold["val_start"], fold["val_end"])
        pred_df = predict_window(models, val_dates, best_config["fourier_config"])
        pred_df = post_process(pred_df)
        
        # 4. Score
        metrics = score_fold(pred_df, val_df, fold)
        fold_metrics.append(metrics)
        
        # 5. Save fold predictions
        pred_df.to_csv(f"{OUTPUT_DIR}/fold_{fold['id']}_predictions.csv", index=False)
        
        # 6. Collect OOF predictions for ensemble stacking
        all_oof.append(pred_df)
    
    # Aggregate
    mean_admitted_wape = np.mean([m["primary_admitted_wape"] for m in fold_metrics])
    
    # Save OOF predictions (for ensemble stacking in §4.2)
    oof_df = pd.concat(all_oof, ignore_index=True)
    oof_df.to_csv(f"{OUTPUT_DIR}/oof_predictions.csv", index=False)
    
    return fold_metrics, mean_admitted_wape
```

#### 6.2 Metrics Collection

Per fold, collect:
- `primary_admitted_wape` — THE ranking metric
- `total_wape`, `admitted_wape`
- `total_rmse`, `admitted_rmse`
- By-site WAPE breakdown
- By-block WAPE breakdown

#### 6.3 WAPE Computation

```python
def compute_wape(actual, predicted):
    """Weighted Absolute Percentage Error.
    
    WAPE = sum(|actual - predicted|) / sum(actual)
    """
    return np.sum(np.abs(actual - predicted)) / np.sum(actual)
```

#### 6.4 Diagnostic Report

```
═══════════════════════════════════════════════════════════
 PIPELINE D: GLM/GAM WITH FOURIER — EVALUATION REPORT
═══════════════════════════════════════════════════════════

 CONFIGURATION:
   Weekly Fourier order:  3
   Annual Fourier order:  10
   Regularization (α):   0.1
   COVID policy:         downweight (0.1×)
   GLM family:           Poisson (log link)
   Admit model:          Quasi-Binomial (logit link)

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

### Eval Check for Step 6
- [ ] **Primary metric**: Mean Admitted WAPE — this is Pipeline D's final score for cross-pipeline comparison.
- [ ] **Sanity check (§2.4)**: No single site WAPE > 2× best site WAPE.
- [ ] **Block stability**: Block-level WAPE should be roughly similar. If Block 0 (overnight) has WAPE >> others, the model can't capture the overnight pattern (fewer Fourier harmonics may be needed for low-volume blocks, or consider a separate model for Block 0).
- [ ] **Per-fold variance**: Compute `std(fold_wapes) / mean(fold_wapes)`. If > 0.3, investigate which fold is the outlier and why (e.g., fold 4 = summer may have different patterns).
- [ ] **Naive baseline comparison**: Compute same-period-last-year WAPE as a floor. Pipeline D MUST beat this. If it doesn't, something is fundamentally broken.
- [ ] **Compare to Pipeline A/B**: If Pipeline D WAPE > 1.5× Pipeline A WAPE, it's a dropout candidate (per master strategy §8). Document but don't eliminate yet — its ensemble contribution depends on diversity, not raw WAPE.
- [ ] **OOF file**: `oof_predictions.csv` has correct shape (sum of all 4 folds' validation rows). This file feeds into the ensemble stacking meta-learner (§4.2 of master strategy).
- [ ] **Save report**: Write to `OUTPUT_DIR/evaluation_report.json`.

---

## Step 7: Orchestrator (`run_pipeline.py`)

### Purpose
Single entry point that runs the entire Pipeline D end-to-end.

### Usage

```bash
# Full pipeline: data → features → tune → train → predict → evaluate
python Pipelines/pipeline_D/run_pipeline.py --mode cv

# Skip tuning (use default config) — for quick iteration
python Pipelines/pipeline_D/run_pipeline.py --mode cv --skip-tune

# Final Sept-Oct 2025 submission
python Pipelines/pipeline_D/run_pipeline.py --mode submit

# Single fold for debugging
python Pipelines/pipeline_D/run_pipeline.py --mode fold --fold-id 1

# Tune only (no final predictions)
python Pipelines/pipeline_D/run_pipeline.py --mode tune
```

### Pipeline Flow

```
1. Load config (Step 0)
2. Load & preprocess data (Step 1)
   → CHECKPOINT: validate data shape, print summary
3. Build design matrix infrastructure (Step 2)
   → CHECKPOINT: feature count, NaN audit, multicollinearity check
4. [Optional] Tune hyperparameters (Step 4)
   → CHECKPOINT: best config, tuning convergence
5. Train 32 models per fold (Step 3)
   → CHECKPOINT: convergence, coefficient sanity, deviance check
6. Predict on validation/test window (Step 5)
   → CHECKPOINT: constraint checks, distribution comparison
7. Evaluate (Step 6)
   → CHECKPOINT: WAPE scores, sanity thresholds
8. Save all artifacts
```

### Execution Logic

```python
import argparse
import json
import time
from config import *
from data_loader import load_master_data, get_fold_data
from features import build_design_matrix
from training import train_all_models
from tuning import tune_pipeline_d
from predict import predict_window, post_process
from evaluate import run_cv_evaluation, score_fold


def main():
    parser = argparse.ArgumentParser(description="Pipeline D: GLM/GAM with Fourier")
    parser.add_argument("--mode", choices=["cv", "submit", "fold", "tune"],
                        default="cv", help="Execution mode")
    parser.add_argument("--fold-id", type=int, default=None, help="Single fold ID (1-4)")
    parser.add_argument("--skip-tune", action="store_true", help="Use default config")
    args = parser.parse_args()
    
    print("=" * 60)
    print("PIPELINE D: GLM/GAM WITH FOURIER — Starting")
    print("=" * 60)
    
    t0 = time.time()
    
    # Step 1: Load data
    print("\n[Step 1] Loading master data...")
    master_df = load_master_data()
    print(f"  Loaded {len(master_df)} rows, date range: "
          f"{master_df['Date'].min()} to {master_df['Date'].max()}")
    
    # Step 4 (run early): Tune hyperparameters
    if args.mode == "tune" or (args.mode in ["cv", "submit"] and not args.skip_tune):
        print("\n[Step 4] Tuning hyperparameters...")
        best_config = tune_pipeline_d(master_df)
        with open(f"{MODEL_DIR}/best_config.json", "w") as f:
            json.dump(best_config, f, indent=2)
        print(f"  Best config: {best_config}")
        if args.mode == "tune":
            return
    else:
        best_config = load_default_config()
    
    if args.mode == "cv":
        # Full 4-fold cross-validation
        fold_metrics, mean_wape = run_cv_evaluation(master_df, best_config)
        print(f"\n{'='*60}")
        print(f"PIPELINE D — Mean Admitted WAPE: {mean_wape:.4f}")
        print(f"{'='*60}")
    
    elif args.mode == "fold":
        # Single fold
        fold = FOLDS[args.fold_id - 1]
        train_df, val_df = get_fold_data(master_df, fold)
        models = train_all_models(train_df, best_config)
        # ... predict & evaluate single fold
    
    elif args.mode == "submit":
        # Final submission: train on ALL data through Aug 2025
        print("\n[FINAL] Training on full history, forecasting Sept-Oct 2025...")
        train_df = master_df[master_df["Date"] <= "2025-08-31"]
        models = train_all_models(train_df, best_config)
        forecast_dates = pd.date_range("2025-09-01", "2025-10-31")
        pred_df = predict_window(models, forecast_dates, best_config)
        pred_df = post_process(pred_df)
        assert len(pred_df) == 976, f"Expected 976 rows, got {len(pred_df)}"
        pred_df.to_csv(f"{OUTPUT_DIR}/final_sept_oct_2025.csv", index=False)
        print(f"  Saved 976-row submission to {OUTPUT_DIR}/final_sept_oct_2025.csv")
    
    elapsed = time.time() - t0
    print(f"\nPipeline D completed in {elapsed/60:.1f} minutes.")


if __name__ == "__main__":
    main()
```

### Output Artifacts

```
Pipelines/pipeline_D/
├── output/
│   ├── fold_1_predictions.csv          # Submission-shaped per fold
│   ├── fold_2_predictions.csv
│   ├── fold_3_predictions.csv
│   ├── fold_4_predictions.csv
│   ├── oof_predictions.csv             # All folds combined (for ensemble stacking)
│   ├── final_sept_oct_2025.csv         # Final competition submission (976 rows)
│   └── evaluation_report.json          # Full eval results
├── models/
│   ├── best_config.json                # Tuned hyperparameters
│   ├── fold_1/
│   │   ├── total_model_A_0.pkl         # 16 Poisson GLM models
│   │   ├── rate_model_A_0.pkl          # 16 Binomial GLM models
│   │   ├── coefficients_A_0.csv        # Coefficient tables
│   │   └── ...                         # (all 16 site-block combos)
│   ├── fold_2/
│   ├── fold_3/
│   └── fold_4/
├── diagnostics/
│   ├── coefficient_summary.csv         # All coefficients across all models
│   ├── deviance_table.csv              # Deviance/df_resid per model
│   ├── residual_plots/                 # Per-(site, block) residual diagnostics
│   └── fourier_fit_plots/              # Fitted Fourier curves overlaid on actuals
└── logs/
    └── pipeline_d_run_{timestamp}.log
```

### Eval Check for Step 7
- [ ] **End-to-end smoke test**: Run `--mode fold --fold-id 1 --skip-tune` to verify full pipeline executes without errors. Should complete in < 2 minutes.
- [ ] **Reproducibility**: No random seeds needed (GLM fitting is deterministic for a given design matrix). Verify identical outputs on re-run.
- [ ] **Runtime**: Full CV with tuning should complete in **< 3 hours**. Without tuning: **< 10 minutes**.
- [ ] **Artifact completeness**: After full run, verify all expected output files exist and are non-empty.
- [ ] **Log quality**: Check that the log captures per-step timings, model convergence status, and WAPE scores.

---

## Dependencies

```
statsmodels>=0.14        # Core: GLM, Poisson, Binomial families, regularized fitting
pandas>=2.0
numpy>=1.24
scikit-learn>=1.3        # LabelEncoder, train_test_split utilities (minimal use)
holidays>=0.40           # US holiday calendar
pyarrow>=14.0            # Parquet I/O
scipy>=1.11              # For beta regression fallback, optimization utilities
# Optional:
pygam>=0.9               # GAM alternative (spline-based smoothing for trend)
```

---

## Execution Order & Time Estimates

| Step | File | Estimated Time | Can Skip? |
|------|------|---------------|-----------|
| 0 | `config.py` | — (static) | No |
| 1 | `data_loader.py` | ~10s | No |
| 2 | `features.py` | ~30s (deterministic features, very fast) | No |
| 3 | `training.py` | ~2 min (32 models × ~1s each × 4 folds) | No |
| 4 | `tuning.py` | ~1-2 hours (210 grid + 50 ablation) | Yes (use defaults) |
| 5 | `predict.py` | ~30s | No |
| 6 | `evaluate.py` | ~10s | No |

**Total (with tuning):** ~2-3 hours  
**Total (skip tuning):** ~5-10 minutes

---

## Key Differences from GBDT Pipelines (A/B)

| Aspect | Pipeline A/B (GBDT) | Pipeline D (GLM/GAM) |
|--------|---------------------|---------------------|
| **Model type** | Non-parametric tree ensemble | Parametric linear model |
| **Features** | Lagged targets + rolling stats + calendar | Fourier + calendar + trend (NO lagged targets) |
| **Leakage risk** | High (lags must be shifted carefully) | **Zero** (all features deterministic) |
| **Trend handling** | Cannot extrapolate (explicit trend feature helps but is approximation) | **Native** (linear trend in log-space = exponential growth) |
| **Number of models** | 2 (global across all series) | 32 (per site × block × target) |
| **Parameters per model** | ~10K+ leaf nodes | ~50 coefficients |
| **Training speed** | Minutes per fold | Seconds per fold |
| **Expected standalone WAPE** | Best (primary workhorse) | Moderate (low-variance, not best standalone) |
| **Ensemble value** | Baseline | **High** (maximally diverse from tree-based methods) |
| **Interpretability** | Low (feature importance only) | **High** (readable coefficients with clinical meaning) |

---

## Risk Mitigations

| Risk | Mitigation |
|------|------------|
| **Overdispersion** (Poisson variance < data variance) | Monitor `deviance/df_resid` ratio. Switch to NegativeBinomial family if > 3.0 (tested during tuning). |
| **Fourier overfitting** (high-order terms capture noise) | L2 regularization shrinks high-order Fourier coefficients. Tune `alpha` and Fourier order jointly. |
| **Trend extrapolation divergence** | For long forecast horizons, the exponential trend can overshoot. Cap predictions at `1.5 × max(historical daily total)` as safety rail. |
| **COVID distortion of Fourier phases** | COVID downweighting (0.1× weight) reduces influence on seasonal coefficient estimates. Test COVID exclusion as ablation. |
| **Low-volume (site, block) instability** | Small blocks (e.g., Site D, Block 0: ~2-5 encounters/day) have high Poisson variance. Regularization (`alpha`) stabilizes estimates. Also: consider fitting a shared model for low-volume blocks with site-specific intercepts (partial pooling). |
| **Holiday feature leakage** | Holiday features are deterministic (derived from the date, not the target). No leakage possible. |
| **GLM non-convergence** | Increase `maxiter` to 500. If still fails, increase `alpha` (more regularization). If persistent, check for perfect separation or near-zero-variance features — drop them. |
| **Weather data missing for forecast** | Weather is optional for Pipeline D. If included, use climatology for forecast dates. If excluded entirely, Pipeline D still works (Fourier captures most weather-correlated seasonality). |

---

## Appendix A: pyGAM Alternative Implementation

If pure GLM underperforms due to non-linear trend or weather effects, use `pyGAM` as a drop-in replacement:

```python
from pygam import PoissonGAM, s, f, l, te

def build_gam_model(X, y, weights):
    """Build a PoissonGAM with smooth trend + linear Fourier + factor DOW."""
    
    # Identify feature indices
    trend_idx = feature_names.index("trend")
    fourier_idx = [i for i, n in enumerate(feature_names) if n.startswith("fourier_")]
    dow_idx = [i for i, n in enumerate(feature_names) if n.startswith("dow_")]
    holiday_idx = [i for i, n in enumerate(feature_names) if "holiday" in n or "school" in n]
    
    # Build GAM formula
    # s() = spline smooth, l() = linear, f() = factor
    terms = (
        s(trend_idx, n_splines=15, spline_order=3) +   # Non-linear trend
        sum(l(i) for i in fourier_idx) +                 # Linear Fourier terms
        sum(f(i) for i in dow_idx) +                     # Factor DOW dummies
        sum(l(i) for i in holiday_idx)                   # Linear holiday effects
    )
    
    gam = PoissonGAM(terms)
    gam.fit(X, y, weights=weights)
    return gam
```

**When to use pyGAM over statsmodels GLM:**
- The trend is clearly non-linear (e.g., faster growth post-2023)
- Weather features have non-linear effects (e.g., U-shaped: extreme heat AND extreme cold increase ED visits)
- The pure GLM residuals show systematic time-dependent patterns

**When to stick with statsmodels GLM:**
- Simpler, faster, fewer dependencies
- L2 regularization is better supported
- Coefficient interpretation is cleaner
- Default choice; only switch if ablation shows pyGAM gains > 1% WAPE

---

## Appendix B: Negative Binomial Upgrade Path

If the Poisson model shows overdispersion (`deviance/df_resid > 3`):

```python
# statsmodels supports NegativeBinomial via NegativeBinomial family
family = sm.families.NegativeBinomial(alpha=1.0)  # alpha = dispersion param

# Or use the dedicated class:
model = sm.NegativeBinomial(y_train, X_train, loglike_method="nb2")
result = model.fit(maxiter=200)
```

**Key difference**: NegBin adds one extra parameter (dispersion `α`) that allows `Var[y] = μ + α×μ²` instead of Poisson's strict `Var[y] = μ`. This handles the overdispersion common in ED encounter counts where some days have random surges.

---

## Site D Isolation Enhancement (master_strategy.md §11)

Site D has a WAPE of ~0.47 across GBDT pipelines and **0.94 on Pipeline D** — near-random. The per-series GLM approach (16 separate models) is especially fragile for Site D because:
- Site D Block 0 averages **1.24 admitted/block** with 29% zeros — the Poisson GLM fits a very low λ and can't model zero-inflation
- Site D's 16 series have the least data AND the highest noise
- Separate models can't borrow the seasonal/holiday patterns that are shared across all sites

Pipeline D is the **natural home for formal partial pooling** — unlike GBDTs (which fake it via categorical splits), GLMs can implement genuine mixed-effects models with principled shrinkage.

### Enhancement A: Mixed-Effects GLM (Replace Per-Series with Pooled)

**Replace the 16 separate Poisson GLMs** with a single mixed-effects model that shares fixed effects (Fourier, DOW, holidays, trend) across all series while allowing site×block-specific intercepts that are shrunk toward the population mean.

#### Architecture Change

**Before (current):** 16 independent GLMs, each fit on ~2,800 rows:
```
For each (site, block):
    log(y) ~ Fourier(day_of_year) + DOW + is_holiday + trend + weather
```

**After (upgraded):** 1 mixed-effects GLM fit on all ~44,800 rows:
```
log(y) ~ Fourier(day_of_year) + DOW + is_holiday + trend + weather   [FIXED effects — shared]
          + (1 | site)                                                 [RANDOM intercept per site]
          + (1 | site:block)                                           [RANDOM intercept per site×block]
```

#### Why Mixed-Effects Fixes Site D

The random effects `(1 | site)` and `(1 | site:block)` implement **empirical Bayes shrinkage**: each site×block intercept is a weighted average of:
- Its own data (local MLE)
- The population mean (global MLE)

The weight depends on the **effective sample size and variance** of each group. Low-volume, high-variance groups (Site D Block 0) get pulled **strongly** toward the population mean. High-volume, low-variance groups (Site B Block 2) retain their local estimate. This is exactly the behavior we need — Site D borrows the seasonal shape from the full population while keeping its own volume level.

#### Implementation (`training.py`)

```python
import statsmodels.formula.api as smf
import statsmodels.api as sm
import numpy as np

def train_mixed_effects_model(train_df, target_col="total_enc"):
    """
    Fit a single mixed-effects Poisson GLM across all sites and blocks.
    
    Fixed effects: Fourier terms, DOW dummies, holiday indicators, trend, weather
    Random effects: (1 | site) + (1 | site_block)
    """
    
    # 1. Prepare the data
    df = train_df.copy()
    df["site_block"] = df["site"].astype(str) + "_" + df["block"].astype(str)  # 16 groups
    df["log_offset"] = 0  # No offset for count model; could use log(population) if available
    
    # 2. Build formula
    #    Fixed effects: all Fourier terms + DOW + holidays + trend + weather
    fourier_cols = [c for c in df.columns if c.startswith("fourier_")]
    dow_cols = [c for c in df.columns if c.startswith("dow_")]
    
    fixed_formula_parts = fourier_cols + dow_cols + [
        "is_holiday",
        "is_halloween",
        "school_in_session",
        "trend",  # days_since_epoch, normalized
        "temp_min", "temp_max", "precip", "snowfall",
    ]
    fixed_formula = " + ".join(fixed_formula_parts)
    formula = f"{target_col} ~ {fixed_formula}"
    
    # 3. Fit mixed-effects model with site_block as random effect
    #    Using MixedLM (Gaussian approximation for Poisson via log-transformed target)
    #    Alternative: Use Poisson GEE with exchangeable correlation
    
    # Option A: Log-transformed Gaussian mixed model (simpler, faster)
    df[f"log_{target_col}"] = np.log1p(df[target_col])  # log(y + 1)
    
    model = smf.mixedlm(
        formula=f"log_{target_col} ~ {fixed_formula}",
        data=df,
        groups=df["site"],           # Primary grouping: site
        re_formula="1",              # Random intercept per site
        vc_formula={                  # Variance component: site×block
            "site_block": "0 + C(site_block)"
        },
    )
    
    result = model.fit(
        method="lbfgs",
        maxiter=500,
        reml=True,  # REML estimation (better for small group counts)
    )
    
    # Option B: Poisson GLMM via PQL (if statsmodels supports it)
    # Or use pymer4 for the full R-style lme4 syntax:
    #
    # from pymer4.models import Lmer
    # model = Lmer(f"{target_col} ~ {fixed_formula} + (1|site) + (1|site_block)",
    #              data=df, family="poisson")
    # result = model.fit()
    
    return result

def predict_mixed_effects(model_result, pred_df):
    """Generate predictions from the mixed-effects model."""
    df = pred_df.copy()
    df["site_block"] = df["site"].astype(str) + "_" + df["block"].astype(str)
    
    # Predict on log scale, then expm1
    log_pred = model_result.predict(df)
    pred = np.expm1(log_pred)  # Back to count scale
    
    return pred.clip(lower=0)
```

#### Shrinkage Behavior (What to Expect)

| Site×Block | Volume | Expected Shrinkage | Behavior |
|-----------|--------|-------------------|----------|
| Site B Block 2 | ~54/block | Minimal — retains local estimate | Population adds little signal |
| Site A Block 2 | ~49/block | Minimal | Similar to B |
| Site C Block 0 | ~10/block | Moderate — ~30% toward population | Borrows some seasonal shape |
| **Site D Block 0** | **~7/block** | **Strong — ~50-70% toward population** | Gets population's seasonal/DOW shape, keeps D's intercept |
| **Site D Block 1** | **~17/block** | **Moderate — ~20-40% toward population** | Some borrowing |

The exact shrinkage percentages depend on the estimated variance components (σ²_site, σ²_site_block, σ²_residual) — REML estimates these automatically.

#### Configuration Changes (`config.py`)

```python
# --- REPLACE per-series model config with mixed-effects ---

# Old: 16 separate models
# GLM_FAMILY = "poisson"
# GLM_ALPHA = 0.1

# New: Single mixed-effects model
MODEL_TYPE = "mixed_effects"    # "per_series" for fallback to old behavior
MIXED_EFFECTS_METHOD = "lbfgs"  # Optimization method
MIXED_EFFECTS_REML = True       # REML vs ML estimation
MIXED_EFFECTS_MAXITER = 500

# Admit rate model: Same structure, but with logistic/beta family
ADMIT_MODEL_TYPE = "mixed_effects_logistic"  # Or keep per-series if stable enough
```

#### Eval Notes — Enhancement A
- [ ] **Convergence**: Model must converge (check `result.converged`). If not, try `method="powell"` or increase `maxiter`
- [ ] **Random effects**: Print estimated random effects per site×block. Site D Block 0 should have a negative intercept (lower than population mean). Verify the magnitude is reasonable (~log(7.5) ≈ 2.0 on log scale vs population ~log(25) ≈ 3.2)
- [ ] **Shrinkage verification**: Compare mixed-effects predictions vs per-series predictions for Site D. Mixed-effects should be closer to population mean (higher for Block 0, potentially lower for peak blocks)
- [ ] **Site D WAPE**: Must improve from the catastrophic 0.94 baseline. Target: < 0.60
- [ ] **A/B/C stability**: WAPE for other sites should not degrade > 0.02
- [ ] **Compare Options A/B**: Run both log-Gaussian mixed model and Poisson GEE; pick whichever has lower CV WAPE

### Enhancement B: Admit-Rate Mixed-Effects

The same mixed-effects approach applies to the admit-rate model:

```python
# Logit-transformed admit_rate with mixed effects
df["logit_admit_rate"] = np.log(df["admit_rate"].clip(0.01, 0.99) / 
                                 (1 - df["admit_rate"].clip(0.01, 0.99)))

model = smf.mixedlm(
    formula=f"logit_admit_rate ~ {fixed_formula}",
    data=df,
    groups=df["site"],
    re_formula="1",
    vc_formula={"site_block": "0 + C(site_block)"},
)
```

This is critical because Site D's 17.4% admit rate is 14pp below A/C (~31%). The per-series logistic model gets this right but has high variance. The mixed-effects model shrinks Site D's admit-rate intercept toward the population (~27%), which may slightly bias it upward but dramatically reduces variance.

### Enhancement C: Zero-Inflation Correction (Post-Hoc)

After the mixed-effects model produces predictions for Site D:

```python
# Same zero-inflation correction as GBDT pipelines
# corrected = pred_admitted_d * (1 - p_zero * ZERO_SHRINKAGE)
# Applied to Site D Blocks 0, 1, 3 only
```

### Fallback: Per-Series Models

If the mixed-effects model fails to converge or shows worse CV WAPE than per-series:
1. Revert to 16 separate GLMs
2. Apply **post-hoc shrinkage** for Site D only: `pred_D = α × pred_per_series_D + (1-α) × pred_population_D` with α tuned via CV
3. This is a manual approximation of what the mixed-effects model does automatically

---

## Appendix C: Per-Series GLM Partial Pooling (Deprecated — see Enhancement A above)

The original recommendation was to start with per-series models and add partial pooling only if WAPE > 2× the best. Given Pipeline D's catastrophic 0.94 WAPE on Site D, the mixed-effects approach in Enhancement A is now the **default**. Per-series models are retained as a fallback only.
