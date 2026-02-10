# DSU-2026: Sanford Health ED Volume Forecasting — Strategy Pack (v2)

## Dataset + output contract (from `README.md` + repo scripts)

- **Raw data grain**: visit counts by (`Site`, `Date`, `Hour`, `REASON_VISIT_NAME`)
- **Sites**: `A,B,C,D`
- **Date range**: `2018-01-01` → `2025-08-31` (train history). Forecast horizon is **Sep–Oct 2025**.
- **Targets**:
  - `ED Enc` (total encounters)
  - `ED Enc Admitted` (subset admitted; included in total)
- **Forecast grain (required output)**:
  - Aggregate hours into **6-hour blocks**:
    - `Block=0`: hours 0–5
    - `Block=1`: hours 6–11
    - `Block=2`: hours 12–17
    - `Block=3`: hours 18–23
  - Predict **every** (`Site`, `Date`, `Block`) for `2025-09-01..2025-10-31`.

### Submission / output file format (repo truth)

The repo’s prediction script writes:

- Columns: **`Site, Date, Block, ED Enc, ED Enc Admitted`**
- `Date` formatted as `YYYY-MM-DD`
- Values are **non-negative integers**

Expected row count:
\[
4\ \text{sites} \times 61\ \text{days} \times 4\ \text{blocks} = 976\ \text{rows}
\]

### Hard constraints worth enforcing

- **Non-negativity**: \( \hat y \ge 0 \)
- **Admitted ≤ Total**: \( \widehat{\text{ED Enc Admitted}} \le \widehat{\text{ED Enc}} \)
- **Optional reconciliation constraint** (strongly recommended):
  - Block totals should sum to a coherent daily total per site
  - Same for admitted

### “Reason for visit” usage constraint

- You **must not** forecast by `REASON_VISIT_NAME` as an output dimension.
- You *can* still use it as a **feature**, e.g. via **case-mix shares** (this repo already does that).

## Scoring + validation (what will actually move the needle)

### Likely primary metric

The baseline pipeline artifacts indicate `primary_metric = avg_admitted_enc_wape`.

- **WAPE**: \(\text{WAPE} = \frac{\sum |y - \hat y|}{\sum |y|}\)
- Implication: the metric **weights high-volume rows more** (Site B / busy blocks matter disproportionately).

### Validation that matches the competition (recommended)

The `README.md` describes 4 two-month “forward” validation periods in 2025:

- Train ≤ Dec 2024 → validate Jan–Feb 2025
- Train ≤ Feb 2025 → validate Mar–Apr 2025
- Train ≤ Apr 2025 → validate May–Jun 2025
- Train ≤ Jun 2025 → validate Jul–Aug 2025

This is better aligned than random CV, and it stress-tests:
- **season drift** (winter vs summer)
- **multi-step horizon behavior** (full 2 months)
- **covariate availability rules** (external features can’t leak)

### Target modeling trick (used in repo; keep it)

Model **(total_enc)** and **(admit_rate)**, then derive:

- \(\widehat{adm} = \min(\widehat{total},\ \widehat{total}\cdot \widehat{rate})\)

Why it works:
- Forces **admitted ≤ total**
- Turns a harder count problem into a smoother bounded rate + a count

## Feature table + external data (what you already have)

The repo already supports a strong “winning-style” feature store via `scripts/build_feature_table.py`:

- **Calendar** (`src/dsu_forecast/features/calendar.py`):
  - DOW / DOY sin/cos, month, weekend, US holiday, month start/end, etc.
- **Case-mix shares** (`src/dsu_forecast/features/case_mix.py`):
  - Creates `cmix_*_share` from `REASON_VISIT_NAME` categories
  - For future dates, fills shares using **climatology by (month, day, block)** (no leakage)
- **External covariates** (`src/dsu_forecast/features/external_covariates.py`):
  - **Events** from `config/events.yaml` → daily `event_intensity`, `event_count`
  - **Weather** (Open-Meteo) aggregated to blocks; **future filled by climatology** (no leakage)
  - **NWS alerts** (daily features); also supports optional **LLM-derived** daily alert features (if present)

### External data policy (important for “winning strategies”)

For fair backtesting and “competition-like” behavior:
- Use only information that would have been known at prediction time.
- This repo already implements a safe approach: **fetch historical up to `train_end`, fill future with climatology**.

If you *actually* want to maximize Sept–Oct 2025 accuracy in 2026 and you’re allowed to use realized future weather/alerts, you can do an “oracle covariates” variant — but that’s **leakage** relative to the intended competition setup. I outline it as an optional pipeline variant, not the default.

---

## Distilled “winning team” patterns (mapped to this dataset)

These are the recurring high-signal tricks from demand/volume forecasting competitions (M5 / Store Sales / similar), translated to ED volumes:

- **Global tabular model beats per-series models**: train one model across all series with `Site`/`Block` IDs so patterns share statistical strength.
- **Lags + rolling stats are the feature backbone**: weekly/monthly lags + rolling means/stds dominate importance; exogenous features usually help at the margins unless the event signal is strong.
- **Horizon-aware training helps**: direct multi-step (or horizon features) reduces the “farther out is different” mismatch.
- **Hierarchy constraints improve stability**: forecast at an easier aggregate (daily) and allocate down (blocks), then reconcile so the forecast is coherent.
- **Ensembling is where leaderboard gains come from**: blend multiple diverse models; stacking with time-safe OOF predictions is the “final mile”.
- **Loss/weights should match the metric**: WAPE pushes you to optimize high-volume rows; sample weighting is the practical way to reflect that.
- **No-leak covariates matter more than fancy models**: a simple model with correct availability assumptions will beat a “smart” model that accidentally peeks.

## Pipeline menu (each based on a distinct “winning team” pattern)

Below are multiple pipelines that are intentionally **different** in modeling assumptions and error modes. The goal is to later ensemble them.

For each pipeline I give: **strategy**, **why it wins**, and **implementation details** (what code/modules you’d write or reuse in this repo).

### Pipeline A — “M5-style global GBDT” (direct block-level forecasting)

**Strategy**
- Treat each (`Site`, `Block`) time series as part of a **global** supervised learning problem.
- Use a tree booster (LightGBM or XGBoost) with **lags + rolling stats + calendar + exogenous**.
- Predict **total_enc** and **admit_rate**, then derive **admitted_enc**.

**Why this wins (competition analogs)**
- This is the dominant pattern in **Kaggle M5** / **Store Sales**: GBDTs on a well-built feature table.
- Strong with multiple interacting seasonalities + nonlinear effects + regime shifts.

**Implementation details (repo fit)**
- **Data**: reuse `scripts/build_feature_table.py` output.
- **Features**:
  - Add additional lags beyond `[1,7,14,28]`: try `[1,2,3,7,14,21,28,35,42,56,84,364]` (guard 364 if insufficient history per fold).
  - Rolling features: mean/median/std over `[7,14,28,56,112]` with proper shifts.
  - Add “recent trend” deltas: `roll_mean_7 - roll_mean_28`, `lag_7 - lag_14`, etc.
  - Interaction-friendly encodings: `Site`/`Block` one-hot or target encoding (CatBoost makes this easy).
- **Model choices**:
  - `total_enc`: count model objective (Poisson/Tweedie). In XGB you already use `count:poisson`.
  - `admit_rate`: bounded regression; clamp to `[0,1]`. (Optional: logit transform + regression.)
- **Loss alignment with WAPE**:
  - Train with **sample weights** ≈ `admitted_enc` (or `total_enc`) to better align with WAPE emphasis.
  - Or use a custom evaluation loop that selects hyperparams based on fold WAPE, not MAE.
- **Inference**:
  - Use the existing style in `scripts/predict_sep_oct_2025.py` (it recomputes lags by concatenating history + horizon rows).
- **Post-processing**:
  - Clip at 0; integer round.
  - Enforce admitted ≤ total.

**Deliverables**
- `pipelines/gbdt_block/train.py`, `predict.py`
- Or extend existing `scripts/train_models.py` with LGBM/CatBoost variants.

---

### Pipeline B — “Horizon-specific direct multi-step” (avoid recursive error)

**Strategy**
- Instead of 1 model predicting all future dates uniformly, train **direct models** for specific horizons (or include horizon as a feature).

Two viable variants:
1. **Separate model per horizon bucket** (e.g. 1–7 days, 8–21, 22–61)
2. **Single model with `days_ahead` feature**, trained on shifted targets

**Why this wins (competition analogs)**
- In M5, many top solutions prefer **direct** forecasting per horizon/day to reduce drift from recursive predictions.
- Even if you’re not doing recursive inference here, horizon-conditioned training helps the model learn that “61 days out” behaves differently than “tomorrow”.

**Implementation details**
- Build training rows by creating examples:
  - Features at date \(t\) → target at date \(t+h\)
  - Add `days_ahead = h` and optionally `horizon_bucket`.
- Keep grouping by (`Site`, `Block`).
- Fit two models: `total_enc(h)` + `admit_rate(h)` (or multioutput).
- Validate on the 4 official 2-month windows; compute WAPE across the full window.

**Deliverables**
- `pipelines/direct_multistep/prepare_supervised.py`
- `pipelines/direct_multistep/train.py`, `predict.py`

---

### Pipeline C — “Daily + allocation + exact reconciliation” (hierarchical constraint modeling)

**Strategy**
- Decompose the problem:
  1) Forecast **daily totals** per (`Site`) for total + admitted (or rate)
  2) Forecast **within-day block shares** per (`Site`, `Block`, `Date`)
  3) Allocate daily totals to blocks with **exact-sum integer rounding**

**Why this wins (competition analogs)**
- Hierarchical reconciliation is a common winning trick in M5-like settings:
  - It reduces noise because daily totals are easier than per-block counts.
  - It enforces coherent structure (your final answers “add up”).
- Your repo already has a reconciliation script that scales block forecasts to match daily forecasts.

**Implementation details (best-practice version)**
- **Daily model**:
  - Use `scripts/train_daily_models.py` style but tune/upgrade (LGBM/CatBoost).
- **Block share model** (the big upgrade vs simple scaling):
  - Train a model for \(p_{block}\) using:
    - Softmax regression / Dirichlet regression / multiclass-like approach
    - Inputs: calendar, weather, events, case-mix shares, lagged block shares
  - Predict shares \(\hat p_0..\hat p_3\) that sum to 1.
  - Allocate:
    - \(\widehat{block\_total} = \widehat{daily\_total} \cdot \hat p\)
    - Then use **largest remainder rounding** (repo already implements this) to enforce exact sums.
- **Admitted allocation**:
  - Either allocate admitted totals with their own share model,
  - Or allocate total + model admit_rate by block then cap.

**Deliverables**
- Reuse: `scripts/reconcile_block_to_daily.py` (already has exact-sum rounding)
- Add: `pipelines/daily_allocation/block_share_model.py`

---

### Pipeline D — “Count GLM/GAM with Fourier seasonality” (stable, low-variance model)

**Strategy**
- Fit a statistical count model per (`Site`, `Block`) with explicit seasonal structure:
  - Poisson / Negative Binomial GLM
  - Fourier terms for weekly + yearly seasonality
  - Holiday/event/weather regressors

**Why this wins**
- GBDTs can overfit subtle artifacts; a GLM/GAM is a strong *complement* in an ensemble:
  - Lower variance
  - Better extrapolation when patterns are smooth/stable
- For WAPE, stable models can outperform in high-volume regimes.

**Implementation details**
- Build per-series design matrix:
  - `dow` one-hot, holiday flags, month, trend
  - Fourier terms: \(\sin(2\pi k t / 7)\), \(\cos(...)\) and similarly for 365.25
  - Optional changepoint indicators (COVID-era vs post)
- Fit:
  - `statsmodels` GLM (Poisson or NegBin)
  - For admit_rate: logistic regression / Beta regression (or model admitted counts with offset log(total)).
- Predict 61 days ahead; clip, integer round.

**Deliverables**
- `pipelines/glm_fourier/train.py`, `predict.py`

---

### Pipeline E — “Deep global sequence model” (TFT / DeepAR / N-BEATS)

**Strategy**
- Train a multi-series neural forecaster that learns shared dynamics across series:
  - Static IDs: `Site`, `Block`
  - Known future covariates: calendar, events, (optional) climatology weather features
  - Observed covariates: lagged targets, case-mix shares

**Why this wins (when it wins)**
- Deep models can learn:
  - latent regime shifts
  - non-linear seasonality interactions
  - cross-series sharing (small number of sites/blocks benefits from parameter sharing)
- Often used in top solutions as **one component** of an ensemble, not alone.

**Implementation details**
- Library: `pytorch-forecasting` (Temporal Fusion Transformer / DeepAR) or `darts`.
- Data frame format:
  - `time_idx` (integer day index)
  - `group_id` = (`Site`,`Block`)
  - targets: `total_enc`, plus optionally `admit_rate`
  - covariates: calendar + external + case-mix
- Train with quantile loss (pinball) or MAE-like; select point forecast = median.
- Postprocess to enforce constraints and integer outputs.

**Deliverables**
- `pipelines/deep_global/`
- Requirements update (torch etc) only if you actually want to run it.

---

### Pipeline F — “Reason-mix latent factor model” (use reasons without forecasting them)

**Strategy**
- Use `REASON_VISIT_NAME` only through **compressed latent factors** that capture demand composition shifts.
- Example:
  1) Build category share matrix (already available via `cmix_*_share`)
  2) Reduce to \(k\) latent factors using PCA/NMF over time (per site or global)
  3) Forecast factors forward (simple AR/GBDT on factors)
  4) Use predicted factors as exogenous regressors for total/admit_rate

**Why this wins**
- ED volume surges often come from shifts in *composition* (respiratory season, trauma events, etc.).
- This gives you a structured way to use reasons without exploding the output space.

**Implementation details**
- Train PCA/NMF on historical `cmix_*_share` (no leakage).
- Factor forecast:
  - simplest: “climatology by (month, day, block)” (already what you do for shares)
  - stronger: autoregressive model on factors + calendar
- Final model can be GBDT/GLM using factors as extra regressors.

**Deliverables**
- `pipelines/casemix_factors/`

---

## Ensembling (how winning teams usually finish)

### Ensemble 1 — simple weighted blend (fast, strong baseline)

- Compute each pipeline’s WAPE on the **latest** validation window (Jul–Aug 2025).
- Pick weights \(w_i\) by minimizing WAPE (grid search over simplex or constrained least squares on absolute errors).
- Blend totals and rates (or totals and admitted) then enforce constraints.

### Ensemble 2 — stacking with time-safe OOF predictions (best practice)

- For each validation period, store OOF predictions for every row.
- Train a meta-learner (ridge / lasso / small GBDT) to predict the final target from:
  - pipeline predictions
  - simple context features (Site, Block, DOW)
- Objective: minimize admitted WAPE (or approximate with weighted MAE).

### Post-ensemble reconciliation

Regardless of ensemble:
- Reconcile to daily totals (Pipeline C style) *or*
- At least apply:
  - \(adm \leftarrow \min(adm, total)\)
  - integer rounding

---

## Implementation blueprint (concrete repo integration)

### Common shared pieces (build once)

- **Feature build**: reuse `scripts/build_feature_table.py`
- **Fold generator**: implement the 4×2-month splits described in `README.md` (match competition)
- **Metrics**: use `src/dsu_forecast/modeling/metrics.py#wape`
- **Output writer**: always emit `Site, Date, Block, ED Enc, ED Enc Admitted`

### Suggested folder structure

- `pipelines/`
  - `gbdt_block/`
  - `direct_multistep/`
  - `daily_allocation/`
  - `glm_fourier/`
  - `deep_global/` (optional)
  - `casemix_factors/`
- `scripts/`
  - keep “one-command” wrappers like the existing ones:
    - `build_feature_table.py`
    - `train_*.py`
    - `predict_*.py`

### Minimal “run order” per pipeline (standardize this)

1) `python scripts/build_feature_table.py --out outputs/feature_table.parquet`
2) `python pipelines/<pipeline>/train.py --features outputs/feature_table.parquet --artifacts artifacts/<pipeline>/`
3) `python pipelines/<pipeline>/predict.py --features outputs/feature_table.parquet --artifacts artifacts/<pipeline>/ --out outputs/<pipeline>_sep_oct_2025.csv`
4) Optional: `python scripts/reconcile_block_to_daily.py ...` to enforce daily coherence

---

## Optional “external data expansions” (only if allowed / worth it)

These can be added as **additional covariate joiners** in `src/dsu_forecast/features/external_covariates.py`:

- **CDC ILI / flu** (weekly) → interpolate to daily; respiratory-related ED volume driver.
- **Air quality** (AQI) → respiratory exacerbations.
- **Local school calendars** per site region (start/end, breaks).
- **Major events** beyond `config/events.yaml` (sports, festivals) if sites are near predictable venues.

I’d only add these if:
- you can obtain them consistently for **all folds** (2018–2025),
- and you keep the **no-leak** rule for backtests.

---

## References (external analogs / winning-solution patterns)

- Kaggle M5 Forecasting (Accuracy) winning-solution repos (LGBM + lags/rolling + hierarchy/ensembles):
  - `https://github.com/stephenllh/m5-accuracy`
  - `https://github.com/btrotta/kaggle-m5`
- Kaggle Store Sales – Time Series Forecasting (GBDT + lag features + holidays + ensembling patterns):
  - `https://www.kaggle.com/competitions/store-sales-time-series-forecasting`
