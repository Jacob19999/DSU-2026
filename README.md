# DSU-2026: Sanford Health ED Volume Forecasting

## Dataset Overview

The Sanford Health Emergency Department (ED) dataset contains visit records from 4 medical centers from **January 1, 2018 through August 31, 2025**.

### Dataset Structure

- **Size**: 1,174,310 records
- **Columns**:
  - `Site`: Facility identifier (A, B, C, D)
  - `Date`: Visit date
  - `Hour`: Hour of visit (0-23)
  - `REASON_VISIT_NAME`: Reason for visit (867 unique reasons)
  - `ED Enc`: Total number of encounters
  - `ED Enc Admitted`: Number of encounters admitted to a floor

**Note**: The number of encounters admitted to a floor are included in the total encounter volume.

### Key Dataset Statistics

- **Sites**: 4 facilities (A, B, C, D)
  - Site A: 26.27% of visits (308,478)
  - Site B: 36.03% of visits (423,139) - highest volume
  - Site C: 21.32% of visits (250,370)
  - Site D: 16.38% of visits (192,323)

- **Overall Admission Rate**: 29.15% (342,337 of 1,174,310 visits)
  - Site A: 34.75%
  - Site C: 32.71%
  - Site B: 27.91%
  - Site D: 18.27%

- **Average Daily Volume**: ~419 visits per day
- **Time Range**: 2,800 days (2018-01-01 to 2025-08-31)

### Ground Truth & Evaluation Construction

The competition only ships **one raw file**, `Dataset/DSU-Dataset.csv`, at **hourly grain**:

- `Site, Date, Hour, REASON_VISIT_NAME, ED Enc, ED Enc Admitted`

There is **no separate `train.csv` or `test.csv`**. All training, validation, and evaluation ground truth is derived from this single file as follows:

- **Hour → Block mapping** (canonical, used everywhere in this repo):
  - 00:00–05:59 → `Block = 0`
  - 06:00–11:59 → `Block = 1`
  - 12:00–17:59 → `Block = 2`
  - 18:00–23:59 → `Block = 3`
- **Aggregation to competition grain**:
  - Group `DSU-Dataset.csv` by `(Site, Date, Block)` using the mapping above.
  - Sum `ED Enc` and `ED Enc Admitted` within each group to produce block-level totals.
  - Normalize `Date` to `YYYY-MM-DD` strings for stable joins.

This produces the **block-level truth table** used by the shared evaluator (`Strategies/eval.md`) for all 4 forward-validation windows.

### Synthetic “Test” Grid for Sept–Oct 2025

Because there is no separate `test.csv`, the Sept–Oct 2025 forecast target is defined by a **synthetic grid** constructed from the calendar and site list:

- `Site ∈ {A, B, C, D}`
- `Date ∈ 2025-09-01 .. 2025-10-31` (61 days)
- `Block ∈ {0, 1, 2, 3}`

This yields **4 × 61 × 4 = 976 rows**. Every pipeline’s final forecast must:

- Output **exactly this grid** (no missing or duplicate `(Site, Date, Block)` keys)
- Use the exact column names:
  - `Site, Date, Block, ED Enc, ED Enc Admitted`
- Respect the hard constraints:
  - `ED Enc ≥ 0`
  - `ED Enc Admitted ≥ 0`
  - `ED Enc Admitted ≤ ED Enc`
  - Both targets integer-valued (after rounding)

The evaluation harness under `Pipelines/Eval/` uses these rules consistently for both cross-validation and final forecast scoring.

## Forecasting Objective

Forecast future daily ED volumes for **September and October 2025** (Calendar Year 2025) by:

1. **Facility**: Each of the 4 sites (A, B, C, D)
2. **Date**: Daily forecasts for Sept 1-30, 2025 and Oct 1-31, 2025
3. **Time Blocks**: 6-hour increments
   - **Block 1**: Hours 0-5 (midnight to 5:59 AM)
   - **Block 2**: Hours 6-11 (6:00 AM to 11:59 AM)
   - **Block 3**: Hours 12-17 (noon to 5:59 PM)
   - **Block 4**: Hours 18-23 (6:00 PM to 11:59 PM)

4. **Metrics to Forecast**:
   - Total encounters (`ED Enc`)
   - Admitted encounters (`ED Enc Admitted`)

### Forecasting Requirements

- **NOT by reason of visit**: The `REASON_VISIT_NAME` field was included to aid in understanding cyclical patterns (time of day, month, year) but should **not** be used as a dimension in the forecast output.

- **Output Format**: Forecasts should be provided for each combination of:
  - Facility (Site)
  - Date (Sept 1-30, 2025; Oct 1-31, 2025)
  - 6-hour time block (0-5, 6-11, 12-17, 18-23) → encoded as `Block ∈ {0,1,2,3}` via `Block = Hour // 6`
  - Total encounters (`ED Enc`)
  - Admitted encounters (`ED Enc Admitted`)

## Baseline Model

The original baseline consisted of two conceptual pipelines:

- **Direct volume prediction**: Train a single ML model to predict block-level volume directly from calendar and historical features.
- **Category-based prediction**: Forecast volumes by reason-category, then aggregate back to total volume.

These ideas are now implemented in a more robust multi-pipeline framework under `Pipelines/`.

## Current Pipeline Strategy (`Pipelines/`)

The production code under `Pipelines/` implements the strategy described in `Strategies/Implementation plan/`:

- **Data Source layer** (`Pipelines/data_source/`): Builds a canonical block-level history (`master_block_history.parquet`) from the raw CSV, adds calendar + case-mix + external covariates (events, weather, school calendar, CDC ILI, AQI).
- **Pipeline A — Global GBDT** (`Pipelines/Pipeline A/`): Single LightGBM pair (total volume + admit rate) trained across all (Site, Block) series with safe lags (≥63 days) and heavy feature engineering; this is the primary workhorse model.
- **Pipeline B — Direct Multi-Step GBDT** (`Pipelines/Pipeline B/`): Horizon-aware LightGBM models that keep short-horizon lags (buckets for days 1–15, 16–30, 31–61) to recover signal that Pipeline A discards.
- **Pipeline C — Hierarchical Daily → Block** (`Pipelines/Pipeline C/`): First forecasts smooth daily site-level totals, then predicts block shares and allocates back to blocks, enforcing that block forecasts sum exactly to the daily total.
- **Pipeline D — GLM/GAM with Fourier seasonality** (`Pipelines/Pipeline D/`): Poisson GLMs with explicit weekly/annual Fourier terms and trend; low-variance, highly interpretable baseline that complements tree models.
- **Pipeline E — Reason-Mix Latent Factors** (`Pipelines/Pipeline E/`): Compresses reason-level case mix into PCA/NMF factors, forecasts factor trajectories, and feeds them as extra regressors into a final GBDT for volume + admit rate.
- **Evaluation harness** (`Pipelines/Eval/`): Shared evaluator that scores any pipeline’s submission-shaped CSVs on the 4-fold forward validation scheme defined in `Strategies/eval.md`.
- **Orchestrator** (`Pipelines/run_all_pipelines.py`): Convenience wrapper to run multiple pipelines and compare their validation scores.

All pipelines share the same evaluation contract:

- **Targets**: `ED Enc` (total encounters) and `ED Enc Admitted` (admitted encounters)
- **Granularity**: `(Site, Date, Block)` with 4 sites (A–D), 61 forecast days (2025‑09‑01 to 2025‑10‑31), and 4 daily blocks
- **Metric**: WAPE, with mean admitted WAPE across 4 validation folds as the primary comparison number

### Running the pipelines (from repo root)

1. **Build the unified block history (once):**

```bash
python "Pipelines/data_source/run_data_source.py"
```

2. **Run individual pipelines (each writes fold CSVs and metrics under its own `output/`):**

```bash
python "Pipelines/Pipeline A/run_pipeline.py"
python "Pipelines/Pipeline B/run_pipeline.py"
python "Pipelines/Pipeline C/run_pipeline.py"
python "Pipelines/Pipeline D/run_pipeline.py"
python "Pipelines/Pipeline E/run_pipeline.py"
```

3. **Run cross-pipeline evaluation / comparison (optional):**

```bash
python "Pipelines/Eval/run_eval.py"
```

4. **Generate the final Sept–Oct 2025 forecast for a given pipeline (example for Pipeline A):**

```bash
python "Pipelines/Pipeline A/run_pipeline.py" --final-forecast
```
