# Data Source Strategy: Ingestion & Isolation

**Status:** APPROVED — **IMPLEMENTED & VALIDATED**  
**Objective:** Develop a centralized "Data Source" step that ingests, aggregates, and joins all necessary raw data (visits, events, weather) into a unified format for all pipelines.  
**Constraint:** **NO IMPUTATION.** This step must produce raw data with missing values preserved, allowing each pipeline to apply its own imputation strategy.  
**Architecture Role:** First layer of the three-layer system (Data Source → Pipelines → Eval). See `master_strategy_2.md` §7.1 for the full architecture diagram. Output is consumed by all 5 pipelines (A-E); each pipeline produces submission CSVs scored by the common evaluator (`eval.md`).

---

## 0. Execution Summary (2026-02-10)

### 0.1 Output Files

| File | Path | Description |
|------|------|-------------|
| **Master Parquet** | `Pipelines/Data Source/Data/master_block_history.parquet` | Primary output — **45,776 rows × 48 columns** |
| **Master CSV** | `Pipelines/Data Source/Data/master_block_history.csv` | Identical CSV for inspection |
| **Reason Summary** | `Pipelines/Data Source/Data/reason_category_summary.csv` | Top-20 reason category breakdown |
| Weather Cache | `Pipelines/Data Source/Data/cache/weather_cache.csv` | Cached Open-Meteo API response |
| AQI Cache | `Pipelines/Data Source/Data/cache/aqi_cache.csv` | Cached EPA annual AQI data |
| CDC ILI Cache | `Pipelines/Data Source/Data/cache/cdc_ili_cache.csv` | Cached ILI rate (from Delphi Epidata fallback) |

### 0.2 Source Code Files

| File | Path | Purpose |
|------|------|---------|
| **Config** | `Pipelines/data_source/config.py` | `DataSourceConfig` dataclass — paths, grid dates, flags |
| **Ingestion** | `Pipelines/data_source/ingestion.py` | Core pipeline: load → grid → aggregate → merge |
| **External Data** | `Pipelines/data_source/external_data.py` | All external API fetches (events, weather, school, CDC ILI, AQI) |
| **Runner** | `Pipelines/data_source/run_data_source.py` | CLI entry point: `python -m Pipelines.data_source.run_data_source` |
| **Init** | `Pipelines/data_source/__init__.py` | Package exports |

### 0.3 Column-Level Validation

| Column | Dtype | Coverage | Status | Details |
|--------|-------|----------|--------|---------|
| `site` | object | 100% | PASS | 4 sites: A, B, C, D |
| `date` | datetime64 | 100% | PASS | 2018-01-01 → 2025-10-31 (2,861 unique dates) |
| `block` | int64 | 100% | PASS | 0, 1, 2, 3 |
| `total_enc` | int64 | 100% | PASS | 0-fill for empty blocks |
| `admitted_enc` | int64 | 100% | PASS | 0-fill for empty blocks |
| `count_reason_*` (×20) | int64 | 100% | PASS | Top 20 by historical volume, 0-fill |
| `count_reason_other` | int64 | 100% | PASS | Tail bucket |
| `dow` | int32 | 100% | PASS | 0=Mon … 6=Sun |
| `day` | int32 | 100% | PASS | 1–31 |
| `month` | int32 | 100% | PASS | 1–12 |
| `year` | int32 | 100% | PASS | 2018–2025 |
| `day_of_year` | int32 | 100% | PASS | 1–366 |
| `quarter` | int64 | 100% | PASS | 1–4 |
| `week_of_year` | int64 | 100% | PASS | 1–53 |
| `is_weekend` | bool | 100% | PASS | |
| `days_since_epoch` | int64 | 100% | PASS | Epoch = 2018-01-01 |
| `is_covid_era` | bool | 100% | PASS | Mar 2020 – Jun 2021 |
| `is_halloween` | bool | 100% | PASS | Oct 31 only |
| `event_name` | object | 5.7% | OK (sparse) | Only populated on event dates (holidays, Sturgis) |
| `event_type` | object | 5.7% | OK (sparse) | Values: `holiday`, `crowd_event` |
| `is_holiday` | bool | 100% | PASS | 84 unique holiday dates (2018–2025), US Federal Holidays |
| `event_count` | int64 | 100% | PASS | 0–1 events per date |
| `school_in_session` | bool | 100% | PASS | 47.9% True; winter/spring/Thanksgiving breaks carved out |
| `temp_min` | float64 | 100% | PASS | −39.3°F to 81.1°F (Fargo + Sioux Falls) |
| `temp_max` | float64 | 100% | PASS | −22.0°F to 102.6°F; Site B differs correctly |
| `precip` | float64 | 100% | PASS | 0 to 2.7 in |
| `snowfall` | float64 | 100% | PASS | 0 to 2.7 in |
| `cdc_ili_rate` | float64 | **99.8%** | PASS | Weekly ILI % interpolated to daily; range 0.16–7.75% |
| `aqi` | float64 | 96.6% | PASS | EPA 2018–2025; per-county (Cass ND / Minnehaha SD) |

### 0.4 CDC ILI Rate — Source Note

The CDC FluView ILINet API (`gis.cdc.gov/grasp/flu2/PostPhase02DataDownload`) returns **HTTP 500** (server-side outage as of 2026-02-10). The pipeline automatically falls back to the **CMU Delphi Epidata API** (`api.delphi.cmu.edu/epidata/fluview/`), which mirrors the same ILINet data (HHS Region 8 = ND, SD, CO, MT, UT, WY) with identical `wili` (weighted ILI %) values. **423 weekly observations** were retrieved and linearly interpolated to daily.

Seasonality sanity check (monthly avg ILI %):
- **Winter peak**: Dec 3.88%, Jan 3.68%, Feb 3.52%
- **Summer trough**: Jul 0.81%, Aug 0.89%, Jun 1.07%
- **Fall ramp-up**: Oct 1.63% → Nov 2.64% → Dec 3.88%

### 0.5 Grid Integrity

- **Sites × Dates × Blocks** = 4 × 2,861 × 4 = **45,776 rows** (matches output exactly)
- **Zero-visit blocks preserved** — grid is a full Cartesian product, no holes
- **No imputation performed** — weather/ILI/AQI NaNs left as-is per contract

### 0.6 Top 20 Reason Categories (by historical volume)

```
ABDOMINAL PAIN, CHEST PAIN, FALL, SHORTNESS OF BREATH, BACK PAIN,
FEVER, HEADACHE, VOMITING, BREATHING PROBLEM, COUGH, DIZZINESS,
WEAKNESS, FLANK PAIN, LEG PAIN, LACERATION, FLU-LIKE SYMPTOMS,
ALCOHOL INTOXICATION, EVALUATION, SEIZURES, MOTOR VEHICLE CRASH
```

All other categories → `count_reason_other`.

### 0.7 How to Re-Run

```bash
# Full run (fetches APIs if cache missing)
python -m Pipelines.data_source.run_data_source

# Skip network calls (use cached data only)
python -m Pipelines.data_source.run_data_source --no-fetch

# Custom reason count
python -m Pipelines.data_source.run_data_source --top-n-reasons 25
```

To force a fresh CDC ILI fetch, delete `Pipelines/Data Source/Data/cache/cdc_ili_cache.csv` before re-running.

---


***

## 1. Data Requirements by Pipeline

| Pipeline | Key Data Dependencies | Source Component | Notes |
|----------|----------------------|------------------|-------|
| **A (Global GBDT)** | Block-level `total_enc`, `admitted_enc`, Calendar (dow, month, day, week_of_year, quarter, day_of_year), Weather (temp, precip, snowfall), Events (is_holiday, event_name/type/count, is_halloween), School Calendar, `is_covid_era`, full history for lags/rolling | Master Time Series | Needs `event_count` for interaction features. Uses `is_halloween`. |
| **B (Multi-Step)** | Same as A; needs full uninterrupted history for horizon-adaptive lag sets per bucket | Master Time Series | Uses `is_covid_era` for downweighting. |
| **C (Hierarchical)** | Daily aggregates (sum block→daily), block-level shares, Calendar, Weather | Master Time Series (aggregated in pipeline) | Uses `school_in_session` for share modeling. |
| **D (GLM/GAM)** | `total_enc`, `admitted_enc`, DOW, `day_of_year` (for Fourier term generation), `is_holiday`, Holiday proximity (derivable from events), trend (`days_since_epoch` derived from `date`), `is_covid_era` | Master Time Series | Needs valid `is_holiday` dates for proximity derivation. |
| **E (Reason-Mix)** | Reason-of-visit counts (`count_reason_*`) for share/factor computation, Calendar, full history for factor AR/momentum lags | Case Mix Aggregates + Master Time Series | Requires ~20 top reason categories + "other" bucket. |

**Cross-Cutting (ALL Pipelines):** `is_covid_era` flag (Mar 2020 – Jun 2021), COVID sample-weight policy (derived from `total_enc` in feature engineering, not stored here).

---

## 2. The "Unified Raw Dataset" Schema

The output of the Data Source step will be a **DataFrame** (saved as Parquet) with the following granular schema. This schema ensures that all downstream pipelines have the raw materials they need without any premature imputation.

**Primary Key:** `(Site, Date, Block)`

### 2.1 Core Columns (Targets & Identifiers)
| Column | Type | Description | Source | Note |
|--------|------|-------------|--------|------|
| `site` | Categorical | Site ID (A, B, C, D) | `sites.json`, raw data | Identifier |
| `date` | Date | Visit Date | `visits.csv` | Identifier |
| `block` | Int | Time Block (0-3), where `Block = Hour // 6` | Derived from `Hour` | Identifier — matches eval.md contract |
| `total_enc` | Int | Count of all encounters | Aggregated from `visits.csv` | **Target** |
| `admitted_enc` | Int | Count of admitted encounters | Aggregated from `visits.csv` | **Target** |

### 2.2 Case Mix Columns (Reason Inputs)
Instead of pre-calculating shares (which might require imputation), we store **Raw Counts**. Pipeline E can then compute shares and decide how to handle zeros or missing categories.

| Column | Type | Description | Source | Note |
|--------|------|-------------|--------|------|
| `count_reason_{cat}` | Int | Count of visits for top K reason categories | Aggregated from `visits.csv` | For Pipeline E (Target K=20) |
| `count_reason_other` | Int | Count of all other reasons (tail) | Aggregated from `visits.csv` | For Pipeline E |

**Implementation Note**: Identify top 20 categories by total historical volume across all sites. All other categories sum to `count_reason_other`.

### 2.3 Temporal & Calendar Features
Raw calendar data is deterministic and does not require imputation.

| Column | Type | Description | Source | Note |
|--------|------|-------------|--------|------|
| `dow` | Int | Day of week (0=Mon, 6=Sun) | Calculated | Feature |
| `day` | Int | Day of month (1-31) | Calculated | Feature (Pipeline A §3.1) |
| `week_of_year` | Int | ISO week number (1-53) | Calculated | Feature (Pipeline A §3.1) |
| `month` | Int | Month (1-12) | Calculated | Feature |
| `quarter` | Int | Quarter (1-4) | Calculated | Feature (Pipeline A §3.1) |
| `day_of_year` | Int | Day of year (1-366) | Calculated | For sin/cos encoding downstream |
| `year` | Int | Year | Calculated | Feature |
| `days_since_epoch` | Int | Days since 2018-01-01 | Calculated | **Trend Feature** (for Pipeline D & A) |
| `is_weekend` | Bool | Saturday/Sunday | Calculated | Feature |
| `is_covid_era` | Bool | Mar 2020 – Jun 2021 (inclusive) | Calculated | §3.0 — used by ALL pipelines for downweighting/exclusion/indicator |
| `is_holiday` | Bool | US Federal Holiday match | `events.yaml` | Feature |
| `is_halloween` | Bool | Oct 31 flag | Calculated | Pediatric injury spike, esp. Block 4 (§6) |
| `event_name` | String | Specific event name (e.g., "Sturgis") | `events.yaml` | Feature (Sparse) |
| `event_type` | String | Event category (Holiday, School, Civic) | `events.yaml` | Feature |
| `event_count` | Int | Number of overlapping events on that date | `events.yaml` | Pipeline A needs `event_count` (§3.1) |

### 2.4 External Data (Weather & School)
Joined by `(Site, Date)`. Missing values must be kept as `NaN`. Downstream pipelines will decide on imputation (e.g., forward fill, climatology, or linear interpolation).

| Column | Type | Description | Source | Actual Coverage | Note |
|--------|------|-------------|--------|-----------------|------|
| `temp_min` | float64 | Min Temperature (°F) | Open-Meteo Historical API | **100%** | Fargo + Sioux Falls; −39.3 to 81.1°F |
| `temp_max` | float64 | Max Temperature (°F) | Open-Meteo Historical API | **100%** | −22.0 to 102.6°F |
| `precip` | float64 | Precipitation (in) | Open-Meteo Historical API | **100%** | 0 to 2.7 in |
| `snowfall` | float64 | Snowfall (in) | Open-Meteo Historical API | **100%** | 0 to 2.7 in |
| `school_in_session`| bool | Is school in session? | Heuristic calendar | **100%** | 47.9% True; breaks carved out |

### 2.5 External Enrichment Data (CDC ILI + AQI)
These columns are now **populated and validated**. Available for all dates 2018–2025, no future leakage.

| Column | Type | Description | Source | Actual Coverage | Note |
|--------|------|-------------|--------|-----------------|------|
| `cdc_ili_rate` | float64 | Weekly ILI surveillance rate (interpolated to daily) | CDC FluView → **Delphi Epidata fallback** | **99.8%** | HHS Region 8; 0.16–7.75% range |
| `aqi` | float64 | Air Quality Index | EPA annual county AQI files | **96.6%** | Cass County ND / Minnehaha County SD |

**Status**: Both columns are now live. Pipeline A/E should test via ablation whether they improve forecasts. The `cdc_ili_rate` shows strong winter seasonality (Dec peak 3.88%, Jul trough 0.81%) — likely useful for respiratory ED volume modeling.

---

## 3. Data Gathering Process (Step-by-Step)

### Step 1: Visit Aggregation
**Input:** `visits.csv` (Raw transactional data)
1. **Load** raw CSV.
2. **Standardize**: Ensure `Date` is datetime, `Hour` is 0-23.
3. **Map Blocks** (must match eval.md: `Block = Hour // 6`):
   - 00:00 - 05:59 -> Block 0
   - 06:00 - 11:59 -> Block 1
   - 12:00 - 17:59 -> Block 2
   - 18:00 - 23:59 -> Block 3
4. **Grid Creation (The "Skeleton")**:
   - Create a Cartesian product of `(All Sites) x (All Dates 2018-01-01 to 2025-10-31) x (Blocks 0-3)`.
   - *Crucial*: This ensures we have rows for days with 0 visits (which are valid data points, not missing data).
   - **No fold filtering here**: The data source produces the **full grid** (all dates). Each pipeline is responsible for splitting by `train_end` per the eval.md validation protocol. This preserves separation of concerns — data ingestion is fold-agnostic.
5. **Aggregate Targets**:
   - Group by `Site, Date, Block`.
   - Sum `ED Enc` -> `total_enc`.
   - Sum `ED Enc Admitted` -> `admitted_enc`.
   - **Merge** targets back to Grid. Fill `NaN` targets with `0` (since no record = 0 visits).

### Step 2: Reason Aggregation (Case Mix)
**Input:** `visits.csv`
1. **Identify Top Categories**: Determine top N (e.g., 20) `REASON_VISIT_NAME` codes by volume historically across all sites.
2. **Pivot**: Group by `Site, Date, Block, REASON_VISIT_NAME`.
3. **Filter**: Keep only Top N specific columns, sum others to `count_reason_other`.
4. **Merge** to Grid. Fill `NaN` with `0` (no visits for that reason).

### Step 3: Event & Calendar Enrichment
**Input:** `events.yaml`, Date Logic
1. **Calendar Attributes**: Compute all deterministic calendar columns for every row in Grid:
   - `dow`, `day`, `week_of_year`, `month`, `quarter`, `day_of_year`, `year`, `is_weekend`
   - `is_covid_era`: `True` if date ∈ [2020-03-01, 2021-06-30], else `False`
   - `is_halloween`: `True` if month=10 and day=31
2. **Event Integration**:
   - Load `events.yaml` — **MUST contain historical events for ALL years 2018-2025**, not just 2025.
     - Sturgis Rally dates (annually, early Aug)
     - US Federal Holidays (all years)
     - School calendar start/end dates (Sioux Falls & Fargo districts, all years)
     - Any known regional events (state fair, etc.)
   - Join on `Date` (left join — most dates have no event).
   - Create `is_holiday` flag from holiday-type events.
   - Compute `event_count` = number of distinct events overlapping each date (0 if none).
   - **Note**: Holiday proximity features (`days_since_xmas`, `days_until_thanksgiving`, etc.) are derived in feature engineering, not here. But the raw holiday dates must be present in `events.yaml` for downstream derivation.

### Step 4: Weather Integration
**Input:** Open-Meteo Historical Archive API (free, no key)
1. **Map Sites** (per master strategy §6):
   - **Site B** → Sioux Falls (43.5446°N, 96.7311°W)
   - **Site A, C** → Fargo (46.8772°N, 96.7898°W)
   - **Site D** → Default Fargo (test Sioux Falls as ablation)
   - Mapping is in `SITE_WEATHER_MAP` dict in `external_data.py`.
2. **Fetch**: Chunked by year, °F + inches, `America/Chicago` timezone.
3. **Join**: Merge on `(site, date)` — each site gets its correct station's weather.
4. **Handling Missing**: **LEAVE AS NaN.** (In practice, Open-Meteo returned 100% coverage for 2018–2025.)
5. **Snowfall**: Converted from cm → inches. SD/ND winter storm events drive trauma and respiratory surges.

### Step 5: School Schedule Integration
**Input:** Heuristic calendar (hardcoded in `external_data.py`)
1. **Academic years** defined as `(start, end)` tuples for 2017–2026, including COVID adjustments (2019-20 early closure, 2020-21 late start).
2. **Rules**:
   - `school_in_session = True` on weekdays within academic year bounds.
   - **Breaks carved out**: Winter (Dec 22–Jan 2), Spring (Mar 14–21), Thanksgiving (Wed–Sun).
3. **Result**: 47.9% of dates = True (1,369 school days / 2,861 total).

### Step 6: CDC ILI Rate Integration
**Input:** CDC FluView ILINet API (primary) → CMU Delphi Epidata API (fallback)
1. **Region**: HHS Region 8 (ND, SD, CO, MT, UT, WY).
2. **Primary source**: `gis.cdc.gov/grasp/flu2/PostPhase02DataDownload` — returns weekly weighted ILI %.
3. **Fallback**: If CDC returns HTTP error, automatically tries `api.delphi.cmu.edu/epidata/fluview/` (same data, better uptime).
4. **Interpolation**: Weekly observations → daily via linear interpolation.
5. **Result**: 423 weekly obs → 2,861 daily rows (99.8% non-NaN).

### Step 7: AQI Integration
**Input:** EPA annual `daily_aqi_by_county` ZIP archives
1. **Counties**: Cass County ND (Sites A/C/D), Minnehaha County SD (Site B).
2. **Join**: Merge on `(site, date)`.
3. **Result**: 96.6% coverage (2018–2025).

---

## 4. Implementation (Actual)

### 4.1 Configuration (`Pipelines/data_source/config.py`)

The config is a Python dataclass — no YAML files needed. All paths are derived from project root.

```python
@dataclass
class DataSourceConfig:
    raw_visits:         Path  # Pipelines/Data Source/Data/DSU-Dataset.csv
    master_parquet:     Path  # Pipelines/Data Source/Data/master_block_history.parquet
    master_csv:         Path  # Pipelines/Data Source/Data/master_block_history.csv
    grid_start:         str   = "2018-01-01"
    grid_end:           str   = "2025-10-31"
    top_n_reasons:      int   = 20
    external_cache_dir: Path  # Pipelines/Data Source/Data/cache/
    fetch_apis:         bool  = True  # False → skip network calls, use cached
```

**Site → Weather Station mapping** (in `external_data.py`):

| Site | Location | Lat/Lon | Weather Source |
|------|----------|---------|----------------|
| A | Fargo ND | 46.8772, −96.7898 | Open-Meteo |
| B | Sioux Falls SD | 43.5446, −96.7311 | Open-Meteo |
| C | Fargo ND | 46.8772, −96.7898 | Open-Meteo |
| D | Fargo ND (default) | 46.8772, −96.7898 | Open-Meteo |

**Site → AQI County mapping**:

| Site | State | County |
|------|-------|--------|
| A, C, D | North Dakota | Cass |
| B | South Dakota | Minnehaha |

### 4.2 Pipeline Flow (`Pipelines/data_source/ingestion.py`)

Actual execution order inside `run_data_ingestion()`:

```
1. Load raw visits      → DSU-Dataset.csv (1,174,310 visit rows)
2. Create skeleton grid → 4 sites × 2,861 dates × 4 blocks = 45,776 rows
3. Aggregate targets    → total_enc, admitted_enc (0-fill empty blocks)
4. Aggregate reasons    → Top 20 count_reason_* + count_reason_other (0-fill)
5. Calendar features    → dow, day, month, year, day_of_year, quarter,
                          week_of_year, is_weekend, days_since_epoch,
                          is_covid_era, is_halloween
6. External data        → Calls add_external_features() which runs:
   6a. Events & holidays    → US Federal Holidays + Sturgis Rally (holidays lib)
   6b. School calendar      → Heuristic: Aug 20–May 25, breaks carved out
   6c. Weather (API/cache)  → Open-Meteo Historical Archive, per-site
   6d. CDC ILI (API/cache)  → CDC FluView → Delphi Epidata fallback
   6e. AQI (API/cache)      → EPA annual county ZIP archives
7. Save outputs         → .parquet + .csv + reason_category_summary.csv
```

### 4.3 External Data Sources (in `external_data.py`)

| Feature | API / Source | Endpoint | Key | Caching |
|---------|-------------|----------|-----|---------|
| Events | `holidays` Python lib + hardcoded Sturgis dates | N/A | None | In-memory (deterministic) |
| School | Heuristic calendar | N/A | None | In-memory (deterministic) |
| Weather | Open-Meteo Historical Archive | `archive-api.open-meteo.com/v1/archive` | None | `cache/weather_cache.csv` |
| CDC ILI | CDC FluView (primary) | `gis.cdc.gov/grasp/flu2/PostPhase02DataDownload` | None | `cache/cdc_ili_cache.csv` |
| CDC ILI | CMU Delphi Epidata (fallback) | `api.delphi.cmu.edu/epidata/fluview/` | None | Same cache file |
| AQI | EPA Annual County AQI | `aqs.epa.gov/aqsweb/airdata/daily_aqi_by_county_{year}.zip` | None | `cache/aqi_cache.csv` |

**CDC ILI Fallback**: If CDC FluView returns HTTP 500 (ongoing outage as of Feb 2026), the pipeline automatically tries the Delphi Epidata API which mirrors the same ILINet data (HHS Region 8, weighted ILI %). Both produce identical `cdc_ili_rate` values.

---

## 5. Output Data Files

All outputs are written to: **`Pipelines/Data Source/Data/`**

### 5.1 Primary Data (Parquet)
**`master_block_history.parquet`** — **45,776 rows × 48 columns**
- **Format**: Parquet (Snappy compression).
- **Contents**: The Unified Raw Dataset described in Section 2.
- **Columns**: 48 total = 5 core + 21 reason counts + 11 calendar/temporal + 4 event + 4 weather + 1 school + 2 external (ILI + AQI).
- **Usage**: Input for ALL forecasting pipelines (A, B, C, D, E).

### 5.2 Inspection Data (CSV)
**`master_block_history.csv`** — identical to Parquet version
- **Format**: CSV (comma-separated), ~45K rows.
- **Usage**: Human inspection, quick debugging, or for pipelines that prefer CSV.

### 5.3 Reason Summary
**`reason_category_summary.csv`**
- **Contents**: Volume breakdown of all reason-of-visit categories (useful for verifying top-20 selection).

### 5.4 API Cache Files (`cache/` subdirectory)
| File | Contents | Re-fetch Trigger |
|------|----------|-----------------|
| `weather_cache.csv` | Open-Meteo daily weather (Fargo + Sioux Falls) | Delete file |
| `aqi_cache.csv` | EPA daily AQI (Cass ND + Minnehaha SD) | Delete file |
| `cdc_ili_cache.csv` | Daily interpolated ILI % (HHS Region 8) | Delete file |

### 5.5 Downstream Pipeline Readiness
- **Pipeline A**: Will derive lags/rolling/interactions/cyclical encodings, FFill weather, apply COVID sample weights.
- **Pipeline B**: Same as A, will create horizon-adaptive lag sets per bucket.
- **Pipeline C**: Will aggregate block→daily for daily model; use block shares for allocation.
- **Pipeline D**: Will generate Fourier terms from `day_of_year`, fit GLM on calendar + trend + `is_covid_era`.
- **Pipeline E**: Will compute `cmix_*_share` from `count_reason_*` columns, run PCA/NMF, build factor AR model with momentum.

## 6. Boundary: What This Step Does NOT Produce

The following are **feature engineering** concerns, handled downstream in each pipeline — NOT in data ingestion:
- Lag features (`lag_63`, `lag_364`, etc.)
- Rolling statistics (mean/std/min/max windows)
- Trend deltas (`roll_mean_7 − roll_mean_28`)
- Cyclical encodings (`doy_sin`, `doy_cos`, `dow_sin`, `dow_cos`)
- Fourier terms (sin/cos at period 7, 365.25)
- Holiday proximity features (`days_since_xmas`, `days_until_thanksgiving`)
- School proximity features (`days_until_school_start`, `days_since_school_start`)
- Interaction features (`is_holiday × Block`, `Site × DOW`)
- Case-mix shares and latent factors (PCA/NMF)
- Sample weights (`total_enc`-based for WAPE alignment)
- `admit_rate` derivation (`admitted_enc / total_enc`)
