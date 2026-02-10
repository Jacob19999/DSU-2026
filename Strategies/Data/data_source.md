# Data Source Strategy: Ingestion & Isolation

**Status:** APPROVED  
**Objective:** Develop a centralized "Data Source" step that ingests, aggregates, and joins all necessary raw data (visits, events, weather) into a unified format for all pipelines.  
**Constraint:** **NO IMPUTATION.** This step must produce raw data with missing values preserved, allowing each pipeline to apply its own imputation strategy.  
**Architecture Role:** First layer of the three-layer system (Data Source → Pipelines → Eval). See `master_strategy_2.md` §7.1 for the full architecture diagram. Output is consumed by all 5 pipelines (A-E); each pipeline produces submission CSVs scored by the common evaluator (`eval.md`).

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

| Column | Type | Description | Source | Note |
|--------|------|-------------|--------|------|
| `temp_min` | Float | Min Temperature (°F) | NOAA API / CSV | **NaN if missing** |
| `temp_max` | Float | Max Temperature (°F) | NOAA API / CSV | **NaN if missing** |
| `precip` | Float | Precipitation (in) | NOAA API / CSV | **NaN if missing** |
| `snowfall` | Float | Snowfall (in) | NOAA API / CSV | **NaN if missing** — SD/ND winters drive trauma & respiratory ED volumes |
| `school_in_session`| Bool/NaN | Is school in session? | Manual/External Map | **NaN if unknown** |

### 2.5 Optional External Data (If Allowed)
These are referenced in the master strategy §6 as optional enrichments. They must be available consistently for all 4 CV folds (2018-2025) and must not leak future information.

| Column | Type | Description | Source | Note |
|--------|------|-------------|--------|------|
| `cdc_ili_rate` | Float | Weekly ILI surveillance rate (interpolated to daily) | CDC ILINet | Strong signal for respiratory ED visits |
| `aqi` | Float | Air Quality Index | EPA AQI API | Respiratory exacerbation driver |

**Decision**: Mark as OPTIONAL in config. Pipeline A/E may benefit; test via ablation. If data collection is infeasible, skip entirely — calendar + case-mix captures most seasonal respiratory signal.

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
**Input:** NOAA CSVs (or API fetcher) — columns: `temp_min`, `temp_max`, `precip`, `snowfall`
1. **Map Sites** (per master strategy §6):
   - **Site B** → Sioux Falls (**KFSD**) — confirmed match (Sanford USD Medical Center)
   - **Site A, C** → Fargo (**KFAR**) — confirmed match (Sanford Medical Center Fargo)
   - **Site D** → **Default KFAR, but test KFSD as ablation** — strategy notes "test which fits best"
   - Store this mapping in `configs/data_config.yaml` so it's easy to swap per-site weather station.
2. **Join**: Merge on `(Date)` + Mapped Location → `(Site, Date)`.
3. **Handling Missing**: **LEAVE AS NaN.** Do not forward fill. (Pipelines will decide if they want to FFill, Linear Interp, or Climatology fill).
4. **Snowfall**: NOAA stations KFSD/KFAR report snowfall. Include as a separate column — SD/ND winter storm events drive trauma and respiratory surges.

### Step 5: School Schedule Integration
**Input:** Dictionary/Config
1. **Source**: Use `schools.json` if available.
2. **Fallback Heuristic** (if external data missing, per Pipeline A spec):
   - **Start**: ~Aug 20-25 (Week 34)
   - **End**: ~May 25-30 (Week 21)
   - Mark `school_in_session = True` between Start and End, excluding `is_holiday` dates.
3. Calculate `school_in_session` for each date.

---

## 4. Implementation Plan

### 4.1 Configuration
Each external data source needs a robust configuration in `configs/data_config.yaml`:
```yaml
data_sources:
  visits: "data/raw/visits.csv"
  sites: "data/raw/sites.json"
  events: "data/raw/events.yaml"            # MUST have 2018-2025 historical events
  weather:
    kfsd: "data/external/weather_kfsd.csv"   # Sioux Falls — temp_min, temp_max, precip, snowfall
    kfar: "data/external/weather_kfar.csv"   # Fargo — same columns
  school_calendar: "data/external/school_dates.json"
  # Optional (§2.5)
  cdc_ilinet: "data/external/cdc_ili_weekly.csv"   # Optional — weekly ILI rate
  aqi: "data/external/aqi_daily.csv"               # Optional — daily AQI

site_weather_map:
  A: "kfar"
  B: "kfsd"
  C: "kfar"
  D: "kfar"    # Default KFAR — test KFSD as ablation

covid_era:
  start: "2020-03-01"
  end: "2021-06-30"
```

### 4.2 Code Module (`src/data/ingestion.py`)
This module will be responsible for producing the raw block-level dataset.

```python
def run_data_ingestion(config):
    # 1. Load Visits
    raw_visits = load_visits(config.data_sources.visits)
    
    # 2. Create Skeleton (Block Grid)
    grid = create_block_grid(start_date='2018-01-01', end_date='2025-10-31', sites=all_sites)
    
    # 3. Aggregate Core Targets
    targets = aggregate_core_metrics(raw_visits)
    master = grid.merge(targets, on=['site','date','block'], how='left')
    master = master.fillna({'total_enc': 0, 'admitted_enc': 0})
    
    # 4. Integrate Reason Counts
    reasons = aggregate_reasons(raw_visits, top_n=20)
    master = master.merge(reasons, on=['site','date','block'], how='left').fillna(0)
    
    # 5. Calendar & Temporal (deterministic — no NaN possible)
    master = add_calendar_features(master)  # dow, day, week_of_year, month, quarter,
                                            # day_of_year, year, is_weekend, is_halloween
    master['is_covid_era'] = master['date'].between(
        config.covid_era.start, config.covid_era.end
    )
    
    # 6. Events (Left Join — most dates have no event)
    events = load_events(config.data_sources.events)  # Must cover 2018-2025
    master = master.merge(events, on=['date'], how='left')
    master['event_count'] = master['event_count'].fillna(0).astype(int)
    master['is_holiday'] = master['is_holiday'].fillna(False)
    
    # 7. Weather (Left Join — KEEP NaNs, do NOT impute)
    weather = load_weather(config.data_sources.weather, config.site_weather_map)
    master = master.merge(weather, on=['site', 'date'], how='left')
    # temp_min, temp_max, precip, snowfall remain NaN if missing
    
    # 8. School Calendar
    school = load_school_calendar(config.data_sources.school_calendar)
    master = master.merge(school, on=['site', 'date'], how='left')
    # school_in_session remains NaN if unknown
    
    # 9. Optional: CDC ILINet / AQI (if configured)
    if config.data_sources.get('cdc_ilinet'):
        ili = load_cdc_ilinet(config.data_sources.cdc_ilinet)
        master = master.merge(ili, on=['date'], how='left')
    if config.data_sources.get('aqi'):
        aqi = load_aqi(config.data_sources.aqi, config.site_weather_map)
        master = master.merge(aqi, on=['site', 'date'], how='left')
    
    return master
```

---

## 5. Output Data Files

The pipeline will output the following files to: **`Pipelines/Data Source/Data/`**

### 5.1 Primary Data (Parquet)
**`master_block_history.parquet`**
- **Format**: Parquet (Snappy compression).
- **Contents**: The Unified Raw Dataset described in Section 2.
- **Columns**: ~35-40 (core 5 + ~20 reason counts + 13 calendar/temporal + 4-5 weather + 1 school + 2-3 event + optional external).
- **Usage**: Input for ALL forecasting pipelines (A, B, C, D, E).

### 5.2 Inspection Data (CSV)
**`master_block_history.csv`**
- **Format**: CSV (comma-separated).
- **Contents**: Identical to Parquet version.
- **Usage**: Human inspection, quick debugging, or for pipelines that prefer CSV.
- **Ready for**:
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
