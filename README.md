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
  - 6-hour time block (0-5, 6-11, 12-17, 18-23)
  - Total encounters
  - Admitted encounters

### Analysis Script

The `analyze_dataset.py` script provides comprehensive exploratory data analysis including:
- Dataset overview and missing values
- Site-level statistics
- Time-based analysis (hourly, monthly trends)
- Visit reason distributions
- Admission rate analysis by site and reason

Run with: `python analyze_dataset.py`
