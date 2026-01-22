# Pipeline 2: Category-Based Prediction with Temporal Splitting

## Overview

Pipeline 2 implements the comprehensive approach from Pipeline 1 but with **category-based splitting**. The training data is split into subsets by reason category (e.g., Injury/Trauma, Cardiovascular, Respiratory) to allow models to learn category-specific cyclical patterns. Predictions are then aggregated back to the (Site, Date, Block) level.

## Key Features

### 1. Category-Based Data Splitting
- Splits training data by reason category using `reason_categories.json`
- Each category gets its own subset with category-specific patterns
- Small categories (< 50 samples) are merged into "Other/Unspecified"
- Allows models to learn distinct seasonal/cyclical patterns per category

### 2. Comprehensive Feature Engineering
Following Pipeline 1's feature engineering approach:

**Temporal Features:**
- Calendar features (day of week, month, quarter, year)
- Cyclical encoding (sin/cos) for month, day of week, block, quarter
- Holiday indicators
- Month start/end flags

**Lag Features:**
- Lags: 1, 7, 14, 28, 30, 90 days
- Applied to: total_enc, admitted_enc, admit_rate
- Grouped by (Site, Block)

**Rolling Window Features:**
- Rolling means: 7, 14, 28, 90 days
- Rolling medians: 7, 14, 28 days
- Rolling std: 7, 14, 28 days
- Applied to: total_enc, admitted_enc, admit_rate

**Site/Block Features:**
- Site encoding
- Block-specific flags (morning, afternoon, evening, night)
- Interaction features (site×block, site×dow, block×dow, site×month)

**Target-Derived Features:**
- Volume trends (slope over recent periods)
- Volatility measures (coefficient of variation)

### 3. Model Architecture
- **Two-model approach per category:**
  - Model 1: Predict `total_enc` (Poisson regression)
  - Model 2: Predict `admit_rate` (squared error)
  - Model 3: Predict `admitted_enc` directly (Poisson) - used for ensemble
- **Ensemble prediction:** Average of direct and derived (total × rate) predictions
- **XGBoost** with optimized hyperparameters

### 4. Prediction Aggregation
- Predictions from all categories are aggregated by summing
- Final output: (Site, Date, Block) level predictions
- Maintains temporal consistency

## Usage

```bash
# Run pipeline2 with validation
python baseline_model/pipieline2/pipeline2.py
```

The pipeline will:
1. Load and validate the dataset
2. Create 4 validation splits (as per competition requirements)
3. For each split:
   - Split training data by category
   - Train category-specific models
   - Make predictions
   - Aggregate predictions
4. Calculate and report metrics (MAE, WAPE)
5. Save results to `baseline_model/pipieline2/results/`

## Output

### Validation Metrics
- `validation_metrics.csv`: Metrics for each validation period
- `predictions_period_{1-4}.csv`: Predictions for each period

### Metrics Reported
- **MAE Encounters**: Mean Absolute Error for total encounters
- **MAE Admitted**: Mean Absolute Error for admitted encounters
- **WAPE Encounters**: Weighted Absolute Percentage Error for total encounters
- **WAPE Admitted**: Weighted Absolute Percentage Error for admitted encounters

## Advantages Over Pipeline 1

1. **Noise Reduction**: Category-specific models reduce cross-category noise
2. **Pattern Learning**: Each model learns category-specific cyclical patterns
   - Sports injuries may peak in certain seasons
   - Cardiovascular issues may have different patterns
   - Respiratory issues may correlate with weather/season
3. **Ensemble Effect**: Aggregating multiple specialized models improves robustness
4. **Better Generalization**: Models focus on homogeneous data subsets

## Implementation Details

### Category Subset Creation
- Minimum samples per category: 50 (for validation), configurable
- Categories with insufficient data are merged into "Other/Unspecified"
- Aggregation by (Site, Date, Block, Category)

### Feature Engineering
- All features from Pipeline 1 are included
- Features are computed per category subset
- Lag/rolling features are grouped by (Site, Block) within each category

### Model Training
- XGBoost with Poisson objective for count data
- Default hyperparameters (can be tuned):
  - n_estimators: 200
  - max_depth: 6
  - learning_rate: 0.05
  - subsample: 0.8
  - colsample_bytree: 0.8

### Prediction Strategy
- For each category, predict total_enc and admit_rate
- Ensemble: average of direct prediction and derived (total × rate)
- Aggregate across categories by summing
- Apply constraints: admitted ≤ total, no negatives

## Dependencies

- pandas
- numpy
- xgboost
- scikit-learn
- dsu_forecast package (for calendar features, lag/rolling features, metrics)

## Files

- `pipeline2.py`: Main implementation
- `README.md`: This file
- `results/`: Output directory (created automatically)
