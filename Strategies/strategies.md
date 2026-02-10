# Forecasting Strategies & Implementation Plan

## 1. Objective Analysis
The goal is to forecast daily ED encounters (Total and Admitted) for Sept-Oct 2025 across 4 sites in 6-hour blocks.
**Key Constraints & Characteristics:**
- **Horizon:** 61 days (Sept 1 - Oct 31).
- **Seasonality:** Strong weekly and yearly seasonality expected in ED visits.
- **Evaluation:** Average performance across 4 validation periods in 2025.
- **Data:** 4 Sites, high frequency (aggregating hourly to 6-hour blocks).

## 2. External Data Strategy
Based on successful Kaggle solutions (e.g., *Recruit Restaurant Visitor*, *Rossmann Sales*), incorporating external signals is critical for maximizing accuracy.
*   **Holidays:** US Federal Holidays are mandatory. ED volume often spikes/drops on specific holidays.
*   **School Schedules:** Local school calendars (if available/generalizable) often correlate with pediatric ED surges. *Action: Use general school year boundaries.*
*   **Weather:** Temperature/Precipitation extremes drive ED volume (e.g., heat exhaustion, slipping on ice). *Action: If historical weather data for the region is accessible, join by date.* (For this plan, we will focus on time-deterministic features first to avoid data leakage/complexity, but Holidays are a must).

## 3. Pipeline Strategies
We will implement three distinct pipelines ranging from tree-based ensembles to deep learning, culminating in a robust weighted ensemble.

### Pipeline 1: "The Recruit" - Advanced Gradient Boosting
*Inspired by the winning solution of the Recruit Restaurant Visitor Forecasting competition.*

**Logic:** LightGBM/XGBoost excels at tabular time-series when provided with rich historical features. It handles non-linearities and interactions (e.g., Site A on weekends vs Site B) extremely well.

**Implementation Details:**
- **Model:** LightGBM (LGBMRegressor) - generally faster and often slightly more accurate than XGBoost for this scale.
- **Target Transformation:** Log1p transform of target (if skewed) or direct count prediction with Poisson/Tweedie objective.
- **Feature Engineering:**
    - **Lag Features:** Critical. 
        - `lag_56` (8 weeks), `lag_364` (52 weeks) - heavily referencing the same day of week in past periods.
        - *Note:* Since we forecast 60 days out, immediate lags (lag_1) are not available without recursive forecasting. We will use a **Direct Approach** (training separate models for different horizons) OR use lags > 60 days.
    - **Rolling Statistics:** Mean/Std/Min/Max of visits over previous 7, 14, 28, 60 days (shifted by prediction horizon).
    - **Time Features:** HourBlock (0-3), DayOfWeek, Month, DayOfYear, WeekOfYear, IsWeekend.
    - **Cyclical Encoding:** Sine/Cosine transform for Hour, Month, DayOfWeek.
    - **Holidays:** `IsHoliday`, `DaysSinceHoliday`, `DaysUntilHoliday`.
- **Validation:** TimeSeriesSplit matching the competition structure.

### Pipeline 2: "The Rossmann" - Neural Network with Entity Embeddings
*Inspired by the 3rd place solution in Rossmann Store Sales.*

**Logic:** Neural Networks can learn shared representations across sites. "Entity Embeddings" allow the network to learn rich vector representations for categorical variables (Site, DayOfWeek) rather than just one-hot encoding, capturing latent relationships (e.g., predicting Site A implies knowing it matches Site C's summer trend).

**Implementation Details:**
- **Architecture:** Feed-Forward Neural Network (FastAI Tabular or PyTorch).
    - Layers: Input -> Embeddings + Continuous -> BatchNorm -> Dropout -> Dense -> ReLU -> ... -> Output.
- **Embeddings:** 
    - `Site` (4 -> dim 3), `DayOfWeek` (7 -> dim 4), `Month` (12 -> dim 6), `HourBlock` (4 -> dim 3).
- **Continuous Inputs:**
    - Trend Features: Linear trend (Unix timestamp), Rolling averages (normalized).
- **Loss Function:** MSE or MAE (depending on metric focus).
- **Pros:** Captures interaction between 'Site' and 'Season' automatically.

### Pipeline 3: "The Prophet" - Decomposable Time Series
*Standard robust baseline for seasonal data.*

**Logic:** ED volume is driven by distinct human cycles (daily, weekly, yearly). Prophet explicitly models these additive components. While often beaten by GBMs on pure accuracy, it offers explainability and robustness to outliers (e.g., COVID anomalies if not filtered).

**Implementation Details:**
- **Model:** Facebook Prophet (or NeuralProphet for autoregression).
- **Components:**
    - Yearly Seasonality (Order 10 Fourier).
    - Weekly Seasonality (Order 3).
    - Daily/Block Seasonality: Add regressor for time-of-day or forecast at daily level and split by historical ratios.
    - **Holidays:** Strong custom list of US holidays.
- **Strategy:** Train one model per Site.

## 4. Final Ensemble Strategy
*Inspired by M5 Forecasting winners.*

Combine predictions from Pipelines 1, 2, and 3.
- **Level 1:** Generate out-of-fold predictions (or validation set predictions) for all timelines.
- **Level 2:** Weighted Average.
    - Weights determined by finding the linear combination that minimizes error on the most recent val period (Jul-Aug 2025).
    - `Final_Pred = w1*LGBM + w2*NN + w3*Prophet`

## 5. Implementation Roadmap

### Phase 1: Data Prep & Feature Engineering
- [ ] **Ingestion:** Standardize `load_dataset`. Create `Holiday` table.
- [ ] **Transformation:** Convert hourly source data to 6-hour blocks target format.
- [ ] **Feature Store:** Create a function `generate_features(df, horizon_date)` to produce lags and rolling stats safely preventing leakage.

### Phase 2: Pipeline Development
- [ ] **Pipeline 1 (LGBM):** Implement `LGBM_Forecaster` class. Hyperparameter tune (Optuna) on 2024 data.
- [ ] **Pipeline 2 (NN):** Implement `EntityEmbeddingNN` using PyTorch/FastAI.
- [ ] **Pipeline 3 (Prophet):** Implement `Prophet_Wrapper`.

### Phase 3: Validation & Tuning
- [ ] Run the 4-period validation loop (Jan-Feb, Mar-Apr, May-Jun, Jul-Aug 2025).
- [ ] Collect metrics (MAPE/MAE) for each pipeline.
- [ ] Optimize Ensemble weights.

### Phase 4: Final Forecast
- [ ] Retrain best models on full history (Jan 2018 - Aug 2025).
- [ ] Generate Sept-Oct 2025 submission.
