"""
Pipeline 1: Direct Volume Prediction

Step 1: Data Preprocessing
This step prepares raw encounter data for modeling by:
1.1 Data Loading - Load and validate the raw DSU dataset
1.2 Data Cleaning - Handle missing values, outliers, and data quality issues
1.3 Data Aggregation - Aggregate encounters to 6-hour time blocks
1.4 Data Splitting - Create temporal validation splits for robust evaluation

Step 2: Feature Engineering
This step creates comprehensive features for volume prediction by:
2.1 Temporal Features - Add calendar, seasonal, and time-based features
2.2 Lag Features - Create historical lag variables for temporal patterns
2.3 Rolling Window Features - Generate moving averages and statistical aggregations
2.4-2.5 Site & Block Features - Add site-specific and time-block characteristics
2.6 Interaction Features - Create feature interactions and polynomial terms
2.7 External Covariates - Join weather, events, and external data sources (optional)
2.8 Target-Derived Features - Add target-related statistical features
2.9 Feature Selection & Cleaning - Remove redundant features and handle multicollinearity

Step 3: Model Training
This step trains machine learning models for volume prediction by:
3.1 Model Selection - Choose XGBoost as primary model with ensemble alternatives
3.2 Model Architecture - Implement two-model approach (total_enc + admit_rate prediction)
3.3 Training Configuration - Set hyperparameters and training parameters
3.4 Training Process - Train models on each validation period with early stopping
3.5 Data Leakage Prevention - Ensure temporal correctness and no future information leakage

Step 4: Hyperparameter Tuning
This step optimizes model hyperparameters for improved performance by:
4.1 Hyperparameter Search Space - Define comprehensive parameter ranges for XGBoost
4.2 Tuning Process - Use Bayesian Optimization (Optuna) across validation periods
4.3 Aggregate Results - Select stable hyperparameters that perform well across periods
4.4 Train Final Tuned Models - Train production models with optimized parameters
4.5 Validation of Tuned Models - Evaluate improvements over baseline models
"""

import logging
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Import existing utilities
import sys
from pathlib import Path as PathLib

# Add paths to import modules
baseline_root = PathLib(__file__).parent.parent  # Go up to baseline_model root
project_root = baseline_root.parent  # Go up to DSU-2026 root
sys.path.insert(0, str(baseline_root))  # For data_ingestion
sys.path.insert(0, str(project_root / "src"))  # For dsu_forecast

from data_ingestion.loader import (
    get_dataset_path,
    load_dataset,
    aggregate_to_blocks,
    create_validation_splits,
    get_validation_periods
)

# Import feature engineering modules
try:
    from dsu_forecast.features.calendar import add_calendar_features
    from dsu_forecast.features.external_covariates import join_external_covariates
    from dsu_forecast.config import load_sites_config
    EXTERNAL_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"External feature modules not available: {e}")
    logger.warning("Will skip external covariates")
    EXTERNAL_MODULES_AVAILABLE = False

    def add_calendar_features(df, date_col="Date"):
        df = df.copy()
        dts = pd.to_datetime(df[date_col])
        df["dow"] = dts.dt.dayofweek
        df["month"] = dts.dt.month
        df["day"] = dts.dt.day
        df["doy"] = dts.dt.dayofyear
        df["is_weekend"] = (df["dow"] >= 5).astype(int)
        df["is_month_start"] = dts.dt.is_month_start.astype(int)
        df["is_month_end"] = dts.dt.is_month_end.astype(int)
        return df

    def join_external_covariates(df, **kwargs):
        return df

    def load_sites_config():
        return {}

# Import machine learning libraries for Step 3
try:
    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    from sklearn.model_selection import TimeSeriesSplit
    import joblib
    ML_MODULES_AVAILABLE = True
except ImportError as e:
    logger.warning(f"ML modules not available: {e}")
    logger.warning("Step 3 (training) will not be available")
    ML_MODULES_AVAILABLE = False

# Import hyperparameter tuning libraries for Step 4
try:
    import optuna
    from optuna.samplers import TPESampler
    from optuna.pruners import MedianPruner
    OPTUNA_AVAILABLE = True
except ImportError:
    logger.warning("Optuna not available. Will use grid search fallback for tuning.")
    OPTUNA_AVAILABLE = False

# Import visualization libraries for Step 5
try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    PLOTTING_AVAILABLE = True
except ImportError:
    logger.warning("Plotting libraries not available. Step 5 visualizations will be limited.")
    PLOTTING_AVAILABLE = False

try:
    from sklearn.model_selection import ParameterGrid, ParameterSampler
    TUNING_MODULES_AVAILABLE = True
except ImportError:
    logger.warning("sklearn tuning modules not available. Step 4 (tuning) will not be available")
    TUNING_MODULES_AVAILABLE = False


def get_data_dir() -> Path:
    """Get the data directory path for saving intermediate results."""
    data_dir = Path(__file__).parent / "data"
    data_dir.mkdir(exist_ok=True)
    return data_dir


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.1 Temporal Features
    - Calendar features: day of week, month, day of month, day of year, week of year, quarter, year
    - Cyclical encoding: day of week, month, day of year, hour block (sin/cos)
    - Boolean flags: is weekend, is month start/end, is quarter start/end
    """
    logger.info("Adding temporal features...")

    df_temp = df.copy()
    dates = pd.to_datetime(df_temp['Date'])

    # Basic calendar features (using existing function where possible)
    df_temp = add_calendar_features(df_temp, date_col='Date')

    # Add additional calendar features
    df_temp['year'] = dates.dt.year
    df_temp['quarter'] = dates.dt.quarter
    df_temp['week_of_year'] = dates.dt.isocalendar().week.astype(int)

    # Cyclical encoding for temporal features
    # Day of week (0-6) -> sin/cos encoding
    dow_sin = np.sin(2 * np.pi * df_temp['dow'] / 7)
    dow_cos = np.cos(2 * np.pi * df_temp['dow'] / 7)
    df_temp['dow_sin'] = dow_sin
    df_temp['dow_cos'] = dow_cos

    # Month (1-12) -> sin/cos encoding
    month_sin = np.sin(2 * np.pi * (df_temp['month'] - 1) / 12)
    month_cos = np.cos(2 * np.pi * (df_temp['month'] - 1) / 12)
    df_temp['month_sin'] = month_sin
    df_temp['month_cos'] = month_cos

    # Day of year (1-365/366) -> sin/cos encoding (already in calendar.py but let's ensure it's there)
    if 'doy_sin' not in df_temp.columns:
        doy_sin = np.sin(2 * np.pi * df_temp['doy'] / 365.25)
        doy_cos = np.cos(2 * np.pi * df_temp['doy'] / 365.25)
        df_temp['doy_sin'] = doy_sin
        df_temp['doy_cos'] = doy_cos

    # Block (0-3) -> sin/cos encoding
    block_sin = np.sin(2 * np.pi * df_temp['Block'] / 4)
    block_cos = np.cos(2 * np.pi * df_temp['Block'] / 4)
    df_temp['block_sin'] = block_sin
    df_temp['block_cos'] = block_cos

    # Quarter start/end flags
    df_temp['is_quarter_start'] = dates.dt.is_quarter_start.astype(int)
    df_temp['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)

    logger.info(f"Added temporal features. Shape: {df_temp.shape}")
    logger.info(f"Temporal features added: {['dow', 'month', 'day', 'doy', 'week_of_year', 'quarter', 'year', 'is_weekend', 'is_month_start', 'is_month_end', 'is_quarter_start', 'is_quarter_end', 'dow_sin', 'dow_cos', 'month_sin', 'month_cos', 'doy_sin', 'doy_cos', 'block_sin', 'block_cos']}")

    return df_temp


def add_lag_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.2 Lag Features
    - Create lagged values grouped by (Site, Block):
      * Lag 1: Previous day's volume
      * Lag 7: Same day of week, previous week
      * Lag 14: Same day of week, 2 weeks ago
      * Lag 28: Same day of week, 4 weeks ago
      * Lag 30: Previous month, same day
      * Lag 90: Previous quarter, same day
      * Lag 365: Previous year, same day (if available)
    - Apply to: total_enc, admitted_enc, admit_rate
    """
    logger.info("Adding lag features...")

    df_lag = df.copy()

    # Sort by Site, Block, Date to ensure proper lag ordering
    df_lag = df_lag.sort_values(['Site', 'Block', 'Date']).reset_index(drop=True)

    # Define target columns for lagging
    target_cols = ['total_enc', 'admitted_enc', 'admit_rate']

    # Group by Site and Block for lagging
    group_cols = ['Site', 'Block']

    # Lag 1: Previous day (same site, block)
    logger.info("  Adding lag 1 (previous day)...")
    for col in target_cols:
        lag_col = f'{col}_lag1'
        df_lag[lag_col] = df_lag.groupby(group_cols)[col].shift(1)

    # Lag 7: Same day of week, previous week
    logger.info("  Adding lag 7 (previous week, same day)...")
    for col in target_cols:
        lag_col = f'{col}_lag7'
        df_lag[lag_col] = df_lag.groupby(group_cols)[col].shift(7)

    # Lag 14: Same day of week, 2 weeks ago
    logger.info("  Adding lag 14 (2 weeks ago, same day)...")
    for col in target_cols:
        lag_col = f'{col}_lag14'
        df_lag[lag_col] = df_lag.groupby(group_cols)[col].shift(14)

    # Lag 28: Same day of week, 4 weeks ago
    logger.info("  Adding lag 28 (4 weeks ago, same day)...")
    for col in target_cols:
        lag_col = f'{col}_lag28'
        df_lag[lag_col] = df_lag.groupby(group_cols)[col].shift(28)

    # Lag 30: Previous month, same day
    logger.info("  Adding lag 30 (previous month, same day)...")
    for col in target_cols:
        lag_col = f'{col}_lag30'
        df_lag[lag_col] = df_lag.groupby(group_cols)[col].shift(30)

    # Lag 90: Previous quarter, same day
    logger.info("  Adding lag 90 (previous quarter, same day)...")
    for col in target_cols:
        lag_col = f'{col}_lag90'
        df_lag[lag_col] = df_lag.groupby(group_cols)[col].shift(90)

    # Lag 365: Previous year, same day (if available)
    logger.info("  Adding lag 365 (previous year, same day)...")
    for col in target_cols:
        lag_col = f'{col}_lag365'
        df_lag[lag_col] = df_lag.groupby(group_cols)[col].shift(365)

    # Count lag features added
    lag_features_count = sum(1 for col in df_lag.columns if '_lag' in col)
    logger.info(f"Added {lag_features_count} lag features")

    return df_lag


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.3 Rolling Window Features
    - Rolling statistics grouped by (Site, Block):
      * Rolling mean: 7-day, 14-day, 28-day, 90-day windows
      * Rolling median: 7-day, 14-day, 28-day windows
      * Rolling std: 7-day, 14-day, 28-day windows
      * Rolling min/max: 7-day, 14-day windows
    - Apply to: total_enc, admitted_enc, admit_rate
    """
    logger.info("Adding rolling window features...")

    df_roll = df.copy()

    # Sort by Site, Block, Date to ensure proper rolling window ordering
    df_roll = df_roll.sort_values(['Site', 'Block', 'Date']).reset_index(drop=True)

    # Define target columns for rolling windows
    target_cols = ['total_enc', 'admitted_enc', 'admit_rate']
    group_cols = ['Site', 'Block']

    # Define rolling window sizes and statistics
    windows = [7, 14, 28, 90]
    stats = ['mean', 'median', 'std', 'min', 'max']

    # For each target column
    for col in target_cols:
        logger.info(f"  Adding rolling features for {col}...")

        # For each window size
        for window in windows:
            # Skip certain stats for larger windows to avoid too many features
            if window > 28 and col == 'admit_rate':
                continue  # Skip median, std for admit_rate on larger windows

            # Rolling mean
            if 'mean' in stats:
                roll_col = f'{col}_roll{window}_mean'
                df_roll[roll_col] = df_roll.groupby(group_cols)[col].shift(1).rolling(window=window, min_periods=1).mean()

            # Rolling median (only for smaller windows)
            if 'median' in stats and window <= 28:
                roll_col = f'{col}_roll{window}_median'
                df_roll[roll_col] = df_roll.groupby(group_cols)[col].shift(1).rolling(window=window, min_periods=1).median()

            # Rolling std (only for smaller windows)
            if 'std' in stats and window <= 28:
                roll_col = f'{col}_roll{window}_std'
                df_roll[roll_col] = df_roll.groupby(group_cols)[col].shift(1).rolling(window=window, min_periods=1).std()

            # Rolling min/max (only for smaller windows)
            if window <= 14:
                if 'min' in stats:
                    roll_col = f'{col}_roll{window}_min'
                    df_roll[roll_col] = df_roll.groupby(group_cols)[col].shift(1).rolling(window=window, min_periods=1).min()

                if 'max' in stats:
                    roll_col = f'{col}_roll{window}_max'
                    df_roll[roll_col] = df_roll.groupby(group_cols)[col].shift(1).rolling(window=window, min_periods=1).max()

    # Count rolling features added
    rolling_features_count = sum(1 for col in df_roll.columns if '_roll' in col)
    logger.info(f"Added {rolling_features_count} rolling window features")

    return df_roll


def add_site_block_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.4 Site-Specific Features & 2.5 Block-Specific Features
    - Site identifier (one-hot encoded)
    - Site-specific historical averages: mean daily volume, mean admission rate
    - Block identifier (already present)
    - Block-specific historical averages: mean volume per block
    """
    logger.info("Adding site-specific and block-specific features...")

    df_sb = df.copy()

    # 2.4 Site-Specific Features
    logger.info("  Adding site-specific features...")

    # One-hot encode Site (A, B, C, D)
    site_dummies = pd.get_dummies(df_sb['Site'], prefix='site', dtype=int)
    df_sb = pd.concat([df_sb, site_dummies], axis=1)

    # Site-specific historical averages (using all historical data)
    site_stats = df_sb.groupby('Site').agg({
        'total_enc': 'mean',
        'admit_rate': 'mean'
    }).rename(columns={
        'total_enc': 'site_mean_total_enc',
        'admit_rate': 'site_mean_admit_rate'
    }).reset_index()

    df_sb = df_sb.merge(site_stats, on='Site', how='left')

    # 2.5 Block-Specific Features
    logger.info("  Adding block-specific features...")

    # Block is already present (0, 1, 2, 3)
    # Block-specific historical averages
    block_stats = df_sb.groupby('Block').agg({
        'total_enc': 'mean',
        'admitted_enc': 'mean',
        'admit_rate': 'mean'
    }).rename(columns={
        'total_enc': 'block_mean_total_enc',
        'admitted_enc': 'block_mean_admitted_enc',
        'admit_rate': 'block_mean_admit_rate'
    }).reset_index()

    df_sb = df_sb.merge(block_stats, on='Block', how='left')

    logger.info(f"Added site/block features. Shape: {df_sb.shape}")

    return df_sb


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.6 Interaction Features
    - Site × Block interactions
    - Site × Day of week interactions
    - Block × Day of week interactions
    - Site × Month interactions
    """
    logger.info("Adding interaction features...")

    df_int = df.copy()

    # Site × Block interactions
    logger.info("  Adding Site × Block interactions...")
    df_int['site_block'] = df_int['Site'] + '_block' + df_int['Block'].astype(str)

    # Create one-hot encoding for site_block combinations
    site_block_dummies = pd.get_dummies(df_int['site_block'], prefix='site_block', dtype=int)
    df_int = pd.concat([df_int, site_block_dummies], axis=1)

    # Site × Day of week interactions
    logger.info("  Adding Site × Day of week interactions...")
    df_int['site_dow'] = df_int['Site'] + '_dow' + df_int['dow'].astype(str)

    # Create one-hot encoding for site_dow combinations
    site_dow_dummies = pd.get_dummies(df_int['site_dow'], prefix='site_dow', dtype=int)
    df_int = pd.concat([df_int, site_dow_dummies], axis=1)

    # Block × Day of week interactions
    logger.info("  Adding Block × Day of week interactions...")
    df_int['block_dow'] = 'block' + df_int['Block'].astype(str) + '_dow' + df_int['dow'].astype(str)

    # Create one-hot encoding for block_dow combinations
    block_dow_dummies = pd.get_dummies(df_int['block_dow'], prefix='block_dow', dtype=int)
    df_int = pd.concat([df_int, block_dow_dummies], axis=1)

    # Site × Month interactions
    logger.info("  Adding Site × Month interactions...")
    df_int['site_month'] = df_int['Site'] + '_month' + df_int['month'].astype(str)

    # Create one-hot encoding for site_month combinations
    site_month_dummies = pd.get_dummies(df_int['site_month'], prefix='site_month', dtype=int)
    df_int = pd.concat([df_int, site_month_dummies], axis=1)

    # Drop temporary string columns used for creating dummies
    temp_columns_to_drop = ['site_block', 'site_dow', 'block_dow', 'site_month']
    df_int = df_int.drop(columns=[col for col in temp_columns_to_drop if col in df_int.columns])

    # Count interaction features added
    interaction_features_count = sum(1 for col in df_int.columns if any(prefix in col for prefix in ['site_block_', 'site_dow_', 'block_dow_', 'site_month_']))
    logger.info(f"Added {interaction_features_count} interaction features")

    return df_int


def add_target_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.8 Target-Derived Features
    - Historical admission rate (rolling average) - already covered in rolling features
    - Volume trends (slope over recent periods)
    - Volatility measures (coefficient of variation)
    """
    logger.info("Adding target-derived features...")

    df_td = df.copy()

    # Sort by Site, Block, Date
    df_td = df_td.sort_values(['Site', 'Block', 'Date']).reset_index(drop=True)

    group_cols = ['Site', 'Block']

    # Volume trends: slope over recent 7-day and 28-day periods
    logger.info("  Adding volume trend features...")

    # Calculate trends for total_enc over 7-day windows
    for window in [7, 28]:
        # Linear trend (slope) over recent window
        trend_col = f'total_enc_trend_{window}d'

        # Use rolling window to calculate slope (simplified as recent - older)
        recent_mean = df_td.groupby(group_cols)['total_enc'].shift(1).rolling(window=window, min_periods=window//2).mean()
        older_mean = df_td.groupby(group_cols)['total_enc'].shift(window//2 + 1).rolling(window=window//2, min_periods=window//4).mean()

        df_td[trend_col] = recent_mean - older_mean

    # Volatility measures: coefficient of variation
    logger.info("  Adding volatility features...")

    for window in [7, 28]:
        # Coefficient of variation for total_enc
        vol_col = f'total_enc_cv_{window}d'

        rolling_mean = df_td.groupby(group_cols)['total_enc'].shift(1).rolling(window=window, min_periods=window//2).mean()
        rolling_std = df_td.groupby(group_cols)['total_enc'].shift(1).rolling(window=window, min_periods=window//2).std()

        # Coefficient of variation = std / mean (with zero division handling)
        df_td[vol_col] = np.where(rolling_mean > 0, rolling_std / rolling_mean, 0)

        # Coefficient of variation for admission rate
        vol_admit_col = f'admit_rate_cv_{window}d'

        rolling_mean_admit = df_td.groupby(group_cols)['admit_rate'].shift(1).rolling(window=window, min_periods=window//2).mean()
        rolling_std_admit = df_td.groupby(group_cols)['admit_rate'].shift(1).rolling(window=window, min_periods=window//2).std()

        df_td[vol_admit_col] = np.where(rolling_mean_admit > 0, rolling_std_admit / rolling_mean_admit, 0)

    # Additional target-derived features
    logger.info("  Adding peak/off-peak indicators...")

    # Peak hours indicator (based on block averages)
    block_avg_volume = df_td.groupby('Block')['total_enc'].transform('mean')
    overall_avg_volume = df_td['total_enc'].mean()
    df_td['is_peak_block'] = (block_avg_volume > overall_avg_volume).astype(int)

    # Weekend vs weekday volume ratio (rolling)
    weekend_mask = df_td['is_weekend'] == 1
    weekday_mask = df_td['is_weekend'] == 0

    # Calculate weekend/weekday averages by site and block
    for name, group in df_td.groupby(['Site', 'Block']):
        site_mask = (df_td['Site'] == name[0]) & (df_td['Block'] == name[1])

        weekend_avg = df_td.loc[site_mask & weekend_mask, 'total_enc'].rolling(4, min_periods=1).mean()
        weekday_avg = df_td.loc[site_mask & weekday_mask, 'total_enc'].rolling(5, min_periods=1).mean()

        # Create weekend/weekday ratio feature
        ratio_col = f'site_{name[0]}_block_{name[1]}_weekend_weekday_ratio'
        df_td.loc[site_mask, ratio_col] = weekend_avg / weekday_avg.replace(0, 1)  # avoid div by zero

    # Fill NaN values with 1.0 (neutral ratio)
    weekend_ratio_cols = [col for col in df_td.columns if 'weekend_weekday_ratio' in col]
    for col in weekend_ratio_cols:
        df_td[col] = df_td[col].fillna(1.0)

    target_derived_count = sum(1 for col in df_td.columns if any(suffix in col for suffix in ['_trend_', '_cv_', '_ratio', 'is_peak_block']))
    logger.info(f"Added {target_derived_count} target-derived features")

    return df_td


def select_and_clean_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    2.9 Feature Selection
    - Remove constant features (zero variance)
    - Remove highly correlated features (correlation > 0.95)
    - Handle missing values in engineered features:
      * Forward fill for lag features
      * Mean imputation for rolling features
      * Drop rows with critical missing values if necessary
    """
    logger.info("Performing feature selection and cleaning...")

    df_clean = df.copy()
    initial_features = len(df_clean.columns)
    initial_rows = len(df_clean)

    # Identify feature columns (exclude target and metadata columns)
    exclude_cols = ['Site', 'Date', 'Block', 'total_enc', 'admitted_enc', 'admit_rate']
    feature_cols = [col for col in df_clean.columns if col not in exclude_cols]

    logger.info(f"Initial feature count: {len(feature_cols)}")

    # Step 1: Handle missing values in engineered features
    logger.info("  Handling missing values...")

    # Forward fill for lag features
    lag_cols = [col for col in feature_cols if '_lag' in col]
    if lag_cols:
        df_clean[lag_cols] = df_clean.groupby(['Site', 'Block'])[lag_cols].fillna(method='ffill')
        logger.info(f"    Forward-filled {len(lag_cols)} lag features")

    # Mean imputation for rolling features
    rolling_cols = [col for col in feature_cols if '_roll' in col]
    if rolling_cols:
        for col in rolling_cols:
            if df_clean[col].isna().sum() > 0:
                # Use site-block specific means for imputation
                site_block_means = df_clean.groupby(['Site', 'Block'])[col].transform('mean')
                df_clean[col] = df_clean[col].fillna(site_block_means)
        logger.info(f"    Mean-imputed {len(rolling_cols)} rolling features")

    # Fill remaining NaN values with 0 for other features
    remaining_features = [col for col in feature_cols if col not in lag_cols + rolling_cols]
    if remaining_features:
        df_clean[remaining_features] = df_clean[remaining_features].fillna(0)

    # Step 2: Remove constant features (zero variance)
    logger.info("  Removing constant features...")
    constant_features = []
    for col in feature_cols:
        if df_clean[col].nunique() <= 1:
            constant_features.append(col)

    if constant_features:
        df_clean = df_clean.drop(columns=constant_features)
        feature_cols = [col for col in feature_cols if col not in constant_features]
        logger.info(f"    Removed {len(constant_features)} constant features: {constant_features}")

    # Step 3: Remove highly correlated features (correlation > 0.95)
    logger.info("  Removing highly correlated features...")
    if len(feature_cols) > 1:
        # Calculate correlation matrix for numeric features only
        numeric_features = [col for col in feature_cols if df_clean[col].dtype in ['int64', 'float64']]
        if len(numeric_features) > 1:
            corr_matrix = df_clean[numeric_features].corr().abs()

            # Find features with correlation > 0.95
            to_drop = set()
            for i in range(len(corr_matrix.columns)):
                for j in range(i):
                    if corr_matrix.iloc[i, j] > 0.95:
                        col_i = corr_matrix.columns[i]
                        col_j = corr_matrix.columns[j]
                        # Keep the one with more variance (less likely to be redundant)
                        if df_clean[col_i].var() < df_clean[col_j].var():
                            to_drop.add(col_i)
                        else:
                            to_drop.add(col_j)

            if to_drop:
                df_clean = df_clean.drop(columns=list(to_drop))
                feature_cols = [col for col in feature_cols if col not in to_drop]
                logger.info(f"    Removed {len(to_drop)} highly correlated features: {list(to_drop)}")

    # Step 4: Final cleanup - drop rows with any remaining missing values
    remaining_na = df_clean.isna().sum().sum()
    if remaining_na > 0:
        logger.warning(f"  Found {remaining_na} remaining missing values after imputation")
        # For critical features, drop rows with missing values
        critical_features = ['Site', 'Date', 'Block', 'total_enc', 'admitted_enc']
        df_clean = df_clean.dropna(subset=critical_features)

        rows_dropped = initial_rows - len(df_clean)
        if rows_dropped > 0:
            logger.warning(f"    Dropped {rows_dropped} rows with missing critical values")

    final_features = len(feature_cols)
    final_rows = len(df_clean)

    logger.info("Feature selection summary:")
    logger.info(f"  Initial features: {initial_features}")
    logger.info(f"  Final features: {final_features} ({initial_features - final_features} removed)")
    logger.info(f"  Initial rows: {initial_rows}")
    logger.info(f"  Final rows: {final_rows} ({initial_rows - final_rows} removed)")

    return df_clean


def load_raw_data() -> pd.DataFrame:
    """
    1.1 Data Loading
    - Load DSU-Dataset.csv from Dataset/ directory
    - Convert Date column to datetime format
    - Validate data integrity (check for missing values, data types)
    - Log basic statistics (shape, date range, site distribution)
    """
    logger.info("=" * 80)
    logger.info("STEP 1.1: DATA LOADING")
    logger.info("=" * 80)
    
    dataset_path = get_dataset_path()
    logger.info(f"Loading dataset from: {dataset_path}")
    
    df = pd.read_csv(dataset_path)
    logger.info(f"Raw dataset loaded: {df.shape[0]:,} rows × {df.shape[1]} columns")
    
    # Convert Date column to datetime
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    logger.info("Date column converted to datetime format")
    
    # Log basic statistics
    logger.info(f"\nDataset Statistics:")
    logger.info(f"  Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    logger.info(f"  Columns: {list(df.columns)}")
    logger.info(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    logger.info(f"  Sites: {sorted(df['Site'].unique())}")
    logger.info(f"  Hour range: {df['Hour'].min():.0f} to {df['Hour'].max():.0f}")
    
    # Site distribution
    site_counts = df['Site'].value_counts().sort_index()
    logger.info(f"\nSite Distribution:")
    for site, count in site_counts.items():
        pct = (count / len(df)) * 100
        logger.info(f"  Site {site}: {count:,} records ({pct:.2f}%)")

    # Save raw data for validation
    data_dir = get_data_dir()
    output_path = data_dir / "step1_1_raw_data.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved raw data to: {output_path}")

    return df


def clean_data(df: pd.DataFrame, include_blank_reasons: bool = True) -> pd.DataFrame:
    """
    1.2 Data Cleaning
    - Handle missing values
    - Validate data consistency
    - Ensure ED Enc Admitted <= ED Enc for all records
    - Check for negative values
    - Validate Site and Hour values
    
    Args:
        df: Input dataframe
        include_blank_reasons: If True, keep records with blank/missing REASON_VISIT_NAME.
                             If False, drop records with blank/missing REASON_VISIT_NAME.
                             Default: True
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1.2: DATA CLEANING")
    logger.info("=" * 80)
    logger.info(f"include_blank_reasons: {include_blank_reasons}")
    
    initial_rows = len(df)
    df_cleaned = df.copy()
    
    # Check for missing values
    logger.info("\nChecking for missing values...")
    missing = df_cleaned.isnull().sum()
    if missing.sum() > 0:
        logger.warning("Missing values found:")
        for col, count in missing[missing > 0].items():
            pct = (count / len(df_cleaned)) * 100
            logger.warning(f"  {col}: {count:,} ({pct:.2f}%)")
    else:
        logger.info("  No missing values found")
    
    # Handle blank/missing REASON_VISIT_NAME based on parameter
    blank_reasons = df_cleaned['REASON_VISIT_NAME'].isna() | (df_cleaned['REASON_VISIT_NAME'].astype(str).str.strip() == '')
    blank_count = blank_reasons.sum()
    
    if blank_count > 0:
        if include_blank_reasons:
            logger.info(f"Keeping {blank_count:,} records with blank/missing REASON_VISIT_NAME (include_blank_reasons=True)")
        else:
            logger.info(f"Dropping {blank_count:,} records with blank/missing REASON_VISIT_NAME (include_blank_reasons=False)")
            df_cleaned = df_cleaned[~blank_reasons].copy()
    
    # Handle missing dates: drop records with invalid dates
    invalid_dates = df_cleaned['Date'].isna()
    if invalid_dates.sum() > 0:
        logger.warning(f"Dropping {invalid_dates.sum():,} records with invalid dates")
        df_cleaned = df_cleaned[~invalid_dates].copy()
    
    # Validate and handle missing Site values
    invalid_sites = df_cleaned['Site'].isna()
    if invalid_sites.sum() > 0:
        logger.warning(f"Dropping {invalid_sites.sum():,} records with missing Site")
        df_cleaned = df_cleaned[~invalid_sites].copy()
    
    # Validate Site values are in {A, B, C, D}
    valid_sites = {'A', 'B', 'C', 'D'}
    invalid_site_values = ~df_cleaned['Site'].isin(valid_sites)
    if invalid_site_values.sum() > 0:
        logger.warning(f"Dropping {invalid_site_values.sum():,} records with invalid Site values")
        df_cleaned = df_cleaned[~invalid_site_values].copy()
    
    # Validate Hour values are in {0-23}
    invalid_hours = (df_cleaned['Hour'] < 0) | (df_cleaned['Hour'] > 23) | df_cleaned['Hour'].isna()
    if invalid_hours.sum() > 0:
        logger.warning(f"Dropping {invalid_hours.sum():,} records with invalid Hour values")
        df_cleaned = df_cleaned[~invalid_hours].copy()
    
    # Handle missing ED Enc/ED Enc Admitted: set to 0
    if df_cleaned['ED Enc'].isna().sum() > 0:
        logger.info(f"Filling {df_cleaned['ED Enc'].isna().sum():,} missing ED Enc values with 0")
        df_cleaned['ED Enc'] = df_cleaned['ED Enc'].fillna(0)
    
    if df_cleaned['ED Enc Admitted'].isna().sum() > 0:
        logger.info(f"Filling {df_cleaned['ED Enc Admitted'].isna().sum():,} missing ED Enc Admitted values with 0")
        df_cleaned['ED Enc Admitted'] = df_cleaned['ED Enc Admitted'].fillna(0)
    
    # Check for negative values in count columns
    negative_enc = (df_cleaned['ED Enc'] < 0).sum()
    negative_admitted = (df_cleaned['ED Enc Admitted'] < 0).sum()
    
    if negative_enc > 0:
        logger.warning(f"Found {negative_enc:,} records with negative ED Enc values. Setting to 0.")
        df_cleaned.loc[df_cleaned['ED Enc'] < 0, 'ED Enc'] = 0
    
    if negative_admitted > 0:
        logger.warning(f"Found {negative_admitted:,} records with negative ED Enc Admitted values. Setting to 0.")
        df_cleaned.loc[df_cleaned['ED Enc Admitted'] < 0, 'ED Enc Admitted'] = 0
    
    # Validate data consistency: Ensure ED Enc Admitted <= ED Enc
    inconsistent = df_cleaned['ED Enc Admitted'] > df_cleaned['ED Enc']
    if inconsistent.sum() > 0:
        logger.warning(f"Found {inconsistent.sum():,} records where ED Enc Admitted > ED Enc. Correcting...")
        # Cap admitted encounters at total encounters
        df_cleaned.loc[inconsistent, 'ED Enc Admitted'] = df_cleaned.loc[inconsistent, 'ED Enc']
    
    rows_dropped = initial_rows - len(df_cleaned)
    logger.info(f"\nCleaning Summary:")
    logger.info(f"  Initial rows: {initial_rows:,}")
    logger.info(f"  Final rows: {len(df_cleaned):,}")
    logger.info(f"  Rows dropped: {rows_dropped:,} ({rows_dropped/initial_rows*100:.2f}%)")

    # Save cleaned data for validation
    data_dir = get_data_dir()
    suffix = "with_blanks" if include_blank_reasons else "without_blanks"
    output_path = data_dir / f"step1_2_cleaned_data_{suffix}.csv"
    df_cleaned.to_csv(output_path, index=False)
    logger.info(f"\nSaved cleaned data to: {output_path}")

    return df_cleaned


def aggregate_to_time_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    1.3 Data Aggregation
    - Aggregate hourly data to 6-hour time blocks:
      * Block 0: Hours 0-5 (midnight to 5:59 AM)
      * Block 1: Hours 6-11 (6:00 AM to 11:59 AM)
      * Block 2: Hours 12-17 (noon to 5:59 PM)
      * Block 3: Hours 18-23 (6:00 PM to 11:59 PM)
    - Group by: Site, Date, Block
    - Aggregate: Sum of ED Enc, Sum of ED Enc Admitted
    - Calculate derived metrics: Admission rate
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1.3: DATA AGGREGATION")
    logger.info("=" * 80)
    
    df_agg = aggregate_to_blocks(df)
    
    # Calculate admission rate with zero-division handling
    df_agg['admit_rate'] = np.where(
        df_agg['ED Enc'] > 0,
        df_agg['ED Enc Admitted'] / df_agg['ED Enc'],
        0.0
    )
    
    # Rename columns for clarity
    df_agg = df_agg.rename(columns={
        'ED Enc': 'total_enc',
        'ED Enc Admitted': 'admitted_enc'
    })
    
    logger.info(f"Aggregated dataset: {df_agg.shape[0]:,} rows × {df_agg.shape[1]} columns")
    logger.info(f"\nBlock Distribution:")
    block_counts = df_agg['Block'].value_counts().sort_index()
    for block, count in block_counts.items():
        logger.info(f"  Block {block}: {count:,} records")
    
    logger.info(f"\nSummary Statistics (after aggregation):")
    logger.info(f"  Total encounters - Mean: {df_agg['total_enc'].mean():.2f}, "
                f"Std: {df_agg['total_enc'].std():.2f}")
    logger.info(f"  Admitted encounters - Mean: {df_agg['admitted_enc'].mean():.2f}, "
                f"Std: {df_agg['admitted_enc'].std():.2f}")
    logger.info(f"  Admission rate - Mean: {df_agg['admit_rate'].mean():.4f}, "
                f"Std: {df_agg['admit_rate'].std():.4f}")

    # Save aggregated data for validation
    data_dir = get_data_dir()
    output_path = data_dir / "step1_3_aggregated_data.csv"
    df_agg.to_csv(output_path, index=False)
    logger.info(f"\nSaved aggregated data to: {output_path}")

    return df_agg


def create_preprocessed_splits(df: pd.DataFrame) -> Dict:
    """
    1.4 Data Splitting
    - Create validation splits using time-based cross-validation
    - Final training set: All data up to 2025-08-31
    - Test set: September-October 2025 (to be forecasted)
    
    Returns:
        Dictionary containing:
        - 'validation_splits': List of validation period splits
        - 'final_train': Final training set (up to 2025-08-31)
        - 'test_period': Test period definition (Sept-Oct 2025)
        - 'preprocessed_df': Full preprocessed dataframe
    """
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1.4: DATA SPLITTING")
    logger.info("=" * 80)
    
    # Create validation splits
    validation_splits = create_validation_splits(df)
    
    logger.info(f"\nCreated {len(validation_splits)} validation periods:")
    for split in validation_splits:
        logger.info(f"\n  Period {split['period_id']}: {split['description']}")
        logger.info(f"    Train: {split['train_df']['Date'].min()} to {split['train_df']['Date'].max()}")
        logger.info(f"    Train shape: {split['train_df'].shape}")
        logger.info(f"    Test: {split['test_df']['Date'].min()} to {split['test_df']['Date'].max()}")
        logger.info(f"    Test shape: {split['test_df'].shape}")
    
    # Final training set: All data up to 2025-08-31
    final_train_end = pd.to_datetime('2025-08-31')
    final_train = df[df['Date'] <= final_train_end].copy()
    
    # Test set: September-October 2025 (to be forecasted)
    test_start = pd.to_datetime('2025-09-01')
    test_end = pd.to_datetime('2025-10-31')
    test_period = {
        'start': test_start,
        'end': test_end,
        'description': 'September-October 2025 (forecast period)'
    }
    
    logger.info(f"\nFinal Training Set:")
    logger.info(f"  Date range: {final_train['Date'].min()} to {final_train['Date'].max()}")
    logger.info(f"  Shape: {final_train.shape}")
    
    logger.info(f"\nTest Period (forecast target):")
    logger.info(f"  Date range: {test_period['start']} to {test_period['end']}")

    # Save validation splits and final training set for validation
    data_dir = get_data_dir()

    # Save each validation split
    for split in validation_splits:
        period_id = split['period_id']
        train_path = data_dir / f"step1_4_validation_period{period_id}_train.csv"
        test_path = data_dir / f"step1_4_validation_period{period_id}_test.csv"

        split['train_df'].to_csv(train_path, index=False)
        split['test_df'].to_csv(test_path, index=False)
        logger.info(f"  Saved validation period {period_id} - Train: {train_path}")
        logger.info(f"  Saved validation period {period_id} - Test: {test_path}")

    # Save final training set
    final_train_path = data_dir / "step1_4_final_train.csv"
    final_train.to_csv(final_train_path, index=False)
    logger.info(f"\nSaved final training set to: {final_train_path}")

    # Save full preprocessed dataframe
    preprocessed_path = data_dir / "step1_4_preprocessed_full.csv"
    df.to_csv(preprocessed_path, index=False)
    logger.info(f"Saved full preprocessed dataframe to: {preprocessed_path}")

    return {
        'validation_splits': validation_splits,
        'final_train': final_train,
        'test_period': test_period,
        'preprocessed_df': df
    }


def run_step1(include_blank_reasons: bool = True) -> Dict:
    """
    Execute Step 1: Complete Data Preprocessing Pipeline
    
    Args:
        include_blank_reasons: If True, keep records with blank/missing REASON_VISIT_NAME.
                              If False, drop records with blank/missing REASON_VISIT_NAME.
                              Default: True
                              This allows testing whether including or excluding blank reasons
                              improves model performance.
    
    Returns:
        Dictionary containing all preprocessed data and splits
    """
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE 1 - STEP 1: DATA PREPROCESSING")
    logger.info("=" * 80)
    logger.info(f"Configuration: include_blank_reasons = {include_blank_reasons}")
    
    # Step 1.1: Data Loading
    df_raw = load_raw_data()
    
    # Step 1.2: Data Cleaning
    df_cleaned = clean_data(df_raw, include_blank_reasons=include_blank_reasons)
    
    # Step 1.3: Data Aggregation
    df_aggregated = aggregate_to_time_blocks(df_cleaned)
    
    # Step 1.4: Data Splitting
    splits = create_preprocessed_splits(df_aggregated)
    
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nFinal preprocessed dataset shape: {splits['preprocessed_df'].shape}")
    logger.info(f"Validation splits created: {len(splits['validation_splits'])}")
    logger.info(f"Final training set shape: {splits['final_train'].shape}")
    
    return splits


def run_step2(step1_results: Dict, include_external_covariates: bool = True) -> pd.DataFrame:
    """
    Execute Step 2: Feature Engineering Pipeline

    Args:
        step1_results: Results from run_step1() containing preprocessed data
        include_external_covariates: Whether to include weather/events/NWS data

    Returns:
        DataFrame with engineered features ready for modeling
    """
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE 1 - STEP 2: FEATURE ENGINEERING")
    logger.info("=" * 80)
    logger.info(f"Configuration: include_external_covariates = {include_external_covariates}")

    # Get preprocessed data from Step 1
    df = step1_results['preprocessed_df'].copy()
    logger.info(f"Starting with preprocessed data: {df.shape[0]:,} rows × {df.shape[1]} columns")

    # Step 2.1: Temporal Features
    logger.info("\n" + "-" * 60)
    logger.info("STEP 2.1: TEMPORAL FEATURES")
    logger.info("-" * 60)
    df = add_temporal_features(df)

    # Step 2.2: Lag Features
    logger.info("\n" + "-" * 60)
    logger.info("STEP 2.2: LAG FEATURES")
    logger.info("-" * 60)
    df = add_lag_features(df)

    # Step 2.3: Rolling Window Features
    logger.info("\n" + "-" * 60)
    logger.info("STEP 2.3: ROLLING WINDOW FEATURES")
    logger.info("-" * 60)
    df = add_rolling_features(df)

    # Step 2.4-2.5: Site-Specific and Block-Specific Features
    logger.info("\n" + "-" * 60)
    logger.info("STEP 2.4-2.5: SITE & BLOCK FEATURES")
    logger.info("-" * 60)
    df = add_site_block_features(df)

    # Step 2.6: Interaction Features
    logger.info("\n" + "-" * 60)
    logger.info("STEP 2.6: INTERACTION FEATURES")
    logger.info("-" * 60)
    df = add_interaction_features(df)

    # Step 2.7: External Covariates (Optional)
    if include_external_covariates:
        logger.info("\n" + "-" * 60)
        logger.info("STEP 2.7: EXTERNAL COVARIATES")
        logger.info("-" * 60)
        try:
            # Load site configuration
            site_meta = load_sites_config()
            # Use final training end date for external features
            train_end = step1_results['final_train']['Date'].max().strftime('%Y-%m-%d')
            df = join_external_covariates(df, site_meta=site_meta, train_end=train_end)
            logger.info("Successfully added external covariates")
        except Exception as e:
            logger.warning(f"Failed to add external covariates: {e}")
            logger.warning("Continuing without external covariates")

    # Step 2.8: Target-Derived Features
    logger.info("\n" + "-" * 60)
    logger.info("STEP 2.8: TARGET-DERIVED FEATURES")
    logger.info("-" * 60)
    df = add_target_derived_features(df)

    # Step 2.9: Feature Selection and Cleaning
    logger.info("\n" + "-" * 60)
    logger.info("STEP 2.9: FEATURE SELECTION & CLEANING")
    logger.info("-" * 60)
    df = select_and_clean_features(df)

    # Save feature-engineered data
    data_dir = get_data_dir()
    output_path = data_dir / "step2_feature_engineered.csv"
    df.to_csv(output_path, index=False)
    logger.info(f"\nSaved feature-engineered data to: {output_path}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 2 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info(f"\nFinal feature-engineered dataset shape: {df.shape}")
    logger.info(f"Feature columns: {len([col for col in df.columns if col not in ['Site', 'Date', 'Block', 'total_enc', 'admitted_enc', 'admit_rate']])}")

    return df


def run_step3(step1_results: Dict, feature_df: pd.DataFrame) -> Dict:
    """
    Execute Step 3: Model Training Pipeline

    Args:
        step1_results: Results from run_step1() containing validation splits
        feature_df: Feature-engineered DataFrame from run_step2()

    Returns:
        Dictionary containing trained models, validation results, and metrics
    """
    if not ML_MODULES_AVAILABLE:
        logger.error("ML modules not available. Cannot run Step 3 (training).")
        raise ImportError("Required ML libraries (XGBoost, sklearn) are not installed.")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE 1 - STEP 3: MODEL TRAINING")
    logger.info("=" * 80)

    # Get validation periods for training
    validation_periods = get_validation_periods()
    logger.info(f"Training on {len(validation_periods)} validation periods")

    # Prepare features and targets
    target_cols = ['total_enc', 'admitted_enc', 'admit_rate']
    feature_cols = [col for col in feature_df.columns if col not in target_cols + ['Site', 'Date', 'Block']]

    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Target columns: {target_cols}")
    logger.info(f"Dataset shape: {feature_df.shape}")

    # Initialize results storage
    results = {
        'validation_results': [],
        'models': {},
        'metrics': {},
        'feature_importance': {}
    }

    # 3.1 Model Selection and Configuration
    logger.info("\n" + "-" * 60)
    logger.info("STEP 3.1: MODEL SELECTION & CONFIGURATION")
    logger.info("-" * 60)

    # XGBoost hyperparameters (initial configuration)
    xgb_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'mae',
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 0.1,
        'random_state': 42,
        'verbosity': 1,
        'early_stopping_rounds': 20  # Move early_stopping_rounds to params
    }

    logger.info(f"XGBoost parameters: {xgb_params}")

    # 3.2 Model Architecture: Two-model approach
    logger.info("\n" + "-" * 60)
    logger.info("STEP 3.2: MODEL ARCHITECTURE - TWO-MODEL APPROACH")
    logger.info("-" * 60)

    # Model 1: Predict total encounters
    model_total = xgb.XGBRegressor(**xgb_params)
    logger.info("Model 1: XGBoost for total_enc prediction")

    # Model 2: Predict admission rate
    model_rate = xgb.XGBRegressor(**xgb_params)
    model_rate.set_params(objective='reg:squarederror')  # Ensure regression objective
    logger.info("Model 2: XGBoost for admit_rate prediction")

    # 3.3 Training Process
    logger.info("\n" + "-" * 60)
    logger.info("STEP 3.3: TRAINING PROCESS")
    logger.info("-" * 60)

    for period_idx, period in enumerate(validation_periods):
        logger.info(f"\n--- VALIDATION PERIOD {period_idx + 1} ---")
        logger.info(f"Train end: {period['train_end']}")
        logger.info(f"Test start: {period['test_start']}")
        logger.info(f"Test end: {period['test_end']}")

        # Get training data for this period
        train_mask = feature_df['Date'] <= period['train_end']
        test_mask = (feature_df['Date'] >= period['test_start']) & (feature_df['Date'] <= period['test_end'])

        train_data = feature_df[train_mask].copy()
        test_data = feature_df[test_mask].copy()

        logger.info(f"Training samples: {len(train_data)}")
        logger.info(f"Test samples: {len(test_data)}")

        if len(train_data) == 0 or len(test_data) == 0:
            logger.warning(f"Insufficient data for period {period_idx + 1}, skipping")
            continue

        # Prepare training features and targets
        X_train = train_data[feature_cols]
        y_train_total = train_data['total_enc']
        y_train_rate = train_data['admit_rate']

        X_test = test_data[feature_cols]
        y_test_total = test_data['total_enc']
        y_test_rate = test_data['admit_rate']
        y_test_admitted = test_data['admitted_enc']

        # Train Model 1: Total encounters
        logger.info("Training total_enc model...")
        eval_set_total = [(X_train, y_train_total), (X_test, y_test_total)]
        model_total.fit(
            X_train, y_train_total,
            eval_set=eval_set_total,
            verbose=False
        )

        # Train Model 2: Admission rate
        logger.info("Training admit_rate model...")
        eval_set_rate = [(X_train, y_train_rate), (X_test, y_test_rate)]
        model_rate.fit(
            X_train, y_train_rate,
            eval_set=eval_set_rate,
            verbose=False
        )

        # Generate predictions
        pred_total = model_total.predict(X_test)
        pred_rate = model_rate.predict(X_test)

        # Apply constraints
        pred_rate = np.clip(pred_rate, 0, 1)  # Admission rate between 0 and 1
        pred_admitted = pred_total * pred_rate
        pred_admitted = np.minimum(pred_admitted, pred_total)  # Cannot exceed total
        pred_admitted = np.maximum(pred_admitted, 0)  # Non-negative

        # Calculate metrics
        metrics = {
            'period': period_idx + 1,
            'train_samples': len(train_data),
            'test_samples': len(test_data),
            'total_enc_mae': mean_absolute_error(y_test_total, pred_total),
            'total_enc_rmse': np.sqrt(mean_squared_error(y_test_total, pred_total)),
            'admit_rate_mae': mean_absolute_error(y_test_rate, pred_rate),
            'admitted_enc_mae': mean_absolute_error(y_test_admitted, pred_admitted),
            'admitted_enc_rmse': np.sqrt(mean_squared_error(y_test_admitted, pred_admitted))
        }

        # Calculate WAPE (Weighted Absolute Percentage Error) for admitted encounters
        wape_numerator = np.sum(np.abs(y_test_admitted - pred_admitted))
        wape_denominator = np.sum(y_test_admitted)
        metrics['admitted_enc_wape'] = wape_numerator / wape_denominator if wape_denominator > 0 else np.nan

        logger.info(f"Period {period_idx + 1} metrics:")
        logger.info(f"  Total Enc MAE: {metrics['total_enc_mae']:.2f}")
        logger.info(f"  Admit Rate MAE: {metrics['admit_rate_mae']:.4f}")
        logger.info(f"  Admitted Enc MAE: {metrics['admitted_enc_mae']:.2f}")
        logger.info(f"  Admitted Enc WAPE: {metrics['admitted_enc_wape']:.4f}")

        results['validation_results'].append({
            'period': period_idx + 1,
            'metrics': metrics,
            'predictions': {
                'actual_total': y_test_total,
                'actual_admitted': y_test_admitted,
                'pred_total': pred_total,
                'pred_admitted': pred_admitted,
                'test_dates': test_data['Date'],
                'test_sites': test_data['Site'],
                'test_blocks': test_data['Block']
            }
        })

    # 3.4 Final Model Training
    logger.info("\n" + "-" * 60)
    logger.info("STEP 3.4: FINAL MODEL TRAINING")
    logger.info("-" * 60)

    # Train final models on all available data up to final_train
    final_train_data = feature_df[feature_df['Date'] <= step1_results['final_train']['Date'].max()].copy()

    logger.info(f"Final training on {len(final_train_data)} samples")
    logger.info(f"Final training period: {final_train_data['Date'].min()} to {final_train_data['Date'].max()}")

    X_final = final_train_data[feature_cols]
    y_final_total = final_train_data['total_enc']
    y_final_rate = final_train_data['admit_rate']

    # Create parameters without early stopping for final training (no validation set)
    final_params = xgb_params.copy()
    final_params.pop('early_stopping_rounds', None)

    # Train final models
    logger.info("Training final total_enc model...")
    model_total_final = xgb.XGBRegressor(**final_params)
    model_total_final.fit(X_final, y_final_total, verbose=False)

    logger.info("Training final admit_rate model...")
    model_rate_final = xgb.XGBRegressor(**final_params)
    model_rate_final.fit(X_final, y_final_rate, verbose=False)

    # Store final models
    results['models'] = {
        'total_enc_model': model_total_final,
        'admit_rate_model': model_rate_final
    }

    # 3.5 Feature Importance Analysis
    logger.info("\n" + "-" * 60)
    logger.info("STEP 3.5: FEATURE IMPORTANCE ANALYSIS")
    logger.info("-" * 60)

    # Get feature importance from final models
    total_importance = model_total_final.get_booster().get_score(importance_type='gain')
    rate_importance = model_rate_final.get_booster().get_score(importance_type='gain')

    results['feature_importance'] = {
        'total_enc_top_features': sorted(total_importance.items(), key=lambda x: x[1], reverse=True)[:20],
        'admit_rate_top_features': sorted(rate_importance.items(), key=lambda x: x[1], reverse=True)[:20]
    }

    logger.info("Top 5 features for total_enc model:")
    for feature, importance in results['feature_importance']['total_enc_top_features'][:5]:
        logger.info(f"  {feature}: {importance:.2f}")

    logger.info("Top 5 features for admit_rate model:")
    for feature, importance in results['feature_importance']['admit_rate_top_features'][:5]:
        logger.info(f"  {feature}: {importance:.2f}")

    # 3.6 Summary Statistics
    logger.info("\n" + "-" * 60)
    logger.info("STEP 3.6: TRAINING SUMMARY")
    logger.info("-" * 60)

    # Aggregate metrics across validation periods
    if results['validation_results']:
        avg_metrics = {}
        for key in results['validation_results'][0]['metrics'].keys():
            if key.startswith(('total_enc_', 'admit_rate_', 'admitted_enc_')):
                values = [r['metrics'][key] for r in results['validation_results']]
                avg_metrics[f'avg_{key}'] = np.mean(values)

        results['metrics'] = avg_metrics

        logger.info("Average validation metrics across all periods:")
        for key, value in avg_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

    # Save models
    data_dir = get_data_dir()
    model_dir = data_dir / "models"
    model_dir.mkdir(exist_ok=True)

    joblib.dump(model_total_final, model_dir / "total_enc_model.pkl")
    joblib.dump(model_rate_final, model_dir / "admit_rate_model.pkl")

    logger.info(f"\nModels saved to: {model_dir}")
    logger.info(f"  - total_enc_model.pkl")
    logger.info(f"  - admit_rate_model.pkl")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 3 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return results


def run_step4(step1_results: Dict, step3_results: Dict, feature_df: pd.DataFrame,
              n_trials: int = 50, timeout: int = 3600) -> Dict:
    """
    Execute Step 4: Hyperparameter Tuning Pipeline

    Uses Bayesian Optimization (Optuna) to tune XGBoost hyperparameters across
    validation periods for robust model selection.

    Args:
        step1_results: Results from run_step1() containing validation splits
        step3_results: Results from run_step3() containing baseline models
        feature_df: Feature-engineered DataFrame from run_step2()
        n_trials: Number of optimization trials per validation period
        timeout: Maximum time in seconds for optimization per period

    Returns:
        Dictionary containing tuned models, optimization results, and best parameters
    """
    if not TUNING_MODULES_AVAILABLE:
        logger.error("Tuning modules not available. Cannot run Step 4 (tuning).")
        raise ImportError("Required tuning libraries are not installed.")

    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE 1 - STEP 4: HYPERPARAMETER TUNING")
    logger.info("=" * 80)

    # Get validation periods for tuning
    validation_periods = get_validation_periods()
    logger.info(f"Tuning across {len(validation_periods)} validation periods")
    logger.info(f"Optimization trials per period: {n_trials}")
    logger.info(f"Timeout per period: {timeout} seconds")

    # Prepare features and targets
    target_cols = ['total_enc', 'admitted_enc', 'admit_rate']
    feature_cols = [col for col in feature_df.columns if col not in target_cols + ['Site', 'Date', 'Block']]

    # Initialize results storage
    tuning_results = {
        'optimization_results': [],
        'best_params_per_period': {},
        'aggregated_best_params': {},
        'tuned_models': {},
        'tuning_metrics': {}
    }

    # 4.1 Hyperparameter Search Space Definition
    logger.info("\n" + "-" * 60)
    logger.info("STEP 4.1: HYPERPARAMETER SEARCH SPACE")
    logger.info("-" * 60)

    def get_param_space(trial):
        """Define hyperparameter search space for Optuna optimization"""
        return {
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 0.0, 1.0),
            'reg_lambda': trial.suggest_float('reg_lambda', 0.0, 1.0),
            'gamma': trial.suggest_float('gamma', 0.0, 1.0)
        }

    def objective(trial, period_idx, target_type, X_train, y_train, X_val, y_val):
        """Objective function for Optuna optimization"""
        params = get_param_space(trial)
        params.update({
            'objective': 'reg:squarederror',
            'eval_metric': 'mae',
            'random_state': 42,
            'verbosity': 0,
            'early_stopping_rounds': 20
        })

        model = xgb.XGBRegressor(**params)

        if target_type == 'admit_rate':
            # For admission rate, ensure predictions stay within [0,1]
            params['objective'] = 'reg:squarederror'  # Keep squared error for rate

        eval_set = [(X_train, y_train), (X_val, y_val)]
        model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

        # Generate predictions
        pred_val = model.predict(X_val)

        if target_type == 'admit_rate':
            pred_val = np.clip(pred_val, 0, 1)

        # Calculate WAPE for primary metric (or MAE for secondary)
        if target_type == 'total_enc':
            # For total_enc, use MAE as it's a direct prediction
            score = mean_absolute_error(y_val, pred_val)
        elif target_type == 'admit_rate':
            # For admit_rate, use MAE on the rate
            score = mean_absolute_error(y_val, pred_val)
        else:
            # For derived admitted_enc, calculate WAPE
            if target_type == 'admitted_enc':
                score = np.sum(np.abs(y_val - pred_val)) / np.sum(y_val) if np.sum(y_val) > 0 else np.nan

        return score if not np.isnan(score) else float('inf')

    # 4.2 Tuning Process
    logger.info("\n" + "-" * 60)
    logger.info("STEP 4.2: TUNING PROCESS")
    logger.info("-" * 60)

    # Store best parameters for each period and target
    period_best_params = {}

    for period_idx, period in enumerate(validation_periods):
        logger.info(f"\n--- TUNING VALIDATION PERIOD {period_idx + 1} ---")
        logger.info(f"Train end: {period['train_end']}")
        logger.info(f"Val start: {period['test_start']}")
        logger.info(f"Val end: {period['test_end']}")

        # Get training and validation data for this period
        train_mask = feature_df['Date'] <= period['train_end']
        val_mask = (feature_df['Date'] >= period['test_start']) & (feature_df['Date'] <= period['test_end'])

        train_data = feature_df[train_mask].copy()
        val_data = feature_df[val_mask].copy()

        if len(train_data) == 0 or len(val_data) == 0:
            logger.warning(f"Insufficient data for period {period_idx + 1}, skipping tuning")
            continue

        X_train = train_data[feature_cols]
        X_val = val_data[feature_cols]

        period_best_params[period_idx] = {}

        # Tune hyperparameters for each target
        for target_type in ['total_enc', 'admit_rate']:
            logger.info(f"Tuning hyperparameters for {target_type}...")

            y_train = train_data[target_type]
            y_val = val_data[target_type]

            # Create Optuna study
            study_name = f"period_{period_idx + 1}_{target_type}"
            if OPTUNA_AVAILABLE:
                sampler = TPESampler(seed=42)
                pruner = MedianPruner()
                study = optuna.create_study(
                    direction='minimize',
                    sampler=sampler,
                    pruner=pruner,
                    study_name=study_name
                )

                # Run optimization
                study.optimize(
                    lambda trial: objective(trial, period_idx, target_type, X_train, y_train, X_val, y_val),
                    n_trials=n_trials,
                    timeout=timeout
                )

                best_params = study.best_params
                best_score = study.best_value

                logger.info(f"Best {target_type} score: {best_score:.4f}")
                logger.info(f"Best {target_type} params: {best_params}")

                # Store optimization results
                tuning_results['optimization_results'].append({
                    'period': period_idx + 1,
                    'target': target_type,
                    'study': study,
                    'best_params': best_params,
                    'best_score': best_score,
                    'n_trials': len(study.trials)
                })

            else:
                # Fallback to grid search if Optuna not available
                logger.warning("Using grid search fallback (limited search space)")

                # Define small grid for fallback
                param_grid = {
                    'n_estimators': [100, 200, 500],
                    'max_depth': [4, 6, 8],
                    'learning_rate': [0.05, 0.1, 0.2],
                    'subsample': [0.8, 0.9],
                    'colsample_bytree': [0.8, 0.9]
                }

                best_score = float('inf')
                best_params = None

                for params in ParameterGrid(param_grid):
                    params.update({
                        'objective': 'reg:squarederror',
                        'eval_metric': 'mae',
                        'random_state': 42,
                        'verbosity': 0,
                        'early_stopping_rounds': 20
                    })

                    model = xgb.XGBRegressor(**params)
                    eval_set = [(X_train, y_train), (X_val, y_val)]
                    model.fit(X_train, y_train, eval_set=eval_set, verbose=False)

                    pred_val = model.predict(X_val)
                    if target_type == 'admit_rate':
                        pred_val = np.clip(pred_val, 0, 1)

                    score = mean_absolute_error(y_val, pred_val)

                    if score < best_score:
                        best_score = score
                        best_params = params.copy()

                logger.info(f"Best {target_type} score (grid search): {best_score:.4f}")

                tuning_results['optimization_results'].append({
                    'period': period_idx + 1,
                    'target': target_type,
                    'best_params': best_params,
                    'best_score': best_score,
                    'method': 'grid_search'
                })

            period_best_params[period_idx][target_type] = best_params

        tuning_results['best_params_per_period'][period_idx + 1] = period_best_params[period_idx]

    # 4.3 Aggregate Results and Select Best Hyperparameters
    logger.info("\n" + "-" * 60)
    logger.info("STEP 4.3: AGGREGATE RESULTS & SELECT BEST PARAMETERS")
    logger.info("-" * 60)

    # Aggregate best parameters across periods for stability
    aggregated_params = {}

    for target_type in ['total_enc', 'admit_rate']:
        target_params = []

        for period_params in period_best_params.values():
            if target_type in period_params:
                target_params.append(period_params[target_type])

        if target_params:
            # For numerical parameters, take median
            # For categorical, take most frequent
            aggregated = {}
            for param in target_params[0].keys():
                values = [p[param] for p in target_params]
                if isinstance(values[0], (int, float)):
                    aggregated[param] = np.median(values)
                else:
                    # For non-numerical, take most common
                    from collections import Counter
                    aggregated[param] = Counter(values).most_common(1)[0][0]

            aggregated_params[target_type] = aggregated

            logger.info(f"Aggregated best params for {target_type}:")
            for param, value in aggregated.items():
                logger.info(f"  {param}: {value}")

    tuning_results['aggregated_best_params'] = aggregated_params

    # 4.4 Train Final Tuned Models
    logger.info("\n" + "-" * 60)
    logger.info("STEP 4.4: TRAIN FINAL TUNED MODELS")
    logger.info("-" * 60)

    # Train final models on all available data using best parameters
    final_train_data = feature_df[feature_df['Date'] <= step1_results['final_train']['Date'].max()].copy()

    logger.info(f"Training final tuned models on {len(final_train_data)} samples")
    logger.info(f"Final training period: {final_train_data['Date'].min()} to {final_train_data['Date'].max()}")

    X_final = final_train_data[feature_cols]

    tuned_models = {}

    for target_type in ['total_enc', 'admit_rate']:
        if target_type in aggregated_params:
            params = aggregated_params[target_type].copy()
            params.update({
                'objective': 'reg:squarederror',
                'eval_metric': 'mae',
                'random_state': 42,
                'verbosity': 1
            })

            y_final = final_train_data[target_type]

            logger.info(f"Training final tuned {target_type} model...")
            model = xgb.XGBRegressor(**params)
            model.fit(X_final, y_final, verbose=False)

            tuned_models[f'{target_type}_model'] = model

            # Get feature importance
            importance = model.get_booster().get_score(importance_type='gain')
            tuning_results[f'{target_type}_importance'] = sorted(
                importance.items(), key=lambda x: x[1], reverse=True
            )[:20]

    tuning_results['tuned_models'] = tuned_models

    # 4.5 Validation of Tuned Models
    logger.info("\n" + "-" * 60)
    logger.info("STEP 4.5: TUNED MODEL VALIDATION")
    logger.info("-" * 60)

    # Evaluate tuned models on validation periods
    tuned_validation_metrics = []

    for period_idx, period in enumerate(validation_periods):
        val_mask = (feature_df['Date'] >= period['test_start']) & (feature_df['Date'] <= period['test_end'])
        val_data = feature_df[val_mask].copy()

        if len(val_data) == 0:
            continue

        X_val = val_data[feature_cols]

        # Generate predictions with tuned models
        pred_total = tuned_models['total_enc_model'].predict(X_val)
        pred_rate = tuned_models['admit_rate_model'].predict(X_val)
        pred_rate = np.clip(pred_rate, 0, 1)
        pred_admitted = pred_total * pred_rate
        pred_admitted = np.minimum(pred_admitted, pred_total)

        # Calculate metrics
        metrics = {
            'period': period_idx + 1,
            'total_enc_mae': mean_absolute_error(val_data['total_enc'], pred_total),
            'admit_rate_mae': mean_absolute_error(val_data['admit_rate'], pred_rate),
            'admitted_enc_mae': mean_absolute_error(val_data['admitted_enc'], pred_admitted),
        }

        # WAPE for admitted encounters
        wape_num = np.sum(np.abs(val_data['admitted_enc'] - pred_admitted))
        wape_den = np.sum(val_data['admitted_enc'])
        metrics['admitted_enc_wape'] = wape_num / wape_den if wape_den > 0 else np.nan

        tuned_validation_metrics.append(metrics)

        logger.info(f"Period {period_idx + 1} tuned model metrics:")
        logger.info(f"  Total Enc MAE: {metrics['total_enc_mae']:.2f}")
        logger.info(f"  Admit Rate MAE: {metrics['admit_rate_mae']:.4f}")
        logger.info(f"  Admitted Enc MAE: {metrics['admitted_enc_mae']:.2f}")
        logger.info(f"  Admitted Enc WAPE: {metrics['admitted_enc_wape']:.4f}")

    # Calculate average metrics across periods
    if tuned_validation_metrics:
        avg_tuned_metrics = {}
        for key in tuned_validation_metrics[0].keys():
            if key != 'period':
                values = [m[key] for m in tuned_validation_metrics]
                avg_tuned_metrics[f'avg_tuned_{key}'] = np.mean(values)

        tuning_results['tuning_metrics'] = {
            'validation_metrics': tuned_validation_metrics,
            'avg_metrics': avg_tuned_metrics
        }

        logger.info("\nAverage tuned model metrics across all periods:")
        for key, value in avg_tuned_metrics.items():
            logger.info(f"  {key}: {value:.4f}")

        # Compare with baseline (step 3) metrics
        if 'metrics' in step3_results:
            logger.info("\nComparison with baseline (Step 3) metrics:")
            baseline_wape = step3_results['metrics'].get('avg_admitted_enc_wape', np.nan)
            tuned_wape = avg_tuned_metrics.get('avg_tuned_admitted_enc_wape', np.nan)

            if not np.isnan(baseline_wape) and not np.isnan(tuned_wape):
                improvement = (baseline_wape - tuned_wape) / baseline_wape * 100
                logger.info(f"  Baseline WAPE: {baseline_wape:.4f}")
                logger.info(f"  Tuned WAPE: {tuned_wape:.4f}")
                logger.info(f"  Improvement: {improvement:.1f}%")

    # Save tuned models
    data_dir = get_data_dir()
    model_dir = data_dir / "models"
    model_dir.mkdir(exist_ok=True)

    for model_name, model in tuned_models.items():
        model_path = model_dir / f"tuned_{model_name}.pkl"
        joblib.dump(model, model_path)
        logger.info(f"Saved tuned model: {model_path}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 4 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return tuning_results


def run_step5(step1_results: Dict, step3_results: Dict, step4_results: Dict = None,
              feature_df: pd.DataFrame = None, save_plots: bool = True) -> Dict:
    """
    Execute Step 5: Comprehensive Validation and Model Diagnostics

    Performs detailed validation analysis, error analysis, model diagnostics,
    and final model selection based on comprehensive metrics and stability checks.

    Args:
        step1_results: Results from run_step1() containing validation splits
        step3_results: Results from run_step3() containing baseline model training
        step4_results: Optional results from run_step4() containing tuned models
        feature_df: Optional feature-engineered DataFrame for additional analysis
        save_plots: Whether to save diagnostic plots to disk

    Returns:
        Dictionary containing comprehensive validation results, diagnostics, and final model selection
    """
    logger.info("\n" + "=" * 80)
    logger.info("PIPELINE 1 - STEP 5: COMPREHENSIVE VALIDATION & DIAGNOSTICS")
    logger.info("=" * 80)

    # Initialize results storage
    validation_results = {
        'validation_summary': {},
        'error_analysis': {},
        'model_diagnostics': {},
        'final_model_selection': {},
        'plots_saved': []
    }

    # 5.1 Validation Strategy
    logger.info("\n" + "-" * 60)
    logger.info("STEP 5.1: VALIDATION STRATEGY")
    logger.info("-" * 60)

    validation_periods = get_validation_periods()
    logger.info(f"Time-based cross-validation with {len(validation_periods)} periods")
    logger.info("No random shuffling - preserving temporal order")

    # Use tuned models if available, otherwise use baseline models
    if step4_results and step4_results.get('tuned_models'):
        logger.info("Using tuned models from Step 4 for validation")
        models_to_evaluate = {
            'tuned_total_enc': step4_results['tuned_models']['total_enc_model'],
            'tuned_admit_rate': step4_results['tuned_models']['admit_rate_model']
        }
    else:
        logger.info("Using baseline models from Step 3 for validation")
        models_to_evaluate = {
            'baseline_total_enc': step3_results['models']['total_enc_model'],
            'baseline_admit_rate': step3_results['models']['admit_rate_model']
        }

    # 5.2 Validation Metrics
    logger.info("\n" + "-" * 60)
    logger.info("STEP 5.2: COMPREHENSIVE VALIDATION METRICS")
    logger.info("-" * 60)

    # Define all metrics to calculate
    metrics_functions = {
        'mae': lambda y_true, y_pred: mean_absolute_error(y_true, y_pred),
        'rmse': lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        'mape': lambda y_true, y_pred: np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1e-6))) * 100,
        'r2': lambda y_true, y_pred: 1 - np.sum((y_true - y_pred)**2) / np.maximum(np.sum((y_true - np.mean(y_true))**2), 1e-6),
        'wape': lambda y_true, y_pred: np.sum(np.abs(y_true - y_pred)) / np.maximum(np.sum(y_true), 1e-6)
    }

    logger.info("Primary metrics: WAPE (competition metric), MAE, RMSE")
    logger.info("Secondary metrics: MAPE, R², Median Absolute Error")

    # 5.3 Validation Process
    logger.info("\n" + "-" * 60)
    logger.info("STEP 5.3: VALIDATION PROCESS")
    logger.info("-" * 60)

    # Prepare features
    target_cols = ['total_enc', 'admitted_enc', 'admit_rate']
    feature_cols = [col for col in feature_df.columns if col not in target_cols + ['Site', 'Date', 'Block']] if feature_df is not None else None

    detailed_results = []

    for period_idx, period in enumerate(validation_periods):
        logger.info(f"\n--- VALIDATION PERIOD {period_idx + 1} ---")
        logger.info(f"Train end: {period['train_end']}")
        logger.info(f"Test start: {period['test_start']}")
        logger.info(f"Test end: {period['test_end']}")

        # Get validation data
        if feature_df is not None:
            train_mask = feature_df['Date'] <= period['train_end']
            test_mask = (feature_df['Date'] >= period['test_start']) & (feature_df['Date'] <= period['test_end'])

            X_test = feature_df[test_mask][feature_cols] if feature_cols else None
            y_test_total = feature_df[test_mask]['total_enc'].values
            y_test_admitted = feature_df[test_mask]['admitted_enc'].values
            y_test_rate = feature_df[test_mask]['admit_rate'].values

            test_metadata = feature_df[test_mask][['Site', 'Date', 'Block']].copy()
        else:
            # Load test data from CSV files saved in step 1
            data_dir = get_data_dir()
            test_path = data_dir / f"step1_4_validation_period{period_idx + 1}_test.csv"
            test_data = pd.read_csv(test_path)
            test_data['Date'] = pd.to_datetime(test_data['Date'])

            y_test_total = test_data['total_enc'].values
            y_test_admitted = test_data['admitted_enc'].values
            y_test_rate = test_data['admit_rate'].values
            test_metadata = test_data[['Site', 'Date', 'Block']].copy()
            X_test = None  # No features available when feature_df is None

        logger.info(f"Test samples: {len(y_test_total)}")

        # Generate predictions
        if X_test is not None:
            pred_total = models_to_evaluate['tuned_total_enc' if 'tuned' in models_to_evaluate else 'baseline_total_enc'].predict(X_test)
            pred_rate = models_to_evaluate['tuned_admit_rate' if 'tuned' in models_to_evaluate else 'baseline_admit_rate'].predict(X_test)
        else:
            # Use pre-computed predictions from step3 if available
            step3_period = step3_results['validation_results'][period_idx]
            pred_total = step3_period['predictions']['pred_total']
            pred_rate = np.full_like(pred_total, np.mean(y_test_rate))  # Simple baseline for rate

        # Apply constraints
        pred_rate = np.clip(pred_rate, 0, 1)
        pred_admitted = pred_total * pred_rate
        pred_admitted = np.minimum(pred_admitted, pred_total)
        pred_admitted = np.maximum(pred_admitted, 0)

        # Calculate comprehensive metrics
        period_metrics = {
            'period': period_idx + 1,
            'samples': len(y_test_total)
        }

        # Total encounters metrics
        for metric_name, metric_func in metrics_functions.items():
            period_metrics[f'total_enc_{metric_name}'] = metric_func(y_test_total, pred_total)

        # Admission rate metrics
        for metric_name, metric_func in metrics_functions.items():
            period_metrics[f'admit_rate_{metric_name}'] = metric_func(y_test_rate, pred_rate)

        # Admitted encounters metrics (primary target)
        for metric_name, metric_func in metrics_functions.items():
            period_metrics[f'admitted_enc_{metric_name}'] = metric_func(y_test_admitted, pred_admitted)

        # Store predictions and metadata
        period_result = {
            'metrics': period_metrics,
            'predictions': {
                'actual_total': y_test_total,
                'actual_admitted': y_test_admitted,
                'actual_rate': y_test_rate,
                'pred_total': pred_total,
                'pred_admitted': pred_admitted,
                'pred_rate': pred_rate
            },
            'metadata': test_metadata
        }

        detailed_results.append(period_result)

        logger.info(f"Period {period_idx + 1} results:")
        logger.info(".2f")
        logger.info(".4f")

    # 5.4 Error Analysis
    logger.info("\n" + "-" * 60)
    logger.info("STEP 5.4: ERROR ANALYSIS")
    logger.info("-" * 60)

    error_analysis = {}

    # Aggregate predictions across all periods
    all_actual = np.concatenate([r['predictions']['actual_admitted'] for r in detailed_results])
    all_predicted = np.concatenate([r['predictions']['pred_admitted'] for r in detailed_results])
    all_metadata = pd.concat([r['metadata'] for r in detailed_results])

    # Calculate overall errors
    errors = all_actual - all_predicted
    abs_errors = np.abs(errors)
    pct_errors = abs_errors / np.maximum(all_actual, 1e-6)

    error_analysis['overall'] = {
        'mean_error': np.mean(errors),
        'mean_abs_error': np.mean(abs_errors),
        'median_abs_error': np.median(abs_errors),
        'rmse': np.sqrt(np.mean(errors**2)),
        'mean_pct_error': np.mean(pct_errors) * 100,
        'median_pct_error': np.median(pct_errors) * 100
    }

    # Site-specific error analysis
    site_errors = {}
    for site in all_metadata['Site'].unique():
        site_mask = all_metadata['Site'] == site
        site_actual = all_actual[site_mask]
        site_predicted = all_predicted[site_mask]

        site_errors[site] = {
            'samples': len(site_actual),
            'mae': mean_absolute_error(site_actual, site_predicted),
            'wape': np.sum(np.abs(site_actual - site_predicted)) / np.maximum(np.sum(site_actual), 1e-6),
            'mean_pct_error': np.mean(np.abs(site_actual - site_predicted) / np.maximum(site_actual, 1e-6)) * 100
        }

    error_analysis['by_site'] = site_errors

    # Block-specific error analysis
    block_errors = {}
    for block in all_metadata['Block'].unique():
        block_mask = all_metadata['Block'] == block
        block_actual = all_actual[block_mask]
        block_predicted = all_predicted[block_mask]

        block_errors[block] = {
            'samples': len(block_actual),
            'mae': mean_absolute_error(block_actual, block_predicted),
            'wape': np.sum(np.abs(block_actual - block_predicted)) / np.maximum(np.sum(block_actual), 1e-6),
            'mean_pct_error': np.mean(np.abs(block_actual - block_predicted) / np.maximum(block_actual, 1e-6)) * 100
        }

    error_analysis['by_block'] = block_errors

    # Time-based error patterns
    all_metadata_with_errors = all_metadata.copy()
    all_metadata_with_errors['error'] = errors
    all_metadata_with_errors['abs_error'] = abs_errors
    all_metadata_with_errors['pct_error'] = pct_errors

    # Monthly error patterns (convert periods to strings to avoid tuple keys)
    monthly_errors = all_metadata_with_errors.groupby(all_metadata_with_errors['Date'].dt.to_period('M')).agg({
        'error': ['mean', 'std', 'count'],
        'abs_error': 'mean',
        'pct_error': 'mean'
    }).round(4)

    # Convert PeriodIndex to string keys
    monthly_dict = {}
    for period, row in monthly_errors.iterrows():
        period_str = str(period)
        monthly_dict[period_str] = {
            'error_mean': row[('error', 'mean')],
            'error_std': row[('error', 'std')],
            'count': row[('error', 'count')],
            'abs_error_mean': row[('abs_error', 'mean')],
            'pct_error_mean': row[('pct_error', 'mean')]
        }

    error_analysis['by_month'] = monthly_dict

    logger.info("Overall error analysis:")
    logger.info(".2f")
    logger.info(".2f")
    logger.info(".1f")

    logger.info("\nSite-specific errors:")
    for site, metrics in site_errors.items():
        logger.info(".2f")

    logger.info("\nBlock-specific errors:")
    for block, metrics in block_errors.items():
        logger.info(".2f")

    # 5.5 Model Diagnostics
    logger.info("\n" + "-" * 60)
    logger.info("STEP 5.5: MODEL DIAGNOSTICS")
    logger.info("-" * 60)

    diagnostics = {}

    # Feature importance analysis (if available)
    if step3_results.get('feature_importance'):
        diagnostics['feature_importance'] = step3_results['feature_importance']

        logger.info("Top 10 features for total_enc model:")
        for i, (feature, importance) in enumerate(diagnostics['feature_importance']['total_enc_top_features'][:10]):
            logger.info(f"  {i+1:2d}. {feature}: {importance:.4f}")

    # Prediction quality checks
    diagnostics['quality_checks'] = {
        'negative_predictions_total': np.sum(all_predicted < 0),
        'negative_predictions_admitted': np.sum(pred_admitted < 0),
        'admitted_exceeds_total': np.sum(pred_admitted > pred_total),
        'unrealistic_spikes': len([x for x in all_predicted if x > np.percentile(all_actual, 99) * 2]),
        'zero_predictions': np.sum(all_predicted == 0)
    }

    logger.info("Prediction quality checks:")
    for check, count in diagnostics['quality_checks'].items():
        logger.info(f"  {check}: {count}")

    # Stability analysis across validation periods
    period_wape = [r['metrics']['admitted_enc_wape'] for r in detailed_results]
    diagnostics['stability'] = {
        'wape_mean': np.mean(period_wape),
        'wape_std': np.std(period_wape),
        'wape_cv': np.std(period_wape) / np.mean(period_wape) if np.mean(period_wape) > 0 else np.nan,
        'period_wape': period_wape
    }

    logger.info("Model stability across validation periods:")
    logger.info(".4f")
    logger.info(".4f")

    # Generate diagnostic plots if plotting is available
    if PLOTTING_AVAILABLE and save_plots:
        logger.info("\nGenerating diagnostic plots...")

        plots_dir = get_data_dir() / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Error distribution plot
        fig = px.histogram(x=errors, nbins=50, title="Prediction Error Distribution")
        fig.update_layout(xaxis_title="Error (Actual - Predicted)", yaxis_title="Frequency")
        error_plot_path = plots_dir / "error_distribution.html"
        fig.write_html(str(error_plot_path))
        validation_results['plots_saved'].append(str(error_plot_path))

        # Actual vs Predicted scatter plot
        fig = px.scatter(x=all_actual, y=all_predicted, opacity=0.6,
                        title="Actual vs Predicted Admitted Encounters")
        fig.update_layout(xaxis_title="Actual", yaxis_title="Predicted")
        # Add 45-degree line
        fig.add_trace(go.Scatter(x=[all_actual.min(), all_actual.max()],
                               y=[all_actual.min(), all_actual.max()],
                               mode='lines', name='Perfect Prediction',
                               line=dict(color='red', dash='dash')))
        scatter_plot_path = plots_dir / "actual_vs_predicted.html"
        fig.write_html(str(scatter_plot_path))
        validation_results['plots_saved'].append(str(scatter_plot_path))

        # Time series of errors
        error_df = pd.DataFrame({
            'date': all_metadata['Date'],
            'error': errors,
            'abs_error': abs_errors
        }).groupby('date').mean().reset_index()

        fig = px.line(error_df, x='date', y='abs_error',
                     title="Mean Absolute Error Over Time")
        fig.update_layout(xaxis_title="Date", yaxis_title="Mean Absolute Error")
        time_series_plot_path = plots_dir / "error_time_series.html"
        fig.write_html(str(time_series_plot_path))
        validation_results['plots_saved'].append(str(time_series_plot_path))

        logger.info(f"Saved {len(validation_results['plots_saved'])} diagnostic plots to {plots_dir}")

    # 5.6 Final Model Selection
    logger.info("\n" + "-" * 60)
    logger.info("STEP 5.6: FINAL MODEL SELECTION")
    logger.info("-" * 60)

    # Calculate average metrics across all validation periods
    avg_metrics = {}
    metric_keys = detailed_results[0]['metrics'].keys()
    for key in metric_keys:
        if key != 'period' and key != 'samples':
            values = [r['metrics'][key] for r in detailed_results]
            avg_metrics[f'avg_{key}'] = np.mean(values)

    validation_results['validation_summary'] = {
        'average_metrics': avg_metrics,
        'detailed_period_results': detailed_results,
        'models_evaluated': list(models_to_evaluate.keys())
    }

    validation_results['error_analysis'] = error_analysis
    validation_results['model_diagnostics'] = diagnostics

    # Final model selection criteria
    final_selection = {
        'primary_metric': 'avg_admitted_enc_wape',
        'primary_score': avg_metrics.get('avg_admitted_enc_wape', np.nan),
        'secondary_metrics': {
            'mae': avg_metrics.get('avg_admitted_enc_mae', np.nan),
            'rmse': avg_metrics.get('avg_admitted_enc_rmse', np.nan)
        },
        'stability_score': diagnostics['stability']['wape_cv'],
        'recommendation': 'selected' if diagnostics['stability']['wape_cv'] < 0.2 else 'needs_improvement',
        'model_type': 'tuned' if step4_results else 'baseline'
    }

    validation_results['final_model_selection'] = final_selection

    logger.info("Final model selection:")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(".4f")
    logger.info(f"Model type: {final_selection['model_type']}")
    logger.info(f"Stability (CV): {final_selection['stability_score']:.4f}")
    logger.info(f"Recommendation: {final_selection['recommendation']}")

    # Save validation results
    data_dir = get_data_dir()
    results_dir = data_dir / "validation_results"
    results_dir.mkdir(exist_ok=True)

    import json
    results_file = results_dir / "step5_validation_results.json"

    # Convert numpy types to native Python types for JSON serialization
    def convert_numpy_types(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Period):
            return str(obj)
        elif isinstance(obj, tuple):
            return tuple(convert_numpy_types(item) for item in obj)
        elif isinstance(obj, dict):
            return {convert_numpy_types(key): convert_numpy_types(value) for key, value in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy_types(item) for item in obj]
        else:
            return obj

    with open(results_file, 'w') as f:
        json.dump(convert_numpy_types(validation_results), f, indent=2, default=str)

    logger.info(f"\nSaved comprehensive validation results to: {results_file}")

    logger.info("\n" + "=" * 80)
    logger.info("STEP 5 COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)

    return validation_results


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Pipeline 1: Direct Volume Prediction (Steps 1-5)')
    parser.add_argument(
        '--include-blank-reasons',
        action='store_true',
        default=True,
        help='Include records with blank/missing REASON_VISIT_NAME (default: True)'
    )
    parser.add_argument(
        '--exclude-blank-reasons',
        action='store_true',
        help='Exclude records with blank/missing REASON_VISIT_NAME'
    )
    parser.add_argument(
        '--skip-step2',
        action='store_true',
        help='Skip Step 2 (feature engineering) and only run Step 1'
    )
    parser.add_argument(
        '--skip-external-covariates',
        action='store_true',
        help='Skip external covariates in Step 2'
    )
    parser.add_argument(
        '--skip-step3',
        action='store_true',
        help='Skip Step 3 (model training) and only run Steps 1-2'
    )
    parser.add_argument(
        '--skip-step4',
        action='store_true',
        help='Skip Step 4 (hyperparameter tuning) and only run Steps 1-3'
    )
    parser.add_argument(
        '--tuning-trials',
        type=int,
        default=50,
        help='Number of optimization trials per validation period (default: 50)'
    )
    parser.add_argument(
        '--tuning-timeout',
        type=int,
        default=3600,
        help='Maximum time in seconds for optimization per period (default: 3600)'
    )
    parser.add_argument(
        '--run-step5',
        action='store_true',
        help='Run Step 5 (comprehensive validation and diagnostics)'
    )
    parser.add_argument(
        '--skip-step5-plots',
        action='store_true',
        help='Skip saving diagnostic plots in Step 5'
    )

    args = parser.parse_args()

    # Determine configuration based on arguments
    include_blank = not args.exclude_blank_reasons
    include_external = not args.skip_external_covariates

    # Initialize variables
    feature_df = None
    training_results = None
    tuning_results = None

    # Run Step 1 preprocessing
    logger.info("Running Pipeline 1...")
    results = run_step1(include_blank_reasons=include_blank)

    # Print Step 1 summary
    print("\n" + "=" * 80)
    print("STEP 1 EXECUTION SUMMARY")
    print("=" * 80)
    print(f"\nConfiguration: include_blank_reasons = {include_blank}")
    print(f"Preprocessed dataset shape: {results['preprocessed_df'].shape}")
    print(f"Validation periods: {len(results['validation_splits'])}")
    print(f"Final training set shape: {results['final_train'].shape}")
    print(f"Test period: {results['test_period']['description']}")

    # Run Step 2 if not skipped
    if not args.skip_step2:
        print("\n" + "=" * 80)
        print("STARTING STEP 2: FEATURE ENGINEERING")
        print("=" * 80)

        feature_df = run_step2(results, include_external_covariates=include_external)

        # Print Step 2 summary
        print("\n" + "=" * 80)
        print("STEP 2 EXECUTION SUMMARY")
        print("=" * 80)
        print(f"\nConfiguration: include_external_covariates = {include_external}")
        print(f"Feature-engineered dataset shape: {feature_df.shape}")
        feature_cols = len([col for col in feature_df.columns if col not in ['Site', 'Date', 'Block', 'total_enc', 'admitted_enc', 'admit_rate']])
        print(f"Feature columns created: {feature_cols}")

        # Run Step 3 if not skipped
        training_results = None
        if not args.skip_step3:
            print("\n" + "=" * 80)
            print("STARTING STEP 3: MODEL TRAINING")
            print("=" * 80)

            training_results = run_step3(results, feature_df)

            # Print Step 3 summary
            print("\n" + "=" * 80)
            print("STEP 3 EXECUTION SUMMARY")
            print("=" * 80)
            print(f"\nValidation periods trained: {len(training_results['validation_results'])}")
            if training_results['metrics']:
                print("Average validation metrics:")
                for key, value in training_results['metrics'].items():
                    print(f"  {key}: {value:.4f}")
            print(f"Models saved: {len(training_results['models'])}")
            print(f"Feature importance analyzed for {len(training_results['feature_importance'])} targets")

            # Run Step 4 if not skipped and Step 3 was executed
            if not args.skip_step4:
                print("\n" + "=" * 80)
                print("STARTING STEP 4: HYPERPARAMETER TUNING")
                print("=" * 80)

                tuning_results = run_step4(
                    results,
                    training_results,
                    feature_df,
                    n_trials=args.tuning_trials,
                    timeout=args.tuning_timeout
                )

                # Print Step 4 summary
                print("\n" + "=" * 80)
                print("STEP 4 EXECUTION SUMMARY")
                print("=" * 80)
                print(f"\nPeriods tuned: {len(tuning_results['best_params_per_period'])}")
                print(f"Optimization trials: {len(tuning_results['optimization_results'])}")
                if tuning_results.get('tuning_metrics', {}).get('avg_metrics'):
                    print("Average tuned model metrics:")
                    for key, value in tuning_results['tuning_metrics']['avg_metrics'].items():
                        print(f"  {key}: {value:.4f}")
                print(f"Tuned models saved: {len(tuning_results['tuned_models'])}")

            else:
                print("\nStep 4 (hyperparameter tuning) skipped as requested.")


        else:
            print("\nStep 3 (model training) skipped as requested.")

    # Run Step 5 if requested
    if args.run_step5:
        # Load existing models if step 3 was skipped
        if training_results is None:
            logger.info("Loading existing models for Step 5 validation...")
            try:
                import joblib
                data_dir = get_data_dir()
                model_dir = data_dir / "models"

                total_enc_model = joblib.load(model_dir / "total_enc_model.pkl")
                admit_rate_model = joblib.load(model_dir / "admit_rate_model.pkl")

                # Create a mock training_results structure
                training_results = {
                    'models': {
                        'total_enc_model': total_enc_model,
                        'admit_rate_model': admit_rate_model
                    },
                    'validation_results': [],  # Will be populated in step 5
                    'feature_importance': {}   # Not available from saved models
                }
                logger.info("Successfully loaded existing models")
            except Exception as e:
                logger.error(f"Failed to load existing models: {e}")
                logger.error("Cannot run Step 5 without models")
                training_results = None

        # Load feature data if step 2 was skipped but needed for step 5
        step5_feature_df = feature_df
        if step5_feature_df is None:
            logger.info("Loading feature-engineered data for Step 5 validation...")
            try:
                data_dir = get_data_dir()
                feature_path = data_dir / "step2_feature_engineered.csv"
                step5_feature_df = pd.read_csv(feature_path)
                step5_feature_df['Date'] = pd.to_datetime(step5_feature_df['Date'])
                logger.info(f"Successfully loaded feature data: {step5_feature_df.shape}")
            except Exception as e:
                logger.error(f"Failed to load feature data: {e}")
                step5_feature_df = None

        if training_results is not None and step5_feature_df is not None:
            print("\n" + "=" * 80)
            print("STARTING STEP 5: COMPREHENSIVE VALIDATION & DIAGNOSTICS")
            print("=" * 80)

            validation_results = run_step5(
                results,
                training_results,
                tuning_results,
                step5_feature_df,
                save_plots=not args.skip_step5_plots
            )

            # Print Step 5 summary
            print("\n" + "=" * 80)
            print("STEP 5 EXECUTION SUMMARY")
            print("=" * 80)
            if validation_results.get('validation_summary', {}).get('average_metrics'):
                print("Average validation metrics:")
                for key, value in validation_results['validation_summary']['average_metrics'].items():
                    if 'wape' in key or 'mae' in key:
                        print(f"  {key}: {value:.4f}")
            if validation_results.get('final_model_selection'):
                print("\nFinal model selection:")
                selection = validation_results['final_model_selection']
                print(f"  Primary WAPE: {selection['primary_score']:.4f}")
                print(f"  Stability (CV): {selection['stability_score']:.4f}")
                print(f"  Recommendation: {selection['recommendation']}")
            print(f"Diagnostic plots saved: {len(validation_results.get('plots_saved', []))}")
            print(f"Results saved to: baseline_model/pipeline1/data/validation_results/")
        else:
            print("\nStep 5 (validation) skipped: models or feature data not available")

    print("\n" + "=" * 80)
    print("PIPELINE 1 COMPLETE")
    print("=" * 80)
