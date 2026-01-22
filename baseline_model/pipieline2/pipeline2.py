"""
PIPELINE 2: CATEGORY-BASED PREDICTION WITH TEMPORAL SPLITTING
==============================================================

This pipeline implements the comprehensive approach from Pipeline 1 but with
category-based splitting. The training data is split into subsets by reason
category (e.g., Injury/Trauma, Cardiovascular, Respiratory) to allow models
to learn category-specific cyclical patterns. Predictions are then aggregated
back to the (Site, Date, Block) level.

Key Features:
- Full feature engineering from Pipeline 1
- Category-based data splitting
- Separate models per category
- Aggregated predictions
- Comprehensive validation
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

# Add parent directories to path
REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "src"))
sys.path.insert(0, str(REPO_ROOT))

from baseline_model.data_ingestion.loader import (
    load_dataset,
    create_validation_splits,
    aggregate_to_blocks
)
from dsu_forecast.features.calendar import add_calendar_features
from dsu_forecast.modeling.features import add_lag_features, add_rolling_mean_features
from dsu_forecast.modeling.metrics import mae, wape


def load_category_mapping() -> Dict[str, str]:
    """Load reason to category mapping from JSON file."""
    mapping_path = REPO_ROOT / "reason_categories.json"
    if not mapping_path.exists():
        raise FileNotFoundError(f"Category mapping not found at {mapping_path}")
    with open(mapping_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def validate_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    STEP 1.2: Data Cleaning and Validation
    """
    df = df.copy()
    
    # Validate data consistency
    if 'ED Enc Admitted' in df.columns and 'ED Enc' in df.columns:
        # Ensure ED Enc Admitted <= ED Enc
        invalid = df['ED Enc Admitted'] > df['ED Enc']
        if invalid.any():
            print(f"  Warning: {invalid.sum()} records with ED Enc Admitted > ED Enc, fixing...")
            df.loc[invalid, 'ED Enc Admitted'] = df.loc[invalid, 'ED Enc']
    
    # Check for negative values
    for col in ['ED Enc', 'ED Enc Admitted']:
        if col in df.columns:
            negative = df[col] < 0
            if negative.any():
                print(f"  Warning: {negative.sum()} negative values in {col}, setting to 0")
                df.loc[negative, col] = 0
    
    # Validate Site values
    if 'Site' in df.columns:
        valid_sites = {'A', 'B', 'C', 'D'}
        invalid_sites = ~df['Site'].isin(valid_sites)
        if invalid_sites.any():
            print(f"  Warning: {invalid_sites.sum()} records with invalid Site, dropping...")
            df = df[~invalid_sites]
    
    return df


def aggregate_to_blocks_with_category(
    df: pd.DataFrame,
    category_mapping: Dict[str, str]
) -> pd.DataFrame:
    """
    Aggregate hourly data to 6-hour time blocks while preserving category information.
    
    Args:
        df: DataFrame with Hour and REASON_VISIT_NAME columns
        category_mapping: Dict mapping reason names to categories
    
    Returns:
        Aggregated dataframe with Block and Category columns
    """
    df = df.copy()
    
    # Add category before aggregation
    df['Category'] = df['REASON_VISIT_NAME'].map(category_mapping).fillna('Other/Unspecified')
    df['Block'] = (df['Hour'] // 6).astype(int)
    
    # Aggregate by Site, Date, Block, Category
    agg_df = df.groupby(['Site', 'Date', 'Block', 'Category'], as_index=False).agg({
        'ED Enc': 'sum',
        'ED Enc Admitted': 'sum'
    })
    
    # Calculate admission rate
    agg_df['admit_rate'] = (
        agg_df['ED Enc Admitted'] / agg_df['ED Enc'].clip(lower=1.0)
    ).clip(0, 1)
    
    return agg_df


def create_category_subsets(
    train_df: pd.DataFrame,
    category_mapping: Dict[str, str],
    min_samples_per_category: int = 100
) -> Dict[str, pd.DataFrame]:
    """
    Split training data into subsets by category.
    
    Args:
        train_df: Training dataframe with REASON_VISIT_NAME column
        category_mapping: Dict mapping reason names to categories
        min_samples_per_category: Minimum samples required to create a separate subset
    
    Returns:
        Dict mapping category names to dataframes
    """
    # train_df should already have Category column from aggregation
    if 'Category' not in train_df.columns:
        if 'REASON_VISIT_NAME' in train_df.columns:
            train_df['Category'] = train_df['REASON_VISIT_NAME'].map(
                category_mapping
            ).fillna('Other/Unspecified')
        else:
            # If no REASON_VISIT_NAME, assume all is Other/Unspecified
            train_df['Category'] = 'Other/Unspecified'
    
    # If already aggregated with category, use as-is
    if 'Category' in train_df.columns and 'Block' in train_df.columns:
        category_data = train_df.copy()
        if 'admit_rate' not in category_data.columns:
            category_data['admit_rate'] = (
                category_data['ED Enc Admitted'] / 
                category_data['ED Enc'].clip(lower=1.0)
            ).clip(0, 1)
    else:
        # Fallback: aggregate by category if needed
        category_data = train_df.groupby(
            ['Site', 'Date', 'Block', 'Category'],
            as_index=False
        ).agg({
            'ED Enc': 'sum',
            'ED Enc Admitted': 'sum'
        })
        category_data['admit_rate'] = (
            category_data['ED Enc Admitted'] / 
            category_data['ED Enc'].clip(lower=1.0)
        ).clip(0, 1)
    
    # Split into subsets
    subsets = {}
    category_counts = category_data.groupby('Category').size()
    
    # Track small categories to merge
    small_categories = []
    
    for category in category_data['Category'].unique():
        count = category_counts.get(category, 0)
        if count >= min_samples_per_category:
            subsets[category] = category_data[
                category_data['Category'] == category
            ].copy()
        else:
            small_categories.append(category)
    
    # Merge small categories into "Other/Unspecified"
    if small_categories:
        if 'Other/Unspecified' in subsets:
            # Add small categories to existing Other/Unspecified
            small_data = category_data[
                category_data['Category'].isin(small_categories)
            ].copy()
            small_data['Category'] = 'Other/Unspecified'
            subsets['Other/Unspecified'] = pd.concat([
                subsets['Other/Unspecified'],
                small_data
            ], ignore_index=True)
        else:
            # Create new Other/Unspecified from small categories
            small_data = category_data[
                category_data['Category'].isin(small_categories)
            ].copy()
            small_data['Category'] = 'Other/Unspecified'
            subsets['Other/Unspecified'] = small_data
    
    return subsets


def add_comprehensive_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    STEP 2: Comprehensive Feature Engineering
    
    Adds all features from Pipeline 1:
    - Temporal features (calendar, cyclical encoding)
    - Lag features
    - Rolling window features
    - Site/Block specific features
    - Interaction features
    """
    df = df.copy()
    df['Date'] = pd.to_datetime(df['Date'])
    
    # 2.1 Temporal Features - Calendar features
    df = add_calendar_features(df, date_col='Date')
    
    # Additional calendar features
    df['quarter'] = df['Date'].dt.quarter
    df['week_of_year'] = df['Date'].dt.isocalendar().week
    df['year'] = df['Date'].dt.year
    
    # Cyclical encoding for month (already in calendar features as doy_sin/cos)
    # Add month-specific cyclical encoding
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    
    # Cyclical encoding for day of week
    df['dow_sin'] = np.sin(2 * np.pi * df['dow'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['dow'] / 7)
    
    # Cyclical encoding for block
    df['block_sin'] = np.sin(2 * np.pi * df['Block'] / 4)
    df['block_cos'] = np.cos(2 * np.pi * df['Block'] / 4)
    
    # Quarter cyclical encoding
    df['quarter_sin'] = np.sin(2 * np.pi * df['quarter'] / 4)
    df['quarter_cos'] = np.cos(2 * np.pi * df['quarter'] / 4)
    
    # 2.2 Lag Features - grouped by (Site, Block)
    target_cols = ['ED Enc', 'ED Enc Admitted', 'admit_rate']
    
    # Ensure admit_rate exists
    if 'admit_rate' not in df.columns:
        df['admit_rate'] = (
            df['ED Enc Admitted'] / df['ED Enc'].clip(lower=1.0)
        ).clip(0, 1)
    
    df = add_lag_features(
        df,
        group_cols=['Site', 'Block'],
        sort_cols=['Date'],
        target_cols=target_cols,
        lags=[1, 7, 14, 28, 30, 90]
    )
    
    # 2.3 Rolling Window Features
    df = add_rolling_mean_features(
        df,
        group_cols=['Site', 'Block'],
        sort_cols=['Date'],
        target_cols=target_cols,
        windows=[7, 14, 28, 90]
    )
    
    # Additional rolling statistics
    for window in [7, 14, 28]:
        for target in ['ED Enc', 'ED Enc Admitted']:
            if target in df.columns:
                # Rolling median
                df[f'{target}_rmedian_{window}'] = (
                    df.groupby(['Site', 'Block'])[target]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).median())
                )
                # Rolling std
                df[f'{target}_rstd_{window}'] = (
                    df.groupby(['Site', 'Block'])[target]
                    .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std())
                )
    
    # 2.4 Site-Specific Features
    site_encoder = LabelEncoder()
    df['site_encoded'] = site_encoder.fit_transform(df['Site'])
    
    # Site-specific historical averages (calculated per category later)
    # For now, add site as one-hot would be too many features
    
    # 2.5 Block-Specific Features
    # Block is already numeric (0-3), but add block-specific patterns
    df['is_morning'] = (df['Block'] == 1).astype(int)
    df['is_afternoon'] = (df['Block'] == 2).astype(int)
    df['is_evening'] = (df['Block'] == 3).astype(int)
    df['is_night'] = (df['Block'] == 0).astype(int)
    
    # 2.6 Interaction Features
    df['site_block'] = df['site_encoded'] * 10 + df['Block']
    df['site_dow'] = df['site_encoded'] * 10 + df['dow']
    df['block_dow'] = df['Block'] * 10 + df['dow']
    df['site_month'] = df['site_encoded'] * 100 + df['month']
    
    # 2.8 Target-Derived Features
    # Volume trends (slope over recent periods)
    for window in [7, 28]:
        df[f'volume_trend_{window}'] = (
            df.groupby(['Site', 'Block'])['ED Enc']
            .transform(lambda x: x.shift(1).rolling(window, min_periods=2).apply(
                lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) >= 2 else 0
            ))
        )
    
    # Volatility (coefficient of variation)
    for window in [7, 28]:
        df[f'volume_cv_{window}'] = (
            df.groupby(['Site', 'Block'])['ED Enc']
            .transform(lambda x: (
                x.shift(1).rolling(window, min_periods=2).std() / 
                x.shift(1).rolling(window, min_periods=2).mean().clip(lower=1.0)
            ))
        )
    
    # Collect feature columns (numeric, excluding targets and identifiers)
    exclude_cols = {
        'ED Enc', 'ED Enc Admitted', 'admit_rate',
        'Site', 'Date', 'Block', 'Category', 'REASON_VISIT_NAME'
    }
    
    feature_cols = [
        c for c in df.columns 
        if c not in exclude_cols 
        and pd.api.types.is_numeric_dtype(df[c])
        and df[c].notna().any()
    ]
    
    return df, feature_cols


def train_category_models(
    train_subset: pd.DataFrame,
    feature_cols: List[str],
    hyperparams: Optional[Dict] = None
) -> Dict[str, xgb.XGBRegressor]:
    """
    STEP 3: Train models for a specific category.
    
    Uses two-model approach:
    - Model 1: Predict total_enc (Poisson regression)
    - Model 2: Predict admit_rate (squared error)
    """
    train_subset = train_subset.copy()
    
    # Prepare training data
    train_subset = train_subset.dropna(subset=feature_cols + ['ED Enc', 'ED Enc Admitted'])
    
    if len(train_subset) == 0:
        return {'encounter': None, 'admitted': None, 'rate': None}
    
    X = train_subset[feature_cols].fillna(0)
    
    # Default hyperparameters
    default_params = {
        'n_estimators': 200,
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 3,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'random_state': 42,
        'n_jobs': -1
    }
    
    if hyperparams:
        default_params.update(hyperparams)
    
    # Model 1: Total encounters (Poisson)
    y_enc = train_subset['ED Enc'].values
    model_enc = xgb.XGBRegressor(
        objective='count:poisson',
        **default_params
    )
    model_enc.fit(X, y_enc)
    
    # Model 2: Admission rate (squared error)
    y_rate = train_subset['admit_rate'].values
    model_rate = xgb.XGBRegressor(
        objective='reg:squarederror',
        **default_params
    )
    model_rate.fit(X, y_rate)
    
    # Model 3: Admitted encounters (Poisson) - alternative approach
    y_adm = train_subset['ED Enc Admitted'].values
    model_adm = xgb.XGBRegressor(
        objective='count:poisson',
        **default_params
    )
    model_adm.fit(X, y_adm)
    
    return {
        'encounter': model_enc,
        'rate': model_rate,
        'admitted': model_adm
    }


def predict_category(
    test_subset: pd.DataFrame,
    models: Dict[str, xgb.XGBRegressor],
    feature_cols: List[str]
) -> pd.DataFrame:
    """
    Make predictions for a category subset.
    """
    test_subset = test_subset.copy()
    
    if models['encounter'] is None:
        # Fallback: return zeros
        test_subset['pred_ED_Enc'] = 0
        test_subset['pred_ED_Enc_Admitted'] = 0
        return test_subset
    
    X = test_subset[feature_cols].fillna(0)
    
    # Predict total encounters
    pred_enc = models['encounter'].predict(X)
    pred_enc = np.clip(pred_enc, 0, None)
    
    # Predict admission rate
    pred_rate = models['rate'].predict(X)
    pred_rate = np.clip(pred_rate, 0, 1)
    
    # Predict admitted encounters (use direct model or derived)
    pred_adm_direct = models['admitted'].predict(X)
    pred_adm_direct = np.clip(pred_adm_direct, 0, None)
    
    # Derived: admitted = total * rate
    pred_adm_derived = pred_enc * pred_rate
    
    # Use ensemble: average of direct and derived
    pred_adm = (pred_adm_direct + pred_adm_derived) / 2
    pred_adm = np.clip(pred_adm, 0, pred_enc)  # Ensure admitted <= total
    
    test_subset['pred_ED_Enc'] = pred_enc
    test_subset['pred_ED_Enc_Admitted'] = pred_adm
    
    return test_subset


def aggregate_predictions(
    category_predictions: Dict[str, pd.DataFrame]
) -> pd.DataFrame:
    """
    Aggregate predictions from all categories back to (Site, Date, Block) level.
    """
    all_preds = []
    for category, pred_df in category_predictions.items():
        if len(pred_df) > 0:
            all_preds.append(pred_df[['Site', 'Date', 'Block', 'pred_ED_Enc', 'pred_ED_Enc_Admitted']])
    
    if not all_preds:
        return pd.DataFrame(columns=['Site', 'Date', 'Block', 'pred_ED_Enc', 'pred_ED_Enc_Admitted'])
    
    combined = pd.concat(all_preds, ignore_index=True)
    
    # Aggregate by Site, Date, Block
    aggregated = combined.groupby(['Site', 'Date', 'Block'], as_index=False).agg({
        'pred_ED_Enc': 'sum',
        'pred_ED_Enc_Admitted': 'sum'
    })
    
    return aggregated


def run_pipeline2(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    category_mapping: Dict[str, str],
    min_samples_per_category: int = 100
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Main pipeline2 execution:
    1. Split training data by category
    2. Train separate models for each category
    3. Make predictions for each category
    4. Aggregate predictions
    """
    print("  Creating category subsets...")
    category_subsets = create_category_subsets(
        train_df, 
        category_mapping, 
        min_samples_per_category
    )
    print(f"  Created {len(category_subsets)} category subsets")
    
    # Prepare features for each subset
    print("  Preparing features for each category...")
    prepared_subsets = {}
    feature_cols = None
    
    for category, subset in category_subsets.items():
        prepared, feats = add_comprehensive_features(subset)
        prepared_subsets[category] = prepared
        if feature_cols is None:
            feature_cols = feats
    
    # Train models for each category
    print("  Training category-specific models...")
    models = {}
    for category, subset in prepared_subsets.items():
        print(f"    Training {category} ({len(subset)} samples)...")
        models[category] = train_category_models(subset, feature_cols)
    
    # Prepare test data by category
    # test_df should already have Category column from aggregation
    test_df = test_df.copy()
    if 'Category' not in test_df.columns:
        if 'REASON_VISIT_NAME' in test_df.columns:
            test_df['Category'] = test_df['REASON_VISIT_NAME'].map(
                category_mapping
            ).fillna('Other/Unspecified')
        else:
            # If no REASON_VISIT_NAME, assume all is Other/Unspecified
            test_df['Category'] = 'Other/Unspecified'
    
    # If already aggregated with category, use as-is
    if 'Category' in test_df.columns and 'Block' in test_df.columns:
        test_category_data = test_df.copy()
        if 'admit_rate' not in test_category_data.columns:
            test_category_data['admit_rate'] = (
                test_category_data['ED Enc Admitted'] / 
                test_category_data['ED Enc'].clip(lower=1.0)
            ).clip(0, 1)
    else:
        # Fallback: aggregate by category if needed
        test_category_data = test_df.groupby(
            ['Site', 'Date', 'Block', 'Category'],
            as_index=False
        ).agg({
            'ED Enc': 'sum',
            'ED Enc Admitted': 'sum'
        })
        test_category_data['admit_rate'] = (
            test_category_data['ED Enc Admitted'] / 
            test_category_data['ED Enc'].clip(lower=1.0)
        ).clip(0, 1)
    
    # Make predictions for each category
    print("  Making predictions for each category...")
    category_predictions = {}
    
    for category in category_subsets.keys():
        test_subset = test_category_data[
            test_category_data['Category'] == category
        ].copy()
        
        if len(test_subset) == 0:
            continue
        
        test_subset, _ = add_comprehensive_features(test_subset)
        test_subset = predict_category(test_subset, models[category], feature_cols)
        
        category_predictions[category] = test_subset[
            ['Site', 'Date', 'Block', 'pred_ED_Enc', 'pred_ED_Enc_Admitted']
        ]
    
    # Aggregate predictions
    print("  Aggregating predictions...")
    final_predictions = aggregate_predictions(category_predictions)
    
    # Calculate metrics
    test_actuals = test_df.groupby(['Site', 'Date', 'Block'], as_index=False).agg({
        'ED Enc': 'sum',
        'ED Enc Admitted': 'sum'
    })
    
    evaluation = final_predictions.merge(
        test_actuals,
        on=['Site', 'Date', 'Block'],
        how='inner'
    )
    
    if len(evaluation) == 0:
        metrics = {
            'mae_encounters': np.nan,
            'mae_admitted': np.nan,
            'wape_encounters': np.nan,
            'wape_admitted': np.nan
        }
    else:
        metrics = {
            'mae_encounters': mae(evaluation['ED Enc'], evaluation['pred_ED_Enc']),
            'mae_admitted': mae(evaluation['ED Enc Admitted'], evaluation['pred_ED_Enc_Admitted']),
            'wape_encounters': wape(evaluation['ED Enc'], evaluation['pred_ED_Enc']),
            'wape_admitted': wape(evaluation['ED Enc Admitted'], evaluation['pred_ED_Enc_Admitted'])
        }
    
    return final_predictions, metrics


def main():
    """
    STEP 5: Validation - Run pipeline2 on all validation splits
    """
    print("=" * 80)
    print("PIPELINE 2: CATEGORY-BASED PREDICTION WITH TEMPORAL SPLITTING")
    print("=" * 80)
    
    # Load data
    print("\nSTEP 1: Data Loading and Preprocessing")
    print("-" * 80)
    print("Loading dataset...")
    df = load_dataset()
    print(f"  Dataset shape: {df.shape}")
    print(f"  Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    # Validate data
    print("Validating data...")
    df = validate_data(df)
    
    # Load category mapping
    print("Loading category mapping...")
    category_mapping = load_category_mapping()
    print(f"  Loaded {len(category_mapping)} reason mappings")
    
    # Create validation splits
    print("\nCreating validation splits...")
    splits = create_validation_splits(df)
    print(f"  Created {len(splits)} validation periods")
    
    all_metrics = []
    all_predictions = []
    
    # Process each validation period
    for split in splits:
        print(f"\n{'='*80}")
        print(f"Processing {split['description']}")
        print(f"{'='*80}")
        
        train_df = split['train_df']
        test_df = split['test_df']
        
        # Aggregate to blocks while preserving category information
        print("Aggregating to 6-hour blocks with category information...")
        train_df = aggregate_to_blocks_with_category(train_df, category_mapping)
        test_df = aggregate_to_blocks_with_category(test_df, category_mapping)
        
        print(f"  Train: {train_df.shape[0]} records")
        print(f"  Test: {test_df.shape[0]} records")
        
        # Run pipeline
        predictions, metrics = run_pipeline2(
            train_df,
            test_df,
            category_mapping,
            min_samples_per_category=50  # Lower threshold for validation
        )
        
        metrics['period_id'] = split['period_id']
        metrics['description'] = split['description']
        all_metrics.append(metrics)
        all_predictions.append(predictions)
        
        print(f"\nMetrics for {split['description']}:")
        print(f"  MAE Encounters: {metrics['mae_encounters']:.2f}")
        print(f"  MAE Admitted: {metrics['mae_admitted']:.2f}")
        print(f"  WAPE Encounters: {metrics['wape_encounters']:.4f}")
        print(f"  WAPE Admitted: {metrics['wape_admitted']:.4f}")
    
    # Summary
    print(f"\n{'='*80}")
    print("VALIDATION SUMMARY")
    print(f"{'='*80}")
    
    metrics_df = pd.DataFrame(all_metrics)
    
    print("\nAverage Metrics Across All Periods:")
    print(f"  Average MAE Encounters: {metrics_df['mae_encounters'].mean():.2f}")
    print(f"  Average MAE Admitted: {metrics_df['mae_admitted'].mean():.2f}")
    print(f"  Average WAPE Encounters: {metrics_df['wape_encounters'].mean():.4f}")
    print(f"  Average WAPE Admitted: {metrics_df['wape_admitted'].mean():.4f}")
    
    print("\nPer-Period Metrics:")
    print(metrics_df[['period_id', 'mae_encounters', 'mae_admitted', 
                      'wape_encounters', 'wape_admitted']].to_string(index=False))
    
    # Save results
    output_dir = REPO_ROOT / "baseline_model" / "pipieline2" / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    metrics_df.to_csv(output_dir / "validation_metrics.csv", index=False)
    print(f"\nSaved validation metrics to {output_dir / 'validation_metrics.csv'}")
    
    # Save predictions for each period
    for i, pred in enumerate(all_predictions):
        pred.to_csv(output_dir / f"predictions_period_{i+1}.csv", index=False)
    
    print(f"Saved predictions to {output_dir}")


if __name__ == '__main__':
    main()
