"""Data loading and splitting utilities for baseline model validation."""

import pandas as pd
from pathlib import Path
from typing import Tuple, List, Dict
from datetime import datetime


def get_dataset_path() -> Path:
    """Get the path to the DSU dataset CSV file."""
    repo_root = Path(__file__).parent.parent.parent
    dataset_path = repo_root / "Dataset" / "DSU-Dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    return dataset_path


def load_dataset() -> pd.DataFrame:
    """
    Load the DSU dataset and perform basic preprocessing.
    
    Returns:
        DataFrame with columns: Site, Date, Hour, REASON_VISIT_NAME, ED Enc, ED Enc Admitted
        Date column is converted to datetime
    """
    dataset_path = get_dataset_path()
    df = pd.read_csv(dataset_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df


def split_by_date_range(
    df: pd.DataFrame,
    train_end: str | datetime,
    test_start: str | datetime,
    test_end: str | datetime
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into training and testing sets based on date ranges.
    
    Args:
        df: Input dataframe with Date column
        train_end: Last date (inclusive) for training data (format: 'YYYY-MM-DD')
        test_start: First date (inclusive) for testing data (format: 'YYYY-MM-DD')
        test_end: Last date (inclusive) for testing data (format: 'YYYY-MM-DD')
    
    Returns:
        Tuple of (train_df, test_df)
    """
    train_end = pd.to_datetime(train_end)
    test_start = pd.to_datetime(test_start)
    test_end = pd.to_datetime(test_end)
    
    train_df = df[df['Date'] <= train_end].copy()
    test_df = df[(df['Date'] >= test_start) & (df['Date'] <= test_end)].copy()
    
    return train_df, test_df


def get_validation_periods() -> List[Dict[str, str]]:
    """
    Define the 4 validation periods for 2025 data.
    Each period uses 2-month increments for testing (Jan-Aug 2025).
    This simulates the prediction task of forecasting 2 months ahead.
    
    Returns:
        List of dictionaries with 'train_end', 'test_start', and 'test_end' keys
    """
    periods = [
        {
            'period_id': 1,
            'train_end': '2024-12-31',
            'test_start': '2025-01-01',
            'test_end': '2025-02-28',
            'description': 'Train up to Dec 2024, predict Jan-Feb 2025'
        },
        {
            'period_id': 2,
            'train_end': '2025-02-28',
            'test_start': '2025-03-01',
            'test_end': '2025-04-30',
            'description': 'Train up to Feb 2025, predict Mar-Apr 2025'
        },
        {
            'period_id': 3,
            'train_end': '2025-04-30',
            'test_start': '2025-05-01',
            'test_end': '2025-06-30',
            'description': 'Train up to Apr 2025, predict May-Jun 2025'
        },
        {
            'period_id': 4,
            'train_end': '2025-06-30',
            'test_start': '2025-07-01',
            'test_end': '2025-08-31',
            'description': 'Train up to Jun 2025, predict Jul-Aug 2025'
        }
    ]
    return periods


def create_validation_splits(df: pd.DataFrame | None = None) -> List[Dict]:
    """
    Create all 4 validation splits for cross-validation.
    
    Args:
        df: Optional dataframe. If None, loads the dataset.
    
    Returns:
        List of dictionaries, each containing:
        - period_id: Integer identifier (1-4)
        - train_df: Training dataframe
        - test_df: Testing dataframe
        - train_end: End date of training period
        - test_start: Start date of test period
        - test_end: End date of test period
        - description: Human-readable description
    """
    if df is None:
        df = load_dataset()
    
    periods = get_validation_periods()
    splits = []
    
    for period in periods:
        train_df, test_df = split_by_date_range(
            df,
            train_end=period['train_end'],
            test_start=period['test_start'],
            test_end=period['test_end']
        )
        
        splits.append({
            'period_id': period['period_id'],
            'train_df': train_df,
            'test_df': test_df,
            'train_end': period['train_end'],
            'test_start': period['test_start'],
            'test_end': period['test_end'],
            'description': period['description']
        })
    
    return splits


def aggregate_to_blocks(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate hourly data to 6-hour time blocks as required by the competition.
    
    Block mapping:
    - Block 0: Hours 0-5 (midnight to 5:59 AM)
    - Block 1: Hours 6-11 (6:00 AM to 11:59 AM)
    - Block 2: Hours 12-17 (noon to 5:59 PM)
    - Block 3: Hours 18-23 (6:00 PM to 11:59 PM)
    
    Args:
        df: DataFrame with Hour column
    
    Returns:
        Aggregated dataframe with Block column instead of Hour
    """
    df = df.copy()
    df['Block'] = (df['Hour'] // 6).astype(int)
    
    # Aggregate by Site, Date, Block
    agg_df = df.groupby(['Site', 'Date', 'Block']).agg({
        'ED Enc': 'sum',
        'ED Enc Admitted': 'sum'
    }).reset_index()
    
    return agg_df


if __name__ == '__main__':
    # Example usage
    print("Loading dataset...")
    df = load_dataset()
    print(f"Dataset shape: {df.shape}")
    print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
    
    print("\nCreating validation splits...")
    splits = create_validation_splits(df)
    
    for split in splits:
        print(f"\n{split['description']}")
        print(f"  Train: {split['train_df']['Date'].min()} to {split['train_df']['Date'].max()}")
        print(f"  Train shape: {split['train_df'].shape}")
        print(f"  Test: {split['test_df']['Date'].min()} to {split['test_df']['Date'].max()}")
        print(f"  Test shape: {split['test_df'].shape}")
