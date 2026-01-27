"""Data loader for EDA - loads raw dataset without processing."""

import pandas as pd
from pathlib import Path


def get_dataset_path() -> Path:
    """Get the path to the DSU dataset CSV file."""
    repo_root = Path(__file__).parent.parent
    dataset_path = repo_root / "Dataset" / "DSU-Dataset.csv"
    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset not found at {dataset_path}")
    return dataset_path


def load_dataset() -> pd.DataFrame:
    """
    Load the DSU dataset from CSV file and print dataset statistics.
    
    Returns:
        DataFrame with raw data (no processing applied)
        Columns: Site, Date, Hour, REASON_VISIT_NAME, ED Enc, ED Enc Admitted
    """
    dataset_path = get_dataset_path()
    print(f"Loading dataset from: {dataset_path}")
    df = pd.read_csv(dataset_path)
    
    # Print dataset overview
    print("\n" + "=" * 80)
    print("DATASET OVERVIEW")
    print("=" * 80)
    print(f"Shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"\nColumns: {list(df.columns)}")
    print(f"\nData types:\n{df.dtypes}")
    
    # Missing values
    print("\n" + "=" * 80)
    print("MISSING VALUES")
    print("=" * 80)
    missing = df.isnull().sum()
    missing_pct = (missing / len(df)) * 100
    missing_df = pd.DataFrame({
        'Missing Count': missing,
        'Percentage': missing_pct
    })
    missing_nonzero = missing_df[missing_df['Missing Count'] > 0]
    if len(missing_nonzero) > 0:
        print(missing_nonzero)
    else:
        print("No missing values found.")
    
    # Numeric columns statistics
    print("\n" + "=" * 80)
    print("NUMERIC COLUMNS SUMMARY STATISTICS")
    print("=" * 80)
    numeric_cols = ['Hour', 'ED Enc', 'ED Enc Admitted']
    print(df[numeric_cols].describe())
    
    # Site distribution
    print("\n" + "=" * 80)
    print("SITE DISTRIBUTION")
    print("=" * 80)
    site_counts = df['Site'].value_counts().sort_index()
    print(f"Number of unique sites: {df['Site'].nunique()}")
    print(f"\nVisit counts by site:")
    for site, count in site_counts.items():
        pct = (count / len(df)) * 100
        print(f"  Site {site}: {count:,} ({pct:.2f}%)")
    
    # Date range (convert temporarily for display)
    print("\n" + "=" * 80)
    print("TIME RANGE ANALYSIS")
    print("=" * 80)
    df_temp = df.copy()
    df_temp['Date'] = pd.to_datetime(df_temp['Date'])
    print(f"Date range: {df_temp['Date'].min()} to {df_temp['Date'].max()}")
    print(f"Total days: {(df_temp['Date'].max() - df_temp['Date'].min()).days + 1}")
    print(f"Average visits per day: {len(df) / ((df_temp['Date'].max() - df_temp['Date'].min()).days + 1):.2f}")
    
    # Hour analysis
    print("\n" + "=" * 80)
    print("HOUR OF DAY ANALYSIS")
    print("=" * 80)
    print(f"Hour range: {df['Hour'].min():.0f} to {df['Hour'].max():.0f}")
    hour_counts = df['Hour'].value_counts().sort_index()
    print(f"\nTop 5 busiest hours:")
    for hour, count in hour_counts.head().items():
        print(f"  Hour {hour:.0f}: {count:,} visits")
    print(f"\nBottom 5 quietest hours:")
    for hour, count in hour_counts.tail().items():
        print(f"  Hour {hour:.0f}: {count:,} visits")
    
    # Reason for visit summary
    print("\n" + "=" * 80)
    print("REASON FOR VISIT SUMMARY")
    print("=" * 80)
    reason_counts = df['REASON_VISIT_NAME'].value_counts()
    print(f"Number of unique visit reasons: {df['REASON_VISIT_NAME'].nunique()}")
    print(f"\nTop 10 most common reasons:")
    for reason, count in reason_counts.head(10).items():
        pct = (count / len(df)) * 100
        print(f"  {reason}: {count:,} ({pct:.2f}%)")
    
    # Admission rate
    print("\n" + "=" * 80)
    print("ADMISSION STATISTICS")
    print("=" * 80)
    total_encounters = df['ED Enc'].sum()
    total_admitted = df['ED Enc Admitted'].sum()
    admission_rate = (total_admitted / total_encounters * 100) if total_encounters > 0 else 0
    print(f"Total ED encounters: {total_encounters:,}")
    print(f"Total admitted: {total_admitted:,}")
    print(f"Overall admission rate: {admission_rate:.2f}%")
    
    print("\n" + "=" * 80)
    print("Dataset loaded successfully.")
    print("=" * 80 + "\n")
    
    return df

df = load_dataset()