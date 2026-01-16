import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Dataset/DSU-Dataset.csv')

print("=" * 80)
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
print(missing_df[missing_df['Missing Count'] > 0])

# Convert Date to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Basic statistics for numeric columns
print("\n" + "=" * 80)
print("NUMERIC COLUMNS SUMMARY STATISTICS")
print("=" * 80)
print(df[['Hour', 'ED Enc', 'ED Enc Admitted']].describe())

# Site analysis
print("\n" + "=" * 80)
print("SITE ANALYSIS")
print("=" * 80)
site_counts = df['Site'].value_counts().sort_index()
print(f"Number of unique sites: {df['Site'].nunique()}")
print(f"\nVisit counts by site:\n{site_counts}")
print(f"\nPercentage by site:\n{(site_counts / len(df) * 100).round(2)}%")

# Date range
print("\n" + "=" * 80)
print("TIME RANGE ANALYSIS")
print("=" * 80)
print(f"Date range: {df['Date'].min()} to {df['Date'].max()}")
print(f"Total days: {(df['Date'].max() - df['Date'].min()).days + 1}")
print(f"Average visits per day: {len(df) / ((df['Date'].max() - df['Date'].min()).days + 1):.2f}")

# Hour analysis
print("\n" + "=" * 80)
print("HOUR OF DAY ANALYSIS")
print("=" * 80)
print(f"Hour range: {df['Hour'].min():.0f} to {df['Hour'].max():.0f}")
hour_counts = df['Hour'].value_counts().sort_index()
print(f"\nTop 5 busiest hours:\n{hour_counts.head()}")
print(f"\nBottom 5 quietest hours:\n{hour_counts.tail()}")

# Reason for visit analysis
print("\n" + "=" * 80)
print("REASON FOR VISIT ANALYSIS")
print("=" * 80)
reason_counts = df['REASON_VISIT_NAME'].value_counts()
print(f"Number of unique visit reasons: {df['REASON_VISIT_NAME'].nunique()}")
print(f"\nTop 15 most common reasons:\n{reason_counts.head(15)}")
print(f"\nBottom 10 least common reasons:\n{reason_counts.tail(10)}")

# Admission analysis
print("\n" + "=" * 80)
print("ADMISSION ANALYSIS")
print("=" * 80)
total_visits = len(df)
admitted_count = df['ED Enc Admitted'].sum()
admission_rate = (admitted_count / total_visits) * 100
print(f"Total visits: {total_visits:,}")
print(f"Admitted: {admitted_count:,} ({admission_rate:.2f}%)")
print(f"Not admitted: {total_visits - admitted_count:,} ({100 - admission_rate:.2f}%)")

# Admission rate by site
print("\n" + "=" * 80)
print("ADMISSION RATE BY SITE")
print("=" * 80)
admission_by_site = df.groupby('Site').agg({
    'ED Enc Admitted': ['sum', 'count']
}).reset_index()
admission_by_site.columns = ['Site', 'Admitted', 'Total']
admission_by_site['Admission Rate %'] = (admission_by_site['Admitted'] / admission_by_site['Total'] * 100).round(2)
print(admission_by_site.sort_values('Admission Rate %', ascending=False))

# Admission rate by reason
print("\n" + "=" * 80)
print("TOP 15 REASONS BY ADMISSION RATE")
print("=" * 80)
admission_by_reason = df.groupby('REASON_VISIT_NAME').agg({
    'ED Enc Admitted': ['sum', 'count']
}).reset_index()
admission_by_reason.columns = ['Reason', 'Admitted', 'Total']
admission_by_reason['Admission Rate %'] = (admission_by_reason['Admitted'] / admission_by_reason['Total'] * 100).round(2)
admission_by_reason = admission_by_reason[admission_by_reason['Total'] >= 100]  # Filter for reasons with at least 100 visits
print(admission_by_reason.nlargest(15, 'Admission Rate %')[['Reason', 'Total', 'Admitted', 'Admission Rate %']])

# ED Encounter analysis
print("\n" + "=" * 80)
print("ED ENCOUNTER ANALYSIS")
print("=" * 80)
print(f"ED Enc range: {df['ED Enc'].min()} to {df['ED Enc'].max()}")
print(f"Mean ED Enc per record: {df['ED Enc'].mean():.2f}")
print(f"Median ED Enc per record: {df['ED Enc'].median():.2f}")
enc_distribution = df['ED Enc'].value_counts().sort_index()
print(f"\nED Enc distribution (showing first 10):\n{enc_distribution.head(10)}")

# Monthly trends
print("\n" + "=" * 80)
print("MONTHLY VISIT TRENDS")
print("=" * 80)
df['YearMonth'] = df['Date'].dt.to_period('M')
monthly_visits = df.groupby('YearMonth').size()
print(f"Months covered: {len(monthly_visits)}")
print(f"\nFirst 5 months:\n{monthly_visits.head()}")
print(f"\nLast 5 months:\n{monthly_visits.tail()}")
print(f"\nMonthly average: {monthly_visits.mean():.2f} visits")
print(f"Monthly min: {monthly_visits.min()} visits")
print(f"Monthly max: {monthly_visits.max()} visits")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
