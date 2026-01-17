import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create Visualizations directory if it doesn't exist
os.makedirs('Visualizations', exist_ok=True)

# Load dataset
print("Loading dataset...")
df = pd.read_csv('Dataset/DSU-Dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Load category mapping
print("Loading category mapping...")
reason_to_category = pd.read_csv('reason_categories.csv')
reason_to_category_dict = dict(zip(reason_to_category['REASON_VISIT_NAME'], 
                                    reason_to_category['Category']))
df['Reason_Category'] = df['REASON_VISIT_NAME'].map(reason_to_category_dict)

# Filter for Injury/Trauma only
injury_df = df[df['Reason_Category'] == 'Injury/Trauma'].copy()
print(f"\nTotal injury encounters: {injury_df['ED Enc'].sum():,.0f}")

# Extract year and month
injury_df['Year'] = injury_df['Date'].dt.year
injury_df['Month'] = injury_df['Date'].dt.month
injury_df['MonthName'] = injury_df['Date'].dt.strftime('%b')
injury_df['YearMonth'] = injury_df['Date'].dt.to_period('M').astype(str)

# Get top injury types by volume
injury_type_volume = injury_df.groupby('REASON_VISIT_NAME').agg({
    'ED Enc': 'sum'
}).reset_index().sort_values('ED Enc', ascending=False)

top_10_injuries = injury_type_volume.head(10)['REASON_VISIT_NAME'].tolist()
print(f"\nTop 10 injury types:")
for idx, inj in enumerate(top_10_injuries, 1):
    vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
    print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")

# ============================================================================
# VISUALIZATION 1: Heatmap - Year vs Month for Top Injury Types
# ============================================================================

print("\nCreating year-month injury visualizations...")

# Aggregate by injury type, year, and month
injury_year_month = injury_df[injury_df['REASON_VISIT_NAME'].isin(top_10_injuries)].groupby([
    'REASON_VISIT_NAME', 'Year', 'Month'
]).agg({
    'ED Enc': 'sum'
}).reset_index()

# Create a 2x5 subplot grid for top 10 injuries
fig, axes = plt.subplots(5, 2, figsize=(18, 20))
fig.suptitle('Year vs Month: Top 10 Injury Types', fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()

for idx, injury_type in enumerate(top_10_injuries):
    ax = axes[idx]
    injury_data = injury_year_month[injury_year_month['REASON_VISIT_NAME'] == injury_type]
    
    # Pivot: Year as rows, Month as columns
    heatmap_data = injury_data.pivot(index='Year', columns='Month', values='ED Enc')
    heatmap_data = heatmap_data.sort_index()  # Sort by year
    
    # Month labels
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
    heatmap_data.columns = month_labels
    
    sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, 
                cbar_kws={'label': 'Encounters'}, 
                annot=True, fmt='.0f', annot_kws={'size': 6},
                linewidths=0.5, cbar=True if idx == 0 else False)
    
    ax.set_title(injury_type, fontweight='bold', pad=10, fontsize=11)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Year', fontweight='bold')
    
    # Only show xlabel on bottom row
    if idx < 8:
        ax.set_xlabel('')

plt.tight_layout()
plt.savefig('Visualizations/injury_year_month_heatmap.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/injury_year_month_heatmap.png")

# ============================================================================
# VISUALIZATION 2: Time Series - Top Injury Types Over Year-Month
# ============================================================================

# Aggregate by injury type and year-month
injury_timeseries = injury_df[injury_df['REASON_VISIT_NAME'].isin(top_10_injuries)].groupby([
    'REASON_VISIT_NAME', 'YearMonth'
]).agg({
    'ED Enc': 'sum'
}).reset_index()

# Convert YearMonth to datetime for plotting
injury_timeseries['YearMonth_dt'] = pd.to_datetime(injury_timeseries['YearMonth'])

fig, axes = plt.subplots(5, 2, figsize=(20, 20))
fig.suptitle('Time Series Trends: Top 10 Injury Types (2018-2025)', 
              fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()
colors = sns.color_palette("husl", 10)

for idx, injury_type in enumerate(top_10_injuries):
    ax = axes[idx]
    injury_data = injury_timeseries[injury_timeseries['REASON_VISIT_NAME'] == injury_type]
    injury_data = injury_data.sort_values('YearMonth_dt')
    
    ax.plot(injury_data['YearMonth_dt'], injury_data['ED Enc'], 
            'o-', color=colors[idx], linewidth=2, markersize=4, alpha=0.8)
    ax.fill_between(injury_data['YearMonth_dt'], injury_data['ED Enc'], 
                    alpha=0.2, color=colors[idx])
    
    ax.set_title(injury_type, fontweight='bold', pad=10, fontsize=11)
    ax.set_ylabel('Encounters', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Format x-axis to show year-month
    from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_minor_locator(MonthLocator(interval=6))
    ax.xaxis.set_major_formatter(DateFormatter('%Y'))
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    
    # Only show xlabel on bottom row
    if idx < 8:
        ax.set_xlabel('')
    else:
        ax.set_xlabel('Year', fontweight='bold')

plt.tight_layout()
plt.savefig('Visualizations/injury_year_month_timeseries.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/injury_year_month_timeseries.png")

# ============================================================================
# VISUALIZATION 3: Faceted Heatmap - All Years and Months
# ============================================================================

# Get top 6 injuries for a clearer view
top_6_injuries = top_10_injuries[:6]

fig, axes = plt.subplots(3, 2, figsize=(18, 16))
fig.suptitle('Annual Patterns: Top 6 Injury Types by Year and Month', 
              fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()

for idx, injury_type in enumerate(top_6_injuries):
    ax = axes[idx]
    injury_data = injury_year_month[injury_year_month['REASON_VISIT_NAME'] == injury_type]
    
    # Create a matrix with Year as rows, Month as columns
    pivot_data = injury_data.pivot(index='Year', columns='Month', values='ED Enc')
    pivot_data = pivot_data.sort_index()
    
    # Full month names for columns
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    pivot_data.columns = month_names
    
    sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax,
                cbar_kws={'label': 'Encounters'},
                annot=True, fmt='.0f', annot_kws={'size': 7},
                linewidths=0.5, cbar=True, square=False)
    
    ax.set_title(injury_type, fontweight='bold', pad=12, fontsize=12)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Year', fontweight='bold')

plt.tight_layout()
plt.savefig('Visualizations/injury_year_month_faceted.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/injury_year_month_faceted.png")

# ============================================================================
# VISUALIZATION 4: Stacked Area - Year-Month Trends
# ============================================================================

# Aggregate all injuries by year-month
all_injury_timeseries = injury_df.groupby(['YearMonth', 'REASON_VISIT_NAME']).agg({
    'ED Enc': 'sum'
}).reset_index()

# Get top 8 injury types
top_8_injuries = top_10_injuries[:8]

# Prepare data for stacked area
stacked_data = all_injury_timeseries[all_injury_timeseries['REASON_VISIT_NAME'].isin(top_8_injuries)]
stacked_pivot = stacked_data.pivot(index='YearMonth', columns='REASON_VISIT_NAME', values='ED Enc')

# Convert index to datetime and sort
stacked_pivot.index = pd.to_datetime(stacked_pivot.index)
stacked_pivot = stacked_pivot.sort_index()
stacked_pivot = stacked_pivot.fillna(0)

# Calculate "Other" injuries
top_8_monthly_total = stacked_pivot.sum(axis=1)
all_monthly_total = injury_df.groupby('YearMonth')['ED Enc'].sum()
all_monthly_total.index = pd.to_datetime(all_monthly_total.index)
all_monthly_total = all_monthly_total.sort_index()
other_monthly = all_monthly_total - top_8_monthly_total
stacked_pivot['Other Injuries'] = other_monthly

fig, ax = plt.subplots(figsize=(18, 10))
colors_stacked = sns.color_palette("Set3", 9)

ax.stackplot(stacked_pivot.index, 
             *[stacked_pivot[col] for col in stacked_pivot.columns],
             labels=list(stacked_pivot.columns),
             colors=colors_stacked, alpha=0.7)

ax.set_xlabel('Year', fontweight='bold')
ax.set_ylabel('Total Encounters', fontweight='bold')
ax.set_title('Year-Month Trends: Top 8 Injury Types + Others (Stacked)', 
             fontweight='bold', pad=15)

# Format x-axis
from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
ax.xaxis.set_major_locator(YearLocator())
ax.xaxis.set_minor_locator(MonthLocator(interval=6))
ax.xaxis.set_major_formatter(DateFormatter('%Y'))
ax.tick_params(axis='x', rotation=45, labelsize=9)

ax.legend(loc='upper left', framealpha=0.9, fontsize=9, ncol=2)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('Visualizations/injury_year_month_stacked.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/injury_year_month_stacked.png")

# ============================================================================
# Summary Statistics
# ============================================================================

print("\n" + "=" * 80)
print("YEAR-MONTH INJURY PATTERN SUMMARY")
print("=" * 80)

# Yearly summary
yearly_summary = injury_df.groupby('Year').agg({
    'ED Enc': 'sum'
}).reset_index().sort_values('Year')

print("\nTotal Encounters by Year:")
for _, row in yearly_summary.iterrows():
    pct = (row['ED Enc'] / yearly_summary['ED Enc'].sum()) * 100
    print(f"  {int(row['Year'])}: {row['ED Enc']:>8,.0f} ({pct:>5.2f}%)")

# Month summary (across all years)
monthly_summary = injury_df.groupby('Month').agg({
    'ED Enc': 'sum'
}).reset_index().sort_values('Month')
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
monthly_summary['MonthName'] = monthly_summary['Month'].apply(lambda x: month_names[x-1])

print("\nTotal Encounters by Month (all years):")
for _, row in monthly_summary.iterrows():
    pct = (row['ED Enc'] / monthly_summary['ED Enc'].sum()) * 100
    print(f"  {row['MonthName']:3s}: {row['ED Enc']:>8,.0f} ({pct:>5.2f}%)")

# Year-month combinations
year_month_summary = injury_df.groupby(['Year', 'Month']).agg({
    'ED Enc': 'sum'
}).reset_index()
peak_ym = year_month_summary.loc[year_month_summary['ED Enc'].idxmax()]
print(f"\nPeak Year-Month: {int(peak_ym['Year'])}-{month_names[int(peak_ym['Month'])-1]} ({peak_ym['ED Enc']:,.0f} encounters)")

lowest_ym = year_month_summary.loc[year_month_summary['ED Enc'].idxmin()]
print(f"Lowest Year-Month: {int(lowest_ym['Year'])}-{month_names[int(lowest_ym['Month'])-1]} ({lowest_ym['ED Enc']:,.0f} encounters)")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
