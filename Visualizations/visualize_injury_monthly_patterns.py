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

# Extract month
injury_df['Month'] = injury_df['Date'].dt.month
injury_df['MonthName'] = injury_df['Date'].dt.strftime('%b')

# Get top injury types by volume
injury_type_volume = injury_df.groupby('REASON_VISIT_NAME').agg({
    'ED Enc': 'sum'
}).reset_index().sort_values('ED Enc', ascending=False)

print(f"\nTop 10 injury types by volume:")
for idx, (_, row) in enumerate(injury_type_volume.head(10).iterrows(), 1):
    print(f"{idx:2d}. {row['REASON_VISIT_NAME']:30s} - {row['ED Enc']:>8,.0f} encounters")

# ============================================================================
# VISUALIZATION 1: Heatmap - Top Injury Types by Month
# ============================================================================

# Get top 15 injury types
top_injuries = injury_type_volume.head(15)['REASON_VISIT_NAME'].tolist()

# Aggregate by injury type and month
injury_monthly = injury_df[injury_df['REASON_VISIT_NAME'].isin(top_injuries)].groupby([
    'REASON_VISIT_NAME', 'Month'
]).agg({
    'ED Enc': 'sum'
}).reset_index()

# Pivot for heatmap
heatmap_data = injury_monthly.pivot(index='REASON_VISIT_NAME', columns='Month', values='ED Enc')
heatmap_data = heatmap_data.reindex(top_injuries)  # Reorder by volume

# Month labels
month_labels = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
heatmap_data.columns = [month_labels[int(m)-1] for m in heatmap_data.columns]

fig, ax = plt.subplots(figsize=(14, 10))
fig.suptitle('Top 15 Injury Types: Monthly Volume Patterns', 
              fontsize=16, fontweight='bold', y=0.98)

sns.heatmap(heatmap_data, cmap='YlOrRd', ax=ax, cbar_kws={'label': 'Total Encounters'}, 
            annot=True, fmt='.0f', annot_kws={'size': 7}, linewidths=0.5)
ax.set_title('Heatmap: Injury Encounters by Type and Month', fontweight='bold', pad=15)
ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Injury Type', fontweight='bold')

plt.tight_layout()
plt.savefig('Visualizations/injury_types_monthly_heatmap.png', dpi=300, bbox_inches='tight')
print("\nSaved: Visualizations/injury_types_monthly_heatmap.png")

# ============================================================================
# VISUALIZATION 2: Line Chart - Top Injury Types Over Months
# ============================================================================

# Get top 8 injury types for clearer visualization
top_8_injuries = injury_type_volume.head(8)['REASON_VISIT_NAME'].tolist()

# Aggregate across all years by month (average or total)
injury_monthly_all = injury_df[injury_df['REASON_VISIT_NAME'].isin(top_8_injuries)].groupby([
    'REASON_VISIT_NAME', 'Month'
]).agg({
    'ED Enc': 'sum'
}).reset_index()

fig, axes = plt.subplots(4, 2, figsize=(16, 14))
fig.suptitle('Monthly Patterns: Top 8 Injury Types', fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()
colors = sns.color_palette("husl", 8)

for idx, injury_type in enumerate(top_8_injuries):
    ax = axes[idx]
    injury_data = injury_monthly_all[injury_monthly_all['REASON_VISIT_NAME'] == injury_type]
    injury_data = injury_data.sort_values('Month')
    
    ax.plot(injury_data['Month'], injury_data['ED Enc'], 'o-', 
            color=colors[idx], linewidth=2.5, markersize=8, alpha=0.8)
    ax.fill_between(injury_data['Month'], injury_data['ED Enc'], 
                    alpha=0.3, color=colors[idx])
    
    ax.set_title(injury_type, fontweight='bold', pad=10, fontsize=11)
    ax.set_xlabel('Month', fontweight='bold')
    ax.set_ylabel('Total Encounters', fontweight='bold')
    ax.set_xticks(range(1, 13))
    ax.set_xticklabels(month_labels, rotation=45, ha='right')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Only show xlabel on bottom row
    if idx < 6:
        ax.set_xlabel('')
    
    # Add value labels on peaks
    max_idx = injury_data['ED Enc'].idxmax()
    max_month = injury_data.loc[max_idx, 'Month']
    max_value = injury_data.loc[max_idx, 'ED Enc']
    ax.annotate(f'{max_value:.0f}', 
                xy=(max_month, max_value), 
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Visualizations/injury_types_monthly_trends.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/injury_types_monthly_trends.png")

# ============================================================================
# VISUALIZATION 3: Seasonal Comparison - Top Injury Types
# ============================================================================

# Define seasons
def get_season(month):
    if month in [12, 1, 2]:
        return 'Winter'
    elif month in [3, 4, 5]:
        return 'Spring'
    elif month in [6, 7, 8]:
        return 'Summer'
    else:
        return 'Fall'

injury_df['Season'] = injury_df['Month'].apply(get_season)

# Get top 12 injury types
top_12_injuries = injury_type_volume.head(12)['REASON_VISIT_NAME'].tolist()

injury_seasonal = injury_df[injury_df['REASON_VISIT_NAME'].isin(top_12_injuries)].groupby([
    'REASON_VISIT_NAME', 'Season'
]).agg({
    'ED Enc': 'sum'
}).reset_index()

# Pivot for grouped bar chart
seasonal_pivot = injury_seasonal.pivot(index='REASON_VISIT_NAME', 
                                       columns='Season', values='ED Enc')
seasonal_pivot = seasonal_pivot.reindex(top_12_injuries)
season_order = ['Spring', 'Summer', 'Fall', 'Winter']
seasonal_pivot = seasonal_pivot[season_order]

fig, ax = plt.subplots(figsize=(16, 10))
x = np.arange(len(top_12_injuries))
width = 0.2

colors_season = ['#2ECC71', '#E67E22', '#F39C12', '#3498DB']  # Spring, Summer, Fall, Winter

for idx, season in enumerate(season_order):
    ax.bar(x + idx * width, seasonal_pivot[season], width, 
           label=season, color=colors_season[idx], alpha=0.8)

ax.set_xlabel('Injury Type', fontweight='bold')
ax.set_ylabel('Total Encounters', fontweight='bold')
ax.set_title('Seasonal Patterns: Top 12 Injury Types', fontweight='bold', pad=15)
ax.set_xticks(x + width * 1.5)
ax.set_xticklabels([inj.replace(' ', '\n')[:20] for inj in top_12_injuries], 
                    rotation=45, ha='right', fontsize=9)
ax.legend(loc='upper left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')

plt.tight_layout()
plt.savefig('Visualizations/injury_types_seasonal_patterns.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/injury_types_seasonal_patterns.png")

# ============================================================================
# VISUALIZATION 4: Stacked Area Chart - Monthly Distribution
# ============================================================================

# Aggregate all injuries by month
all_injury_monthly = injury_df.groupby(['Month', 'REASON_VISIT_NAME']).agg({
    'ED Enc': 'sum'
}).reset_index()

# Get top 6 injury types
top_6_injuries = injury_type_volume.head(6)['REASON_VISIT_NAME'].tolist()

# Prepare data for stacked area
stacked_data = all_injury_monthly[all_injury_monthly['REASON_VISIT_NAME'].isin(top_6_injuries)]
stacked_pivot = stacked_data.pivot(index='Month', 
                                   columns='REASON_VISIT_NAME', values='ED Enc')
stacked_pivot = stacked_pivot.reindex(range(1, 13))
stacked_pivot = stacked_pivot.fillna(0)

# Calculate "Other" injuries
top_6_monthly_total = stacked_pivot.sum(axis=1)
all_monthly_total = injury_df.groupby('Month')['ED Enc'].sum()
other_monthly = all_monthly_total - top_6_monthly_total
stacked_pivot['Other Injuries'] = other_monthly

fig, ax = plt.subplots(figsize=(14, 8))
colors_stacked = sns.color_palette("Set2", 7)

ax.stackplot(range(1, 13), 
             *[stacked_pivot[col] for col in stacked_pivot.columns],
             labels=list(stacked_pivot.columns),
             colors=colors_stacked, alpha=0.7)

ax.set_xlabel('Month', fontweight='bold')
ax.set_ylabel('Total Encounters', fontweight='bold')
ax.set_title('Monthly Injury Distribution: Top 6 Types + Others', 
             fontweight='bold', pad=15)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(month_labels)
ax.legend(loc='upper left', framealpha=0.9, fontsize=9)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.savefig('Visualizations/injury_types_monthly_stacked.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/injury_types_monthly_stacked.png")

# Print summary statistics
print("\n" + "=" * 80)
print("INJURY MONTHLY PATTERN SUMMARY")
print("=" * 80)

monthly_summary = injury_df.groupby('Month').agg({
    'ED Enc': 'sum'
}).reset_index()
monthly_summary['MonthName'] = monthly_summary['Month'].apply(lambda x: month_labels[x-1])

print("\nTotal Encounters by Month:")
for _, row in monthly_summary.iterrows():
    pct = (row['ED Enc'] / monthly_summary['ED Enc'].sum()) * 100
    print(f"  {row['MonthName']:3s}: {row['ED Enc']:>8,.0f} ({pct:>5.2f}%)")

peak_month = monthly_summary.loc[monthly_summary['ED Enc'].idxmax()]
print(f"\nPeak Month: {peak_month['MonthName']} ({peak_month['ED Enc']:,.0f} encounters)")
lowest_month = monthly_summary.loc[monthly_summary['ED Enc'].idxmin()]
print(f"Lowest Month: {lowest_month['MonthName']} ({lowest_month['ED Enc']:,.0f} encounters)")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
