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

# Extract time features
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['DayOfWeek'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
df['DayOfWeekName'] = df['Date'].dt.day_name()
df['DayOfYear'] = df['Date'].dt.dayofyear

# Aggregate by time dimensions for analysis
print("\nAggregating data for cyclical analysis...")

# 1. HOURLY PATTERNS (24-hour cycle)
print("Analyzing hourly patterns...")
hourly_agg = df.groupby('Hour').agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
hourly_agg['Admission Rate'] = (hourly_agg['ED Enc Admitted'] / hourly_agg['ED Enc']) * 100

# 2. DAY OF WEEK PATTERNS (weekly cycle)
print("Analyzing day-of-week patterns...")
dow_agg = df.groupby(['DayOfWeek', 'DayOfWeekName']).agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
dow_agg = dow_agg.sort_values('DayOfWeek')
dow_agg['Admission Rate'] = (dow_agg['ED Enc Admitted'] / dow_agg['ED Enc']) * 100

# 3. MONTHLY PATTERNS (seasonal cycle)
print("Analyzing monthly patterns...")
monthly_agg = df.groupby('Month').agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
monthly_agg['MonthName'] = pd.to_datetime(monthly_agg['Month'], format='%m').dt.strftime('%b')
monthly_agg['Admission Rate'] = (monthly_agg['ED Enc Admitted'] / monthly_agg['ED Enc']) * 100

# 4. DAILY TIME SERIES (long-term trends)
print("Analyzing daily time series...")
daily_agg = df.groupby('Date').agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
daily_agg['Admission Rate'] = (daily_agg['ED Enc Admitted'] / daily_agg['ED Enc']) * 100
daily_agg = daily_agg.sort_values('Date')

# 5. YEAR-OVER-YEAR TRENDS
print("Analyzing year-over-year trends...")
yearly_agg = df.groupby('Year').agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
yearly_agg['Admission Rate'] = (yearly_agg['ED Enc Admitted'] / yearly_agg['ED Enc']) * 100

# ============================================================================
# CREATE VISUALIZATIONS
# ============================================================================

print("\nGenerating visualizations...")

# FIGURE 1: Comprehensive Cyclical Patterns Overview
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('Cyclical Patterns in ED Encounters and Admissions', fontsize=16, fontweight='bold', y=0.995)

# 1.1 Hourly Pattern
ax1 = axes[0, 0]
ax1.plot(hourly_agg['Hour'], hourly_agg['ED Enc'], 'o-', color='#2E86AB', linewidth=2.5, 
         markersize=6, label='Total Encounters', alpha=0.8)
ax1.plot(hourly_agg['Hour'], hourly_agg['ED Enc Admitted'], 's-', color='#A23B72', 
         linewidth=2.5, markersize=6, label='Admitted Encounters', alpha=0.8)
ax1.set_xlabel('Hour of Day', fontweight='bold')
ax1.set_ylabel('Number of Encounters', fontweight='bold')
ax1.set_title('24-Hour Cycle: Hourly Patient Volume', fontweight='bold', pad=10)
ax1.set_xticks(range(0, 24, 2))
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.legend(loc='upper left', framealpha=0.9)

# 1.2 Day of Week Pattern
ax2 = axes[0, 1]
x_pos = np.arange(len(dow_agg))
width = 0.35
ax2.bar(x_pos - width/2, dow_agg['ED Enc'], width, color='#2E86AB', alpha=0.7, 
        label='Total Encounters')
ax2.bar(x_pos + width/2, dow_agg['ED Enc Admitted'], width, color='#A23B72', 
        alpha=0.7, label='Admitted Encounters')
ax2.set_xlabel('Day of Week', fontweight='bold')
ax2.set_ylabel('Number of Encounters', fontweight='bold')
ax2.set_title('Weekly Cycle: Patient Volume by Day of Week', fontweight='bold', pad=10)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(dow_agg['DayOfWeekName'], rotation=45, ha='right')
ax2.grid(True, alpha=0.3, linestyle='--', axis='y')
ax2.legend(loc='upper left', framealpha=0.9)

# 1.3 Monthly/Seasonal Pattern
ax3 = axes[1, 0]
ax3.plot(monthly_agg['Month'], monthly_agg['ED Enc'], 'o-', color='#2E86AB', 
         linewidth=2.5, markersize=8, label='Total Encounters', alpha=0.8)
ax3.plot(monthly_agg['Month'], monthly_agg['ED Enc Admitted'], 's-', color='#A23B72', 
         linewidth=2.5, markersize=8, label='Admitted Encounters', alpha=0.8)
ax3.set_xlabel('Month', fontweight='bold')
ax3.set_ylabel('Number of Encounters', fontweight='bold')
ax3.set_title('Seasonal Cycle: Monthly Patient Volume', fontweight='bold', pad=10)
ax3.set_xticks(monthly_agg['Month'])
ax3.set_xticklabels(monthly_agg['MonthName'], rotation=45, ha='right')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper left', framealpha=0.9)

# 1.4 Year-over-Year Trends
ax4 = axes[1, 1]
width = 0.35
x_pos = np.arange(len(yearly_agg))
ax4.bar(x_pos - width/2, yearly_agg['ED Enc'], width, 
        color='#2E86AB', alpha=0.7, label='Total Encounters')
ax4.bar(x_pos + width/2, yearly_agg['ED Enc Admitted'], width, 
        color='#A23B72', alpha=0.7, label='Admitted Encounters')
ax4.set_xlabel('Year', fontweight='bold')
ax4.set_ylabel('Number of Encounters', fontweight='bold')
ax4.set_title('Year-over-Year Trends', fontweight='bold', pad=10)
ax4.set_xticks(x_pos)
ax4.set_xticklabels(yearly_agg['Year'])
ax4.grid(True, alpha=0.3, linestyle='--', axis='y')
ax4.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('Visualizations/cyclical_patterns_overview.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/cyclical_patterns_overview.png")

# FIGURE 2: Time Series Trends
fig, axes = plt.subplots(3, 1, figsize=(16, 12))
fig.suptitle('ED Volume Trends Over Time', fontsize=16, fontweight='bold', y=0.995)

# 2.1 Daily Total Encounters
ax1 = axes[0]
ax1.plot(daily_agg['Date'], daily_agg['ED Enc'], color='#2E86AB', linewidth=1, alpha=0.7)
ax1.fill_between(daily_agg['Date'], daily_agg['ED Enc'], alpha=0.3, color='#2E86AB')
ax1.set_ylabel('Total Encounters', fontweight='bold')
ax1.set_title('Daily Total Encounters Over Time', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--')

# Add rolling average
window = 30
daily_agg['ED Enc MA'] = daily_agg['ED Enc'].rolling(window=window, center=True).mean()
ax1.plot(daily_agg['Date'], daily_agg['ED Enc MA'], color='#F18F01', linewidth=2.5, 
         label=f'{window}-day Moving Average')
ax1.legend(loc='upper left', framealpha=0.9)

# 2.2 Daily Admitted Encounters
ax2 = axes[1]
ax2.plot(daily_agg['Date'], daily_agg['ED Enc Admitted'], color='#A23B72', linewidth=1, alpha=0.7)
ax2.fill_between(daily_agg['Date'], daily_agg['ED Enc Admitted'], alpha=0.3, color='#A23B72')
daily_agg['ED Enc Admitted MA'] = daily_agg['ED Enc Admitted'].rolling(window=window, center=True).mean()
ax2.plot(daily_agg['Date'], daily_agg['ED Enc Admitted MA'], color='#F18F01', linewidth=2.5, 
         label=f'{window}-day Moving Average')
ax2.set_ylabel('Admitted Encounters', fontweight='bold')
ax2.set_title('Daily Admitted Encounters Over Time', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='upper left', framealpha=0.9)

# 2.3 Daily Admission Rate
ax3 = axes[2]
ax3.plot(daily_agg['Date'], daily_agg['Admission Rate'], color='#C73E1D', linewidth=1, alpha=0.7)
daily_agg['Admission Rate MA'] = daily_agg['Admission Rate'].rolling(window=window, center=True).mean()
ax3.plot(daily_agg['Date'], daily_agg['Admission Rate MA'], color='#F18F01', linewidth=2.5, 
         label=f'{window}-day Moving Average')
ax3.set_xlabel('Date', fontweight='bold')
ax3.set_ylabel('Admission Rate (%)', fontweight='bold')
ax3.set_title('Daily Admission Rate Over Time', fontweight='bold', pad=10)
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='upper left', framealpha=0.9)

plt.tight_layout()
plt.savefig('Visualizations/time_series_trends.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/time_series_trends.png")

# FIGURE 3: Heatmap - Hour vs Day of Week
print("Creating heatmap visualizations...")
hour_dow = df.groupby(['DayOfWeek', 'Hour']).agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
hour_dow['Admission Rate'] = (hour_dow['ED Enc Admitted'] / hour_dow['ED Enc']) * 100

# Pivot for heatmap
heatmap_total = hour_dow.pivot(index='DayOfWeek', columns='Hour', values='ED Enc')
heatmap_admitted = hour_dow.pivot(index='DayOfWeek', columns='Hour', values='ED Enc Admitted')
heatmap_rate = hour_dow.pivot(index='DayOfWeek', columns='Hour', values='Admission Rate')

day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
heatmap_total.index = [day_labels[i] for i in heatmap_total.index]
heatmap_admitted.index = [day_labels[i] for i in heatmap_admitted.index]
heatmap_rate.index = [day_labels[i] for i in heatmap_rate.index]

fig, axes = plt.subplots(1, 3, figsize=(20, 6))
fig.suptitle('Heatmap: Patient Volume Patterns (Day of Week × Hour)', fontsize=16, fontweight='bold', y=1.02)

sns.heatmap(heatmap_total, cmap='YlOrRd', ax=axes[0], cbar_kws={'label': 'Total Encounters'}, 
            annot=True, fmt='.0f', annot_kws={'size': 6})
axes[0].set_title('Total Encounters', fontweight='bold', pad=10)
axes[0].set_xlabel('Hour of Day', fontweight='bold')
axes[0].set_ylabel('Day of Week', fontweight='bold')

sns.heatmap(heatmap_admitted, cmap='YlOrRd', ax=axes[1], cbar_kws={'label': 'Admitted Encounters'}, 
            annot=True, fmt='.0f', annot_kws={'size': 6})
axes[1].set_title('Admitted Encounters', fontweight='bold', pad=10)
axes[1].set_xlabel('Hour of Day', fontweight='bold')
axes[1].set_ylabel('Day of Week', fontweight='bold')

sns.heatmap(heatmap_rate, cmap='RdYlBu_r', ax=axes[2], cbar_kws={'label': 'Admission Rate (%)'}, 
            annot=True, fmt='.1f', annot_kws={'size': 6}, center=29.15)
axes[2].set_title('Admission Rate (%)', fontweight='bold', pad=10)
axes[2].set_xlabel('Hour of Day', fontweight='bold')
axes[2].set_ylabel('Day of Week', fontweight='bold')

plt.tight_layout()
plt.savefig('Visualizations/heatmap_hour_dow.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/heatmap_hour_dow.png")

# FIGURE 4: Admission Rate Patterns
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Admission Rate Across Different Time Cycles', fontsize=16, fontweight='bold', y=1.02)

axes[0].plot(hourly_agg['Hour'], hourly_agg['Admission Rate'], 'o-', color='#A23B72', 
             linewidth=2.5, markersize=8, alpha=0.8)
axes[0].axhline(y=29.15, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Overall Average (29.15%)')
axes[0].set_xlabel('Hour of Day', fontweight='bold')
axes[0].set_ylabel('Admission Rate (%)', fontweight='bold')
axes[0].set_title('Admission Rate by Hour', fontweight='bold', pad=10)
axes[0].set_xticks(range(0, 24, 2))
axes[0].grid(True, alpha=0.3, linestyle='--')
axes[0].legend(framealpha=0.9)

axes[1].bar(dow_agg['DayOfWeekName'], dow_agg['Admission Rate'], color='#A23B72', alpha=0.7)
axes[1].axhline(y=29.15, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Overall Average (29.15%)')
axes[1].set_xlabel('Day of Week', fontweight='bold')
axes[1].set_ylabel('Admission Rate (%)', fontweight='bold')
axes[1].set_title('Admission Rate by Day of Week', fontweight='bold', pad=10)
axes[1].set_xticklabels(dow_agg['DayOfWeekName'], rotation=45, ha='right')
axes[1].grid(True, alpha=0.3, linestyle='--', axis='y')
axes[1].legend(framealpha=0.9)

axes[2].plot(monthly_agg['Month'], monthly_agg['Admission Rate'], 'o-', color='#A23B72', 
             linewidth=2.5, markersize=8, alpha=0.8)
axes[2].axhline(y=29.15, color='r', linestyle='--', linewidth=1.5, alpha=0.7, label='Overall Average (29.15%)')
axes[2].set_xlabel('Month', fontweight='bold')
axes[2].set_ylabel('Admission Rate (%)', fontweight='bold')
axes[2].set_title('Admission Rate by Month', fontweight='bold', pad=10)
axes[2].set_xticks(monthly_agg['Month'])
axes[2].set_xticklabels(monthly_agg['MonthName'], rotation=45, ha='right')
axes[2].grid(True, alpha=0.3, linestyle='--')
axes[2].legend(framealpha=0.9)

plt.tight_layout()
plt.savefig('Visualizations/admission_rate_patterns.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/admission_rate_patterns.png")

# Print summary statistics
print("\n" + "=" * 80)
print("CYCLICAL PATTERN SUMMARY")
print("=" * 80)

print("\nHOURLY PATTERNS:")
print(f"  Peak Hour (Total Encounters): Hour {hourly_agg.loc[hourly_agg['ED Enc'].idxmax(), 'Hour']:.0f} ({hourly_agg['ED Enc'].max():,.0f} encounters)")
print(f"  Lowest Hour: Hour {hourly_agg.loc[hourly_agg['ED Enc'].idxmin(), 'Hour']:.0f} ({hourly_agg['ED Enc'].min():,.0f} encounters)")
print(f"  Peak Hour (Admitted): Hour {hourly_agg.loc[hourly_agg['ED Enc Admitted'].idxmax(), 'Hour']:.0f} ({hourly_agg['ED Enc Admitted'].max():,.0f} encounters)")

print("\nWEEKLY PATTERNS:")
print(f"  Busiest Day (Total Encounters): {dow_agg.loc[dow_agg['ED Enc'].idxmax(), 'DayOfWeekName']} ({dow_agg['ED Enc'].max():,.0f} encounters)")
print(f"  Quietest Day: {dow_agg.loc[dow_agg['ED Enc'].idxmin(), 'DayOfWeekName']} ({dow_agg['ED Enc'].min():,.0f} encounters)")

print("\nMONTHLY PATTERNS:")
print(f"  Busiest Month (Total Encounters): {monthly_agg.loc[monthly_agg['ED Enc'].idxmax(), 'MonthName']} ({monthly_agg['ED Enc'].max():,.0f} encounters)")
print(f"  Quietest Month: {monthly_agg.loc[monthly_agg['ED Enc'].idxmin(), 'MonthName']} ({monthly_agg['ED Enc'].min():,.0f} encounters)")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE - All visualizations saved to Visualizations/ folder")
print("=" * 80)
