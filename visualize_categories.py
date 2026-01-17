import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 8)
plt.rcParams['font.size'] = 10

# Create Visualizations directory if it doesn't exist
os.makedirs('Visualizations', exist_ok=True)

# Load category mapping and volume data
print("Loading category data...")
category_volume = pd.read_csv('category_volume_analysis.csv')
category_mapping = pd.read_csv('reason_categories.csv')

# Load full dataset with categories
df = pd.read_csv('Dataset/DSU-Dataset.csv')
df['Date'] = pd.to_datetime(df['Date'])

# Load category mapping
reason_to_category = pd.read_csv('reason_categories.csv')
reason_to_category_dict = dict(zip(reason_to_category['REASON_VISIT_NAME'], 
                                    reason_to_category['Category']))
df['Reason_Category'] = df['REASON_VISIT_NAME'].map(reason_to_category_dict)

# Sort by volume for better visualization
category_volume = category_volume.sort_values('ED Enc', ascending=False)

# ============================================================================
# VISUALIZATION 1: Category Volume Overview
# ============================================================================

fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle('ED Volume by Reason Category', fontsize=16, fontweight='bold', y=0.995)

# 1. Total Encounters by Category (Bar Chart)
ax1 = axes[0, 0]
colors = sns.color_palette("husl", len(category_volume))
bars = ax1.barh(range(len(category_volume)), category_volume['ED Enc'], color=colors, alpha=0.8)
ax1.set_yticks(range(len(category_volume)))
ax1.set_yticklabels(category_volume['Reason_Category'], fontsize=9)
ax1.set_xlabel('Total Encounters', fontweight='bold')
ax1.set_title('Total Encounters by Category', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--', axis='x')
ax1.invert_yaxis()

# Add value labels
for i, (idx, row) in enumerate(category_volume.iterrows()):
    ax1.text(row['ED Enc'] + 2000, i, f"{row['ED Enc']:,.0f}", 
             va='center', fontsize=8, fontweight='bold')

# 2. Admitted Encounters by Category
ax2 = axes[0, 1]
bars = ax2.barh(range(len(category_volume)), category_volume['ED Enc Admitted'], 
                color=colors, alpha=0.8)
ax2.set_yticks(range(len(category_volume)))
ax2.set_yticklabels(category_volume['Reason_Category'], fontsize=9)
ax2.set_xlabel('Admitted Encounters', fontweight='bold')
ax2.set_title('Admitted Encounters by Category', fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--', axis='x')
ax2.invert_yaxis()

# Add value labels
for i, (idx, row) in enumerate(category_volume.iterrows()):
    ax2.text(row['ED Enc Admitted'] + 500, i, f"{row['ED Enc Admitted']:,.0f}", 
             va='center', fontsize=8, fontweight='bold')

# 3. Admission Rate by Category
ax3 = axes[1, 0]
bars = ax3.barh(range(len(category_volume)), category_volume['Admission Rate'], 
                color=colors, alpha=0.8)
ax3.set_yticks(range(len(category_volume)))
ax3.set_yticklabels(category_volume['Reason_Category'], fontsize=9)
ax3.set_xlabel('Admission Rate (%)', fontweight='bold')
ax3.set_title('Admission Rate by Category', fontweight='bold', pad=10)
ax3.axvline(x=29.15, color='r', linestyle='--', linewidth=2, alpha=0.7, label='Overall Average (29.15%)')
ax3.grid(True, alpha=0.3, linestyle='--', axis='x')
ax3.invert_yaxis()
ax3.legend(loc='lower right', framealpha=0.9)

# Add value labels
for i, (idx, row) in enumerate(category_volume.iterrows()):
    ax3.text(row['Admission Rate'] + 1, i, f"{row['Admission Rate']:.1f}%", 
             va='center', fontsize=8, fontweight='bold')

# 4. Category Distribution (Pie Chart - Top 10 by volume)
ax4 = axes[1, 1]
top_categories = category_volume.head(10)
other_volume = category_volume.iloc[10:]['ED Enc'].sum()
other_admitted = category_volume.iloc[10:]['ED Enc Admitted'].sum()

pie_data = list(top_categories['ED Enc']) + [other_volume]
pie_labels = list(top_categories['Reason_Category']) + ['Other Categories']
pie_colors = colors[:10] + ['#cccccc']

wedges, texts, autotexts = ax4.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                    colors=pie_colors, startangle=90, textprops={'fontsize': 8})
ax4.set_title('Top 10 Categories by Volume\n(% of Total Encounters)', 
              fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('Visualizations/category_volume_overview.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/category_volume_overview.png")

# ============================================================================
# VISUALIZATION 2: Category Trends Over Time
# ============================================================================

# Aggregate by category and month
df['YearMonth'] = df['Date'].dt.to_period('M')
category_monthly = df.groupby(['YearMonth', 'Reason_Category']).agg({
    'ED Enc': 'sum',
    'ED Enc Admitted': 'sum'
}).reset_index()
category_monthly['YearMonth'] = category_monthly['YearMonth'].astype(str)

# Convert YearMonth strings to datetime for better x-axis handling
category_monthly['YearMonth_dt'] = pd.to_datetime(category_monthly['YearMonth'])

# Get top 6 categories by volume for trend plot
top_6_categories = category_volume.head(6)['Reason_Category'].tolist()

fig, axes = plt.subplots(3, 2, figsize=(20, 14))
fig.suptitle('Monthly Trends by Category (Top 6)', fontsize=16, fontweight='bold', y=0.995)

axes = axes.flatten()
for idx, category in enumerate(top_6_categories):
    ax = axes[idx]
    cat_data = category_monthly[category_monthly['Reason_Category'] == category].copy()
    cat_data = cat_data.sort_values('YearMonth_dt')
    
    ax.plot(cat_data['YearMonth_dt'], cat_data['ED Enc'], 'o-', color='#2E86AB', 
            linewidth=1.5, markersize=3, alpha=0.7, label='Total Encounters')
    ax.plot(cat_data['YearMonth_dt'], cat_data['ED Enc Admitted'], 's-', color='#A23B72', 
            linewidth=1.5, markersize=3, alpha=0.7, label='Admitted Encounters')
    
    ax.set_title(category, fontweight='bold', pad=10)
    ax.set_ylabel('Encounters', fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Set x-axis to show fewer labels (every 6 months)
    from matplotlib.dates import YearLocator, MonthLocator, DateFormatter
    ax.xaxis.set_major_locator(MonthLocator(interval=6))  # Every 6 months
    ax.xaxis.set_minor_locator(MonthLocator(interval=3))  # Minor ticks every 3 months
    ax.xaxis.set_major_formatter(DateFormatter('%Y-%m'))  # Format: 2018-01
    
    # Rotate labels and set alignment
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='x', which='major', labelsize=8)
    
    # Show only every 6th month label to reduce clutter
    if idx >= 4:  # Bottom row
        ax.set_xlabel('Month', fontweight='bold')
    
    # Show legend only on first subplot
    if idx == 0:
        ax.legend(loc='upper left', framealpha=0.9, fontsize=8)

plt.tight_layout()
plt.savefig('Visualizations/category_trends_over_time.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/category_trends_over_time.png")

# ============================================================================
# VISUALIZATION 3: Category Distribution Summary
# ============================================================================

fig, axes = plt.subplots(1, 2, figsize=(16, 6))
fig.suptitle('Reason Category Distribution', fontsize=16, fontweight='bold', y=1.02)

# Number of reasons per category
category_counts = category_mapping['Category'].value_counts().sort_values(ascending=True)
ax1 = axes[0]
bars = ax1.barh(range(len(category_counts)), category_counts.values, 
                color=sns.color_palette("viridis", len(category_counts)), alpha=0.8)
ax1.set_yticks(range(len(category_counts)))
ax1.set_yticklabels(category_counts.index, fontsize=9)
ax1.set_xlabel('Number of Reasons', fontweight='bold')
ax1.set_title('Number of Unique Reasons per Category', fontweight='bold', pad=10)
ax1.grid(True, alpha=0.3, linestyle='--', axis='x')

# Add value labels
for i, count in enumerate(category_counts.values):
    ax1.text(count + 2, i, f"{count}", va='center', fontsize=9, fontweight='bold')

# Volume percentage pie chart
ax2 = axes[1]
top_8 = category_volume.head(8)
other_vol = category_volume.iloc[8:]['ED Enc'].sum()
pie_data = list(top_8['ED Enc']) + [other_vol]
pie_labels = list(top_8['Reason_Category']) + ['Other Categories']
pie_colors = sns.color_palette("Set2", 9)

wedges, texts, autotexts = ax2.pie(pie_data, labels=pie_labels, autopct='%1.1f%%', 
                                    colors=pie_colors, startangle=90, textprops={'fontsize': 8})
ax2.set_title('Volume Distribution by Category\n(% of Total Encounters)', 
              fontweight='bold', pad=10)

plt.tight_layout()
plt.savefig('Visualizations/category_distribution.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/category_distribution.png")

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print("\nGenerated visualizations:")
print("  1. category_volume_overview.png - Volume and admission rates by category")
print("  2. category_trends_over_time.png - Monthly trends for top 6 categories")
print("  3. category_distribution.png - Category distribution summary")
