import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (14, 10)
plt.rcParams['font.size'] = 11

# Create Visualizations directory if it doesn't exist
os.makedirs('Visualizations', exist_ok=True)

# Load category volume data
print("Loading category volume data...")
category_volume = pd.read_csv('category_volume_analysis.csv')

# Sort by total encounters in descending order
category_volume = category_volume.sort_values('ED Enc', ascending=False)

print("\n" + "=" * 80)
print("CATEGORY RANKINGS BY ENCOUNTER VOLUME")
print("=" * 80)
print("\nRank | Category | Total Encounters | Admitted | Admission Rate (%)")
print("-" * 80)

for idx, (_, row) in enumerate(category_volume.iterrows(), 1):
    print(f"{idx:4d} | {row['Reason_Category']:35s} | {row['ED Enc']:>15,.0f} | {row['ED Enc Admitted']:>8,.0f} | {row['Admission Rate']:>6.2f}%")

# Create pie chart
fig, ax = plt.subplots(figsize=(14, 10))
fig.suptitle('ED Encounters by Reason Category\n(Ranked by Volume)', 
              fontsize=16, fontweight='bold', y=0.98)

# Use all categories for the pie chart
colors = sns.color_palette("Set3", len(category_volume))
if len(category_volume) > 12:
    colors = sns.color_palette("husl", len(category_volume))

# Create labels with volume and percentage
labels = []
for _, row in category_volume.iterrows():
    pct = (row['ED Enc'] / category_volume['ED Enc'].sum()) * 100
    label = f"{row['Reason_Category']}\n({row['ED Enc']:,.0f} - {pct:.1f}%)"
    labels.append(label)

# Create pie chart
wedges, texts, autotexts = ax.pie(
    category_volume['ED Enc'], 
    labels=labels,
    autopct='',  # We're including percentage in labels
    colors=colors,
    startangle=90,
    textprops={'fontsize': 9, 'fontweight': 'normal'},
    pctdistance=0.85
)

# Improve text positioning
for autotext in autotexts:
    autotext.set_color('white')
    autotext.set_fontweight('bold')
    autotext.set_fontsize(9)

# Add total encounters text in center
total_encounters = category_volume['ED Enc'].sum()
ax.text(0, 0, f'Total:\n{total_encounters:,.0f}\nEncounters', 
        ha='center', va='center', fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Visualizations/category_volume_pie_chart.png', dpi=300, bbox_inches='tight')
print("\n" + "=" * 80)
print("Saved: Visualizations/category_volume_pie_chart.png")
print("=" * 80)

# Create a second version with top N categories grouped as "Other"
fig2, ax2 = plt.subplots(figsize=(12, 10))
fig2.suptitle('ED Encounters by Reason Category\n(Top Categories)', 
               fontsize=16, fontweight='bold', y=0.98)

# Show top 10 categories, group the rest as "Other"
top_n = 10
top_categories = category_volume.head(top_n)
other_volume = category_volume.iloc[top_n:]['ED Enc'].sum()
other_admitted = category_volume.iloc[top_n:]['ED Enc Admitted'].sum()

pie_data = list(top_categories['ED Enc']) + [other_volume]
pie_labels = []
for _, row in top_categories.iterrows():
    pct = (row['ED Enc'] / category_volume['ED Enc'].sum()) * 100
    pie_labels.append(f"{row['Reason_Category']}\n({row['ED Enc']:,.0f} - {pct:.1f}%)")

if other_volume > 0:
    other_pct = (other_volume / category_volume['ED Enc'].sum()) * 100
    pie_labels.append(f"Other Categories\n({other_volume:,.0f} - {other_pct:.1f}%)")

pie_colors = colors[:top_n] + ['#cccccc']

wedges2, texts2, autotexts2 = ax2.pie(
    pie_data,
    labels=pie_labels,
    autopct='',
    colors=pie_colors,
    startangle=90,
    textprops={'fontsize': 10, 'fontweight': 'normal'}
)

# Add total encounters text in center
ax2.text(0, 0, f'Total:\n{total_encounters:,.0f}\nEncounters', 
         ha='center', va='center', fontsize=12, fontweight='bold',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

plt.tight_layout()
plt.savefig('Visualizations/category_volume_pie_chart_top10.png', dpi=300, bbox_inches='tight')
print("Saved: Visualizations/category_volume_pie_chart_top10.png")
print("=" * 80)
