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

# Get all injury types by volume
injury_type_volume = injury_df.groupby('REASON_VISIT_NAME').agg({
    'ED Enc': 'sum'
}).reset_index().sort_values('ED Enc', ascending=False)

print(f"\nTotal unique injury types: {len(injury_type_volume)}")

# Aggregate by injury type, year, and month
print("Aggregating data by injury type, year, and month...")
injury_year_month = injury_df.groupby([
    'REASON_VISIT_NAME', 'Year', 'Month'
]).agg({
    'ED Enc': 'sum'
}).reset_index()

# Month names
month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
               'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

def create_faceted_heatmap(injury_list, title_suffix, filename):
    """Create faceted heatmap for a list of injuries"""
    n_injuries = len(injury_list)
    rows = (n_injuries + 1) // 2  # Round up for 2 columns
    if rows == 0:
        rows = 1
    
    fig, axes = plt.subplots(rows, 2, figsize=(18, 4 * rows + 2))
    fig.suptitle(f'Annual Patterns: {title_suffix} by Year and Month', 
                  fontsize=16, fontweight='bold', y=0.995)
    
    # Handle single row case
    if rows == 1:
        axes = axes.reshape(1, -1)
    axes = axes.flatten()
    
    for idx, injury_type in enumerate(injury_list):
        ax = axes[idx]
        injury_data = injury_year_month[injury_year_month['REASON_VISIT_NAME'] == injury_type]
        
        # Create a matrix with Year as rows, Month as columns
        pivot_data = injury_data.pivot(index='Year', columns='Month', values='ED Enc')
        pivot_data = pivot_data.sort_index()
        
        # Reindex to ensure all 12 months are present (fill missing with 0)
        pivot_data = pivot_data.reindex(columns=range(1, 13), fill_value=0)
        pivot_data.columns = month_names
        
        # Calculate total for this injury type
        total_encounters = pivot_data.sum().sum()
        
        sns.heatmap(pivot_data, cmap='YlOrRd', ax=ax,
                    cbar_kws={'label': 'Encounters'},
                    annot=True, fmt='.0f', annot_kws={'size': 7},
                    linewidths=0.5, cbar=True, square=False)
        
        ax.set_title(f"{injury_type}\n(Total: {total_encounters:,.0f})", 
                     fontweight='bold', pad=12, fontsize=11)
        ax.set_xlabel('Month', fontweight='bold')
        ax.set_ylabel('Year', fontweight='bold')
    
    # Hide unused subplots
    for idx in range(len(injury_list), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f'Visualizations/{filename}', dpi=300, bbox_inches='tight')
    print(f"Saved: Visualizations/{filename}")

# Get all injury types ranked by volume
all_injuries = injury_type_volume['REASON_VISIT_NAME'].tolist()

print("\n" + "=" * 80)
print("Creating faceted heatmaps for additional injury types...")
print("=" * 80)

# Group 2: Injuries ranked 7-12 (next 6 after top 6)
if len(all_injuries) >= 12:
    injuries_7_12 = all_injuries[6:12]
    print(f"\nGroup 2: Injuries ranked 7-12")
    for idx, inj in enumerate(injuries_7_12, 7):
        vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
        print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")
    create_faceted_heatmap(injuries_7_12, 'Injury Types #7-12', 
                           'injury_year_month_faceted_7-12.png')

# Group 3: Injuries ranked 13-18
if len(all_injuries) >= 18:
    injuries_13_18 = all_injuries[12:18]
    print(f"\nGroup 3: Injuries ranked 13-18")
    for idx, inj in enumerate(injuries_13_18, 13):
        vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
        print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")
    create_faceted_heatmap(injuries_13_18, 'Injury Types #13-18', 
                           'injury_year_month_faceted_13-18.png')

# Group 4: Injuries ranked 19-24
if len(all_injuries) >= 24:
    injuries_19_24 = all_injuries[18:24]
    print(f"\nGroup 4: Injuries ranked 19-24")
    for idx, inj in enumerate(injuries_19_24, 19):
        vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
        print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")
    create_faceted_heatmap(injuries_19_24, 'Injury Types #19-24', 
                           'injury_year_month_faceted_19-24.png')

# Group 5: Injuries ranked 25-30
if len(all_injuries) >= 30:
    injuries_25_30 = all_injuries[24:30]
    print(f"\nGroup 5: Injuries ranked 25-30")
    for idx, inj in enumerate(injuries_25_30, 25):
        vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
        print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")
    create_faceted_heatmap(injuries_25_30, 'Injury Types #25-30', 
                           'injury_year_month_faceted_25-30.png')

# Group 6: Injuries ranked 31-36
if len(all_injuries) >= 36:
    injuries_31_36 = all_injuries[30:36]
    print(f"\nGroup 6: Injuries ranked 31-36")
    for idx, inj in enumerate(injuries_31_36, 31):
        vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
        print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")
    create_faceted_heatmap(injuries_31_36, 'Injury Types #31-36', 
                           'injury_year_month_faceted_31-36.png')

# Group 7: Injuries ranked 37-42
if len(all_injuries) >= 42:
    injuries_37_42 = all_injuries[36:42]
    print(f"\nGroup 7: Injuries ranked 37-42")
    for idx, inj in enumerate(injuries_37_42, 37):
        vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
        print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")
    create_faceted_heatmap(injuries_37_42, 'Injury Types #37-42', 
                           'injury_year_month_faceted_37-42.png')

# Group 8: Remaining injuries (if any)
if len(all_injuries) > 42:
    remaining_injuries = all_injuries[42:48] if len(all_injuries) >= 48 else all_injuries[42:]
    if len(remaining_injuries) > 0:
        print(f"\nGroup 8: Injuries ranked 43-{43 + len(remaining_injuries) - 1}")
        for idx, inj in enumerate(remaining_injuries, 43):
            vol = injury_type_volume[injury_type_volume['REASON_VISIT_NAME'] == inj]['ED Enc'].values[0]
            print(f"  {idx:2d}. {inj:30s} - {vol:>8,.0f} encounters")
        create_faceted_heatmap(remaining_injuries, 
                               f'Injury Types #43-{43 + len(remaining_injuries) - 1}', 
                               'injury_year_month_faceted_remaining.png')

print("\n" + "=" * 80)
print("VISUALIZATION COMPLETE")
print("=" * 80)
print(f"\nTotal injury types visualized: {len(all_injuries)}")
print(f"Number of visualization groups created: {min((len(all_injuries) + 5) // 6, 8)}")
