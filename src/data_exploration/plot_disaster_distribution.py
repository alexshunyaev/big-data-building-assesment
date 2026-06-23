"""
Disaster Type Distribution Analysis

This script examines the raw xBD dataset JSON labels to calculate and visualize
the distribution of buildings across different disaster types.
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys
from tqdm import tqdm

def analyze_disasters(raw_dir: Path, output_path: str) -> None:
    """
    Scans the raw data directories, counts buildings per disaster type,
    and saves a bar chart.
    """
    if not raw_dir.exists():
        print(f"Error: {raw_dir} does not exist.")
        sys.exit(1)

    splits = ["train", "test", "hold"]
    records = []
    
    for split in splits:
        labels_dir = raw_dir / split / "labels"
        if not labels_dir.exists():
            continue
            
        json_files = list(labels_dir.glob("*_post_disaster.json"))
        if not json_files:
            continue
            
        print(f"Processing '{split}' split...")
        for json_path in tqdm(json_files, desc=split):
            # The disaster name is everything before the first underscore in the filename
            # e.g. hurricane-matthew_000000_post_disaster.json -> hurricane-matthew
            disaster_name = json_path.name.split('_')[0]
            
            with open(json_path, 'r') as f:
                data = json.load(f)
                
            # Count the number of building polygons
            buildings = data.get('features', {}).get('xy', [])
            num_buildings = 0
            for b in buildings:
                # Only count classified damage polygons
                if b.get('properties', {}).get('subtype') in ["no-damage", "minor-damage", "major-damage", "destroyed"]:
                    num_buildings += 1
            
            records.append({
                "Split": split,
                "Disaster Type": disaster_name.replace('-', ' ').title(),
                "Buildings": num_buildings
            })

    if not records:
        print("No label files found to analyze.")
        sys.exit(1)

    # Group by Disaster Type and sum buildings
    df_records = pd.DataFrame(records)
    df_agg = df_records.groupby(["Disaster Type", "Split"])["Buildings"].sum().reset_index()
    
    # Sort disasters by total buildings descending
    total_buildings = df_agg.groupby("Disaster Type")["Buildings"].sum().sort_values(ascending=False).index
    
    # Plotting
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Using a bright palette
    ax = sns.barplot(data=df_agg, x="Buildings", y="Disaster Type", hue="Split", palette="bright", order=total_buildings)
    
    plt.title("Building Distribution by Disaster Type (xBD Dataset)", fontsize=16, weight="bold")
    plt.xlabel("Number of Buildings", fontsize=12)
    plt.ylabel("Disaster Type", fontsize=12)
    
    # Customize legend
    plt.legend(title="Data Split")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    _HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = _HERE.parents[1]
    
    RAW_DIR = PROJECT_ROOT / "data" / "raw"
    
    # Save the output plot
    plot_path = sys.argv[1] if len(sys.argv) > 1 else "disaster_distribution.png"
    analyze_disasters(RAW_DIR, plot_path)
