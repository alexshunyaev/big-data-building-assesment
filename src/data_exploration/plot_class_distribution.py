"""
Dataset Class Imbalance Analysis

This script examines the processed ViT crops to calculate and visualize
the class distribution, highlighting the extreme class imbalance in the dataset.
"""

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path
import sys

def analyze_and_plot(data_dir: Path, output_path: str) -> None:
    """
    Scans the given data directory, counts images per class, and saves a bar chart.
    """
    if not data_dir.exists():
        print(f"Error: {data_dir} does not exist.")
        sys.exit(1)

    splits = ["train", "test", "hold"]
    classes = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    
    records = []
    
    for split in splits:
        split_dir = data_dir / split
        if not split_dir.exists():
            continue
            
        for cls in classes:
            cls_dir = split_dir / cls
            if cls_dir.exists():
                # Count files ending with .png
                count = sum(1 for _ in cls_dir.glob("*.png"))
                records.append({"Split": split, "Class": cls, "Count": count})

    if not records:
        print("No images found to analyze.")
        sys.exit(1)

    df = pd.DataFrame(records)
    
    from matplotlib.ticker import ScalarFormatter

    # Plotting
    plt.figure(figsize=(10, 6))
    sns.set_theme(style="whitegrid")
    
    # Custom green and red palette
    custom_palette = ["#2ecc71", "#e74c3c", "#3498db"] # Green, Red, Blue (if a 3rd split exists)
    ax = sns.barplot(data=df, x="Class", y="Count", hue="Split", palette=custom_palette)
    plt.title("Building Damage Class Distribution (xBD Dataset)", fontsize=16, weight="bold")
    plt.xlabel("Damage Severity", fontsize=12)
    plt.ylabel("Number of Images", fontsize=12)
    plt.yscale("log")  # Keep log scale to see small classes, but fix labels
    
    # Use normal numbers instead of scientific/power-of-10 notation
    formatter = ScalarFormatter()
    formatter.set_scientific(False)
    ax.yaxis.set_major_formatter(formatter)
    
    # Customize legend
    plt.legend(title="Data Split")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    print(f"Plot saved successfully to: {output_path}")

if __name__ == "__main__":
    _HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = _HERE.parents[1]
    
    VIT_DIR = PROJECT_ROOT / "data" / "vit_crops"
    
    # Save the output plot
    plot_path = sys.argv[1] if len(sys.argv) > 1 else "class_distribution.png"
    analyze_and_plot(VIT_DIR, plot_path)
