"""
YOLO Multi-Class Evaluation & Visualization Module

Standalone script for generating training plots and classification metrics.
Reads artifacts produced by train_yolo_multi.py and generates:
    - Training/Validation loss curves and Val Accuracy (PNG)
    - Per-class precision, recall, F1 metrics (CSV)
    - Confusion matrix heatmap (PNG)
    - Classification report (console + CSV)

Usage:
    python src/evaluation/metrics_yolo.py
"""
import csv
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)
from torchvision.datasets import ImageFolder
from ultralytics import YOLO

# Add project src directory to path for imports
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
PROJECT_ROOT = _SRC.parent

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# Damage class names
CLASS_NAMES = ['not-damaged', 'damaged']
RUN_NAME = "single_binary_yolo11l_cls"


def plot_training_curves(history_path: str | Path, output_dir: str | Path):
    """
    Plots training/validation loss, validation accuracy, and learning rate curves from YOLO's results.csv.
    """
    try:
        df = pd.read_csv(history_path)
    except FileNotFoundError:
        print(f"[!] Warning: {history_path} not found. Cannot plot training curves.")
        return
        
    df.columns = [c.strip() for c in df.columns]

    epochs = df['epoch'].values
    train_loss = df['train/loss'].values
    val_loss = df['val/loss'].values
    val_acc = df['metrics/accuracy_top1'].values
    lr = df['lr/pg0'].values

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    axes[0].plot(epochs, train_loss, linewidth=2, color='#e74c3c', label='Train Loss', alpha=0.8)
    axes[0].plot(epochs, val_loss, linewidth=2, color='#c0392b', label='Val Loss', linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Validation Loss', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(epochs, val_acc, linewidth=2, color='#27ae60', label='Val Acc (Top-1)')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # --- Learning Rate ---
    axes[2].plot(epochs, lr, linewidth=2, color='#3498db', label='Learning Rate')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('LR', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plot_path = Path(output_dir) / f"{RUN_NAME}_training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Training curves saved to {plot_path}")


def plot_confusion_matrix(cm: np.ndarray, output_dir: str | Path):
    """
    Plots and saves a confusion matrix heatmap to a PNG file.
    """
    fig, ax = plt.subplots(figsize=(8, 7))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, linecolor='white'
    )
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title(f'Confusion Matrix — {RUN_NAME}', fontsize=14, fontweight='bold')
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)

    plt.tight_layout()
    cm_path = Path(output_dir) / f"{RUN_NAME}_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Confusion matrix saved to {cm_path}")


def save_metrics_csv(all_labels: np.ndarray, all_preds: np.ndarray, output_dir: str | Path):
    """
    Computes classification metrics and saves them to a CSV report.
    """
    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    precision_w = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_w = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    overall_acc = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    metrics_path = Path(output_dir) / f"{RUN_NAME}_metrics.csv"
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow([f"=== YOLO Classification Metrics ({RUN_NAME}) ==="])
        writer.writerow([])

        writer.writerow(["Final Evaluation Accuracy", f"{overall_acc:.4f}"])
        writer.writerow([])

        writer.writerow(["Class", "Precision", "Recall", "F1-Score", "Support"])
        for i, name in enumerate(CLASS_NAMES):
            support = int(np.sum(all_labels == i))
            writer.writerow([
                name,
                f"{precision_per_class[i]:.4f}",
                f"{recall_per_class[i]:.4f}",
                f"{f1_per_class[i]:.4f}",
                support
            ])
        writer.writerow([])

        writer.writerow(["Weighted Avg", f"{precision_w:.4f}", f"{recall_w:.4f}", f"{f1_w:.4f}", len(all_labels)])
        writer.writerow([])

        writer.writerow(["=== Confusion Matrix ==="])
        writer.writerow(["Predicted ->"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + [str(v) for v in cm[i]])

    print(f"[*] Metrics CSV saved to {metrics_path}")
    return cm, overall_acc


def evaluate_model(model_path: str | Path, data_dir: str | Path):
    """
    Loads the trained YOLO model and evaluates it against the validation data.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on device: {device}")

    dataset = ImageFolder(root=str(data_dir))
    samples = dataset.samples

    print(f"[*] Loading best YOLO model from {model_path}...")
    model = YOLO(model_path)

    all_preds = []
    all_labels = []

    print(f"[*] Running inference on {len(samples)} images in batches...")
    
    batch_size = 64
    for i in range(0, len(samples), batch_size):
        batch = samples[i:i+batch_size]
        paths = [b[0] for b in batch]
        labels = [b[1] for b in batch]
        all_labels.extend(labels)
        
        # Verbose=False so it doesn't spam console
        results = model.predict(paths, verbose=False, device=device.index if device.type == "cuda" else "cpu")
        
        # results contains a list of Result objects. Extract top1 class prediction.
        preds = [res.probs.top1 for res in results]
        all_preds.extend(preds)
        
        if (i + batch_size) % 640 == 0 or (i + batch_size) >= len(samples):
            print(f"  -> Processed {min(i+batch_size, len(samples))}/{len(samples)} images")

    return np.array(all_preds), np.array(all_labels)


def run_full_evaluation():
    """
    Orchestrates the evaluation pipeline.
    """
    # Define paths
    data_dir = PROJECT_ROOT / "data" / "yolo_single_binary" / "val"
    yolo_run_dir = PROJECT_ROOT / "results" / "res_yolo" / RUN_NAME
    model_path = yolo_run_dir / "weights" / "best.pt"
    history_path = yolo_run_dir / "results.csv"
    
    # We will output our custom artifacts here
    output_dir = PROJECT_ROOT / "results" / "res_yolo" / f"{RUN_NAME}_custom_eval"
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"  YOLO Classification — Full Evaluation Pipeline")
    print("=" * 60)

    print("\n[Step 1/4] Plotting training curves...")
    plot_training_curves(history_path, output_dir)

    if not model_path.exists():
        print(f"\n[ERROR] Model weights not found at {model_path}")
        print("         Run train_yolo_multi.py first.")
        return

    print("\n[Step 2/4] Running model inference...")
    all_preds, all_labels = evaluate_model(model_path, data_dir)

    print("\n[Step 3/4] Computing and saving metrics...")
    cm, overall_acc = save_metrics_csv(all_labels, all_preds, output_dir)

    print("\n[Step 4/4] Generating confusion matrix heatmap...")
    plot_confusion_matrix(cm, output_dir)

    print("\n" + "=" * 60)
    print("  CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, zero_division=0))
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("\n[*] All evaluation outputs saved to:", output_dir)


if __name__ == "__main__":
    run_full_evaluation()
