"""
Siamese Binary ViT Evaluation & Visualization Module

Standalone script for generating training plots and classification metrics
for the Siamese binary (damaged vs. not-damaged) Vision Transformer model.
"""
import csv
import json
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# Add project src directory to path for imports
_HERE = Path(__file__).resolve().parent
_SRC = _HERE.parent
PROJECT_ROOT = _SRC.parent

if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from torch.utils.data import DataLoader
from data_prep.vit_dataset_binary import BinaryBuildingDamageDataset
from models_vit.vit_binary_siamese import SiameseChangeViT

# Binary class names
CLASS_NAMES = ['not-damaged', 'damaged']


def plot_training_curves(history_path: str | Path, output_dir: str | Path):
    """
    Plots training/validation loss, accuracy, and learning rate curves and saves them to PNG.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    epochs = range(1, len(history['train_loss']) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # --- Loss ---
    axes[0].plot(epochs, history['train_loss'], linewidth=2, color='#e74c3c', label='Train Loss', alpha=0.8)
    axes[0].plot(epochs, history['val_loss'], linewidth=2, color='#c0392b', label='Val Loss', linestyle='--')
    axes[0].set_xlabel('Epoch', fontsize=12)
    axes[0].set_ylabel('Loss', fontsize=12)
    axes[0].set_title('Training & Val Loss (Siamese Binary)', fontsize=14, fontweight='bold')
    axes[0].legend(fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # --- Accuracy ---
    axes[1].plot(epochs, history['train_acc'], linewidth=2, color='#2ecc71', label='Train Acc', alpha=0.8)
    axes[1].plot(epochs, history['val_acc'], linewidth=2, color='#27ae60', label='Val Acc', linestyle='--')
    axes[1].set_xlabel('Epoch', fontsize=12)
    axes[1].set_ylabel('Accuracy', fontsize=12)
    axes[1].set_title('Training & Val Accuracy (Siamese Binary)', fontsize=14, fontweight='bold')
    axes[1].legend(fontsize=11)
    axes[1].grid(True, alpha=0.3)

    # --- Learning Rate ---
    axes[2].plot(epochs, history['lr'], linewidth=2, color='#3498db', label='Learning Rate')
    axes[2].set_xlabel('Epoch', fontsize=12)
    axes[2].set_ylabel('LR', fontsize=12)
    axes[2].set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
    axes[2].legend(fontsize=11)
    axes[2].grid(True, alpha=0.3)
    axes[2].ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

    plt.tight_layout()
    plot_path = Path(output_dir) / "vit_siamese_training_curves.png"
    plt.savefig(str(plot_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Training curves saved to {plot_path}")


def plot_confusion_matrix(cm: np.ndarray, output_dir: str | Path):
    """
    Plots and saves a binary confusion matrix heatmap to a PNG file.
    """
    fig, ax = plt.subplots(figsize=(7, 6))

    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues',
        xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES,
        ax=ax, linewidths=0.5, linecolor='white'
    )
    ax.set_xlabel('Predicted', fontsize=13)
    ax.set_ylabel('Actual', fontsize=13)
    ax.set_title('Confusion Matrix — Siamese Binary ViT', fontsize=14, fontweight='bold')
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)

    plt.tight_layout()
    cm_path = Path(output_dir) / "vit_siamese_confusion_matrix.png"
    plt.savefig(str(cm_path), dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[*] Confusion matrix saved to {cm_path}")


def save_metrics_csv(all_labels: np.ndarray, all_preds: np.ndarray,
                     history_path: str | Path, output_dir: str | Path):
    """
    Computes binary classification metrics and saves them to a CSV report.
    """
    with open(history_path, 'r') as f:
        history = json.load(f)

    precision_per_class = precision_score(all_labels, all_preds, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_preds, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0)

    precision_w = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall_w = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_w = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    # Binary-specific metrics (positive class = damaged = 1)
    precision_bin = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall_bin = recall_score(all_labels, all_preds, average='binary', zero_division=0)
    f1_bin = f1_score(all_labels, all_preds, average='binary', zero_division=0)

    overall_acc = np.mean(all_preds == all_labels)
    cm = confusion_matrix(all_labels, all_preds)

    metrics_path = Path(output_dir) / "vit_siamese_metrics.csv"
    with open(metrics_path, 'w', newline='') as f:
        writer = csv.writer(f)

        writer.writerow(["=== Siamese Binary ViT Classification Metrics ==="])
        writer.writerow([])

        writer.writerow(["Training Summary"])
        writer.writerow(["Total Epochs Run", len(history['train_loss'])])
        writer.writerow(["Final Train Loss", f"{history['train_loss'][-1]:.4f}"])
        writer.writerow(["Final Train Accuracy", f"{history['train_acc'][-1]:.4f}"])
        writer.writerow(["Best Validation Loss", f"{min(history['val_loss']):.4f}"])
        writer.writerow(["Final Evaluation Accuracy", f"{overall_acc:.4f}"])
        writer.writerow([])

        writer.writerow(["Per-Class Metrics"])
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

        writer.writerow(["Binary Metrics (positive class = damaged)"])
        writer.writerow(["Precision", f"{precision_bin:.4f}"])
        writer.writerow(["Recall", f"{recall_bin:.4f}"])
        writer.writerow(["F1-Score", f"{f1_bin:.4f}"])
        writer.writerow([])

        writer.writerow(["=== Confusion Matrix ==="])
        writer.writerow(["Predicted ->"] + CLASS_NAMES)
        for i, name in enumerate(CLASS_NAMES):
            writer.writerow([name] + [str(v) for v in cm[i]])

    print(f"[*] Metrics CSV saved to {metrics_path}")
    return cm, overall_acc


def evaluate_model(model_path: str | Path, data_dir: str | Path, output_dir: str | Path):
    """
    Loads the trained Siamese binary ViT model and evaluates it.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Evaluating on device: {device}")

    dataset = BinaryBuildingDamageDataset(root_dir=data_dir, augment=False)
    loader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=device.type == 'cuda'
    )

    print("[*] Loading best Siamese Binary ViT model...")
    model = SiameseChangeViT(
        img_size=224, patch_size=16, in_channels=3,
        embed_dim=256, depth=6, num_heads=8
    ).to(device)
    model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
    model.eval()

    all_preds = []
    all_labels = []

    print("[*] Running inference...")
    with torch.no_grad():
        for pre_imgs, post_imgs, labels in loader:
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)

            logits = model(pre_imgs, post_imgs).squeeze(1)           # (B,)
            preds = (torch.sigmoid(logits) >= 0.5).long()            # binary threshold

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(np.array(labels))

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    return all_preds, all_labels


def run_full_evaluation():
    """
    Orchestrates the evaluation pipeline.
    """
    data_dir = PROJECT_ROOT / "data" / "vit_crops" / "train"
    results_dir = PROJECT_ROOT / "results" / "res_vit_binary_siamese"
    model_path = PROJECT_ROOT / "results" / "models" / "best_vit_binary_siamese.pth"
    history_path = results_dir / "training_history_binary_siamese.json"

    results_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Siamese Binary ViT — Full Evaluation Pipeline")
    print("=" * 60)

    if history_path.exists():
        print("\n[Step 1/4] Plotting training curves...")
        plot_training_curves(history_path, results_dir)
    else:
        print(f"\n[Step 1/4] SKIPPED — {history_path} not found")

    if not model_path.exists():
        print(f"\n[ERROR] Model not found at {model_path}")
        print("         Run train_vit_binary_siamese.py first.")
        return

    print("\n[Step 2/4] Running model inference...")
    all_preds, all_labels = evaluate_model(model_path, data_dir, results_dir)

    print("\n[Step 3/4] Computing and saving metrics...")
    cm, overall_acc = save_metrics_csv(all_labels, all_preds, history_path, results_dir)

    print("\n[Step 4/4] Generating confusion matrix heatmap...")
    plot_confusion_matrix(cm, results_dir)

    print("\n" + "=" * 60)
    print("  SIAMESE BINARY CLASSIFICATION REPORT")
    print("=" * 60)
    print(classification_report(all_labels, all_preds,
                                target_names=CLASS_NAMES, zero_division=0))
    print(f"Overall Accuracy: {overall_acc:.4f}")
    print(f"\nConfusion Matrix:\n{cm}")
    print("\n[*] All evaluation outputs saved to:", results_dir)


if __name__ == "__main__":
    run_full_evaluation()
