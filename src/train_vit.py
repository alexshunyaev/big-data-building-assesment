"""
ViT Training Script

This module defines the training loop, focal loss, and evaluation metrics
for training the Custom Change Vision Transformer.
"""

import json
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data_prep.vit_dataset import BuildingDamageDataset
from models_vit.vit import CustomChangeViT


class FocalLoss(nn.Module):
    """
    Focal Loss: Down-weights easy/well-classified examples to concentrate
    on hard, misclassified samples. Handles class imbalance via alpha weights.
    """
    def __init__(self, alpha=None, gamma=2.0, label_smoothing=0.0):
        super().__init__()
        self.alpha = alpha              # per-class weight tensor
        self.gamma = gamma              # focusing parameter
        self.label_smoothing = label_smoothing

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            weight=self.alpha,
            label_smoothing=self.label_smoothing,
            reduction='none'
        )
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()


def train_model():
    """
    Main training function. Sets up datasets, dataloaders, model, optimizer,
    and executes the training loop with validation.
    """
    # ---------------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------------
    _HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = _HERE.parent
    
    DATA_DIR = PROJECT_ROOT / "data" / "vit_crops" / "train"
    SAVE_DIR = PROJECT_ROOT / "results" / "models"
    RESULTS_DIR = PROJECT_ROOT / "results" / "res_vit"
    
    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    warmup_epochs = 5
    patience = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")

    # ---------------------------------------------------------------
    # 2. Dataset Initialization & Split
    # ---------------------------------------------------------------
    print("\n[*] Initializing Dataset and splitting 80/20...")
    
    full_train_dataset = BuildingDamageDataset(root_dir=DATA_DIR, augment=True)
    full_val_dataset = BuildingDamageDataset(root_dir=DATA_DIR, augment=False)

    dataset_size = len(full_train_dataset)
    indices = torch.randperm(dataset_size).tolist()
    val_split = int(0.2 * dataset_size)
    
    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    print(f"[*] Total Images: {dataset_size} | Training: {len(train_subset)} | Validation: {len(val_subset)}")

    # Balanced Sampling
    train_labels = [full_train_dataset.labels[i] for i in train_indices]
    class_sample_counts = torch.tensor(
        [train_labels.count(c) for c in range(len(full_train_dataset.class_to_idx))],
        dtype=torch.float
    )
    per_sample_weight = 1.0 / torch.sqrt(class_sample_counts[train_labels])
    train_sampler = WeightedRandomSampler(
        weights=per_sample_weight,
        num_samples=len(per_sample_weight),
        replacement=True
    )
    print(f"[*] Balanced sampler active — inverse class counts: {class_sample_counts.tolist()}")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4 if device.type == 'cuda' else 0, pin_memory=device.type == 'cuda'
    )
    
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0, pin_memory=device.type == 'cuda'
    )

    # ---------------------------------------------------------------
    # 3. Setup Model, Optimizer, Scheduler
    # ---------------------------------------------------------------
    model = CustomChangeViT(
        img_size=224, patch_size=16, in_channels=9, num_classes=4,
        embed_dim=256, depth=6, num_heads=8, drop_path_rate=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = FocalLoss(alpha=None, gamma=2.0, label_smoothing=0.1)

    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return 0.01 + (1.0 - 0.01) * (epoch / warmup_epochs)
        else:
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.01 + (1.0 - 0.01) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---------------------------------------------------------------
    # 4. Training Loop
    # ---------------------------------------------------------------
    print(f"\n[*] Starting Training Loop ({epochs} epochs, patience={patience})...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'epoch_time': []}
    history_path = RESULTS_DIR / "training_history_9ch_withmixup.json"

    for epoch in range(epochs):
        start_time = time.time()
        
        # --- TRAIN PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        
        print(f"\n====> Epoch {epoch+1}/{epochs} starting...")
        for batch_idx, (pre_imgs, post_imgs, labels) in enumerate(train_loader):
            pre_imgs, post_imgs, labels = pre_imgs.to(device), post_imgs.to(device), labels.to(device)

            optimizer.zero_grad()

            use_mixup = random.random() < 0.5
            if use_mixup:
                lam = torch.distributions.Beta(
                    torch.tensor(0.4), torch.tensor(0.4)
                ).sample().item()
                rand_idx = torch.randperm(labels.size(0), device=device)
                pre_imgs = lam * pre_imgs + (1 - lam) * pre_imgs[rand_idx]
                post_imgs = lam * post_imgs + (1 - lam) * post_imgs[rand_idx]

            outputs = model(pre_imgs, post_imgs)

            if use_mixup:
                loss = lam * criterion(outputs, labels) + (1 - lam) * criterion(outputs, labels[rand_idx])
            else:
                loss = criterion(outputs, labels)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            train_correct += torch.sum(preds == labels.data).item()
            train_total += labels.size(0)

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                print(f"  [Train] Batch [{batch_idx+1:^4}/{len(train_loader)}] | "
                      f"Loss: {loss.item():.4f} | Acc: {(train_correct/train_total):.2f}")

        epoch_train_loss = train_loss / len(train_subset)
        epoch_train_acc = train_correct / train_total

        # --- VALIDATION PHASE ---
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        
        with torch.no_grad():
            for pre_imgs, post_imgs, labels in val_loader:
                pre_imgs, post_imgs, labels = pre_imgs.to(device), post_imgs.to(device), labels.to(device)
                outputs = model(pre_imgs, post_imgs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                _, preds = torch.max(outputs, 1)
                val_correct += torch.sum(preds == labels.data).item()
                val_total += labels.size(0)

        epoch_val_loss = val_loss / len(val_subset)
        epoch_val_acc = val_correct / val_total
        epoch_time = time.time() - start_time

        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        history['train_loss'].append(epoch_train_loss)
        history['train_acc'].append(epoch_train_acc)
        history['val_loss'].append(epoch_val_loss)
        history['val_acc'].append(epoch_val_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        print(f"====> Epoch {epoch+1} Summary | "
              f"T-Loss: {epoch_train_loss:.4f} | T-Acc: {epoch_train_acc:.4f} || "
              f"V-Loss: {epoch_val_loss:.4f} | V-Acc: {epoch_val_acc:.4f} | "
              f"Time: {epoch_time:.1f}s")

        # --- CHECKPOINT ---
        if epoch_val_loss < best_val_loss:
            best_val_loss = epoch_val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), SAVE_DIR / "best_vit_ch9_withmixup.pth")
            print(f"      [!] New best model saved (Val Loss={epoch_val_loss:.4f})")
        else:
            epochs_no_improve += 1
            print(f"      [-] No improvement. Early stopping counter: {epochs_no_improve}/{patience}")
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping triggered at epoch {epoch+1}.")
                with open(history_path, 'w') as f:
                    json.dump(history, f, indent=2)
                break

        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)

    print(f"\n[*] Training Complete. Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_model()
