"""
Binary ViT Training Script

Trains the BinaryChangeViT model for damaged-vs-not-damaged classification.
Uses BCEWithLogitsLoss (with optional pos_weight for class imbalance) and
the same training recipe as the multi-class variant (warmup + cosine LR,
mixup, early stopping).
"""

import json
import time
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler

from data_prep.vit_dataset_binary import BinaryBuildingDamageDataset
from models_vit.vit_binary import BinaryChangeViT


def train_binary_model():
    """
    Main training function for the binary damage detection model.
    """
    # ---------------------------------------------------------------
    # 1. Configuration
    # ---------------------------------------------------------------
    _HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = _HERE.parent

    DATA_DIR = PROJECT_ROOT / "data" / "vit_crops" / "train"
    SAVE_DIR = PROJECT_ROOT / "results" / "models"
    RESULTS_DIR = PROJECT_ROOT / "results" / "res_vit_binary"

    SAVE_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    batch_size = 32
    epochs = 50
    learning_rate = 1e-4
    warmup_epochs = 5
    patience = 15

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training BINARY ViT on device: {device}")

    # ---------------------------------------------------------------
    # 2. Dataset Initialization & Split
    # ---------------------------------------------------------------
    print("\n[*] Initializing Binary Dataset and splitting 80/20...")

    full_train_dataset = BinaryBuildingDamageDataset(root_dir=DATA_DIR, augment=True)
    full_val_dataset = BinaryBuildingDamageDataset(root_dir=DATA_DIR, augment=False)

    dataset_size = len(full_train_dataset)
    indices = torch.randperm(dataset_size).tolist()
    val_split = int(0.2 * dataset_size)

    train_indices = indices[val_split:]
    val_indices = indices[:val_split]

    train_subset = Subset(full_train_dataset, train_indices)
    val_subset = Subset(full_val_dataset, val_indices)

    print(f"[*] Total Images: {dataset_size} | Training: {len(train_subset)} | Validation: {len(val_subset)}")
    print(f"[*] Class distribution — Not-damaged: {full_train_dataset.class_counts[0]} | "
          f"Damaged: {full_train_dataset.class_counts[1]}")

    # Balanced Sampling (binary: 2 classes)
    train_labels = [full_train_dataset.labels[i] for i in train_indices]
    class_sample_counts = torch.tensor(
        [train_labels.count(0), train_labels.count(1)],
        dtype=torch.float
    )
    per_sample_weight = 1.0 / torch.sqrt(class_sample_counts[train_labels])
    train_sampler = WeightedRandomSampler(
        weights=per_sample_weight,
        num_samples=len(per_sample_weight),
        replacement=True
    )
    print(f"[*] Balanced sampler active — class counts: {class_sample_counts.tolist()}")

    train_loader = DataLoader(
        train_subset, batch_size=batch_size, sampler=train_sampler,
        num_workers=4 if device.type == 'cuda' else 0, pin_memory=device.type == 'cuda'
    )
    val_loader = DataLoader(
        val_subset, batch_size=batch_size, shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0, pin_memory=device.type == 'cuda'
    )

    # ---------------------------------------------------------------
    # 3. Setup Model, Optimizer, Scheduler, Loss
    # ---------------------------------------------------------------
    model = BinaryChangeViT(
        img_size=224, patch_size=16, in_channels=9,
        embed_dim=256, depth=6, num_heads=8, drop_path_rate=0.2
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)

    # pos_weight compensates for class imbalance: weight for "damaged" class
    # If not-damaged is rarer, pos_weight < 1; if damaged is rarer, pos_weight > 1
    pos_weight = class_sample_counts[0] / class_sample_counts[1]  # ratio of neg/pos
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    print(f"[*] BCEWithLogitsLoss pos_weight={pos_weight.item():.3f}")

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
    print(f"\n[*] Starting Binary Training Loop ({epochs} epochs, patience={patience})...")
    best_val_loss = float('inf')
    epochs_no_improve = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'lr': [], 'epoch_time': []}
    history_path = RESULTS_DIR / "training_history_binary.json"

    for epoch in range(epochs):
        start_time = time.time()

        # --- TRAIN PHASE ---
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0

        print(f"\n====> Epoch {epoch+1}/{epochs} starting...")
        for batch_idx, (pre_imgs, post_imgs, labels) in enumerate(train_loader):
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            labels = labels.float().to(device)              # BCE expects float targets

            optimizer.zero_grad()

            # Mixup (same recipe as multi-class)
            use_mixup = random.random() < 0.5
            if use_mixup:
                lam = torch.distributions.Beta(
                    torch.tensor(0.4), torch.tensor(0.4)
                ).sample().item()
                rand_idx = torch.randperm(labels.size(0), device=device)
                pre_imgs = lam * pre_imgs + (1 - lam) * pre_imgs[rand_idx]
                post_imgs = lam * post_imgs + (1 - lam) * post_imgs[rand_idx]

            logits = model(pre_imgs, post_imgs).squeeze(1)  # (B,)

            if use_mixup:
                loss = lam * criterion(logits, labels) + (1 - lam) * criterion(logits, labels[rand_idx])
            else:
                loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            preds = (torch.sigmoid(logits) >= 0.5).long()
            train_correct += (preds == labels.long()).sum().item()
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
                pre_imgs = pre_imgs.to(device)
                post_imgs = post_imgs.to(device)
                labels = labels.float().to(device)

                logits = model(pre_imgs, post_imgs).squeeze(1)
                loss = criterion(logits, labels)

                val_loss += loss.item() * labels.size(0)
                preds = (torch.sigmoid(logits) >= 0.5).long()
                val_correct += (preds == labels.long()).sum().item()
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
            torch.save(model.state_dict(), SAVE_DIR / "best_vit_binary.pth")
            print(f"      [!] New best BINARY model saved (Val Loss={epoch_val_loss:.4f})")
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

    print(f"\n[*] Binary Training Complete. Best Validation Loss: {best_val_loss:.4f}")


if __name__ == "__main__":
    train_binary_model()
