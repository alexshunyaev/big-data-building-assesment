import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_prep.vit_dataset import BuildingDamageDataset
from models_vit.vit import SiameseViT


def train_model():
    # ---------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data", "vit_crops", "train")

    batch_size = 32
    epochs = 100
    learning_rate = 1e-4
    warmup_epochs = 5        # Freeze backbone and warm up LR for this many epochs
    patience = 15             # Early stopping patience

    save_dir = os.path.join(project_root, "results", "models")
    results_dir = os.path.join(project_root, "results", "res_vit")
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(results_dir, exist_ok=True)

    # ---------------------------------------------------------------
    # 1. Device Auto-Detection
    # ---------------------------------------------------------------
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    if device.type == 'cuda':
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    # ---------------------------------------------------------------
    # 2. Dataset — Full Training (No Validation Split)
    # ---------------------------------------------------------------
    print("\n[*] Initializing Dataset...")
    train_dataset = BuildingDamageDataset(root_dir=data_dir, augment=True)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    # ---------------------------------------------------------------
    # 3. Inverse Class Frequency Weights
    # ---------------------------------------------------------------
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"[*] Inverse Class Frequency Weights: {class_weights}")

    # ---------------------------------------------------------------
    # 4. Siamese ViT — Pretrained Backbone
    # ---------------------------------------------------------------
    print("\n[*] Instantiating Siamese ViT (vit_small_patch16_224, pretrained=True)...")
    model = SiameseViT(
        num_classes=4,
        backbone_name='vit_small_patch16_224',
        pretrained=True
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"[*] Total Parameters: {total_params:,}")

    # Freeze backbone during warmup — train only the classification head
    model.freeze_backbone()
    print(f"[*] Backbone FROZEN for first {warmup_epochs} epochs (head-only training)")

    # ---------------------------------------------------------------
    # 5. Optimizer + Scheduler (AdamW + Linear Warmup + Cosine Annealing)
    # ---------------------------------------------------------------
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Custom LR schedule: linear warmup → cosine decay
    # During warmup: LR ramps from 1e-6 to learning_rate
    # After warmup: cosine annealing from learning_rate to 1e-6
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            # Linear warmup: scale from 0.01 (1e-6/1e-4) to 1.0
            return 0.01 + (1.0 - 0.01) * (epoch / warmup_epochs)
        else:
            # Cosine annealing for remaining epochs
            import math
            progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
            return 0.01 + (1.0 - 0.01) * 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

    # ---------------------------------------------------------------
    # 6. Training Loop
    # ---------------------------------------------------------------
    print(f"\n[*] Starting Training Loop ({epochs} epochs, patience={patience})...")
    best_loss = float('inf')
    epochs_no_improve = 0

    # History for evaluation script to consume
    history = {
        'train_loss': [],
        'train_acc': [],
        'lr': [],
        'epoch_time': []
    }

    for epoch in range(epochs):
        # --- UNFREEZE BACKBONE AFTER WARMUP ---
        if epoch == warmup_epochs:
            model.unfreeze_backbone()
            # Re-create optimizer to include backbone params with proper LR
            optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
            # Step scheduler to current epoch position
            for _ in range(epoch):
                scheduler.step()
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            print(f"\n[!] Backbone UNFROZEN at epoch {epoch+1}. Trainable params: {trainable:,}")

        # --- TRAIN PHASE ---
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        start_time = time.time()

        for batch_idx, (pre_imgs, post_imgs, labels) in enumerate(train_loader):
            pre_imgs = pre_imgs.to(device)
            post_imgs = post_imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(pre_imgs, post_imgs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data).item()
            total_preds += labels.size(0)

            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], "
                      f"Loss: {loss.item():.4f}")

        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_preds / total_preds
        epoch_time = time.time() - start_time

        # Step scheduler
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]

        # Record history
        history['train_loss'].append(epoch_loss)
        history['train_acc'].append(epoch_acc)
        history['lr'].append(current_lr)
        history['epoch_time'].append(epoch_time)

        phase = "WARMUP" if epoch < warmup_epochs else "TRAIN"
        print(f"====> [{phase}] Epoch {epoch+1}/{epochs} | "
              f"Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f} | "
              f"LR: {current_lr:.6f} | Time: {epoch_time:.2f}s")

        # --- CHECKPOINT ---
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            epochs_no_improve = 0
            save_path = os.path.join(save_dir, "best_vit.pth")
            torch.save(model.state_dict(), save_path)
            print(f"      [!] New best model saved (loss={epoch_loss:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"\n[!] Early stopping at epoch {epoch+1} "
                      f"(no improvement for {patience} epochs)")
                break

    # ---------------------------------------------------------------
    # 7. Save Training History
    # ---------------------------------------------------------------
    history_path = os.path.join(results_dir, "training_history.json")
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\n[*] Training history saved to {history_path}")

    print(f"[*] Best model saved to {os.path.join(save_dir, 'best_vit.pth')}")
    print(f"[*] Best training loss: {best_loss:.4f}")
    print("\n[*] Training Complete! Run evaluation/metrics.py for plots and metrics.")


if __name__ == "__main__":
    train_model()
