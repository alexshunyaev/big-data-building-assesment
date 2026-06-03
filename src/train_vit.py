import os
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from data_prep.vit_dataset import BuildingDamageDataset
from models_vit.vit import VisionTransformer

def train_model():
    # Configuration
    # Automatically resolve the project root based on this script's location
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)
    data_dir = os.path.join(project_root, "data", "vit_crops", "train")
    
    batch_size = 32
    epochs = 20
    learning_rate = 1e-4
    save_dir = os.path.join(project_root, "results", "models")
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # 1. Device Auto-Detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Training on device: {device}")
    if device.type == 'cuda':
        print(f"[*] GPU Name: {torch.cuda.get_device_name(0)}")

    # 2. Setup Dataset and DataLoader
    print("\n[*] Initializing Dataset...")
    train_dataset = BuildingDamageDataset(root_dir=data_dir)
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=4 if device.type == 'cuda' else 0, # Use multiprocessing if GPU
        pin_memory=True if device.type == 'cuda' else False
    )
    
    # 3. Setup Class Weights for Imbalance
    class_weights = train_dataset.get_class_weights().to(device)
    print(f"[*] Calculated Class Weights: {class_weights}")
    
    # 4. Instantiate Vision Transformer Model
    print("\n[*] Instantiating Vision Transformer...")
    model = VisionTransformer(
        img_size=64, 
        patch_size=8, 
        in_channels=6, 
        num_classes=4
    ).to(device)
    
    # 5. Optimizer and Loss Function
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    
    # 6. Training Loop
    print("\n[*] Starting Training Loop...")
    best_loss = float('inf')
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        correct_preds = 0
        total_preds = 0
        
        start_time = time.time()
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            
            # Loss calculation
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Statistics
            running_loss += loss.item() * inputs.size(0)
            
            # Calculate accuracy
            _, preds = torch.max(outputs, 1)
            correct_preds += torch.sum(preds == labels.data).item()
            total_preds += labels.size(0)
            
            if (batch_idx + 1) % 50 == 0:
                print(f"  Epoch [{epoch+1}/{epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
                
        epoch_loss = running_loss / len(train_dataset)
        epoch_acc = correct_preds / total_preds
        epoch_time = time.time() - start_time
        
        print(f"====> Epoch {epoch+1}/{epochs} | Loss: {epoch_loss:.4f} | Accuracy: {epoch_acc:.4f} | Time: {epoch_time:.2f}s")
        
        # Save best model
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            save_path = os.path.join(save_dir, "best_vit.pth")
            torch.save(model.state_dict(), save_path)
            print(f"      [!] Saved new best model to {save_path}")

    print("\n[*] Training Complete!")

if __name__ == "__main__":
    train_model()
