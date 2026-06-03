import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

class BuildingDamageDataset(Dataset):
    """
    Custom Dataset for loading multi-temporal building crops for the ViT.
    
    Expects directory structure:
    root_dir/
      no-damage/
        *.png
      minor-damage/
        *.png
      major-damage/
        *.png
      destroyed/
        *.png
        
    The images are expected to be 64x128 (height=64, width=128), containing 
    the Pre-disaster and Post-disaster crops side-by-side.
    """
    def __init__(self, root_dir):
        self.root_dir = root_dir
        
        # Ordinal class mapping
        self.class_to_idx = {
            'no-damage': 0,
            'minor-damage': 1,
            'major-damage': 2,
            'destroyed': 3
        }
        
        self.image_paths = []
        self.labels = []
        
        # Keep track of counts for class weighting
        self.class_counts = {0: 0, 1: 0, 2: 0, 3: 0}
        
        # Scan directories
        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                print(f"Warning: Directory {class_dir} does not exist.")
                continue
                
            img_files = glob.glob(os.path.join(class_dir, "*.png"))
            self.image_paths.extend(img_files)
            self.labels.extend([class_idx] * len(img_files))
            self.class_counts[class_idx] += len(img_files)
            
        print(f"Loaded {len(self.image_paths)} images from {root_dir}")
        print(f"Class distribution: {self.class_counts}")
        
    def get_class_weights(self):
        """
        Calculates inverse frequency class weights to pass to CrossEntropyLoss.
        Mitigates extreme class imbalance (e.g., lots of 'no-damage').
        """
        total_samples = len(self.image_paths)
        weights = []
        for i in range(len(self.class_to_idx)):
            count = self.class_counts[i]
            # Handle potential edge case where a class has 0 samples
            weight = total_samples / (4.0 * count) if count > 0 else 0.0
            weights.append(weight)
            
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.image_paths)
        
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load image (BGR format from cv2)
        img = cv2.imread(img_path)
        
        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Shape is expected to be (64, 128, 3)
        # Split horizontally: Left half is Pre-disaster, Right half is Post-disaster
        # If dimensions are different, this assumes they are concatenated along width
        width = img.shape[1]
        half_w = width // 2
        
        pre_img = img[:, :half_w, :]   # Shape: (64, 64, 3)
        post_img = img[:, half_w:, :]  # Shape: (64, 64, 3)
        
        # Convert to float and normalize to [0, 1]
        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0
        
        # Convert to PyTorch tensors and permute to [Channels, Height, Width]
        # Shape becomes (3, 64, 64)
        pre_tensor = torch.from_numpy(pre_img).permute(2, 0, 1)
        post_tensor = torch.from_numpy(post_img).permute(2, 0, 1)
        
        # Concatenate along channel dimension -> (6, 64, 64)
        combined_tensor = torch.cat((pre_tensor, post_tensor), dim=0)
        
        return combined_tensor, label
