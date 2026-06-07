import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2


# ImageNet normalization constants (required for pretrained ViT backbones)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class BuildingDamageDataset(Dataset):
    """
    Siamese-compatible dataset for loading multi-temporal building crops.

    Returns separate pre-disaster and post-disaster tensors for the Siamese ViT.

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

    Each PNG is a side-by-side concatenation: [pre_crop | post_crop].
    With crop_size=224, images are 448x224 (width x height).

    Returns:
        (pre_tensor [3, H, W], post_tensor [3, H, W], label)
    """

    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.augment = augment

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

    def _apply_augmentation(self, pre_img, post_img):
        """
        Apply identical geometric and color augmentations to both crops.
        Spatial correspondence between pre and post is preserved by using
        the same random decisions for both images.
        """
        # --- Geometric augmentations (identical for both) ---

        # Random rotation: 0, 90, 180, or 270 degrees
        rot_choice = np.random.randint(4)
        if rot_choice > 0:
            # cv2.rotate codes: 0=90CW, 1=180, 2=90CCW (270CW)
            rot_code = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][rot_choice]
            pre_img = cv2.rotate(pre_img, rot_code)
            post_img = cv2.rotate(post_img, rot_code)

        # Random horizontal flip
        if np.random.rand() > 0.5:
            pre_img = np.flip(pre_img, axis=1).copy()
            post_img = np.flip(post_img, axis=1).copy()

        # Random vertical flip
        if np.random.rand() > 0.5:
            pre_img = np.flip(pre_img, axis=0).copy()
            post_img = np.flip(post_img, axis=0).copy()

        # --- Color jitter (identical parameters for both) ---
        # Generate shared random factors once, apply to both crops
        brightness_factor = 1.0 + np.random.uniform(-0.2, 0.2)
        contrast_factor = 1.0 + np.random.uniform(-0.2, 0.2)
        saturation_factor = 1.0 + np.random.uniform(-0.1, 0.1)
        hue_shift = np.random.uniform(-0.05, 0.05) * 180  # OpenCV hue is 0-180

        pre_img = self._apply_color_jitter(pre_img, brightness_factor, contrast_factor, saturation_factor, hue_shift)
        post_img = self._apply_color_jitter(post_img, brightness_factor, contrast_factor, saturation_factor, hue_shift)

        return pre_img, post_img

    @staticmethod
    def _apply_color_jitter(img, brightness, contrast, saturation, hue_shift):
        """Apply color jitter to a single image using shared random parameters."""
        # Brightness: scale pixel values
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)

        # Contrast: blend with mean intensity
        mean_val = img.mean()
        img = np.clip((img - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

        # Saturation & Hue: operate in HSV space
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180        # Hue shift
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)  # Saturation scale
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        # Load image (BGR format from cv2)
        img = cv2.imread(img_path)

        # Convert to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Split horizontally: Left half is Pre-disaster, Right half is Post-disaster
        width = img.shape[1]
        half_w = width // 2

        pre_img = img[:, :half_w, :]
        post_img = img[:, half_w:, :]

        # Apply augmentation (identical transforms to both crops)
        if self.augment:
            pre_img, post_img = self._apply_augmentation(pre_img, post_img)

        # Convert to float32, normalize to [0, 1], then apply ImageNet normalization
        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        # ImageNet normalization: (pixel - mean) / std
        pre_img = (pre_img - IMAGENET_MEAN) / IMAGENET_STD
        post_img = (post_img - IMAGENET_MEAN) / IMAGENET_STD

        # Convert to PyTorch tensors and permute to [Channels, Height, Width]
        pre_tensor = torch.from_numpy(pre_img).permute(2, 0, 1)    # [3, 224, 224]
        post_tensor = torch.from_numpy(post_img).permute(2, 0, 1)  # [3, 224, 224]

        return pre_tensor, post_tensor, label
