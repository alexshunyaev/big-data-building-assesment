"""
ViT Dataset Script

This module defines the PyTorch Dataset class used for loading the building
damage classification crops for Vision Transformer training.
"""

from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

STD_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD_DEV = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class BuildingDamageDataset(Dataset):
    """
    Dataset class for loading building damage pairs.

    Attributes:
        root_dir (Path): Root directory containing damage class folders.
        augment (bool): Whether to apply data augmentation.
    """
    def __init__(self, root_dir: str | Path, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment = augment

        self.class_to_idx = {
            'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3
        }

        self.image_paths = []
        self.labels = []
        self.class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            img_files = list(class_dir.glob("*.png"))
            self.image_paths.extend(img_files)
            self.labels.extend([class_idx] * len(img_files))
            self.class_counts[class_idx] += len(img_files)

    def get_class_weights(self) -> torch.Tensor:
        """
        Calculates weights to balance the dataset during training.

        Returns:
            torch.Tensor: Tensor containing weights for each class.
        """
        total_samples = len(self.image_paths)
        weights = []
        for i in range(len(self.class_to_idx)):
            count = self.class_counts[i]
            weight = total_samples / (4.0 * count) if count > 0 else 0.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.image_paths)

    def _apply_augmentation(self, pre_img: np.ndarray, post_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Applies geometric and color augmentations identically to pre and post crops.
        """
        rot_choice = np.random.randint(4)
        if rot_choice > 0:
            rot_code = [None, cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, cv2.ROTATE_90_COUNTERCLOCKWISE][rot_choice]
            pre_img = cv2.rotate(pre_img, rot_code)
            post_img = cv2.rotate(post_img, rot_code)

        if np.random.rand() > 0.5:
            pre_img = np.flip(pre_img, axis=1).copy()
            post_img = np.flip(post_img, axis=1).copy()

        if np.random.rand() > 0.5:
            pre_img = np.flip(pre_img, axis=0).copy()
            post_img = np.flip(post_img, axis=0).copy()

        brightness = 1.0 + np.random.uniform(-0.2, 0.2)
        contrast = 1.0 + np.random.uniform(-0.2, 0.2)
        saturation = 1.0 + np.random.uniform(-0.1, 0.1)
        hue_shift = np.random.uniform(-0.05, 0.05) * 180

        pre_img = self._apply_color_jitter(pre_img, brightness, contrast, saturation, hue_shift)
        post_img = self._apply_color_jitter(post_img, brightness, contrast, saturation, hue_shift)

        return pre_img, post_img

    @staticmethod
    def _apply_color_jitter(img: np.ndarray, brightness: float, contrast: float, saturation: float, hue_shift: float) -> np.ndarray:
        """
        Applies brightness, contrast, saturation, and hue shift jitter to an image.
        """
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        mean_val = img.mean()
        img = np.clip((img - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Loads and returns a single sample from the dataset.
        """
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        width = img.shape[1]
        half_w = width // 2

        pre_img = img[:, :half_w, :]
        post_img = img[:, half_w:, :]

        if self.augment:
            pre_img, post_img = self._apply_augmentation(pre_img, post_img)

        pre_img = pre_img.astype(np.float32) / 255.0
        post_img = post_img.astype(np.float32) / 255.0

        pre_img = (pre_img - STD_MEAN) / STD_DEV
        post_img = (post_img - STD_MEAN) / STD_DEV

        pre_tensor = torch.from_numpy(pre_img).permute(2, 0, 1)
        post_tensor = torch.from_numpy(post_img).permute(2, 0, 1)

        return pre_tensor, post_tensor, label
