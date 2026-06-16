"""
Binary ViT Dataset Script

This module defines the PyTorch Dataset class for binary damage classification.
It remaps the original 4-class labels into two classes:
    0 → not-damaged   (original class: "no-damage")
    1 → damaged        (original classes: "minor-damage", "major-damage", "destroyed")
"""

from pathlib import Path

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

STD_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD_DEV = np.array([0.5, 0.5, 0.5], dtype=np.float32)


class BinaryBuildingDamageDataset(Dataset):
    """
    Dataset class for binary building damage detection.

    Reads the same folder structure as the 4-class dataset
    (no-damage / minor-damage / major-damage / destroyed)
    but collapses labels into binary:
        0 = not-damaged  (no-damage)
        1 = damaged      (minor-damage, major-damage, destroyed)
    """

    # Maps original class folders → binary label
    BINARY_MAP = {
        'no-damage': 0,
        'minor-damage': 1,
        'major-damage': 1,
        'destroyed': 1,
    }

    def __init__(self, root_dir: str | Path, augment: bool = False):
        self.root_dir = Path(root_dir)
        self.augment = augment

        self.image_paths: list[Path] = []
        self.labels: list[int] = []
        self.class_counts = {0: 0, 1: 0}

        for class_name, binary_label in self.BINARY_MAP.items():
            class_dir = self.root_dir / class_name
            if not class_dir.exists():
                continue

            img_files = list(class_dir.glob("*.png"))
            self.image_paths.extend(img_files)
            self.labels.extend([binary_label] * len(img_files))
            self.class_counts[binary_label] += len(img_files)

    def get_class_weights(self) -> torch.Tensor:
        """
        Returns inverse-frequency weights for [not-damaged, damaged].

        Returns:
            torch.Tensor: Tensor of shape (2,) with per-class weights.
        """
        total = len(self.image_paths)
        weights = []
        for i in range(2):
            count = self.class_counts[i]
            weight = total / (2.0 * count) if count > 0 else 0.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self) -> int:
        return len(self.image_paths)

    # ----- augmentation (identical to multi-class version) -----

    def _apply_augmentation(self, pre_img: np.ndarray, post_img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Applies geometric and color augmentations identically to pre and post crops."""
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
    def _apply_color_jitter(img: np.ndarray, brightness: float, contrast: float,
                            saturation: float, hue_shift: float) -> np.ndarray:
        """Applies brightness, contrast, saturation, and hue shift jitter to an image."""
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        mean_val = img.mean()
        img = np.clip((img - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img

    # ----- item loading -----

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        """
        Loads and returns a single sample.

        Returns:
            (pre_tensor, post_tensor, label)
            label is 0 (not-damaged) or 1 (damaged).
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
