import os
import glob
import torch
import numpy as np
from torch.utils.data import Dataset
import cv2

STD_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
STD_DEV = np.array([0.5, 0.5, 0.5], dtype=np.float32)

class BuildingDamageDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.root_dir = root_dir
        self.augment = augment

        self.class_to_idx = {
            'no-damage': 0, 'minor-damage': 1, 'major-damage': 2, 'destroyed': 3
        }

        self.image_paths = []
        self.labels = []
        self.class_counts = {0: 0, 1: 0, 2: 0, 3: 0}

        for class_name, class_idx in self.class_to_idx.items():
            class_dir = os.path.join(root_dir, class_name)
            if not os.path.exists(class_dir):
                continue

            img_files = glob.glob(os.path.join(class_dir, "*.png"))
            self.image_paths.extend(img_files)
            self.labels.extend([class_idx] * len(img_files))
            self.class_counts[class_idx] += len(img_files)

    def get_class_weights(self):
        total_samples = len(self.image_paths)
        weights = []
        for i in range(len(self.class_to_idx)):
            count = self.class_counts[i]
            weight = total_samples / (4.0 * count) if count > 0 else 0.0
            weights.append(weight)
        return torch.tensor(weights, dtype=torch.float)

    def __len__(self):
        return len(self.image_paths)

    def _apply_augmentation(self, pre_img, post_img):
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
    def _apply_color_jitter(img, brightness, contrast, saturation, hue_shift):
        img = np.clip(img * brightness, 0, 255).astype(np.uint8)
        mean_val = img.mean()
        img = np.clip((img - mean_val) * contrast + mean_val, 0, 255).astype(np.uint8)

        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 0] = (hsv[:, :, 0] + hue_shift) % 180
        hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)

        return img

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        label = self.labels[idx]

        img = cv2.imread(img_path)
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
