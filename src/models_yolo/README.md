# YOLO Classification Architecture

This directory contains the custom training pipelines for the YOLOv11 classification models, adapted specifically for the building damage assessment project. The models use pre-processed single building crops extracted from the original satellite imagery.

## Model Architectures & Parameters

The project utilizes pre-trained YOLO classification models as its backbone, fine-tuned on our specialized dataset.

**Shared Training Strategies:**
*   **Image Size:** Variable based on model (224x224 for multiclass, 320x320 for binary).
*   **Batch Size:** 256
*   **Optimizer:** AdamW with weight decay (`0.05`) and linear warmup (`5.0` epochs), followed by cosine learning rate annealing (`cos_lr=True`).
*   **Regularization:** Label smoothing (`0.1`) and geometric augmentations (flips, slight rotations up to `10.0` degrees).
*   **Frozen Layers:** The first `10` layers of the backbone are frozen during training to retain fundamental feature extractors and speed up convergence.

### 1. Multiclass Classification (YOLOv11m-cls)
*   **Model:** `yolo11m-cls.pt`
*   **Target:** Predicts all 4 xView2 damage levels (`no-damage`, `minor-damage`, `major-damage`, `destroyed`).
*   **Input Size:** 224x224

### 2. Binary Classification (YOLOv11l-cls)
*   **Model:** `yolo11l-cls.pt`
*   **Target:** Predicts 2 consolidated classes (`Damaged` vs. `Not Damaged`).
*   **Input Size:** 320x320

## Handling Class Imbalance

The xBD dataset exhibits extreme class imbalance (mostly undamaged buildings). To mitigate this, we inject custom components into the standard YOLO training loop via the `BalancedClassificationTrainer` and `BalancedYOLO` wrapper classes:

### 1. Weighted Random Sampler
Instead of sampling uniformly, we calculate class distribution at the start of training. A PyTorch `WeightedRandomSampler` is injected into the DataLoader, assigning weights inversely proportional to the square root of the class counts. This ensures rarer damage classes are seen more frequently in every epoch.

### 2. Focal Loss Integration
The default Cross-Entropy criterion in YOLO is dynamically replaced with a custom Focal Loss implementation (`gamma=2.0`). Focal Loss inherently reduces the loss contribution from "easy" well-classified examples (like the abundant undamaged buildings), forcing the network to focus on harder, overlapping severity cases.
