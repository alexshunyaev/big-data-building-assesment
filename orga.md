## 1. Pipeline Architecture
The core scientific challenge is broken down into a multi-step sequential pipeline combining an **Instance Segmentation Foundation** and a custom **Vision Transformer Classifier**.

### The Inference Pipeline Flow
1. **Pre-Disaster Scene Input:** Pass the 1024x1024 pre-disaster image (or tiled sub-patches) into **YOLO26n-seg**.
2. **Mask & Bounding Box Generation:** YOLO localizes all buildings, generating high-fidelity coordinates and binary masks.
3. **Dual-Temporal Masked Cropping:** For each detected building, extract the identical bounding box coordinates from *both* the **Pre-disaster** and **Post-disaster** images. Multiplying these crops by the binary mask zeros out background noise (trees, roads, shadows).
4. **ViT Classification:** Concatenate the pre- and post-disaster building crops and feed them into the **Custom Vision Transformer (ViT)** to predict one of four ordinal classes: *No Damage*, *Minor Damage*, *Major Damage*, or *Destroyed*.

---

## 2. Model Specifications

### Task 1 (Localization): YOLO26n-seg
* **Task:** Semantic & Instance Building Localization (Binary Masking).
* **Implementation Level:** Framework implementation via `ultralytics` with pre-trained initial weights (Transfer Learning).
* **Architectural Justification:**
  * **NMS-Free Architecture:** Eliminates traditional Non-Maximum Suppression bottlenecks, preventing dense urban clusters of adjacent buildings from being skipped or merged.
  * **ProgLoss & STAL Layering:** Advanced feature extraction optimized for tiny object resolution, keeping ultra-small residential structures intact without downsampling loss.

### Task 2 (Classification): Custom Vision Transformer (ViT)
* **Task:** Multi-class Damage Classification (4 classes).
* **Implementation Level:** Fully written from scratch in PyTorch.
* **Architectural Justification:**
  * Utilizes **Multi-Head Self-Attention (MHSA)** to look globally across the entire cropped building canvas. This allows the model to map dependencies between intact structural lines and sprawling debris fields.
  * Captures the multi-temporal relationship effectively by treating the combined pre/post patches as a continuous sequence.
* **Core Components:**
  * `PatchEmbedding`: Linear projection layer slicing a crop into uniform 4x4 or 8x8 tokens.
  * `PositionalEncoding`: Learnable 1D spatial indicators added to tokens to preserve topological context.
  * `MultiHeadSelfAttention`: Scaled dot-product attention mapping Query, Key, and Value vectors across multiple heads.
  * `TransformerBlock`: Stacking Pre-Layer Normalization, MHSA, Residual Connections, and an MLP block.

---

## 3. Dataset Configuration & Preprocessing

* **Training Set:** ~7.8 GB | 5,598 Scenes.
* **Test Set:** ~2.6 GB | 1,866 Scenes.
* **Holdout Set:** ~2.6 GB | 1,866 Scenes.

### Preprocessing Steps
1. **Patch Tiling:** Sub-divide large 1024x1024 scenes into 512x512 patches to optimize GPU mini-batch packing (idk if really need this).
2. **Label Transformation:** Convert the native xView2 JSON polygon metadata into standard YOLO segmentation text formats.
3. **Masked Patch Extraction:** Save the cropped coordinates of buildings as distinct arrays for direct ViT dataloader streaming.


## 5. Critical Engineering concerns

1. **Class Imbalance:** Minor and major damage classes are vastly outnumbered by "no damage" structures. Mitigated by using **Focal Loss** or **Class-Weighted Cross-Entropy** within the ViT loss optimizer.
2. **Transformer Overfitting:** Mitigated by executing heavy pixel-level data augmentations and applying **Stochastic Depth (DropPath)** inside the custom transformer blocks.
3. **Background Occlusion:** Addressed by multiplying the raw rectangular crop by YOLO's precise binary segmentation mask, forcing the transformer to look *only* at the building structure