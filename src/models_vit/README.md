# Vision Transformer (ViT) Architecture

This directory contains the custom Vision Transformer (ViT) architectures engineered for the building damage assessment project. The models are specifically adapted to process multitemporal satellite image pairs (pre- and post-disaster).

## Model Architectures & Parameters

Both the Multiclass and Binary variants share a lightweight 6-block Transformer encoder backbone to prevent overfitting on the limited dataset size.

**Shared Hyperparameters:**
*   **Image Size:** 224x224
*   **Patch Size:** 16x16
*   **Embedding Dimension (`embed_dim`):** 256
*   **Transformer Blocks (`depth`):** 6
*   **Attention Heads (`num_heads`):** 8
*   **MLP Expansion Ratio (`mlp_ratio`):** 4.0
*   **Regularization:** Standard Dropout (0.1) and Progressive Stochastic Depth (DropPath rate up to 0.2).

### 1. Multiclass Classification (Early Fusion)
*   **Class:** `CustomChangeViT`
*   **Input Channels:** 9
*   **Fusion Strategy:** Early fusion. Pre-disaster, post-disaster, and explicitly calculated absolute difference images are concatenated channel-wise before patch embedding. This allows the self-attention mechanism to identify subtle structural changes from the very first layer.

### 2. Binary Classification (Late Fusion Siamese)
*   **Class:** `BinaryChangeViT` / Siamese Architecture
*   **Input Channels:** 3 (Processed independently)
*   **Fusion Strategy:** Late fusion. Pre-disaster and post-disaster images are processed through identical shared ViT weights independently. The final `[CLS]` tokens from both branches are concatenated along with their absolute difference, and then passed to the classification head.

## Loss Functions

The models address the severe class imbalance in the xBD dataset directly at the loss level:

### Multiclass Model (`FocalLoss`)
*   **Implementation:** Replaces standard Cross-Entropy with a custom Focal Loss (`gamma=2.0`).
*   **Description:** Focal Loss dynamically down-weights "easy" background or no-damage examples, forcing the optimizer to concentrate on the harder, overlapping severity classes (like minor vs. major damage). It also incorporates label smoothing (`0.1`) to prevent overconfidence.

### Binary Model (`BCEWithLogitsLoss`)
*   **Implementation:** Uses PyTorch's standard Binary Cross Entropy with Logits Loss.
*   **Description:** Applies a dynamically calculated `pos_weight` based on the class distribution in the training set. This explicitly counteracts the massive imbalance between the "Not Damaged" and "Damaged" building classes at a batch level.
