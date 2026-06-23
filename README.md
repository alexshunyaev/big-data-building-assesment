# Building Damage Assessment Project

This project aims to automate the assessment of building damage from natural disasters using satellite imagery. The models analyze pre- and post-disaster images to identify and classify the severity of structural damage. 

## Dataset
We use the **xBD Dataset** released in conjunction with the [xView2 Challenge](https://xview2.org/). It is one of the largest datasets for building segmentation and damage assessment, covering various disaster types with high-resolution optical satellite imagery.

### Where to Put the Images
1. Download the dataset from the xView2 website.
2. Place the raw unzipped folders in `data/raw/` (e.g., `data/raw/train`, `data/raw/test`, `data/raw/hold`).
   * Images should go into `data/raw/<split>/images/`.
   * Labels (GeoJSON metadata) should go into `data/raw/<split>/labels/`.

## Setup
Ensure you have Python 3.9+ installed. Set up your virtual environment and install the required dependencies:
```bash
pip install -r requirements.txt
```

## Data Preprocessing for ViT
Because Vision Transformers (ViT) require localized data without excessive background noise, the project includes a preprocessing pipeline to extract building footprints:
1. **Run the script:**
   ```bash
   python src/data_prep/create_vit_crops.py --split train
   ```
2. **What it does:** It extracts building polygons using the xView2 JSON metadata, applies minor padding, and crops both pre- and post-disaster images from the exact same spatial coordinates.
3. **Output:** Crops are resized to 224x224, concatenated side-by-side (448x224), and saved into categorized class folders (`no-damage`, `minor-damage`, `major-damage`, `destroyed`) inside `data/vit_crops/<split>/`.

## Data Preprocessing for YOLO
We also implemented YOLOv11 classification models as an alternative to ViT. The YOLO models require a different directory structure:
1. **Run the YOLO formatting script:**
   ```bash
   python src/data_prep/prepare_single_crops_yolo.py
   ```
2. **What it does:** It takes the generated ViT crops, extracts individual building images, pads them to a square aspect ratio, and organizes them into both Multiclass and Binary classification directory structures required by YOLO.

## Training Process
The project supports different training paradigms for the Vision Transformer (ViT) and YOLO classification models:

**ViT Training:**
*   **Multiclass Training (4 damage levels):** `python src/train_vit.py`
*   **Binary Training (Damaged vs. Not Damaged):** `python src/train_vit_binary.py`
*   *See [src/models_vit/README.md](src/models_vit/README.md) for ViT architecture details.*

**YOLO Training:**
*   **Multiclass Training:** `python src/models_yolo/train_yolo_multi.py`
*   **Binary Training:** `python src/models_yolo/train_yolo_binary.py`
*   *See [src/models_yolo/README.md](src/models_yolo/README.md) for YOLO architecture details.*

**Key Training Mechanics:**
*   **Handling Class Imbalance:** We use `WeightedRandomSampler` to ensure rare disaster/damage classes are sampled proportionally. In YOLO, we inject this via a custom `BalancedClassificationTrainer`.
*   **Loss Functions:** The models use custom Focal Loss to counteract extreme class imbalances.
*   **Optimization:** Trained using the AdamW optimizer with weight decay, coupled with learning rate schedulers.

## Our Results

### Vision Transformer (ViT)
*   The evaluation metrics, training history curves, and confusion matrices are logged into the `results/res_vit/` and `results/res_vit_binary/` directories.
*   The final model weights are saved as `.pth` files in `results/models/`.

### YOLOv11 Classification
*   The YOLO training loop automatically handles logging. Evaluation metrics and curves are saved in `results/res_yolo/`.
*   The final model weights (`best.pt` and `last.pt`) are stored within the respective YOLO run directories inside `results/res_yolo/`.
