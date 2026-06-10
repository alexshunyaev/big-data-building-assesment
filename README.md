# Building Damage Assessment Project (YOLOv11)

## Model Parameters

### 1. `damage_yolo_run1-3`
*   **Architecture:** YOLO11n (Nano)
*   **Image Size (imgsz):** 1024
*   **Batch Size:** 4
*   **Epochs:** 30
*   **Class Weighting (cls_pw):** 0.0

### 2. `damage_yolo11s_100ep-3`
*   **Architecture:** YOLO11s (Small)
*   **Image Size (imgsz):** 640
*   **Batch Size:** 8
*   **Epochs:** 100 (patience=20)
*   **Class Weighting (cls_pw):** 0.0

### 3. `damage_yolo11s_1024_balanced`
*   **Architecture:** YOLO11s (Small)
*   **Image Size (imgsz):** 1024
*   **Batch Size:** 4
*   **Epochs:** 100 (patience=20)
*   **Class Weighting (cls_pw):** 1.0
*   **Degrees (rotation):** 180.0
*   **Vertical Flip (flipud):** 0.5
*   **Horizontal Flip (fliplr):** 0.5

---

## Repository Structure

```text
DL/ (Repository Root)
├── .gitignore                       # File exclusion rules (dataset and base weights)
├── prepare.ipynb                    # Data preprocessing notebook (WKT JSON -> YOLO format)
├── Train.ipynb                      # Notebook used to train all model versions
├── Test_model.ipynb                 # Notebook for model testing and validation inference
└── yolo-damage/                     
    ├── data.yaml                    # YOLO path and class name configurations
    └── runs/                        # Training outputs of the three completed runs
        ├── damage_yolo_run1-3/      # Weights, logs, and plots for Model 1 (YOLO11n)
        ├── damage_yolo11s_100ep-3/  # Weights, logs, and plots for Model 2 (YOLO11s, 640px)
        └── damage_yolo11s_1024_balanced/ # Weights, logs, and plots for Model 3 (YOLO11s, 1024px)
```

Each of the three run folders inside `runs/` contains:
*   **`weights/best.pt`** — Trained model weights at the epoch with the best validation performance (recommended for testing and inference).
*   **`weights/last.pt`** — Model weights at the final epoch of the training process.
*   **`results.csv` & `results.png`** — Numerical logs and plots tracking training/validation losses and metrics over epochs.
*   **`confusion_matrix.png`** — Classification confusion matrix across the 4 building damage levels.
*   **Validation plots** — Performance metrics curves and visual predictions (e.g., `BoxPR_curve.png`, `BoxF1_curve.png`).
