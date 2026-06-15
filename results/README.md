# Results Directory

This directory stores all the artifacts generated during the training and evaluation of the building damage assessment models. 

## Directory Structure

- **`models/`**: Contains the best performing, finalized model weights (e.g., `best_vit.pth`, `best_yolo.pt`). These are the primary models used by `pipeline.py` for inference.
- **`res_vit/`**: Contains evaluation artifacts for the Vision Transformer (ViT) model, including:
  - `vit_training_curves.png`: Graphs of loss and accuracy over epochs.
  - `vit_confusion_matrix.png`: Heatmap showing classification accuracy across damage levels.
  - `vit_metrics.csv`: Detailed precision, recall, and F1-score breakdown per class.
  - `training_history.json`: Raw telemetry data from the training loop.
- **`res_yolo/`**: Dedicated folder for custom YOLO evaluation outputs.

## Usage

When running inference using `src/pipeline.py`, ensure that the target models exist within the `models/` directory as configured. 

The evaluation scripts in `src/evaluation/` will automatically write their generated reports and plots back into the relevant subdirectories here.