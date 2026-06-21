import torch
from pathlib import Path
from ultralytics import YOLO

# Paths
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]

# ==========================================
# MODE SELECTION (Uncomment target)
# ==========================================

# MODEL 1: MULTICLASS (uncomment to use)
# DATA_DIR = PROJECT_ROOT / "data" / "yolo_single_multi"
# RUN_NAME = "single_multi_yolo11s_cls"

# MODEL 2: BINARY (2 classes)
DATA_DIR = PROJECT_ROOT / "data" / "yolo_single_binary"
RUN_NAME = "single_binary_yolo11s_cls"

# ==========================================

if __name__ == '__main__':

    print("PyTorch version:", torch.__version__)
    print("CUDA IS:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU!!!:", torch.cuda.get_device_name(0))
    else:
        print("GPU NOT FOUND!!!")

    # Load YOLO11s classification! model
    model = YOLO("yolo11s-cls.pt") 

    
    train_params = {
        "data": str(DATA_DIR.resolve()), 
        "epochs": 100,
        "patience": 15,            
        "imgsz": 224,              
        "batch": 128,              
        "label_smoothing": 0.1,    
        "project": str((PROJECT_ROOT / "results" / "res_yolo").resolve()), 
        "name": RUN_NAME,          
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 8,
        "deterministic": True
    }

    torch.cuda.empty_cache()
    print(f"\n[*] Starting training on: {DATA_DIR.name}")
    model.train(**train_params)
    print(f"[+] Training complete. Artifacts saved to results/res_yolo/{RUN_NAME}")