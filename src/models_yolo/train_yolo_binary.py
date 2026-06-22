import torch
from pathlib import Path
from ultralytics import YOLO

# Paths
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]


DATA_DIR = PROJECT_ROOT / "data" / "yolo_single_binary"
RUN_NAME = "single_binary_yolo11l_cls"

# ==========================================

if __name__ == '__main__':

    print("PyTorch version:", torch.__version__)
    print("CUDA IS:", torch.cuda.is_available())
    if torch.cuda.is_available():
        print("GPU!!!:", torch.cuda.get_device_name(0))
    else:
        print("GPU NOT FOUND!!!")

    model = YOLO("yolo11l-cls.pt") 
    
    train_params = {
        "data": str(DATA_DIR.resolve()), 
        "epochs": 100,             
        "patience": 20,            
        "imgsz": 224,              
        "batch": 512,          
        
        "project": str((PROJECT_ROOT / "results" / "res_yolo").resolve()), 
        "name": RUN_NAME,          
        "device": 0 if torch.cuda.is_available() else "cpu",
        "workers": 2,
        "deterministic": True,
        

        "optimizer": "AdamW",       
        "lr0": 0.0005,        
        "weight_decay": 0.05, 
        "warmup_epochs": 5.0,      
        "cos_lr": True,            
        
   
        "freeze": 10,        
        "label_smoothing": 0.1, 

    
        "degrees": 180.0,           
        "flipud": 0.5,              
        "fliplr": 0.5,             
        
    
        "hsv_h": 0.0,        
        "hsv_s": 0.2,         
        "hsv_v": 0.2,       
        
    
        "erasing": 0.0,     
        "mixup": 0.0,         
        "copy_paste": 0.0,    
        "scale": 0.0,        
        "translate": 0.0,     
        "auto_augment": False
    }

    torch.cuda.empty_cache()
    print(f"\n[*] Starting training on: {DATA_DIR.name}")
    model.train(**train_params)
    print(f"[+] Training complete. Artifacts saved to results/res_yolo/{RUN_NAME}")
