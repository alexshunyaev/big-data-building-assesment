import shutil
from pathlib import Path

# Paths
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]

VIT_DIR = PROJECT_ROOT / "data" / "vit_crops"
BINARY_DIR = PROJECT_ROOT / "data" / "yolo_cls_binary"

BINARY_MAP = {
    'no-damage': 'not-damaged',
    'minor-damage': 'damaged',
    'major-damage': 'damaged',
    'destroyed': 'damaged'
}

def create_binary_dataset():
    if not VIT_DIR.exists():
        print(f"Error: {VIT_DIR} not found.")
        return

    for split_dir in VIT_DIR.iterdir():
        if not split_dir.is_dir():
            continue
            
        split_name = split_dir.name
        # YOLO expects 'val', rename 'test'
        out_split_name = 'val' if split_name == 'test' else split_name

        print(f"\nProcessing: {split_name} -> {out_split_name}")
        
        for orig_class, bin_class in BINARY_MAP.items():
            src_class_dir = split_dir / orig_class
            dst_class_dir = BINARY_DIR / out_split_name / bin_class
            
            if not src_class_dir.exists():
                continue
                
            dst_class_dir.mkdir(parents=True, exist_ok=True)
            
            # Copy images
            imgs = list(src_class_dir.glob("*.png"))
            print(f"  Copying {len(imgs)} imgs: {orig_class} -> {bin_class}")
            for img_path in imgs:
                shutil.copy2(img_path, dst_class_dir / img_path.name)

if __name__ == '__main__':
    # Clear existing dir
    if BINARY_DIR.exists():
        shutil.rmtree(BINARY_DIR)
    
    create_binary_dataset()
    print(f"\n Dataset ready: {BINARY_DIR}")