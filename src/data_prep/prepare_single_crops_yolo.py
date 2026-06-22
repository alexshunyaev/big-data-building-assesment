import cv2
import shutil
import numpy as np
from pathlib import Path
from tqdm import tqdm

# Paths
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]

VIT_DIR = PROJECT_ROOT / "data" / "vit_crops"
MULTI_DIR = PROJECT_ROOT / "data" / "yolo_single_multi"
BINARY_DIR = PROJECT_ROOT / "data" / "yolo_single_binary"

# 4-to-2 mapping
BINARY_MAP = {
    'no-damage': 'not-damaged',
    'minor-damage': 'damaged',
    'major-damage': 'damaged',
    'destroyed': 'damaged'
}

def process_single_crops():
    if not VIT_DIR.exists():
        print(f"[!] Error: {VIT_DIR} not found.")
        return

    splits = [d for d in VIT_DIR.iterdir() if d.is_dir()]
    print(f"[*] Found {len(splits)} splits: {[d.name for d in splits]}")

    for split_dir in splits:
        split_name = split_dir.name
        # YOLO expects 'val'
        out_split_name = 'val' if split_name == 'test' else split_name

        print(f"\n--- Processing: {split_name} -> {out_split_name} ---")
        
        for orig_class, bin_class in BINARY_MAP.items():
            src_class_dir = split_dir / orig_class
            if not src_class_dir.exists():
                continue

            # Setup target directories
            dst_multi = MULTI_DIR / out_split_name / orig_class
            dst_binary = BINARY_DIR / out_split_name / bin_class
            dst_multi.mkdir(parents=True, exist_ok=True)
            dst_binary.mkdir(parents=True, exist_ok=True)

            imgs = list(src_class_dir.glob("*.png"))
            if not imgs:
                continue

            # Process and split images
            desc = f"{orig_class} -> multi & binary"
            for img_path in tqdm(imgs, desc=desc, unit="img", dynamic_ncols=True):
                # Read image
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                
                # We add padding because YOLO crops images to squares
                h, w, c = img.shape
                square_size = max(h, w)
                padded_img = np.zeros((square_size, square_size, c), dtype=np.uint8)
                
                y_offset = (square_size - h) // 2
                x_offset = (square_size - w) // 2
                padded_img[y_offset:y_offset+h, x_offset:x_offset+w] = img
                
                # Save padded image to both directories
                cv2.imwrite(str(dst_multi / img_path.name), padded_img)
                cv2.imwrite(str(dst_binary / img_path.name), padded_img)

if __name__ == '__main__':
    # Clear existing dirs
    for d in [MULTI_DIR, BINARY_DIR]:
        if d.exists():
            print(f"[*] Clearing {d.name}...")
            shutil.rmtree(d)
    
    process_single_crops()
    print(f"\n[+] Multi dataset ready: {MULTI_DIR}")
    print(f"[+] Binary dataset ready: {BINARY_DIR}")