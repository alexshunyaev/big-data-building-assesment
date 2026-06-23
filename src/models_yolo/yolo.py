"""
YOLO Data Preparation and Training Script

This module processes xView2 data into YOLO format and trains a YOLOv11 model
for 1-class detection (building detection).
"""

import json
import shutil
import random
from pathlib import Path

import torch
from shapely.wkt import loads as load_wkt
from ultralytics import YOLO

# Dynamic path resolution relative to this script's location
_HERE = Path(__file__).resolve().parent
PROJECT_ROOT = _HERE.parents[1]  # Navigate up from src/models_yolo to project root

# Input paths
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "train"
IMAGES_DIR = RAW_DATA_DIR / "images"
LABELS_DIR = RAW_DATA_DIR / "labels"

# Output paths
YOLO_DATA_ROOT = PROJECT_ROOT / "data" / "yolo_format"
YAML_PATH = YOLO_DATA_ROOT / "data.yaml"
YOLO_PROJECT_RUNS = PROJECT_ROOT / "results" / "yolo_runs"

# 1-class map: 0 = building (any state)
DAMAGE_MAP = {
    "no-damage": 0,
    "minor-damage": 0,
    "major-damage": 0,
    "destroyed": 0,
}


def poly_to_yolo(wkt_str: str, img_width: int = 1024, img_height: int = 1024) -> tuple[float, float, float, float]:
    """
    Converts a WKT polygon string to YOLO format bounding box coordinates.

    Args:
        wkt_str (str): WKT polygon string.
        img_width (int): Image width in pixels.
        img_height (int): Image height in pixels.

    Returns:
        tuple: (x_center, y_center, width, height) normalized to [0, 1].
    """
    poly = load_wkt(wkt_str)
    minx, miny, maxx, maxy = poly.bounds
    xc = ((minx + maxx) / 2) / img_width
    yc = ((miny + maxy) / 2) / img_height
    bw = (maxx - minx) / img_width
    bh = (maxy - miny) / img_height
    return xc, yc, bw, bh


def process_file(json_path: Path, target_img_dir: Path, target_lbl_dir: Path) -> bool:
    """
    Parses a JSON file, extracts building polygons, and saves them in YOLO format.

    Args:
        json_path (Path): Path to the input JSON file.
        target_img_dir (Path): Destination directory for the image.
        target_lbl_dir (Path): Destination directory for the YOLO label file.

    Returns:
        bool: True if the file was processed successfully, False otherwise.
    """
    with open(json_path) as f:
        data = json.load(f)

    img_name = json_path.name.replace(".json", ".png")
    img_path = IMAGES_DIR / img_name
    if not img_path.exists():
        return False

    lines = []
    for feat in data["features"]["xy"]:
        props = feat["properties"]
        subtype = props.get("subtype")
        if subtype not in DAMAGE_MAP:
            continue

        wkt_str = feat["wkt"]
        xc, yc, bw, bh = poly_to_yolo(wkt_str)
        if bw <= 0 or bh <= 0:
            continue

        cls = DAMAGE_MAP[subtype]
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return False

    shutil.copy2(img_path, target_img_dir / img_name)
    (target_lbl_dir / f"{json_path.stem}.txt").write_text("\n".join(lines))
    return True


def prepare_yolo_dataset() -> None:
    """
    Cleans up old data, processes JSON files into YOLO format, and splits
    the dataset into 90/10 train/validation sets.
    """
    if YOLO_DATA_ROOT.exists():
        shutil.rmtree(YOLO_DATA_ROOT)

    for split in ["train", "val"]:
        (YOLO_DATA_ROOT / split / "images").mkdir(parents=True, exist_ok=True)
        (YOLO_DATA_ROOT / split / "labels").mkdir(parents=True, exist_ok=True)

    json_files = sorted(list(LABELS_DIR.glob("*_post_disaster.json")))
    if not json_files:
        print("Error: No JSON label files found. Ensure raw data is in place.")
        return

    random.seed(42)
    random.shuffle(json_files)
    split_idx = int(len(json_files) * 0.9)
    train_files = json_files[:split_idx]
    val_files = json_files[split_idx:]

    print(f"Dataset split: {len(train_files)} train, {len(val_files)} val")

    train_count = sum(1 for jp in train_files if process_file(
        jp, YOLO_DATA_ROOT / "train" / "images", YOLO_DATA_ROOT / "train" / "labels"
    ))
    
    val_count = sum(1 for jp in val_files if process_file(
        jp, YOLO_DATA_ROOT / "val" / "images", YOLO_DATA_ROOT / "val" / "labels"
    ))

    # Write data.yaml mapping file
    yaml_content = f"""path: {YOLO_DATA_ROOT.as_posix()}
train: train/images
val: val/images

names:
  0: building
"""
    YAML_PATH.write_text(yaml_content, encoding="utf-8")
    print(f"Dataset prepared. Train: {train_count} | Val: {val_count}")


def train_yolo_model() -> None:
    """
    Initializes and trains the YOLOv11s model on the prepared dataset.
    """
    train_params = {
        "data": str(YAML_PATH),
        "epochs": 100,
        "patience": 20,
        "imgsz": 1024,
        "batch": 4,
        "degrees": 180.0,
        "flipud": 0.5,
        "fliplr": 0.5,
        "project": str(YOLO_PROJECT_RUNS),
        "name": "damage_yolo11s_1024_1class",
        "device": 0
    }

    model = YOLO("yolo11s.pt")
    torch.cuda.empty_cache()
    model.train(**train_params)


if __name__ == '__main__':
    prepare_yolo_dataset()
    train_yolo_model()