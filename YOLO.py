import json
import shutil
import random
from pathlib import Path
from shapely.wkt import loads as load_wkt
from ultralytics import YOLO
import torch

# paths setup
base_dir = Path(r"C:\Users\misha\OneDrive\Desktop\DL")
source_train_dir = base_dir / "train"
images_dir = source_train_dir / "images"
labels_dir = source_train_dir / "labels"
out_root = base_dir / "yolo-damage" / "data"
yaml_path = base_dir / "yolo-damage" / "data.yaml"
project_path = base_dir / "yolo-damage"

# 1-class map: 0 = building (any state)
damage_map = {
    "no-damage": 0,
    "minor-damage": 0,
    "major-damage": 0,
    "destroyed": 0,
}

# convert wkt poly to yolo format
def poly_to_yolo(wkt_str, img_width=1024, img_height=1024):
    poly = load_wkt(wkt_str)
    minx, miny, maxx, maxy = poly.bounds
    xc = ((minx + maxx) / 2) / img_width
    yc = ((miny + maxy) / 2) / img_height
    bw = (maxx - minx) / img_width
    bh = (maxy - miny) / img_height
    return xc, yc, bw, bh

# parse json, check bounds, copy img, save txt
def process_file(json_path, target_img_dir, target_lbl_dir):
    with open(json_path) as f:
        data = json.load(f)

    img_name = json_path.name.replace(".json", ".png")
    img_path = images_dir / img_name
    if not img_path.exists():
        return False

    lines = []
    for feat in data["features"]["xy"]:
        props = feat["properties"]
        subtype = props.get("subtype")
        if subtype not in damage_map:
            continue

        wkt_str = feat["wkt"]
        xc, yc, bw, bh = poly_to_yolo(wkt_str)
        if bw <= 0 or bh <= 0:
            continue

        cls = damage_map[subtype]
        lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

    if not lines:
        return False

    shutil.copy2(img_path, target_img_dir / img_name)
    (target_lbl_dir / f"{json_path.stem}.txt").write_text("\n".join(lines))
    return True

# rtx 4070 limits, batch 4 keeps vram safe, 1-class detection
train_params = {
    "data": str(yaml_path),
    "epochs": 100,
    "patience": 20,
    "imgsz": 1024,
    "batch": 4,
    "degrees": 180.0,
    "flipud": 0.5,
    "fliplr": 0.5,
    "project": str(project_path / "runs"),
    "name": "damage_yolo11s_1024_1class",
    "device": 0
}

# run execution safely
if __name__ == '__main__':
    # nuke old data
    if out_root.exists():
        shutil.rmtree(out_root)

    # make clean dirs
    for split in ["train", "val"]:
        (out_root / split / "images").mkdir(parents=True, exist_ok=True)
        (out_root / split / "labels").mkdir(parents=True, exist_ok=True)

    # find and split files 90/10
    json_files = sorted(list(labels_dir.glob("*_post_disaster.json")))
    if len(json_files) == 0:
        print("Error: No files found")
    else:
        random.seed(42)
        random.shuffle(json_files)
        split_idx = int(len(json_files) * 0.9)
        train_files = json_files[:split_idx]
        val_files = json_files[split_idx:]

        print(f"split: {len(train_files)} train, {len(val_files)} val")

        # save train files
        train_count = sum(1 for jp in train_files if process_file(jp, out_root / "train" / "images", out_root / "train" / "labels"))
        
        # save val files
        val_count = sum(1 for jp in val_files if process_file(jp, out_root / "val" / "images", out_root / "val" / "labels"))

        # write 1-class data.yaml
        yaml_content = f"""path: C:/Users/misha/OneDrive/Desktop/DL/yolo-damage/data
train: train/images
val: val/images

names:
  0: building
"""
        yaml_path.write_text(yaml_content, encoding="utf-8")
        print(f"train: {train_count} | val: {val_count}")

    # yolov11s goes here
    model = YOLO("yolo11s.pt")

    # empty cuda cache
    torch.cuda.empty_cache()

    # start training
    results = model.train(**train_params)