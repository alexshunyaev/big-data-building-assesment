import os
import cv2
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from models_vit.vit import CustomChangeViT

DAMAGE_CLASSES = {0: "no-damage", 1: "minor-damage", 2: "major-damage", 3: "destroyed"}
NORM_MEAN = np.array([0.5, 0.5, 0.5], dtype=np.float32)
NORM_STD  = np.array([0.5, 0.5, 0.5], dtype=np.float32)
CROP_SIZE = 224
CROP_PAD  = 2


def load_yolo(weights_path: str, device: torch.device) -> YOLO:
    model = YOLO(weights_path)
    model.to(device)
    return model


def load_vit(weights_path: str, device: torch.device) -> CustomChangeViT:
    model = CustomChangeViT(
        img_size=224, patch_size=16, in_channels=6, num_classes=4,
        embed_dim=256, depth=6, num_heads=8, drop_path_rate=0.2
    )
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def _preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    """BGR crop → normalised CHW float tensor."""
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - NORM_MEAN) / NORM_STD
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _detect_buildings(yolo: YOLO, post_img_bgr: np.ndarray, conf: float = 0.25) -> list[list[int]]:
    """
    Run YOLO on post-disaster image.
    Returns list of pixel-space bboxes [x1, y1, x2, y2] — class ignored.
    """
    h, w = post_img_bgr.shape[:2]
    results = yolo.predict(source=post_img_bgr, imgsz=1024, conf=conf, verbose=False)
    boxes = []
    for box in results[0].boxes.xyxy.cpu().numpy():
        x1, y1, x2, y2 = box[:4]
        x1 = int(max(0, x1 - CROP_PAD))
        y1 = int(max(0, y1 - CROP_PAD))
        x2 = int(min(w,  x2 + CROP_PAD))
        y2 = int(min(h,  y2 + CROP_PAD))
        if (x2 - x1) >= 10 and (y2 - y1) >= 10:
            boxes.append([x1, y1, x2, y2])
    return boxes


def _classify_building(vit: CustomChangeViT, pre_bgr: np.ndarray, post_bgr: np.ndarray,
                        device: torch.device) -> tuple[str, float]:
    """Crop pair → damage class label and confidence."""
    pre_t  = _preprocess_crop(pre_bgr).unsqueeze(0).to(device)
    post_t = _preprocess_crop(post_bgr).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = vit(pre_t, post_t)
        probs  = torch.softmax(logits, dim=1)[0]
        cls_id = int(probs.argmax())

    return DAMAGE_CLASSES[cls_id], float(probs[cls_id])


def run_pipeline(
    pre_image_path:  str,
    post_image_path: str,
    yolo_weights:    str,
    vit_weights:     str,
    conf:            float = 0.25,
) -> list[dict]:
    """
    Full inference pipeline for a single pre/post image pair.

    Returns a list of dicts:
        {
            "bbox":       [x1, y1, x2, y2],   # pixel coords in original image
            "class":      "major-damage",
            "confidence": 0.91
        }
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_bgr  = cv2.imread(pre_image_path)
    post_bgr = cv2.imread(post_image_path)
    if pre_bgr is None or post_bgr is None:
        raise FileNotFoundError(f"Could not load images:\n  {pre_image_path}\n  {post_image_path}")

    yolo = load_yolo(yolo_weights, device)
    vit  = load_vit(vit_weights, device)

    boxes = _detect_buildings(yolo, post_bgr, conf=conf)
    print(f"[YOLO] Detected {len(boxes)} buildings.")

    results = []
    for bbox in boxes:
        x1, y1, x2, y2 = bbox

        pre_crop  = cv2.resize(pre_bgr[y1:y2, x1:x2],   (CROP_SIZE, CROP_SIZE))
        post_crop = cv2.resize(post_bgr[y1:y2, x1:x2],  (CROP_SIZE, CROP_SIZE))

        label, conf_score = _classify_building(vit, pre_crop, post_crop, device)
        results.append({"bbox": bbox, "class": label, "confidence": round(conf_score, 4)})

    return results


def run_pipeline_batch(
    pairs:        list[tuple[str, str]],
    yolo_weights: str,
    vit_weights:  str,
    conf:         float = 0.25,
) -> list[list[dict]]:
    """
    Run the pipeline over multiple (pre, post) image path pairs.
    Models are loaded once and reused across all pairs.

    Returns one result list per pair (same order as input).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    yolo = load_yolo(yolo_weights, device)
    vit  = load_vit(vit_weights, device)

    all_results = []
    for i, (pre_path, post_path) in enumerate(pairs):
        print(f"[{i+1}/{len(pairs)}] Processing: {Path(post_path).name}")
        pre_bgr  = cv2.imread(pre_path)
        post_bgr = cv2.imread(post_path)
        if pre_bgr is None or post_bgr is None:
            print(f"  [!] Skipping — could not load images.")
            all_results.append([])
            continue

        boxes = _detect_buildings(yolo, post_bgr, conf=conf)
        print(f"  [YOLO] {len(boxes)} buildings detected.")

        scene_results = []
        for bbox in boxes:
            x1, y1, x2, y2 = bbox
            pre_crop  = cv2.resize(pre_bgr[y1:y2, x1:x2],  (CROP_SIZE, CROP_SIZE))
            post_crop = cv2.resize(post_bgr[y1:y2, x1:x2], (CROP_SIZE, CROP_SIZE))
            label, conf_score = _classify_building(vit, pre_crop, post_crop, device)
            scene_results.append({"bbox": bbox, "class": label, "confidence": round(conf_score, 4)})

        all_results.append(scene_results)

    return all_results


if __name__ == "__main__":
    current_dir  = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    PRE_IMAGE   = os.path.join(project_root, "data", "raw", "test", "images", "guatemala-volcano_00000000_pre_disaster.png")
    POST_IMAGE  = os.path.join(project_root, "data", "raw", "test", "images", "guatemala-volcano_00000000_post_disaster.png")
    YOLO_WEIGHTS = os.path.join(project_root, "yolo-damage", "runs", "damage_yolo11s_1024_balanced", "weights", "best.pt")
    VIT_WEIGHTS  = os.path.join(project_root, "results", "models", "best_vit.pth")

    detections = run_pipeline(PRE_IMAGE, POST_IMAGE, YOLO_WEIGHTS, VIT_WEIGHTS)

    print(f"\n=== Results: {len(detections)} buildings ===")
    for i, d in enumerate(detections):
        print(f"  [{i+1}] {d['class']:<15} conf={d['confidence']:.2f}  bbox={d['bbox']}")
