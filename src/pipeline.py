"""
Inference Pipeline Script

This module runs the full pipeline for detecting buildings and classifying
their damage levels using a YOLO object detection model and a Vision Transformer
(ViT) classification model.
"""

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


def load_yolo(weights_path: str | Path, device: torch.device) -> YOLO:
    """
    Loads the YOLOv11 model.

    Args:
        weights_path (str | Path): Path to the trained YOLO weights.
        device (torch.device): Target device (CPU or CUDA).

    Returns:
        YOLO: Loaded YOLO model.
    """
    model = YOLO(str(weights_path))
    model.to(device)
    return model


def load_vit(weights_path: str | Path, device: torch.device) -> CustomChangeViT:
    """
    Loads the Custom Change Vision Transformer model.

    Args:
        weights_path (str | Path): Path to the trained ViT weights.
        device (torch.device): Target device (CPU or CUDA).

    Returns:
        CustomChangeViT: Loaded and initialized ViT model in evaluation mode.
    """
    model = CustomChangeViT(
        img_size=224, patch_size=16, in_channels=9, num_classes=4,
        embed_dim=256, depth=6, num_heads=8, drop_path_rate=0.2
    )
    model.load_state_dict(torch.load(str(weights_path), map_location=device))
    model.to(device)
    model.eval()
    return model


def _preprocess_crop(crop_bgr: np.ndarray) -> torch.Tensor:
    """
    Converts a BGR crop to a normalized CHW float tensor.

    Args:
        crop_bgr (np.ndarray): BGR image crop.

    Returns:
        torch.Tensor: Normalized tensor.
    """
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    rgb = (rgb - NORM_MEAN) / NORM_STD
    return torch.from_numpy(rgb).permute(2, 0, 1)


def _detect_buildings(yolo: YOLO, post_img_bgr: np.ndarray, conf: float = 0.25) -> list[list[int]]:
    """
    Runs YOLO on a post-disaster image to detect building bounding boxes.

    Args:
        yolo (YOLO): Loaded YOLO model.
        post_img_bgr (np.ndarray): Post-disaster BGR image.
        conf (float, optional): Confidence threshold. Defaults to 0.25.

    Returns:
        list[list[int]]: List of pixel-space bounding boxes [x1, y1, x2, y2].
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
    """
    Classifies building damage from pre and post disaster image crops.

    Args:
        vit (CustomChangeViT): Loaded ViT model.
        pre_bgr (np.ndarray): Pre-disaster BGR image crop.
        post_bgr (np.ndarray): Post-disaster BGR image crop.
        device (torch.device): Computation device.

    Returns:
        tuple[str, float]: Predicted damage class and its confidence score.
    """
    pre_t  = _preprocess_crop(pre_bgr).unsqueeze(0).to(device)
    post_t = _preprocess_crop(post_bgr).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = vit(pre_t, post_t)
        probs  = torch.softmax(logits, dim=1)[0]
        cls_id = int(probs.argmax())

    return DAMAGE_CLASSES[cls_id], float(probs[cls_id])


def run_pipeline(
    pre_image_path:  str | Path,
    post_image_path: str | Path,
    yolo_weights:    str | Path,
    vit_weights:     str | Path,
    conf:            float = 0.25,
) -> list[dict]:
    """
    Runs the full inference pipeline for a single pre/post image pair.

    Args:
        pre_image_path (str | Path): Path to pre-disaster image.
        post_image_path (str | Path): Path to post-disaster image.
        yolo_weights (str | Path): Path to YOLO weights.
        vit_weights (str | Path): Path to ViT weights.
        conf (float, optional): Confidence threshold. Defaults to 0.25.

    Returns:
        list[dict]: List of detection dictionaries containing bbox, class, and confidence.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    pre_bgr  = cv2.imread(str(pre_image_path))
    post_bgr = cv2.imread(str(post_image_path))
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


def draw_and_show_results(image_path: str | Path, detections: list[dict]):
    """
    Draws bounding boxes and class labels on an image and displays it on screen.

    Args:
        image_path (str | Path): Path to the original post-disaster image.
        detections (list[dict]): List of detections from run_pipeline.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        return

    # Colors in BGR format
    color_map = {
        "no-damage": (0, 255, 0),       # Green
        "minor-damage": (0, 255, 255),  # Yellow
        "major-damage": (0, 165, 255),  # Orange
        "destroyed": (0, 0, 255)        # Red
    }

    for d in detections:
        x1, y1, x2, y2 = d['bbox']
        cls = d['class']
        conf = d['confidence']
        color = color_map.get(cls, (255, 255, 255))

        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

        # Draw label background
        label = f"{cls} {conf:.2f}"
        (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(img, (x1, y1 - 20), (x1 + w, y1), color, -1)

        # Draw label text
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    cv2.imshow("Detection Results", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    _HERE = Path(__file__).resolve().parent
    PROJECT_ROOT = _HERE.parent

    PRE_IMAGE   = PROJECT_ROOT / "data" / "raw" / "test" / "images" / "hurricane-florence_00000150_pre_disaster.png"
    POST_IMAGE  = PROJECT_ROOT / "data" / "raw" / "test" / "images" / "hurricane-florence_00000150_post_disaster.png"
    YOLO_WEIGHTS = PROJECT_ROOT / "results" / "models" / "best_yolo.pt"
    VIT_WEIGHTS  = PROJECT_ROOT / "results" / "models" / "best_vit.pth"

    try:
        detections = run_pipeline(PRE_IMAGE, POST_IMAGE, YOLO_WEIGHTS, VIT_WEIGHTS)
        print(f"\n=== Results: {len(detections)} buildings ===")
        for i, d in enumerate(detections):
            print(f"  [{i+1}] {d['class']:<15} conf={d['confidence']:.2f}  bbox={d['bbox']}")

        draw_and_show_results(POST_IMAGE, detections)

    except FileNotFoundError as e:
        print(f"Setup Notice: {e}")
        print("Please ensure the data and weights paths exist to run this example.")
