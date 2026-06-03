import json
from pathlib import Path

import cv2
from shapely import wkt
from tqdm import tqdm


def process_xview2_to_yolo(raw_dir: Path, out_dir: Path):
    """
    Converts xView2 pre-disaster JSON polygons into YOLO segmentation format.
    """
    raw_images_dir = raw_dir / "images"
    raw_labels_dir = raw_dir / "labels"

    yolo_images_dir = out_dir / "images" / "train"
    yolo_labels_dir = out_dir / "labels" / "train"

    yolo_images_dir.mkdir(parents=True, exist_ok=True)
    yolo_labels_dir.mkdir(parents=True, exist_ok=True)

    # We only care about pre-disaster images for localization
    pre_jsons = list(raw_labels_dir.glob("*_pre_disaster.json"))

    print(f"Processing {len(pre_jsons)} pre-disaster scenes for YOLO...")

    for json_path in tqdm(pre_jsons):
        with open(json_path, 'r') as f:
            data = json.load(f)

        img_name = data['metadata']['img_name']
        img_path = raw_images_dir / img_name

        if not img_path.exists():
            continue

        # Get image dimensions for YOLO normalization
        img = cv2.imread(str(img_path))
        height, width = img.shape[:2]

        yolo_label_path = yolo_labels_dir / f"{img_path.stem}.txt"

        with open(yolo_label_path, 'w') as label_file:
            # xView2 stores buildings in the 'features.xy' array
            for feature in data['features']['xy']:
                if feature['properties']['feature_type'] != 'building':
                    continue

                # Parse the WKT (Well-Known Text) polygon string
                polygon_wkt = feature['wkt']
                poly = wkt.loads(polygon_wkt)

                if poly.is_empty:
                    continue

                # Extract coordinates and normalize them to [0, 1] for YOLO
                coords = list(poly.exterior.coords)
                yolo_coords = []
                for x, y in coords:
                    yolo_coords.append(str(max(0.0, min(1.0, x / width))))
                    yolo_coords.append(str(max(0.0, min(1.0, y / height))))

                # Format: <class_id> <x1> <y1> <x2> <y2> ... (Class 0 = Building)
                yolo_line = f"0 {' '.join(yolo_coords)}\n"
                label_file.write(yolo_line)

        # Copy image to YOLO directory (Consider symlinking to save disk space!)
        cv2.imwrite(str(yolo_images_dir / img_name), img)


if __name__ == "__main__":
    _HERE = Path(__file__).parent
    RAW_DATA = _HERE / "../../data/raw/train"
    YOLO_OUT = _HERE / "../../data/yolo_format"
    process_xview2_to_yolo(RAW_DATA, YOLO_OUT)
