import json
from pathlib import Path

import cv2
import numpy as np
from shapely import wkt
from tqdm import tqdm


def create_vit_dataset(raw_dir: Path, out_dir: Path, crop_size: int = 64):
    """
    Extracts building pairs (Pre and Post) and saves them into class folders.
    """
    raw_images_dir = raw_dir / "images"
    raw_labels_dir = raw_dir / "labels"

    # xView2 Damage Classes
    classes = ["no-damage", "minor-damage", "major-damage", "destroyed"]
    for c in classes:
        (out_dir / c).mkdir(parents=True, exist_ok=True)

    post_jsons = list(raw_labels_dir.glob("*_post_disaster.json"))
    print(f"Extracting ViT crops from {len(post_jsons)} post-disaster scenes...")

    for post_json_path in tqdm(post_jsons):
        with open(post_json_path, 'r') as f:
            post_data = json.load(f)

        post_img_name = post_data['metadata']['img_name']
        pre_img_name = post_img_name.replace("_post_disaster", "_pre_disaster")

        post_img_path = raw_images_dir / post_img_name
        pre_img_path = raw_images_dir / pre_img_name

        if not post_img_path.exists() or not pre_img_path.exists():
            continue

        pre_img = cv2.imread(str(pre_img_path))
        post_img = cv2.imread(str(post_img_path))

        for idx, feature in enumerate(post_data['features']['xy']):
            damage_type = feature['properties'].get('subtype', 'un-classified')

            if damage_type not in classes:
                continue  # Skip unclassified buildings

            poly = wkt.loads(feature['wkt'])
            if poly.is_empty:
                continue

            # Get bounding box [min_x, min_y, max_x, max_y]
            min_x, min_y, max_x, max_y = [int(v) for v in poly.bounds]

            # Add a small padding (e.g., 2 pixels) to capture edges
            pad = 2
            min_y, max_y = max(0, min_y - pad), min(pre_img.shape[0], max_y + pad)
            min_x, max_x = max(0, min_x - pad), min(pre_img.shape[1], max_x + pad)

            # Ignore tiny noisy polygons
            if max_y - min_y < 10 or max_x - min_x < 10:
                continue

            # Crop both images using the exact same coordinates
            pre_crop = pre_img[min_y:max_y, min_x:max_x]
            post_crop = post_img[min_y:max_y, min_x:max_x]

            # Resize to standardized ViT patch size
            pre_crop = cv2.resize(pre_crop, (crop_size, crop_size))
            post_crop = cv2.resize(post_crop, (crop_size, crop_size))

            # Concatenate horizontally: Shape becomes (crop_size, crop_size * 2, 3)
            # The ViT will see the 'before' on the left and 'after' on the right
            combined_crop = np.hstack((pre_crop, post_crop))

            # Save to the respective damage class folder
            uid = feature['properties'].get('uid', f"building_{idx}")
            save_path = out_dir / damage_type / f"{uid}.png"
            cv2.imwrite(str(save_path), combined_crop)


if __name__ == "__main__":
    _HERE = Path(__file__).parent
    RAW_DATA = _HERE / "../../data/raw/train"
    VIT_OUT = _HERE / "../../data/vit_crops/train"

    # 64x64 is excellent for ViTs processing buildings. Combined, the image is 128x64.
    create_vit_dataset(RAW_DATA, VIT_OUT, crop_size=64)
