"""
infer.py
--------
Run vein segmentation inference on a single image or folder.
Outputs annotated images with colored vessel masks overlaid.

Usage:
    python scripts/infer.py --model models/runs/vein_seg_v1/weights/best.pt --source path/to/image.jpg
    python scripts/infer.py --model models/runs/vein_seg_v1/weights/best.pt --source dataset/images/test/
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

CLASS_COLORS = {
    0: (0, 0, 255),    # spider_vein  → red
    1: (255, 100, 0),  # facial_vein  → blue-orange
}

CLASS_NAMES = {0: "spider_vein", 1: "facial_vein"}


def overlay_masks(image: np.ndarray, result, alpha: float = 0.45) -> np.ndarray:
    """Draw segmentation masks and bounding boxes on image."""
    vis = image.copy()

    if result.masks is None:
        return vis

    masks = result.masks.data.cpu().numpy()   # (N, H, W)
    classes = result.boxes.cls.cpu().numpy().astype(int)
    confs = result.boxes.conf.cpu().numpy()

    h, w = image.shape[:2]

    for mask, cls, conf in zip(masks, classes, confs):
        color = CLASS_COLORS.get(cls, (200, 200, 200))
        mask_resized = cv2.resize(mask, (w, h))
        mask_bool = mask_resized > 0.5

        # Colored fill
        overlay = vis.copy()
        overlay[mask_bool] = color
        vis = cv2.addWeighted(vis, 1 - alpha, overlay, alpha, 0)

        # Contour outline
        contours, _ = cv2.findContours(mask_bool.astype(np.uint8),
                                       cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, contours, -1, color, 1)

        # Label
        if contours:
            cx, cy = contours[0][0][0]
            label = f"{CLASS_NAMES[cls]} {conf:.2f}"
            cv2.putText(vis, label, (cx, cy - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)

    return vis


def run(model_path: str, source: str, conf: float = 0.25, out_dir: str = "output"):
    model = YOLO(model_path)
    source_path = Path(source)
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    if source_path.is_file():
        images = [source_path]
    else:
        images = list(source_path.glob("*.jpg")) + list(source_path.glob("*.png"))

    print(f"Running inference on {len(images)} image(s)...")

    for img_path in images:
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"  Could not read {img_path}")
            continue

        results = model.predict(img, conf=conf, verbose=False)
        vis = overlay_masks(img, results[0])

        out_file = out_path / f"pred_{img_path.name}"
        cv2.imwrite(str(out_file), vis)
        print(f"  Saved: {out_file}")

    print("Done.")


def main():
    parser = argparse.ArgumentParser(description="Vein segmentation inference")
    parser.add_argument("--model",  required=True, help="Path to best.pt or model.onnx")
    parser.add_argument("--source", required=True, help="Image file or folder")
    parser.add_argument("--conf",   type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--out",    default="output", help="Output folder for annotated images")
    args = parser.parse_args()
    run(args.model, args.source, args.conf, args.out)


if __name__ == "__main__":
    main()
