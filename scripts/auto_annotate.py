"""
auto_annotate.py
----------------
Automatic vein pre-labeler using SAM2 + vessel detection.

Pipeline per image:
  1. Frangi tubeness filter  — highlights tubular structures (vessels)
  2. HSV color thresholding  — isolates red/purple vein hues
  3. Combined candidate mask — union of both methods
  4. SAM2 prompt             — each candidate region used as a point/box prompt
  5. SAM2 mask refinement    — produces clean polygon contours
  6. YOLO seg export         — saves .txt label file alongside each image

Usage:
    pip install torch torchvision opencv-python scikit-image numpy tqdm
    pip install git+https://github.com/facebookresearch/sam2.git
    # Download SAM2 checkpoint:
    #   https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt
    # Place in: models/sam2/

    python scripts/auto_annotate.py \
        --images  dataset/images/train \
        --labels  dataset/labels/train \
        --checkpoint models/sam2/sam2.1_hiera_small.pt \
        --conf 0.4 \
        --preview
"""

import argparse
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm
import json

# ── Vessel Detection (no GPU needed) ─────────────────────────────────────────

def frangi_vessel_mask(img_bgr: np.ndarray, scale_range=(1, 6)) -> np.ndarray:
    """
    Frangi tubeness filter — highlights elongated tubular structures.
    Returns float32 mask [0..1].
    """
    from skimage.filters import frangi
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY).astype(np.float32) / 255.0
    # Invert for dark vessels on bright skin background
    gray_inv = 1.0 - gray
    tube = frangi(gray_inv, sigmas=range(*scale_range), black_ridges=False)
    # Also run on original (some veins are darker)
    tube2 = frangi(gray, sigmas=range(*scale_range), black_ridges=True)
    combined = np.maximum(tube, tube2)
    # Normalize
    if combined.max() > 0:
        combined /= combined.max()
    return combined.astype(np.float32)


def hsv_vein_mask(img_bgr: np.ndarray) -> np.ndarray:
    """
    HSV color thresholding for red/purple vein hues on skin.
    Returns uint8 binary mask.
    """
    img_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    h, w = img_bgr.shape[:2]

    # Red hues wrap around 0/180° in OpenCV HSV
    red_low1  = cv2.inRange(img_hsv, (0,   40, 60), (15,  255, 255))
    red_low2  = cv2.inRange(img_hsv, (160, 40, 60), (180, 255, 255))
    # Purple/violet
    purple    = cv2.inRange(img_hsv, (125, 30, 50), (160, 255, 220))
    # Blue-green (reticular / subsurface veins)
    blue_green = cv2.inRange(img_hsv, (85, 30, 30), (130, 200, 180))

    combined = cv2.bitwise_or(red_low1, red_low2)
    combined = cv2.bitwise_or(combined, purple)
    combined = cv2.bitwise_or(combined, blue_green)

    # Morphological cleanup
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    combined = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel, iterations=2)
    combined = cv2.morphologyEx(combined, cv2.MORPH_OPEN,  kernel, iterations=1)
    return combined


def build_candidate_mask(img_bgr: np.ndarray,
                          frangi_thresh: float = 0.15,
                          color_weight: float = 0.5) -> np.ndarray:
    """
    Combine Frangi + HSV into a single binary candidate mask.
    """
    frangi_f = frangi_vessel_mask(img_bgr)
    hsv_mask = hsv_vein_mask(img_bgr).astype(np.float32) / 255.0

    # Weighted combination
    combined = frangi_f * (1 - color_weight) + hsv_mask * color_weight
    binary = (combined > frangi_thresh).astype(np.uint8) * 255

    # Remove tiny noise blobs
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(binary)
    min_area = (img_bgr.shape[0] * img_bgr.shape[1]) * 0.0003  # 0.03% of image
    clean = np.zeros_like(binary)
    for i in range(1, num_labels):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            clean[labels == i] = 255
    return clean


def extract_candidate_boxes(mask: np.ndarray, img_shape,
                              min_area_frac=0.0003, max_area_frac=0.5):
    """
    Extract bounding boxes from candidate regions for SAM prompting.
    Returns list of [x1, y1, x2, y2] in pixel coords.
    """
    h, w = img_shape[:2]
    min_area = h * w * min_area_frac
    max_area = h * w * max_area_frac

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask)
    boxes = []
    for i in range(1, num_labels):
        area = stats[i, cv2.CC_STAT_AREA]
        if not (min_area <= area <= max_area):
            continue
        x, y, bw, bh = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP], \
                        stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        # Add small padding
        pad = 4
        boxes.append([max(0, x-pad), max(0, y-pad),
                       min(w, x+bw+pad), min(h, y+bh+pad)])
    return boxes


# ── SAM2 Segmentation ─────────────────────────────────────────────────────────

def load_sam2(checkpoint: str, device: str = "auto"):
    """Load SAM2 predictor. Falls back gracefully if not installed."""
    try:
        import torch
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"

        # Infer model config from checkpoint filename
        ckpt = Path(checkpoint).name
        if "large" in ckpt:
            cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
        elif "base" in ckpt or "base_plus" in ckpt:
            cfg = "configs/sam2.1/sam2.1_hiera_b+.yaml"
        elif "small" in ckpt:
            cfg = "configs/sam2.1/sam2.1_hiera_s.yaml"
        else:
            cfg = "configs/sam2.1/sam2.1_hiera_t.yaml"  # tiny

        sam2_model = build_sam2(cfg, checkpoint, device=device)
        predictor = SAM2ImagePredictor(sam2_model)
        print(f"[SAM2] Loaded {ckpt} on {device}")
        return predictor, device

    except ImportError:
        print("[SAM2] Not installed — falling back to contour-only mode.")
        print("       Install: pip install git+https://github.com/facebookresearch/sam2.git")
        return None, "cpu"
    except Exception as e:
        print(f"[SAM2] Load error: {e} — using contour-only mode.")
        return None, "cpu"


def sam2_predict(predictor, img_rgb: np.ndarray, boxes: list) -> list:
    """
    Run SAM2 on a list of box prompts.
    Returns list of binary masks (H×W uint8).
    """
    import torch
    import numpy as np

    predictor.set_image(img_rgb)
    masks_out = []

    for box in boxes:
        box_np = np.array(box, dtype=np.float32)
        with torch.no_grad():
            masks, scores, _ = predictor.predict(
                box=box_np,
                multimask_output=True,
            )
        # Pick highest-scoring mask
        best = masks[np.argmax(scores)]
        masks_out.append(best.astype(np.uint8) * 255)

    return masks_out


# ── Polygon → YOLO seg format ─────────────────────────────────────────────────

def mask_to_yolo_polygon(mask: np.ndarray, img_shape,
                          class_id: int,
                          epsilon_frac: float = 0.005,
                          min_points: int = 4) -> list[str]:
    """
    Convert binary mask to YOLO segmentation lines.
    One line per connected region: <class> x1 y1 x2 y2 ... (normalized).
    Returns list of YOLO annotation strings.
    """
    h, w = img_shape[:2]
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    lines = []

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < (h * w * 0.0001):  # skip tiny fragments
            continue
        # Simplify polygon
        epsilon = epsilon_frac * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, epsilon, True)
        pts = approx.reshape(-1, 2)
        if len(pts) < min_points:
            continue
        # Normalize
        norm = []
        for px, py in pts:
            norm.extend([round(px / w, 6), round(py / h, 6)])
        coords = " ".join(map(str, norm))
        lines.append(f"{class_id} {coords}")

    return lines


# ── Classify vein type per region ────────────────────────────────────────────

def classify_vein(img_bgr: np.ndarray, mask: np.ndarray) -> int:
    """
    Heuristic class assignment based on color inside the mask.
    0 = spider_vein (red/purple)
    1 = facial_vein (red/capillary, often finer)
    """
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    region_h = hsv[:, :, 0][mask > 0]
    if len(region_h) == 0:
        return 0
    mean_h = float(np.mean(region_h))
    # Purple/violet hues → spider_vein; reddish → facial_vein
    if 125 <= mean_h <= 160:
        return 0  # spider_vein (purple)
    else:
        return 1  # facial_vein (red/capillary)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def annotate_image(img_path: Path, label_path: Path,
                   predictor, conf_thresh: float,
                   preview: bool = False) -> int:
    """Process a single image. Returns number of annotations written."""
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"  Could not read: {img_path.name}")
        return 0

    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Step 1: vessel candidate detection
    candidate_mask = build_candidate_mask(img_bgr)
    boxes = extract_candidate_boxes(candidate_mask, img_bgr.shape)

    if not boxes:
        print(f"  {img_path.name}: no vessel candidates found")
        return 0

    yolo_lines = []

    # Step 2: SAM2 refinement (if available)
    if predictor is not None:
        masks = sam2_predict(predictor, img_rgb, boxes)
        for mask in masks:
            class_id = classify_vein(img_bgr, mask)
            yolo_lines.extend(mask_to_yolo_polygon(mask, img_bgr.shape, class_id))
    else:
        # Fallback: use raw candidate mask contours
        for box in boxes:
            x1, y1, x2, y2 = box
            roi_mask = np.zeros((h, w), dtype=np.uint8)
            roi_mask[y1:y2, x1:x2] = candidate_mask[y1:y2, x1:x2]
            class_id = classify_vein(img_bgr, roi_mask)
            yolo_lines.extend(mask_to_yolo_polygon(roi_mask, img_bgr.shape, class_id))

    if not yolo_lines:
        return 0

    # Step 3: write YOLO label file
    label_path.parent.mkdir(parents=True, exist_ok=True)
    label_path.write_text("\n".join(yolo_lines) + "\n")

    # Step 4: optional preview image
    if preview:
        vis = img_bgr.copy()
        for line in yolo_lines:
            parts = line.strip().split()
            cls = int(parts[0])
            coords = list(map(float, parts[1:]))
            pts = np.array([(int(coords[i]*w), int(coords[i+1]*h))
                            for i in range(0, len(coords), 2)], dtype=np.int32)
            color = (0, 0, 255) if cls == 0 else (0, 140, 255)
            cv2.polylines(vis, [pts], True, color, 2)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [pts], color)
            vis = cv2.addWeighted(vis, 0.65, overlay, 0.35, 0)
        preview_dir = img_path.parent.parent.parent / "previews"
        preview_dir.mkdir(exist_ok=True)
        cv2.imwrite(str(preview_dir / f"preview_{img_path.name}"), vis)

    return len(yolo_lines)


def main():
    parser = argparse.ArgumentParser(description="Auto-annotate vein images with SAM2 + vessel detection")
    parser.add_argument("--images",     default="dataset/images/train",
                        help="Folder of input images")
    parser.add_argument("--labels",     default="dataset/labels/train",
                        help="Output folder for YOLO .txt label files")
    parser.add_argument("--checkpoint", default="models/sam2/sam2.1_hiera_small.pt",
                        help="SAM2 checkpoint path")
    parser.add_argument("--conf",       type=float, default=0.4,
                        help="Frangi confidence threshold (0–1)")
    parser.add_argument("--device",     default="auto",
                        help="'cuda', 'cpu', or 'auto'")
    parser.add_argument("--preview",    action="store_true",
                        help="Save annotated preview images to dataset/previews/")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip images that already have a label file")
    args = parser.parse_args()

    img_dir   = Path(args.images)
    label_dir = Path(args.labels)
    img_paths = sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))

    if not img_paths:
        print(f"No images found in {img_dir}")
        return

    print(f"Found {len(img_paths)} images in {img_dir}")

    # Load SAM2 (optional — falls back to contour mode if not installed)
    predictor, device = load_sam2(args.checkpoint, args.device)

    total_annotations = 0
    skipped = 0

    for img_path in tqdm(img_paths, desc="Auto-annotating"):
        label_path = label_dir / f"{img_path.stem}.txt"

        if args.skip_existing and label_path.exists():
            skipped += 1
            continue

        n = annotate_image(img_path, label_path, predictor,
                           args.conf, args.preview)
        total_annotations += n

    print(f"\nDone.")
    print(f"  Images processed : {len(img_paths) - skipped}")
    print(f"  Images skipped   : {skipped}")
    print(f"  Total annotations: {total_annotations}")
    print(f"  Labels written to: {label_dir}")
    if args.preview:
        print(f"  Previews in      : {img_dir.parent.parent}/previews/")
    print("\nNext: open Label Studio to review and correct the auto-annotations.")
    print("      See annotation_tool/annotation_guide.md for import instructions.")


if __name__ == "__main__":
    main()
