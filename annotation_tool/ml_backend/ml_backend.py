"""
ml_backend.py
-------------
Label Studio ML backend for live vein auto-annotation.

Label Studio calls this server when you open an image.
It runs the SAM2 + vessel detection pipeline and returns
pre-filled polygon suggestions -- you just review and correct.

Setup:
    pip install label-studio-ml flask
    python annotation_tool/ml_backend/ml_backend.py

Then in Label Studio:
  Settings -> Machine Learning -> Add Model
  URL: http://localhost:9090
"""

import os
import io
import base64
import logging
import numpy as np
import cv2
from pathlib import Path
from flask import Flask, request, jsonify

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from auto_annotate import (
    build_candidate_mask,
    extract_candidate_boxes,
    classify_vein,
    mask_to_yolo_polygon,
    load_sam2,
    sam2_predict,
)

SAM2_CHECKPOINT = os.environ.get(
    "SAM2_CHECKPOINT",
    str(Path(__file__).parent.parent.parent / "models/sam2/sam2.1_hiera_small.pt")
)
DEVICE      = os.environ.get("DEVICE", "auto")
CONF_THRESH = float(os.environ.get("CONF_THRESH", "0.35"))
PORT        = int(os.environ.get("PORT", "9090"))

CLASS_NAMES = {0: "spider_vein", 1: "facial_vein"}

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

predictor, device = load_sam2(SAM2_CHECKPOINT, DEVICE)
logger.info(f"ML backend ready. SAM2={'loaded' if predictor else 'fallback mode'}")


def decode_image(data_url):
    if data_url.startswith("data:image"):
        header, encoded = data_url.split(",", 1)
        img_bytes = base64.b64decode(encoded)
        arr = np.frombuffer(img_bytes, np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)
    else:
        return cv2.imread(data_url)


def masks_to_ls_predictions(img_bgr, yolo_lines, from_name="label", to_name="image"):
    h, w = img_bgr.shape[:2]
    results = []
    for line in yolo_lines:
        parts = line.strip().split()
        if len(parts) < 7:
            continue
        cls = int(parts[0])
        coords = list(map(float, parts[1:]))
        points = [{"x": coords[i] * 100, "y": coords[i+1] * 100}
                  for i in range(0, len(coords), 2)]
        results.append({
            "from_name": from_name,
            "to_name":   to_name,
            "type":      "polygonlabels",
            "value": {
                "points":        points,
                "polygonlabels": [CLASS_NAMES[cls]],
                "closed":        True,
            },
            "score": 0.8,
        })
    return results


def run_pipeline(img_bgr):
    h, w = img_bgr.shape[:2]
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    candidate_mask = build_candidate_mask(img_bgr, frangi_thresh=CONF_THRESH)
    boxes = extract_candidate_boxes(candidate_mask, img_bgr.shape)
    if not boxes:
        return []
    yolo_lines = []
    if predictor is not None:
        masks = sam2_predict(predictor, img_rgb, boxes)
        for mask in masks:
            cls = classify_vein(img_bgr, mask)
            yolo_lines.extend(mask_to_yolo_polygon(mask, img_bgr.shape, cls))
    else:
        for box in boxes:
            x1, y1, x2, y2 = box
            roi = np.zeros((h, w), dtype=np.uint8)
            roi[y1:y2, x1:x2] = candidate_mask[y1:y2, x1:x2]
            cls = classify_vein(img_bgr, roi)
            yolo_lines.extend(mask_to_yolo_polygon(roi, img_bgr.shape, cls))
    return yolo_lines


@app.route("/predict", methods=["POST"])
def predict():
    body = request.json
    predictions = []
    for task in body.get("tasks", []):
        img_src = task.get("data", {}).get("image", "")
        if not img_src:
            continue
        img_bgr = decode_image(img_src)
        if img_bgr is None:
            logger.warning(f"Could not decode image for task {task.get('id')}")
            continue
        yolo_lines = run_pipeline(img_bgr)
        ls_results = masks_to_ls_predictions(img_bgr, yolo_lines)
        predictions.append({
            "task":    task.get("id"),
            "result":  ls_results,
            "score":   0.8 if ls_results else 0.0,
            "model_version": "vein-sam2-v1",
        })
        logger.info(f"Task {task.get('id')}: {len(ls_results)} annotations predicted")
    return jsonify({"predictions": predictions})


@app.route("/train", methods=["POST"])
def train():
    return jsonify({"status": "ok", "message": "Training not yet implemented"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "sam2": "loaded" if predictor else "fallback", "device": device})


@app.route("/setup", methods=["POST"])
def setup():
    return jsonify({
        "model_version": "vein-sam2-v1",
        "schema": {"type": "PolygonLabels", "labels": list(CLASS_NAMES.values())},
    })


if __name__ == "__main__":
    print(f"\n{'='*55}")
    print(f"  Vein Auto-Annotation ML Backend")
    print(f"  SAM2: {'loaded' if predictor else 'not found -- using contour fallback'}")
    print(f"  Device: {device}")
    print(f"  Listening on: http://localhost:{PORT}")
    print(f"{'='*55}")
    print(f"\n  In Label Studio: Settings -> Machine Learning -> Add Model")
    print(f"  URL: http://localhost:{PORT}\n")
    app.run(host="0.0.0.0", port=PORT, debug=False)
