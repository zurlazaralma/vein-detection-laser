# Vein Detection Laser System

Superficial blood vessel detection and segmentation for aesthetic laser targeting.  
Targets: **spider veins** (telangiectasias) and **facial veins** (broken capillaries, rosacea vessels).

---

## Project Structure

```
vein-detection-laser/
├── dataset/
│   ├── data.yaml               # YOLO dataset config (classes, paths)
│   ├── images/
│   │   ├── train/              # 640×640 training images
│   │   ├── val/
│   │   └── test/
│   ├── labels/
│   │   ├── train/              # YOLO seg .txt annotation files
│   │   ├── val/
│   │   └── test/
│   └── raw_downloads/          # Raw images before split/resize
├── scripts/
│   ├── collect_images.py       # Download from ISIC, Wikimedia, Roboflow
│   ├── prepare_dataset.py      # Deduplicate, resize, split into train/val/test
│   ├── train.py                # YOLOv8-seg training
│   └── infer.py                # Inference with mask overlay
├── annotation_tool/
│   └── annotation_guide.md     # Full annotation instructions (Label Studio / LabelImg)
├── models/                     # Trained weights (gitignored, tracked via releases)
└── requirements.txt
```

---

## Quickstart

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Collect images
```bash
# From ISIC dermoscopy archive
python scripts/collect_images.py --source isic --n 50 --out dataset/raw_downloads

# From Wikimedia Commons (free license)
python scripts/collect_images.py --source wikimedia --n 30 --out dataset/raw_downloads

# From Roboflow (requires free API key from roboflow.com)
python scripts/collect_images.py --source roboflow --api-key YOUR_KEY --out dataset/raw_downloads
```

### 3. Annotate
See **[annotation_tool/annotation_guide.md](annotation_tool/annotation_guide.md)** for full instructions.  
Recommended tool: [Label Studio](https://labelstud.io/) — browser-based polygon segmentation.

### 4. Prepare dataset
```bash
python scripts/prepare_dataset.py --raw dataset/raw_downloads --split 70 20 10
```
Deduplicates, letterbox-resizes to 640×640, and creates the train/val/test split.

### 5. Train
```bash
python scripts/train.py --model yolov8s-seg.pt --epochs 100 --batch 16
```
Weights saved to `models/runs/vein_seg_v1/weights/best.pt`. ONNX exported automatically.

### 6. Inference
```bash
python scripts/infer.py \
    --model models/runs/vein_seg_v1/weights/best.pt \
    --source path/to/patient/image.jpg \
    --out output/
```

---

## Dataset Classes

| ID | Class | Description |
|----|-------|-------------|
| 0 | `spider_vein` | Telangiectasias — fine red/purple web-like vessels, typically on legs |
| 1 | `facial_vein` | Broken capillaries, rosacea vessels, perinasal/cheek telangiectasias |

---

## Model

**YOLOv8-seg** — real-time instance segmentation  
- Input: 640×640 RGB  
- Output: bounding boxes + polygon segmentation masks per detected vessel  
- Base: `yolov8s-seg.pt` pretrained on COCO, fine-tuned on vein dataset

---

## Roadmap

- [ ] Collect and annotate 50–100 pilot images
- [ ] Train v1 model, evaluate mAP@50
- [ ] Add reticular vein class (blue-green feeder veins)
- [ ] Integrate with Alma laser targeting system
- [ ] Real-time camera inference pipeline
