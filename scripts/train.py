"""
train.py
--------
Fine-tune YOLOv8 segmentation on the vein dataset.

Usage:
    pip install ultralytics
    python scripts/train.py --model yolov8n-seg --epochs 100 --imgsz 640
"""

import argparse
from pathlib import Path
from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for vein detection")
    parser.add_argument("--model", default="yolov8n-seg.pt",
                        choices=["yolov8n-seg.pt", "yolov8s-seg.pt",
                                 "yolov8m-seg.pt", "yolov8l-seg.pt"],
                        help="YOLO base model (n=nano, s=small, m=medium, l=large)")
    parser.add_argument("--data",   default="dataset/data.yaml")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz",  type=int, default=640)
    parser.add_argument("--batch",  type=int, default=16)
    parser.add_argument("--device", default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--project", default="models/runs")
    parser.add_argument("--name",    default="vein_seg_v1")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=args.project,
        name=args.name,
        # Augmentation tuned for skin/vein images
        hsv_h=0.015,    # slight hue shift (skin tones)
        hsv_s=0.5,      # saturation
        hsv_v=0.3,      # brightness
        degrees=15,     # rotation (limbs can be at angles)
        flipud=0.2,
        fliplr=0.5,
        mosaic=0.5,
        close_mosaic=10,
        # Keep fine vessel detail
        scale=0.5,
        translate=0.1,
    )

    # Export best model to ONNX for deployment
    best = Path(args.project) / args.name / "weights/best.pt"
    if best.exists():
        YOLO(str(best)).export(format="onnx", imgsz=args.imgsz)
        print(f"\nExported ONNX model: {best.with_suffix('.onnx')}")

    print(f"\nTraining complete. Results in: {args.project}/{args.name}/")


if __name__ == "__main__":
    main()
