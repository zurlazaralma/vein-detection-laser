"""
prepare_dataset.py
------------------
Organizes raw downloaded images into the YOLO train/val/test split.
Deduplicates by perceptual hash, resizes to 640×640 (letterboxed),
and produces a dataset manifest (manifest.json).

Usage:
    pip install pillow imagehash tqdm
    python scripts/prepare_dataset.py \
        --raw dataset/raw_downloads \
        --split 70 20 10
"""

import os
import json
import shutil
import argparse
import random
from pathlib import Path
from tqdm import tqdm
from PIL import Image, ImageOps

try:
    import imagehash
    DEDUP = True
except ImportError:
    DEDUP = False
    print("Warning: imagehash not installed — skipping deduplication. pip install imagehash")

IMG_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
TARGET_SIZE = 640  # YOLO standard


def letterbox(img: Image.Image, size: int = 640, fill=(114, 114, 114)) -> Image.Image:
    """Resize image keeping aspect ratio, pad to square."""
    img.thumbnail((size, size), Image.LANCZOS)
    result = Image.new("RGB", (size, size), fill)
    offset = ((size - img.width) // 2, (size - img.height) // 2)
    result.paste(img, offset)
    return result


def collect_images(raw_dir: Path) -> list:
    imgs = []
    for ext in IMG_EXTENSIONS:
        imgs.extend(raw_dir.rglob(f"*{ext}"))
        imgs.extend(raw_dir.rglob(f"*{ext.upper()}"))
    return sorted(set(imgs))


def deduplicate(paths: list, threshold: int = 8) -> list:
    if not DEDUP:
        return paths
    seen_hashes = set()
    unique = []
    for p in tqdm(paths, desc="Deduplicating"):
        try:
            h = imagehash.phash(Image.open(p))
            if not any(abs(h - s) < threshold for s in seen_hashes):
                seen_hashes.add(h)
                unique.append(p)
        except Exception:
            pass
    print(f"  {len(paths)} → {len(unique)} unique images after dedup")
    return unique


def split(paths: list, train_pct: int, val_pct: int, test_pct: int):
    assert train_pct + val_pct + test_pct == 100
    random.shuffle(paths)
    n = len(paths)
    n_train = int(n * train_pct / 100)
    n_val   = int(n * val_pct   / 100)
    return paths[:n_train], paths[n_train:n_train + n_val], paths[n_train + n_val:]


def copy_and_resize(paths: list, dest: Path, size: int = 640):
    dest.mkdir(parents=True, exist_ok=True)
    for p in tqdm(paths, desc=f"  → {dest.name}"):
        try:
            img = Image.open(p).convert("RGB")
            img = letterbox(img, size)
            img.save(dest / f"{p.stem}.jpg", "JPEG", quality=95)
        except Exception as e:
            print(f"  Warning: {p.name} — {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw",  default="dataset/raw_downloads")
    parser.add_argument("--out",  default="dataset/images")
    parser.add_argument("--split", nargs=3, type=int, default=[70, 20, 10],
                        metavar=("TRAIN", "VAL", "TEST"))
    parser.add_argument("--size", type=int, default=640)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    raw_dir = Path(args.raw)
    out_dir = Path(args.out)

    print(f"\nScanning {raw_dir} for images...")
    all_imgs = collect_images(raw_dir)
    print(f"Found {len(all_imgs)} images")

    all_imgs = deduplicate(all_imgs)

    train, val, test = split(all_imgs, *args.split)
    print(f"\nSplit → train:{len(train)}  val:{len(val)}  test:{len(test)}")

    for subset, paths in [("train", train), ("val", val), ("test", test)]:
        print(f"\nCopying {subset}...")
        copy_and_resize(paths, out_dir / subset, args.size)

    # Write manifest
    manifest = {
        "total": len(all_imgs),
        "train": len(train),
        "val": len(val),
        "test": len(test),
        "size": args.size,
        "split": args.split,
    }
    manifest_path = Path("dataset/manifest.json")
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest written to {manifest_path}")
    print("Done. Next step: annotate images in dataset/images/ using Label Studio or LabelImg.")


if __name__ == "__main__":
    main()
