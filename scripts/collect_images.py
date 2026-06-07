"""
collect_images.py
-----------------
Downloads vein images from multiple public sources for dataset building.

Sources:
  1. ISIC Archive API  — dermoscopy images with visible vessels
  2. Wikimedia Commons — free-use medical/dermatology images
  3. Roboflow Universe — pre-annotated vein datasets (requires API key)

Usage:
    pip install requests tqdm pillow
    python scripts/collect_images.py --source isic --n 50 --out dataset/raw_downloads
    python scripts/collect_images.py --source wikimedia --n 30 --out dataset/raw_downloads
    python scripts/collect_images.py --source roboflow --api-key YOUR_KEY --out dataset/raw_downloads
"""

import os
import argparse
import requests
import urllib3
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import io
import json
import time

# Corporate network: SSL inspection may break cert chain.
# Suppress InsecureRequestWarning to keep output clean.
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
VERIFY_SSL = False  # Set to True or a path to your CA bundle if you have it

RAW_DIR = Path("dataset/raw_downloads")


# ── ISIC Archive ─────────────────────────────────────────────────────────────

ISIC_API = "https://api.isic-archive.com/api/v2"

# ISIC v2: filter by diagnosis keywords in the clinical metadata.
# These map to vascular / vessel-rich dermoscopy images.
ISIC_DIAGNOSIS_FILTERS = [
    "angioma",
    "telangiectasia",
    "vascular",
    "rosacea",
]

# Shared session
_isic_session = requests.Session()
_isic_session.verify = VERIFY_SSL
_isic_session.headers.update({
    "User-Agent": "VeinDatasetCollector/1.0 (research; almalasers.com)",
    "Accept": "application/json",
})


def _isic_image_url(item: dict) -> str:
    """Extract the full-resolution S3 URL from an ISIC v2 image record."""
    return item.get("files", {}).get("full", {}).get("url", "")


def download_isic(n: int, out_dir: Path):
    """Download up to n vascular-lesion images from ISIC archive (v2 API)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for keyword in ISIC_DIAGNOSIS_FILTERS:
        if downloaded >= n:
            break
        print(f"\n[ISIC] Searching diagnosis keyword: '{keyword}'")
        # ISIC v2 supports ?search= which scans metadata fields
        params = {
            "limit": min(100, n - downloaded),
            "offset": 0,
            "search": keyword,
        }
        try:
            resp = _isic_session.get(f"{ISIC_API}/images/", params=params, timeout=30)
            resp.raise_for_status()
            data = resp.json()
            results = data.get("results", [])
            print(f"  Total matching: {data.get('count', '?')} — fetched {len(results)}")

            for item in tqdm(results, desc=f"  Downloading '{keyword}'"):
                if downloaded >= n:
                    break
                isic_id = item.get("isic_id", "")
                img_url = _isic_image_url(item)
                if not isic_id or not img_url:
                    continue
                try:
                    img_resp = _isic_session.get(img_url, timeout=60)
                    img_resp.raise_for_status()
                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    # Skip small images
                    if img.width < 300 or img.height < 300:
                        continue
                    fname = out_dir / f"isic_{isic_id}.jpg"
                    img.save(fname, "JPEG", quality=95)
                    downloaded += 1
                    print(f"  Saved: {fname.name}")
                except Exception as e:
                    print(f"  Warning: {isic_id} — {e}")
                time.sleep(0.3)
        except Exception as e:
            print(f"  [ISIC] Error for '{keyword}': {e}")

    print(f"\n[ISIC] Downloaded {downloaded} images to {out_dir}")


# ── Wikimedia Commons ─────────────────────────────────────────────────────────

WIKI_API = "https://commons.wikimedia.org/w/api.php"

WIKI_QUERIES = [
    "telangiectasia skin",
    "spider veins",
    "varicose veins",
    "rosacea face",
    "broken capillaries",
]

# Wikimedia requires a descriptive User-Agent or returns 403
_wiki_session = requests.Session()
_wiki_session.verify = VERIFY_SSL
_wiki_session.headers.update({
    "User-Agent": "VeinDatasetCollector/1.0 (research project; contact: zur.lazar@almalasers.com)",
    "Accept": "application/json",
})


def download_wikimedia(n: int, out_dir: Path):
    """Download free-use vein images from Wikimedia Commons."""
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0

    for query in WIKI_QUERIES:
        if downloaded >= n:
            break
        print(f"\n[Wikimedia] Searching: '{query}'")
        params = {
            "action": "query",
            "format": "json",
            "generator": "search",
            "gsrnamespace": 6,   # File namespace only
            "gsrsearch": query,
            "gsrlimit": 20,
            "prop": "imageinfo",
            "iiprop": "url|mime|size",
        }
        try:
            resp = _wiki_session.get(WIKI_API, params=params, timeout=30)
            resp.raise_for_status()
            pages = resp.json().get("query", {}).get("pages", {})
            if not pages:
                print(f"  No results for '{query}'")
                continue

            for page in tqdm(pages.values(), desc=f"  Downloading"):
                if downloaded >= n:
                    break
                ii = page.get("imageinfo", [{}])[0]
                mime = ii.get("mime", "")
                url = ii.get("url", "")
                width = ii.get("width", 0)
                height = ii.get("height", 0)
                if not url or mime not in ("image/jpeg", "image/png"):
                    continue
                if width < 300 or height < 300:
                    continue
                try:
                    img_resp = _wiki_session.get(url, timeout=60)
                    img_resp.raise_for_status()
                    img = Image.open(io.BytesIO(img_resp.content)).convert("RGB")
                    title = page.get("title", f"wiki_{downloaded}").replace(" ", "_").replace("/", "_")
                    fname = out_dir / f"wiki_{title[:60]}.jpg"
                    img.save(fname, "JPEG", quality=95)
                    downloaded += 1
                    print(f"  Saved: {fname.name}")
                except Exception as e:
                    print(f"  Warning: {e}")
                time.sleep(0.4)
        except Exception as e:
            print(f"  [Wikimedia] Error for '{query}': {e}")

    print(f"\n[Wikimedia] Downloaded {downloaded} images to {out_dir}")


# ── Roboflow Universe ─────────────────────────────────────────────────────────

def download_roboflow(api_key: str, out_dir: Path):
    """
    Download the public 'Vein-Segmentation' dataset from Roboflow Universe.
    Dataset: https://universe.roboflow.com/vein-detection/vein-segmentation
    Already in YOLO format — images + labels together.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from roboflow import Roboflow
        rf = Roboflow(api_key=api_key)
        project = rf.workspace("vein-detection").project("vein-segmentation")
        dataset = project.version(1).download("yolov8", location=str(out_dir / "roboflow_vein"))
        print(f"[Roboflow] Downloaded to {out_dir / 'roboflow_vein'}")
    except ImportError:
        print("[Roboflow] Install: pip install roboflow")
    except Exception as e:
        print(f"[Roboflow] Error: {e}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect vein images from public sources")
    parser.add_argument("--source", choices=["isic", "wikimedia", "roboflow", "all"],
                        default="all", help="Image source to download from")
    parser.add_argument("--n", type=int, default=50, help="Number of images to download")
    parser.add_argument("--out", type=str, default="dataset/raw_downloads",
                        help="Output directory")
    parser.add_argument("--api-key", type=str, default="",
                        help="Roboflow API key (only needed for --source roboflow)")
    args = parser.parse_args()

    out_dir = Path(args.out)

    if args.source in ("isic", "all"):
        download_isic(args.n, out_dir / "isic")
    if args.source in ("wikimedia", "all"):
        download_wikimedia(args.n, out_dir / "wikimedia")
    if args.source in ("roboflow", "all") and args.api_key:
        download_roboflow(args.api_key, out_dir)
    elif args.source == "roboflow" and not args.api_key:
        print("[Roboflow] Provide --api-key. Get a free key at roboflow.com")


if __name__ == "__main__":
    main()
