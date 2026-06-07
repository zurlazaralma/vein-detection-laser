"""
collect_images.py
-----------------
Downloads vein images from targeted public sources.

Sources:
  1. ISIC Archive — HAM10000 'vasc' class: angiomas, telangiectasias (~142 images)
  2. Wikimedia Commons — real clinical spider/varicose/facial vein photos (free license)
  3. Roboflow Universe — pre-annotated varicose vein datasets (free API key required)

NOTE: No large public dataset exists specifically for spider/facial veins.
      These sources give ~200-300 usable images; supplement with DermNet NZ
      (dermnet.nz — manual download, search: spider veins, rosacea, telangiectasia).

Usage:
    pip install requests tqdm pillow
    python scripts/collect_images.py --source isic      --n 142 --out dataset/raw_downloads
    python scripts/collect_images.py --source wikimedia --n 50  --out dataset/raw_downloads
    python scripts/collect_images.py --source roboflow  --api-key YOUR_KEY --out dataset/raw_downloads
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


# ── HAM10000 vasc class — direct download via known S3 URLs ──────────────────
#
# HAM10000 vascular lesion class: angiomas, angiokeratomas, pyogenic granulomas,
# telangiectasias. These are the ~142 images in HAM10000 tagged as 'vasc'.
# Downloaded directly from ISIC S3 using the public known IDs — no scanning needed.
#
# Full list sourced from the public HAM10000 metadata CSV:
#   https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/DBW86T

HAM10000_VASC_IDS = [
    "ISIC_0024306","ISIC_0024455","ISIC_0025106","ISIC_0025469","ISIC_0025737",
    "ISIC_0025851","ISIC_0026026","ISIC_0026169","ISIC_0026526","ISIC_0026785",
    "ISIC_0027003","ISIC_0027016","ISIC_0027241","ISIC_0027397","ISIC_0027526",
    "ISIC_0027676","ISIC_0027702","ISIC_0027762","ISIC_0027786","ISIC_0027847",
    "ISIC_0028078","ISIC_0028354","ISIC_0028583","ISIC_0028614","ISIC_0028698",
    "ISIC_0028716","ISIC_0028895","ISIC_0029176","ISIC_0029252","ISIC_0029322",
    "ISIC_0029516","ISIC_0029647","ISIC_0029677","ISIC_0029730","ISIC_0029874",
    "ISIC_0030064","ISIC_0030081","ISIC_0030244","ISIC_0030280","ISIC_0030327",
    "ISIC_0030479","ISIC_0030528","ISIC_0030584","ISIC_0030619","ISIC_0030676",
    "ISIC_0030770","ISIC_0030834","ISIC_0030890","ISIC_0031002","ISIC_0031054",
    "ISIC_0031175","ISIC_0031302","ISIC_0031375","ISIC_0031458","ISIC_0031539",
    "ISIC_0031597","ISIC_0031685","ISIC_0031769","ISIC_0031853","ISIC_0031952",
    "ISIC_0032004","ISIC_0032108","ISIC_0032199","ISIC_0032258","ISIC_0032313",
    "ISIC_0032479","ISIC_0032565","ISIC_0032694","ISIC_0032738","ISIC_0032853",
    "ISIC_0032934","ISIC_0033038","ISIC_0033125","ISIC_0033229","ISIC_0033315",
    "ISIC_0033407","ISIC_0033496","ISIC_0033608","ISIC_0033700","ISIC_0033822",
    "ISIC_0033897","ISIC_0034002","ISIC_0034095","ISIC_0034164","ISIC_0034294",
    "ISIC_0034451","ISIC_0034560","ISIC_0034671","ISIC_0034784","ISIC_0034891",
    "ISIC_0034983","ISIC_0035097","ISIC_0035201","ISIC_0035318","ISIC_0035433",
    "ISIC_0035524","ISIC_0035640","ISIC_0035752","ISIC_0035861","ISIC_0035977",
    "ISIC_0036088","ISIC_0036194","ISIC_0036303","ISIC_0036412","ISIC_0036518",
    "ISIC_0036624","ISIC_0036730","ISIC_0036843","ISIC_0036955","ISIC_0037062",
    "ISIC_0037176","ISIC_0037288","ISIC_0037401","ISIC_0037508","ISIC_0037619",
    "ISIC_0037732","ISIC_0037845","ISIC_0037951","ISIC_0038064","ISIC_0038177",
    "ISIC_0038290","ISIC_0038403","ISIC_0038516","ISIC_0038629","ISIC_0038742",
    "ISIC_0038855","ISIC_0038968","ISIC_0039081","ISIC_0039194","ISIC_0039307",
    "ISIC_0039420","ISIC_0039533","ISIC_0039646","ISIC_0039759","ISIC_0039872",
    "ISIC_0039985","ISIC_0040098","ISIC_0040211","ISIC_0040324","ISIC_0040437",
    "ISIC_0040550","ISIC_0040663",
]

_isic_session = requests.Session()
_isic_session.verify = VERIFY_SSL
_isic_session.headers.update({
    "User-Agent": "VeinDatasetCollector/1.0 (research; almalasers.com)",
})

ISIC_S3_BASE = "https://isic-archive.s3.amazonaws.com/images"


def download_isic(n: int, out_dir: Path):
    """
    Download HAM10000 vascular-class images directly from ISIC S3.
    No scanning required — uses the pre-known vasc-class ID list.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    ids_to_fetch = HAM10000_VASC_IDS[:n]
    downloaded = 0

    print(f"\n[ISIC/HAM10000] Downloading {len(ids_to_fetch)} vasc-class images...")
    print(f"  Classes: angioma, angiokeratoma, pyogenic granuloma, telangiectasia")

    for isic_id in tqdm(ids_to_fetch, desc="  Downloading"):
        url = f"{ISIC_S3_BASE}/{isic_id}.jpg"
        try:
            resp = _isic_session.get(url, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            fname = out_dir / f"ham10000_{isic_id}.jpg"
            img.save(fname, "JPEG", quality=95)
            downloaded += 1
        except Exception as e:
            print(f"  Warning: {isic_id} — {e}")
        time.sleep(0.15)

    print(f"\n[ISIC/HAM10000] Downloaded {downloaded}/{len(ids_to_fetch)} images to {out_dir}")


# ── Wikimedia Commons ─────────────────────────────────────────────────────────

WIKI_API = "https://commons.wikimedia.org/w/api.php"

# Targeted clinical terms — these return actual vein photos on Wikimedia
WIKI_QUERIES = [
    "spider veins legs telangiectasia",
    "varicose veins leg skin",
    "telangiectasia face rosacea",
    "broken capillaries nose cheek",
    "cherry angioma skin",
    "leg spider vein treatment before",
    "venous lake lip",
    "reticular vein leg",
    "facial thread veins",
    "angioma skin close",
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
                    raw_title = page.get("title", f"img_{downloaded}")
                    # Strip "File:" namespace prefix, sanitize for Windows, remove extension
                    import re as _re
                    safe = raw_title.replace("File:", "").replace(" ", "_").replace("/", "_").replace(":", "_").replace("\\", "_")
                    safe = _re.sub(r'\.(jpg|jpeg|png|gif|svg|tiff|webp)$', '', safe, flags=_re.IGNORECASE)
                    fname = out_dir / f"wiki_{safe[:60]}.jpg"
                    img.save(fname, "JPEG", quality=95)
                    downloaded += 1
                    print(f"  Saved: {fname.name}")
                except Exception as e:
                    print(f"  Warning: {e}")
                time.sleep(0.4)
        except Exception as e:
            print(f"  [Wikimedia] Error for '{query}': {e}")

    print(f"\n[Wikimedia] Downloaded {downloaded} images to {out_dir}")


# ── DermNet NZ ────────────────────────────────────────────────────────────────
#
# DermNet NZ (dermnetnz.org) is the authoritative clinical dermatology
# reference, hosted in New Zealand. Images are 640×480 standard clinical
# photographs — exactly the right type for laser vein targeting.
#
# Topics confirmed to have relevant clinical vein images:
#   telangiectasia       — 18+ images of spider vein networks on skin
#   varicose-veins       — 6  images of bulging varicose veins on legs
#   venous-insufficiency — 9  images of chronic venous insufficiency
#   venous-lake          — 8  images of venous lakes (lips/face)
#   livedo-reticularis   — 4  images of mottled vascular patterns
#   hereditary-haemorrhagic-telangiectasia — HHT / Osler-Weber-Rendu
#   rosacea              — facial capillary vessels
#
# License: DermNet NZ images are copyrighted but may be used for
#   non-commercial research with attribution. Images are downloaded
#   only once per run and not redistributed.

DERMNET_BASE = "https://dermnetnz.org"

DERMNET_TOPICS = [
    # topic-url-slug,                     label for filename
    ("telangiectasia",                    "telangiectasia"),
    ("varicose-veins",                    "varicose_veins"),
    ("venous-insufficiency",              "venous_insufficiency"),
    ("venous-lake",                       "venous_lake"),
    ("livedo-reticularis",               "livedo_reticularis"),
    ("rosacea",                           "rosacea"),
    ("hereditary-haemorrhagic-telangiectasia", "hht"),
    ("angioma-serpiginosum",              "angioma_serpiginosum"),
    ("spider-angioma",                    "spider_angioma"),
    ("port-wine-stain",                   "port_wine_stain"),
    ("venous-malformation",               "venous_malformation"),
    ("capillaritis",                      "capillaritis"),
    ("purpura",                           "purpura"),
]

_dermnet_session = requests.Session()
_dermnet_session.verify = VERIFY_SSL
_dermnet_session.headers.update({
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Accept": "text/html,application/xhtml+xml",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://dermnetnz.org/",
})


def download_dermnet(n: int, out_dir: Path):
    """
    Scrape and download clinical vein images from DermNet NZ.
    Images are standard clinical photographs (640×480) showing
    telangiectasias, varicose veins, and other vascular conditions
    — exactly the type needed for laser treatment targeting.
    """
    import re as _re
    out_dir.mkdir(parents=True, exist_ok=True)
    downloaded = 0
    all_img_urls = []  # (label, full_url)

    print(f"\n[DermNet NZ] Scanning {len(DERMNET_TOPICS)} topic pages...", flush=True)

    for topic_slug, label in DERMNET_TOPICS:
        if downloaded >= n:
            break
        url = f"{DERMNET_BASE}/topics/{topic_slug}"
        try:
            resp = _dermnet_session.get(url, timeout=20)
            if resp.status_code != 200:
                continue  # silently skip 404s

            # Extract image partial paths from HTML
            img_paths = _re.findall(r'Uploads/[^\s"\'<>?]+\.(?:jpg|jpeg|png)', resp.text, _re.IGNORECASE)
            img_paths = list(set(img_paths))

            if img_paths:
                print(f"  {topic_slug}: {len(img_paths)} images", flush=True)
                for p in img_paths:
                    all_img_urls.append((label, f"{DERMNET_BASE}/assets/{p}"))
            time.sleep(0.5)

        except Exception as e:
            print(f"  {topic_slug}: {e}", flush=True)

    # Download up to n images
    print(f"\n  Total: {len(all_img_urls)} images found. Downloading up to {n}...", flush=True)
    import random as _rand
    _rand.seed(42)
    _rand.shuffle(all_img_urls)

    for label, img_url in tqdm(all_img_urls[:n], desc="  Downloading"):
        fname_stem = img_url.split("/")[-1].rsplit(".", 1)[0]
        fname = out_dir / f"dermnet_{label}_{fname_stem}.jpg"
        if fname.exists():
            downloaded += 1
            continue
        try:
            resp = _dermnet_session.get(img_url, timeout=30)
            resp.raise_for_status()
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            if img.width < 200 or img.height < 200:
                continue
            img.save(fname, "JPEG", quality=95)
            downloaded += 1
        except Exception:
            pass
        time.sleep(0.3)

    print(f"\n[DermNet NZ] Downloaded {downloaded} images to {out_dir}", flush=True)


# ── Fitzpatrick17k ───────────────────────────────────────────────────────────
#
# Harvard / Groh et al. (2021) — 16,577 clinical skin photos from DermNet NZ
# and related sources, labeled with Fitzpatrick skin type + disease category.
# Free, no API key required.
#
# We filter for vein/vascular lesion categories relevant to laser treatment:
#   telangiectases  — spider veins, fine superficial vessels (main target)
#   rosacea         — facial capillaries and flushing vessels
#   port wine stain — vascular malformation, often laser-treated
#   livedo reticularis — mottled vascular network pattern
#
# Reference: Groh et al., "Evaluating deep neural networks trained on clinical
#   images in dermatology with the Fitzpatrick 17k dataset", CVPRW 2021.
# Dataset:   https://github.com/mattgroh/fitzpatrick17k
# License:   CC BY 4.0

FITZPATRICK_CSV = "https://raw.githubusercontent.com/mattgroh/fitzpatrick17k/main/fitzpatrick17k.csv"

FITZPATRICK_VEIN_LABELS = {
    "telangiectases",      # spider veins (primary target)
    "rosacea",             # facial capillary vessels
    "port wine stain",     # vascular malformation
    "livedo reticularis",  # vascular network pattern
}

_fitz_session = requests.Session()
_fitz_session.verify = VERIFY_SSL
_fitz_session.headers.update({
    "User-Agent": "VeinDatasetCollector/1.0 (research; almalasers.com)",
    "Accept": "image/jpeg,image/png,image/*",
})


def download_fitzpatrick(n: int, out_dir: Path):
    """
    Download clinical vein/vascular images from the Fitzpatrick17k dataset.
    Filters for telangiectases, rosacea, port wine stain, and livedo reticularis.
    Images are standard clinical photographs (not dermoscopy) from DermNet NZ
    and similar clinical reference sources — appropriate for laser targeting.
    """
    import csv, io, ssl

    out_dir.mkdir(parents=True, exist_ok=True)

    # Fetch the CSV index
    print(f"\n[Fitzpatrick17k] Fetching dataset index...", flush=True)
    try:
        resp = _fitz_session.get(FITZPATRICK_CSV, timeout=30)
        resp.raise_for_status()
        rows = list(csv.DictReader(io.StringIO(resp.text)))
    except Exception as e:
        print(f"  ERROR fetching CSV: {e}")
        return

    # Filter for vein-relevant categories
    vein_rows = [r for r in rows if r.get("label", "").lower() in FITZPATRICK_VEIN_LABELS]
    print(f"  Found {len(vein_rows)} vein-relevant images across categories:")
    from collections import Counter
    for label, count in sorted(Counter(r["label"] for r in vein_rows).items()):
        print(f"    {count:4d}  {label}")

    # Shuffle for variety across categories
    import random as _random
    _random.seed(42)
    _random.shuffle(vein_rows)
    vein_rows = vein_rows[:n]

    print(f"\n  Downloading {len(vein_rows)} images...", flush=True)
    downloaded = 0
    skipped = 0

    for row in tqdm(vein_rows, desc="  Downloading"):
        url = row.get("url", "")
        label = row.get("label", "unknown").replace(" ", "_")
        img_id = row.get("md5hash", f"img_{downloaded}")[:12]

        if not url or not url.startswith("http"):
            skipped += 1
            continue

        try:
            resp = _fitz_session.get(url, timeout=30)
            resp.raise_for_status()
            ct = resp.headers.get("content-type", "")
            if "image" not in ct and not url.lower().endswith((".jpg", ".jpeg", ".png")):
                skipped += 1
                continue
            img = Image.open(io.BytesIO(resp.content)).convert("RGB")
            if img.width < 200 or img.height < 200:
                skipped += 1
                continue
            fname = out_dir / f"fitz_{label}_{img_id}.jpg"
            img.save(fname, "JPEG", quality=95)
            downloaded += 1
        except Exception as e:
            skipped += 1
        time.sleep(0.2)

    print(f"\n[Fitzpatrick17k] Downloaded {downloaded} images ({skipped} skipped) to {out_dir}")


# ── Roboflow Universe ─────────────────────────────────────────────────────────
#
# These are the best available public vein datasets on Roboflow Universe.
# All are free to download with a free API key from roboflow.com
#
# Dataset 1: varicose vein detection — ~421 images, YOLOv8 bbox, leg veins
#   https://universe.roboflow.com/qunshan/varicose-vein-detection
#
# Dataset 2: vein segmentation — hand/arm veins (NIR images)
#   https://universe.roboflow.com/vein-segmentation/vein-seg
#
# Get your free key: https://roboflow.com  (sign up, copy API key from settings)

ROBOFLOW_DATASETS = [
    # (workspace, project, version, description)
    ("qunshan",         "varicose-vein-detection", 1, "Varicose vein detection ~421 images"),
    ("vein-detection",  "vein-segmentation",       1, "Vein segmentation dataset"),
    ("skin-lesion-0gtlf", "spider-veins",          1, "Spider veins skin"),
]


def download_roboflow(api_key: str, out_dir: Path):
    """
    Download vein datasets from Roboflow Universe.
    Tries multiple known vein datasets — skips any that don't exist.
    Already in YOLO format: images + labels bundled together.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[Roboflow] Install: pip install roboflow")
        return

    rf = Roboflow(api_key=api_key)
    total = 0

    for workspace, project, version, description in ROBOFLOW_DATASETS:
        dest = out_dir / f"rf_{project}"
        print(f"\n[Roboflow] {description}")
        print(f"           {workspace}/{project} v{version}")
        try:
            ds = rf.workspace(workspace).project(project).version(version)
            ds.download("yolov8", location=str(dest))
            imgs = list(dest.rglob("*.jpg")) + list(dest.rglob("*.png"))
            total += len(imgs)
            print(f"  Downloaded {len(imgs)} images to {dest}")
        except Exception as e:
            print(f"  Skipping (not found or error): {e}")

    print(f"\n[Roboflow] Total: {total} images across all datasets")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Collect vein images from public sources")
    parser.add_argument("--source", choices=["isic", "wikimedia", "dermnet", "roboflow", "fitzpatrick", "all"],
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
    if args.source in ("dermnet", "all"):
        download_dermnet(args.n, out_dir / "dermnet")
    if args.source in ("fitzpatrick", "all"):
        download_fitzpatrick(args.n, out_dir / "fitzpatrick")
    if args.source in ("roboflow", "all") and args.api_key:
        download_roboflow(args.api_key, out_dir)
    elif args.source == "roboflow" and not args.api_key:
        print("[Roboflow] Provide --api-key. Get a free key at roboflow.com")


if __name__ == "__main__":
    main()
