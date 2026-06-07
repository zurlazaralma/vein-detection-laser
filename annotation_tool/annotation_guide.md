# Vein Dataset Annotation Guide

## Tool: Label Studio (recommended)

Label Studio provides a browser-based UI for polygon segmentation — ideal for fine vessel tracing.

### Setup
```bash
pip install label-studio
label-studio start
```
Open http://localhost:8080, create a project, and use this label config:

```xml
<View>
  <Image name="image" value="$image"/>
  <PolygonLabels name="label" toName="image" strokeWidth="2">
    <Label value="spider_vein" background="#FF0000"/>
    <Label value="facial_vein" background="#FF6400"/>
  </PolygonLabels>
</View>
```

After annotating, export as **YOLO** format.

---

## Alternative: LabelImg (lightweight)

```bash
pip install labelImg
labelImg
```

Set save format to **YOLO Seg** mode. Label map:
```
spider_vein
facial_vein
```

---

## YOLO Segmentation Label Format

Each image `dataset/images/train/img001.jpg` needs a matching label file  
`dataset/labels/train/img001.txt` with one line per annotated vessel region:

```
<class_id> <x1> <y1> <x2> <y2> ... <xN> <yN>
```

- `class_id`: `0` = spider_vein, `1` = facial_vein
- Coordinates are **normalized** (0.0–1.0) relative to image width/height
- Polygon vertices go clockwise, minimum 3 points
- Multiple vessels in one image → multiple lines

**Example (two vessels in one 640×640 image):**
```
0 0.32 0.45 0.35 0.47 0.38 0.44 0.36 0.42
1 0.61 0.22 0.65 0.25 0.68 0.23 0.64 0.20
```

---

## Annotation Guidelines for Veins

### What to annotate
- Trace the **visible vessel lumen** (the colored/visible portion of the vein)
- For spider vein networks: annotate **each branch** as a separate polygon
- For diffuse redness without clear vessels: annotate the reddened region as `facial_vein`

### Polygon resolution
- Use **8–20 points** per vessel segment — enough to capture the shape without over-tracing
- For thin linear vessels, a 4-point narrow polygon is fine (2 points each side)

### What NOT to annotate
- Normal skin pigmentation or moles
- Hair strands crossing the skin
- Image artifacts, reflections, or blur zones

### Quality checks
- Every vessel visible to the naked eye should be marked
- Minimum vessel width to annotate: ~3px in the 640×640 image
- When in doubt about class (spider vs facial), use the location:
  - Face/neck/décolletage → `facial_vein`
  - Legs/ankles → `spider_vein`

---

## Recommended Image Sources for Manual Collection

| Source | URL | Notes |
|--------|-----|-------|
| DermNet NZ | dermnet.nz | Free educational use; search "spider veins", "rosacea" |
| ISIC Archive | isic-archive.com | Dermoscopy; run `collect_images.py --source isic` |
| Wikimedia Commons | commons.wikimedia.org | Free license; run `collect_images.py --source wikimedia` |
| Roboflow Universe | universe.roboflow.com | Search "vein segmentation" — some pre-annotated sets |
| Google Images | images.google.com | Manual download; check license filter "Creative Commons" |

---

## Dataset Split After Annotation

```bash
python scripts/prepare_dataset.py --raw dataset/raw_downloads --split 70 20 10
```

This resizes all images to 640×640 (letterboxed) and splits into train/val/test.
