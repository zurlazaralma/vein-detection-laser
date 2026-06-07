# Label Studio ML Backend — Vein Auto-Annotation

## What it does
When you open an image in Label Studio, this backend automatically detects veins
and pre-fills polygon annotations. You just correct mistakes instead of drawing from scratch.

## Architecture
```
Label Studio (UI) <---> ml_backend.py (Flask) <---> SAM2 + Frangi vessel detection
```

## Setup

### 1. Install dependencies
```
pip install flask label-studio-ml opencv-python scikit-image numpy
```

### 2. (Optional) Download SAM2 checkpoint
For best results, download the SAM2 Small model:
  https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt

Save to: `models/sam2/sam2.1_hiera_small.pt`

Install SAM2:
  pip install git+https://github.com/facebookresearch/sam2.git

**Without SAM2**, the backend still works using contour-based fallback mode.

### 3. Start the backend
Double-click `start_backend.bat`  
or run: `python annotation_tool/ml_backend/ml_backend.py`

### 4. Connect to Label Studio
1. Start Label Studio: `label-studio start`
2. Open your project -> Settings -> Machine Learning
3. Click "Add Model"
4. URL: `http://localhost:9090`
5. Click "Validate and Save"

### 5. Annotate
- Open any image task
- Auto-annotations appear immediately as red/orange polygons
- Green polygon = spider_vein, Orange = facial_vein
- Accept, delete, or reshape them as needed
- Submit when done

## Environment variables
| Variable | Default | Description |
|----------|---------|-------------|
| SAM2_CHECKPOINT | models/sam2/sam2.1_hiera_small.pt | Path to SAM2 .pt file |
| DEVICE | auto | cuda / cpu / auto |
| CONF_THRESH | 0.35 | Vessel detection sensitivity (lower = more detections) |
| PORT | 9090 | Backend port |

## Health check
Open: http://localhost:9090/health
