# EcoWatch — Litter & Illegal Dumping Detection App

A Streamlit web application that runs **YOLOv8** object detection to identify
litter, garbage, and illegal dumping in images, videos, and live webcam feeds.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Project Structure](#project-structure)
4. [Requirements](#requirements)
5. [Installation](#installation)
6. [Running the App](#running-the-app)
7. [Usage Guide](#usage-guide)
8. [Configuration](#configuration)
9. [Training a Custom Model](#training-a-custom-model)
10. [Module Reference](#module-reference)
11. [Contributing](#contributing)

---

## Overview

EcoWatch uses the **Ultralytics YOLOv8** framework to perform real-time object
detection for environmental monitoring tasks. Out of the box it loads
`yolov8m.pt` (the medium-size COCO-pretrained model), but the intended
use-case is to drop in a **custom-trained model** fine-tuned on aerial imagery
of litter and dumping sites.

Key capabilities:

| Feature                | Description                                                                                                                         |
| ---------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| Image inference        | Upload a single image; get annotated bounding boxes, detection stats table, bar chart, and download buttons for the image and a CSV |
| Video inference        | Upload a video file; processed frame-by-frame with live preview and progress bar                                                    |
| Webcam inference       | Stream from the host machine's default camera (index 0)                                                                             |
| Dynamic model selector | All `.pt` files in `weights/` appear in the sidebar — switch models without editing code                                            |
| Confidence control     | Sidebar slider to tune the detection confidence threshold (30 – 100 %)                                                              |

---

## Architecture

```
┌─────────────────────────────────────────┐
│              Streamlit UI               │
│  ┌─────────────┐   ┌─────────────────┐  │
│  │   Sidebar   │   │   Main Canvas   │  │
│  │  - Task     │   │  - Images       │  │
│  │  - Conf.    │   │  - Video frames │  │
│  │  - Source   │   │  - Count text   │  │
│  └──────┬──────┘   └────────▲────────┘  │
│         │ select             │ render    │
└─────────┼────────────────────┼──────────┘
          │                    │
     ┌────▼────────────────────┴────┐
     │         utils.py             │
     │  load_model()               │
     │  infer_uploaded_image()      │
     │  infer_uploaded_video()      │
     │  infer_uploaded_webcam()     │
     │  _display_detected_frames()  │
     └────────────┬─────────────────┘
                  │
     ┌────────────▼─────────────────┐
     │       YOLOv8 (ultralytics)   │
     │   weights/yolov8m.pt         │
     └──────────────────────────────┘
```

---

## Project Structure

```
litter-and-dumping-detection-app/
├── app.py           # Streamlit entry point — UI layout and routing
├── config.py        # Centralised configuration (paths, sources, counters)
├── utils.py         # Inference helpers (model loading + three input modes)
├── requirements.txt # Python dependencies (ultralytics, opencv-python, pandas)
├── packages.txt     # System-level packages (for Streamlit Cloud deployment)
├── .gitignore       # Ignores weights, datasets, run artefacts, venvs, etc.
├── README.md        # This file
└── weights/
    └── yolov8m.pt   # YOLOv8-medium pretrained weights (replace with custom)
```

---

## Requirements

### Python packages

| Package         | Purpose                                 |
| --------------- | --------------------------------------- | --- | -------- | ----------------------------------------- |
| `streamlit`     | Web UI framework                        |
| `ultralytics`   | YOLOv8 training & inference             |
| `opencv-python` | Video/webcam capture and frame decoding |
| `Pillow`        | Image loading for uploaded files        |     | `pandas` | Detection statistics table and CSV export |

### System packages (Linux / Streamlit Cloud)

```
libgl1-mesa-glx   # OpenCV dependency (GUI rendering)
libglib2.0-0      # GLib runtime required by OpenCV
```

### Hardware

- **CPU** — supported; inference will be slower.
- **NVIDIA GPU** — recommended; CUDA is detected automatically by PyTorch.
- **AMD GPU (ROCm)** — supported via the `HSA_OVERRIDE_GFX_VERSION` env-var set
  in `app.py` for RDNA2 cards. Adjust or remove if using a different GPU.

---

## Installation

```bash
# 1. Clone the repository
git clone https://github.com/yaminsadik/litter-and-dumping-detection-app
cd litter-and-dumping-detection-app

# 2. (Recommended) Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

# 3. Install Python dependencies
pip install -r requirements.txt

# 4. (Linux only) Install system packages if OpenCV fails to import
sudo apt-get install -y libgl1-mesa-glx libglib2.0-0
```

---

## Running the App

```bash
streamlit run app.py
```

Streamlit will print a local URL (typically `http://localhost:8501`) and open
it automatically in your default browser.

---

## Usage Guide

### 1. Image Detection

1. In the sidebar set **Select Source** → `Image`.
2. (Optional) Choose which model to run from **Select Model Weights**.
3. Click **Browse files** and upload a `.jpg`, `.jpeg`, `.png`, `.bmp`, or
   `.webp` image.
4. The original image appears on the left.
5. Click **Execution** — the annotated result appears on the right.
6. Use the **Download Annotated Image** button to save the result as a PNG.
7. Expand **Detection Results** to see a table of class names, confidence
   scores, and bounding-box coordinates, a bar chart of detections by class,
   and a **Download Detections CSV** button.

### 2. Video Detection

1. Set **Select Source** → `Video`.
2. Upload any common video format (`.mp4`, `.avi`, …).
3. A preview of the video is shown.
4. Click **Execution** — frames are processed sequentially, displayed live,
   and a **progress bar** tracks the current frame against the total.

### 3. Webcam Detection

1. Set **Select Source** → `Webcam`.
2. The feed starts immediately using device index 0.
3. Click **Stop running** to end the session.

> **Note:** Webcam mode requires the app to run locally. It is not available
> on Streamlit Cloud or other hosted environments.

### Dynamic Model Selector

Place any YOLOv8 `.pt` weights file in the `weights/` directory. The
**Select Model Weights** sidebar dropdown is populated automatically at
startup — no code changes needed.

```bash
# Example: add a custom-trained model
cp runs/train/litter_detector/weights/best.pt weights/litter_v1.pt
# Then restart the app and pick "litter_v1.pt" from the sidebar.
```

### Confidence Threshold

Use the **Select Model Confidence** slider (30 – 100 %) to control detection
sensitivity. Lower values detect more objects but increase false positives;
higher values are more conservative.

---

## Configuration

All tunable parameters live in `config.py`:

| Variable              | Default                        | Description                             |
| --------------------- | ------------------------------ | --------------------------------------- |
| `SOURCES_LIST`        | `["Image", "Video", "Webcam"]` | Ordered input source options            |
| `WEIGHTS_DIR`         | `weights/`                     | Directory scanned for `.pt` model files |
| `DETECTION_MODEL_DIR` | `weights/yolov8m.pt`           | Default model weights file              |
| `OBJECT_COUNTER`      | `None`                         | Global counter for exiting objects      |
| `OBJECT_COUNTER1`     | `None`                         | Global counter for entering objects     |

To switch to a custom-trained model, either:

- **No-code approach:** drop a `.pt` file into `weights/` and select it from
  the sidebar dropdown.
- **Hard-code approach:** update `DETECTION_MODEL_DIR` in `config.py`:

```python
# config.py
DETECTION_MODEL_DIR = WEIGHTS_DIR / 'my_litter_model.pt'
```

---

## Training a Custom Model

The app is built around **inference** with a YOLOv8 model, but training is
done separately using the Ultralytics CLI or Python API.

### Quick-start training example

```python
from ultralytics import YOLO

# Load a pretrained model to fine-tune
model = YOLO("yolov8m.pt")

# Train on a custom dataset (YOLO format)
results = model.train(
    data="dataset.yaml",   # path to your dataset config
    epochs=100,
    imgsz=640,
    batch=16,
    project="runs/train",
    name="litter_detector"
)
```

### Dataset YAML structure (`dataset.yaml`)

```yaml
path: /path/to/dataset
train: images/train
val: images/val
test: images/test # optional

nc: 3 # number of classes
names:
  0: litter
  1: illegal_dumping
  2: garbage_bag
```

### After training

Copy the best weights to the `weights/` directory:

```bash
cp runs/train/litter_detector/weights/best.pt weights/litter_v1.pt
```

Restart the app and select `litter_v1.pt` from the **Select Model Weights**
dropdown — no code changes required.

---

## Module Reference

### `app.py`

Entry point for the Streamlit application.

- Configures the page layout and sidebar.
- Loads the YOLOv8 model via `utils.load_model`.
- Routes user interaction to the appropriate inference function based on the
  selected input source.

### `config.py`

Centralised settings module.

- Adds the project root to `sys.path` for reliable imports.
- Exposes `SOURCES_LIST`, `DETECTION_MODEL_DIR`, `OBJECT_COUNTER`, and
  `OBJECT_COUNTER1` as module-level variables consumed by `app.py` and
  `utils.py`.

### `utils.py`

Inference utility functions.

| Function                                                           | Description                                                           |
| ------------------------------------------------------------------ | --------------------------------------------------------------------- |
| `load_model(model_path)`                                           | Load and Streamlit-cache a YOLOv8 model                               |
| `_display_detected_frames(conf, model, st_count, st_frame, image)` | Annotate and render a single frame                                    |
| `infer_uploaded_image(conf, model)`                                | Image upload + inference + stats table + bar chart + download buttons |
| `infer_uploaded_video(conf, model)`                                | Video upload + frame-by-frame inference + progress bar                |
| `infer_uploaded_webcam(conf, model)`                               | Live webcam inference                                                 |

---

## Contributing

Contributions are welcome! Please:

1. Fork the repository and create a feature branch.
2. Make your changes with clear commit messages.
3. Open a Pull Request describing what was changed and why.
4. For bug reports, open an Issue with reproduction steps and your environment
   details (OS, Python version, GPU).
