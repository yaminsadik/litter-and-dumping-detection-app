"""
app.py - EcoWatch: Litter and Illegal Dumping Detection App
===========================================================

This is the main entry point for the EcoWatch Streamlit application.
It provides an interactive web UI for running YOLOv8-based object detection
to identify litter, garbage, and illegal dumping in:
  - Uploaded images
  - Uploaded videos
  - Live webcam feed

The app loads a pre-trained (or custom-trained) YOLOv8 model from the path
specified in config.py and exposes model confidence tuning via the sidebar.

Features
--------
- Dynamic model selector: all .pt files in weights/ appear in the sidebar.
- Confidence threshold slider (30 – 100 %).
- Image inference with annotated result, detection stats table, bar chart,
  and one-click download for the annotated image and a detections CSV.
- Video inference with live frame-by-frame preview and a progress bar.
- Live webcam inference with a stop button.

Usage:
    streamlit run app.py

Dependencies:
    - streamlit
    - ultralytics (YOLOv8)
    - opencv-python
    - Pillow
    - pandas
"""

from os import putenv

# AMD GPU compatibility fix: override the GFX version for ROCm-based GPUs
# (e.g., RX 6700 XT / RDNA2). Safe to remove on NVIDIA or CPU-only systems.
putenv("HSA_OVERRIDE_GFX_VERSION", "10.3.0")


from pathlib import Path
import streamlit as st

import config
from utils import load_model, infer_uploaded_image, infer_uploaded_video, infer_uploaded_webcam



# ── Page Layout ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="EcoWatch",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
    )

# Main page heading
st.title("Welcome to EcoWatch")


# ── Sidebar: Model Configuration ─────────────────────────────────────────────
st.sidebar.header("DL Model Config")

# Task selector (currently only Detection is implemented; placeholder for
# future tasks such as Segmentation or Classification)
task_type = st.sidebar.selectbox(
    "Select Task",
    ["Detection"]
)

# Confidence threshold: converted from integer percentage to float (0.0 – 1.0)
confidence = float(st.sidebar.slider(
    "Select Model Confidence", 30, 100, 50)) / 100


# ── Model Loading ─────────────────────────────────────────────────────────────
# Scan the weights/ directory and list every .pt file so the user can switch
# between the stock model and any custom-trained variants without editing code.
weights_dir = Path(config.WEIGHTS_DIR)
available_models = sorted([p.name for p in weights_dir.glob("*.pt")])

if not available_models:
    st.error(
        f"No `.pt` model files found in `{weights_dir}/`.  "
        "Please add a YOLOv8 weights file and restart."
    )
    st.stop()

default_idx = (
    available_models.index("yolov8m.pt")
    if "yolov8m.pt" in available_models
    else 0
)
selected_model = st.sidebar.selectbox(
    "Select Model Weights",
    available_models,
    index=default_idx,
    help="All .pt files found in the weights/ directory are listed here.",
)
model_path = weights_dir / selected_model

try:
    model = load_model(model_path)
except Exception as e:
    st.error(f"Unable to load model '{selected_model}': {e}")
    st.stop()

# ── Sidebar: Input Source Configuration ───────────────────────────────────────
st.sidebar.header("Image/Video Config")
source_selectbox = st.sidebar.selectbox(
    "Select Source",
    config.SOURCES_LIST
)

# Route to the appropriate inference function based on the selected input source
source_img = None
if source_selectbox == config.SOURCES_LIST[0]:   # Image upload
    infer_uploaded_image(confidence, model)
elif source_selectbox == config.SOURCES_LIST[1]: # Video upload
    infer_uploaded_video(confidence, model)
elif source_selectbox == config.SOURCES_LIST[2]: # Live webcam
    infer_uploaded_webcam(confidence, model)
else:
    st.error("Currently only 'Image' and 'Video' source are implemented")

# ── Sidebar: About ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown(
    """
    **EcoWatch** uses [YOLOv8](https://github.com/ultralytics/ultralytics)
    to detect litter and illegal dumping in images, videos, and live
    webcam feeds.

    Place custom-trained `.pt` weights in the `weights/` folder to switch
    models without changing any code.

    [GitHub](https://github.com/yaminsadik/litter-and-dumping-detection-app)
    """
)