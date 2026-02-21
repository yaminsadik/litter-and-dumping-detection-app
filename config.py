"""
config.py - Application Configuration
======================================

Centralised configuration module for the EcoWatch litter and illegal dumping
detection application.  All file paths, model locations, and global state
variables are defined here so that other modules can import them from a single
source of truth.

Variables
---------
ROOT : Path
    Project root directory relative to the current working directory.
SOURCES_LIST : list[str]
    Ordered list of input source options shown in the Streamlit sidebar.
DETECTION_MODEL_DIR : Path
    Path to the YOLOv8 model weights file used for object detection.
OBJECT_COUNTER : dict | None
    Global counter tracking objects leaving a monitored zone ("out" direction).
    Updated by the vehicle counting logic in utils.py.
OBJECT_COUNTER1 : dict | None
    Global counter tracking objects entering a monitored zone ("in" direction).
    Updated by the vehicle counting logic in utils.py.
"""

from pathlib import Path
import sys

# ── Path Setup ────────────────────────────────────────────────────────────────
# Resolve the absolute path of this config file
file_path = Path(__file__).resolve()

# The project root is the directory that contains this file
root_path = file_path.parent

# Ensure the project root is on sys.path so all internal modules can be
# imported regardless of where the app is launched from
if root_path not in sys.path:
    sys.path.append(str(root_path))

# Relative path from the current working directory – used for constructing
# model and asset paths that are portable across environments
ROOT = root_path.relative_to(Path.cwd())

# ── Input Sources ────────────────────────────────────────────────────────────
# Ordered list of inference input modes presented in the Streamlit sidebar.
# Index 0 → Image upload, Index 1 → Video upload, Index 2 → Live webcam
SOURCES_LIST = ["Image", "Video", "Webcam"]

# ── Model Paths ─────────────────────────────────────────────────────────────
# Directory that holds all YOLOv8 .pt weight files.
# The app scans this folder at startup and lists every .pt file in the
# "Select Model Weights" sidebar dropdown.
WEIGHTS_DIR = ROOT / 'weights'

# Default model weights file (YOLOv8-medium, COCO pretrained).
# Replace with a custom-trained .pt file fine-tuned on litter / dumping data.
DETECTION_MODEL_DIR = WEIGHTS_DIR / "yolov8m.pt"

# ── Global Object Counters ──────────────────────────────────────────────────────
# These are intentionally module-level mutable variables so that
# _display_detected_frames() in utils.py can update them across Streamlit
# reruns without requiring session_state persistence.

# Counts of objects exiting the monitored region (populated during video/webcam inference)
OBJECT_COUNTER = None

# Counts of objects entering the monitored region (populated during video/webcam inference)
OBJECT_COUNTER1 = None
