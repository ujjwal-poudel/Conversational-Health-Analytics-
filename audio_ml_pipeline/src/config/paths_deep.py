"""
Path configuration for deep learning features training pipeline.

Version 6: PCA (768→100) + Segment Pooling (3 segments) Hybrid.
Uses _pca_segment_v6 suffix for all output directories.
"""

import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Path to the .npy datasets created by build_tabular_dataset.py
from ..utils.config import MODEL_DIR as NPY_DATA_DIR

# Reports directory with v6 suffix (PCA + segment pooling experiment)
REPORTS_DIR_DEEP = os.path.join(BASE_DIR, "audio_ml_pipeline", "reports")
LGBM_REPORT_DIR_DEEP = os.path.join(REPORTS_DIR_DEEP, "lgbm_pca_segment_v6")
XGB_REPORT_DIR_DEEP = os.path.join(REPORTS_DIR_DEEP, "xgb_pca_segment_v6")

# Models directory with v6 suffix
MODELS_DIR_DEEP = os.path.join(BASE_DIR, "audio_ml_pipeline", "models", "pca_segment_v6")

# Create directories
os.makedirs(LGBM_REPORT_DIR_DEEP, exist_ok=True)
os.makedirs(XGB_REPORT_DIR_DEEP, exist_ok=True)
os.makedirs(MODELS_DIR_DEEP, exist_ok=True)