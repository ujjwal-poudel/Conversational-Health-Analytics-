import os

# Base directory of the project
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))

# Dataset directory
DATASET_DIR = os.path.join(BASE_DIR, "audio_ml_pipeline", "datasets")

# Reports directory
REPORTS_DIR = os.path.join(BASE_DIR, "audio_ml_pipeline", "reports")

# Ensure reports subfolders exist for both models
LGBM_REPORT_DIR = os.path.join(REPORTS_DIR, "lgbm")
XGB_REPORT_DIR = os.path.join(REPORTS_DIR, "xgb")

os.makedirs(LGBM_REPORT_DIR, exist_ok=True)
os.makedirs(XGB_REPORT_DIR, exist_ok=True)