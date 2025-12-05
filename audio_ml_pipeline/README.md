# Audio ML Pipeline

End-to-end machine learning pipeline for depression detection (PHQ-8 scores) from audio features extracted from the DAIC-WOZ dataset.

## Project Structure

```
audio_ml_pipeline/
├── datasets/               # Generated train/dev/test CSV files
├── models/                 # Saved model artifacts (.joblib)
├── reports/                # Training reports and plots
│   ├── lgbm/              # LightGBM results
│   └── xgb/               # XGBoost results
└── src/
    ├── config/            # Path configurations
    ├── datautils/         # Dataset loading utilities
    ├── evaluation/        # Metrics and visualization
    ├── features/          # Feature extraction modules
    ├── labels/            # PHQ label loading
    ├── modeling/          # Model wrappers (LightGBM, XGBoost)
    ├── preprocessing/     # Audio preprocessing (diarization, filtering)
    ├── scripts/           # Pipeline scripts
    ├── tuning/            # Hyperparameter tuning (Optuna)
    ├── training/          # Training pipeline
    └── utils/             # Legacy config (audio params)
```

## Quick Start

```bash
cd audio_ml_pipeline
source ../.newvenv/bin/activate
```

## Pipeline Steps

### Step 1: Preprocess Audio
Extract participant-only audio from raw DAIC-WOZ recordings and apply high-pass filtering:
```bash
python -m src.preprocessing.pipeline
```

### Step 2: Extract Features
Extract audio features (MFCC, chroma, spectral, RMS, ZCR, tonnetz) and compute summary statistics:
```bash
python -m src.features.run_feature_extraction
```

### Step 3: Build Dataset
Combine feature summaries into train/dev/test CSV files:
```bash
python -m src.scripts.build_dataset
```

### Step 4: Train Models
Run hyperparameter tuning and train LightGBM/XGBoost models:
```bash
python -m src.train
```

## Dependencies

```bash
pip install pandas numpy librosa scipy scikit-learn lightgbm xgboost optuna matplotlib
```

## Output

After training, results are saved to:
- `models/` - Trained model artifacts (`.joblib`)
- `reports/lgbm/` - LightGBM metrics, plots, and best parameters
- `reports/xgb/` - XGBoost metrics, plots, and best parameters

## Troubleshooting

**`ModuleNotFoundError: No module named 'src'`**  
Run scripts as modules with `-m` flag from the `audio_ml_pipeline` directory.

**`ValueError: No summary files found`**  
Run preprocessing and feature extraction first (Steps 1-3).
