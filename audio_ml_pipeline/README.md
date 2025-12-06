# Audio ML Pipeline

Depression severity prediction from conversational audio using acoustic features.

---

## Overview

This pipeline extracts acoustic features from conversation audio and predicts PHQ-8 depression scores. It was designed to complement the text-based model for multimodal fusion.

### Key Results

| Model | Test MAE | Test RMSE | R² |
|-------|----------|-----------|-----|
| **Lasso (K=55)** | **4.71** | **5.78** | **0.18** |
| XGBoost | 4.93 | 5.94 | 0.14 |
| LightGBM | 5.13 | 6.20 | 0.06 |

---

## Feature Extraction Pipeline

```
Audio → Preprocess → Feature Extraction → Pooling → Model
```

### 1. Preprocessing

| Step | Details |
|------|---------|
| Sample Rate | 16 kHz (mono) |
| Normalization | Amplitude to [-1, 1] |
| High-pass Filter | 300 Hz Butterworth (order 5) |

### 2. Feature Extraction

Two parallel feature streams are extracted:

#### Wav2Vec2 Embeddings

| Property | Value |
|----------|-------|
| Model | `superb/wav2vec2-base-superb-er` |
| Pretrained On | Speech emotion recognition |
| Frame Dim | 768 |
| Processing | 20s chunks, frame subsampling (×2) |

#### Prosodic Features

| Feature | Dimensions |
|---------|------------|
| F0 (Pitch via YIN) | 1 |
| RMS Energy | 1 |
| Zero-Crossing Rate | 1 |
| Spectral Centroid | 1 |
| Spectral Bandwidth | 1 |
| Spectral Rolloff | 1 |
| Spectral Contrast | 7 |
| **Total per frame** | **13** |

### 3. Dimensionality Reduction

| Step | Input → Output |
|------|----------------|
| PCA (Wav2Vec2) | 768 → 200 dims (96.3% variance) |

### 4. Segment Pooling

Audio is split into 3 temporal segments (Beginning, Middle, End), with 4 statistics computed per segment:

| Statistic | Description |
|-----------|-------------|
| Mean | Average value |
| Std | Standard deviation |
| Min | Minimum value |
| Max | Maximum value |

#### Final Feature Dimensions

| Feature Set | Calculation | Dimensions |
|-------------|-------------|------------|
| Wav2Vec2 (PCA) | 200 × 3 segments × 4 stats | 2,400 |
| Prosody | 13 × 3 segments × 4 stats | 156 |
| **Total** | | **2,556** |

---

## Best Model: Lasso Regression

After feature selection, the Lasso model uses 55 features:

| Configuration | Value |
|---------------|-------|
| Feature Selection | SelectKBest (f_regression) |
| Selected Features | K = 55 |
| Regularization (α) | 0.376 |
| Non-zero Coefficients | 11 |

### Performance

| Split | MAE | RMSE | R² |
|-------|-----|------|-----|
| Cross-Val (5-fold) | 4.27 | 5.20 | — |
| **Test Set** | **4.71** | **5.78** | **0.18** |

---

## Experiment Summary

### Phase 1: Baseline Models

| Model | Features | Test MAE |
|-------|----------|----------|
| LightGBM | Wav2Vec2 + Prosody (raw) | 5.36 |
| XGBoost | Wav2Vec2 + Prosody (raw) | 5.38 |

### Phase 2: Feature Engineering

| Version | Improvement | Test MAE |
|---------|-------------|----------|
| V3 - Prosody Only | Baseline | 5.26 |
| V4 - Segment Pooling | +3 segments | 4.93 |
| V5 - PCA | +dim reduction | 5.19 |
| V6 - PCA + Pooling | Combined | 5.20 |
| V7 - Linear Models | Regularization | 4.77 |
| **V8 - Lasso (K=55)** | **Best** | **4.71** |

### Linear Models Comparison (V7)

| Model | K | Test MAE |
|-------|---|----------|
| Lasso | 55 | 4.71 |
| Lasso | 100 | 4.77 |
| Ridge | 100 | 4.86 |
| ElasticNet | No FS | 5.05 |

---

## Multimodal Fusion

Combining audio and text models improves performance:

| Strategy | MAE | Improvement |
|----------|-----|-------------|
| Audio Only (Lasso) | 4.71 | — |
| Text Only (RoBERTa) | 4.73 | — |
| **Min Fusion** | **4.26** | **+9.5%** |
| Weighted Average | 4.34 | +7.9% |

---

## Directory Structure

```
audio_ml_pipeline/
├── src/
│   ├── features/           # Feature extractors
│   │   ├── wav2vec2_extractor.py
│   │   └── prosody_extractor.py
│   ├── preprocessing/      # Audio preprocessing
│   │   ├── filtering.py    # 300Hz high-pass
│   │   └── pipeline.py
│   ├── training/           # Model training scripts
│   ├── evaluation/         # Evaluation scripts
│   └── utils/              # Pooling, logging
├── models/                 # Saved models
│   └── lasso_final_v8/
│       ├── lasso_model.joblib
│       ├── scaler.joblib
│       └── selector.joblib
├── reports/                # Experiment results
└── data/                   # Feature matrices
```

---

## Usage

### Inference

```python
from audio_inference_service import AudioInferenceService

# Load models once
service = AudioInferenceService()
service.load_models()

# Predict
score = service.predict("path/to/audio.wav")
print(f"PHQ Score: {score:.2f}")
```

### Training

```bash
# Extract features
python -m src.features.build_tabular_dataset

# Train model
python -m src.training.train_lasso_final
```

---

## Conclusion

- **Best single-modality model**: Lasso (K=55) with MAE = 4.71
- **Feature engineering key insight**: Segment pooling (Begin/Mid/End) captures temporal patterns
- **Linear models outperform** tree-based models on this dataset
- **Multimodal fusion** with min strategy achieves 9.5% improvement over best single modality
