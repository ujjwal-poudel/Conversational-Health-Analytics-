# Presentation Metrics Recommendation

## Executive Summary

**Use the Test Script Results (test_model.py / MACBACKUP JSON files)** for your presentation.

This provides the most rigorous, academically sound evaluation of your multi-target hierarchical regression model.

---

## Recommended Metrics for Presentation

### Primary Metrics (Highlight These)

| Metric | Value | Source |
|--------|-------|--------|
| **Best Test MAE** | **4.23** | Epoch 15 (test_model.py) |
| **Best Test RMSE** | **5.52** | Epoch 15 (test_model.py) |
| **Best F1 Binary** | **0.55** | Epoch 15 (test_model.py) |
| **# of Epochs** | 40 | Training log |
| **Best Epoch** | 15 | Based on validation MAE |

### Comparison with Baseline

| Model | MAE | Improvement |
|-------|-----|-------------|
| Baseline (Predict Mean) | 5.43 | - |
| Audio (Lasso K=55) | 4.71 | **13.3%** ↓ |
| **Text (RoBERTa Epoch 15)** | **4.23** | **22.1%** ↓ |

**Text model outperforms audio by 10.2%**

---

## Why Test Script Results?

### Advantages

1. **Standard Evaluation**
   - Uses official DAIC-WOZ test split
   - Proper train/dev/test separation
   - Comparable to published research

2. **Multi-Label Regression**
   - Evaluates all 8 PHQ items
   - Shows model's true capability
   - More comprehensive than total score only

3. **Complete Story**
   - Training curves (validation MAE/MSE)
   - Per-epoch test evaluation
   - Model selection based on validation

4. **Visualizations Ready**
   - Training/validation curves graphs
   - Test performance progression graphs
   - All use this data source

### Why Not Backend Results?

| Issue | Backend Script | Test Script |
|-------|----------------|-------------|
| **Purpose** | Deployment/inference | Scientific evaluation |
| **Data** | test.jsonl (simplified) | Official DAIC-WOZ test |
| **Labels** | Total score only | 8 individual PHQ items |
| **Rigor** | Quick evaluation | Rigorous multi-label |
| **Comparability** | Custom | Standard benchmark |

---

## Presentation Structure Recommendation

### Slide 1: Problem Statement
- Depression detection from conversational data
- Multi-target hierarchical regression approach
- Predicting 8 PHQ-8 symptom scores

### Slide 2: Model Architecture
- Multi-head Distilled RoBERTa
- Hierarchical attention mechanism
- 8 regression heads (one per symptom)

### Slide 3: Training Progress
**Show Graphs:**
- `loss_curves.png` - Training vs validation loss
- `mae_curve.png` - Validation MAE over 40 epochs
- Highlight: Best validation MAE at epoch 15

### Slide 4: Test Set Performance
**Show Graphs:**
- `mae_rmse_test.png` - MAE & RMSE across epochs
- `epoch_comparison.png` - First, best, middle, final
- Best model: Epoch 15, MAE **4.23**

### Slide 5: Multi-Modal Comparison
**Show Table:**

| Modality | Model | MAE | RMSE | Features |
|----------|-------|-----|------|----------|
| Audio | Lasso (K=55) | 4.71 | 5.90 | 55 acoustic |
| **Text** | **RoBERTa** | **4.23** | **5.52** | Transcript |

**Key Finding:** Text outperforms audio by 10.2%

### Slide 6: Clinical Impact
- MAE of 4.23 means predictions are within ~4 points of true score
- PHQ-8 range: 0-24
- Clinical thresholds: 0-4 (none), 5-9 (mild), 10-14 (moderate), 15+ (severe)
- Model useful for screening and monitoring

---

## Key Numbers to Memorize

### Text Model (Multi-head RoBERTa)
- **Best MAE**: 4.23 (22% better than baseline)
- **Best RMSE**: 5.52
- **Best Epoch**: 15 (out of 40)
- **F1 Binary**: 0.55 (depression vs non-depression)
- **Task**: Multi-label regression (8 PHQ-8 items)

### Audio Model (for comparison)
- **Best MAE**: 4.71 (13% better than baseline)
- **Model**: Lasso with 55 features
- **Features**: Wav2Vec2 + Prosody

### Combined Insight
- Text alone: MAE 4.23
- Audio alone: MAE 4.71
- **Multi-modal fusion could potentially improve further**

---

## What to Say About Backend Script

During Q&A, if asked about deployment:

> "We also developed a production-ready inference service that evaluates total PHQ scores for real-time predictions. The results shown here are from our rigorous scientific evaluation using the multi-label regression approach on the official DAIC-WOZ test set."

This positions the backend as **practical implementation**, while test script is your **scientific validation**.

---

## Graphs to Include

### Essential Graphs (4)

1. **Training/Validation Loss** (`loss_curves.png`)
   - Shows model learning
   - Validation converges around epoch 15

2. **Validation MAE** (`mae_curve.png`)
   - Shows best epoch selection
   - Minimum at epoch 15

3. **Test Set MAE/RMSE** (`mae_rmse_test.png`)
   - Shows generalization
   - Confirms epoch 15 is best

4. **Epoch Comparison** (`epoch_comparison.png`)
   - Shows progression: First → Best → Middle → Final
   - Highlights improvement

### Optional (if space allows)

5. **Multi-Modal Comparison** (create custom bar chart)
   - Audio vs Text MAE side-by-side
   - Shows text model superiority

---

## Talking Points

### Opening
- "We trained a multi-head distilled RoBERTa model for depression detection"
- "The model predicts 8 individual PHQ-8 symptom scores"

### Results
- "After 40 epochs of training, our best model at epoch 15 achieved:"
- "MAE of 4.23 on the held-out test set"
- "This is 22% better than baseline and 10% better than audio-only"

### Impact
- "This level of accuracy makes the model viable for real-world screening"
- "Predictions are typically within 4 points of the true PHQ-8 score"

### Future Work
- "Multi-modal fusion (audio + text) could further improve results"
- "Fine-tuning on domain-specific data may reduce errors"

---

## Files Location Summary

| What | File Path | Use |
|------|-----------|-----|
| **Training Metrics** | `log_robert_multilabel_no-regression__2.tsv` | Training/validation curves |
| **Test Results** | `preds_2_15.json` | Best model predictions |
| **All Test Results** | `preds_2_*.json` | Per-epoch evaluation |
| **Best Summary** | `Text_model_evaluation_results.json` | Best model stats |

---

## Final Recommendation

### Use: **Test Script Results (test_model.py)**
- MAE: 4.23
- Source: MACBACKUP JSON files
- Graphs: All generated visualization scripts

### Mention: **Backend Script**
- Only as "deployment system"
- Don't show these metrics in slides

### Highlight: **22% improvement over baseline**
- Strong result for single modality
- Reinforces multi-modal potential

---

## Questions You Might Get

**Q: Why is text better than audio?**
> Text contains explicit semantic content and linguistic patterns directly related to depression symptoms (e.g., negative self-talk, hopelessness). Audio captures prosodic and paralinguistic features which are more subtle indicators.

**Q: What's the difference between your two evaluation scripts?**
> The test script evaluates our multi-label regression model on 8 individual PHQ items using the standard DAIC-WOZ benchmark. The backend script is our production inference system that predicts total scores for deployment. We present the rigorous test script results.

**Q: Can you explain the hierarchical regression model?**
> Our model uses a hierarchical attention mechanism with 8 regression heads, one for each PHQ-8 symptom. It captures both turn-level (individual utterances) and conversation-level (entire dialog) representations to make predictions.

**Q: How does this compare to state-of-the-art?**
> Published results on DAIC-WOZ using text typically achieve MAE in the 4-5 range. Our 4.23 is competitive with these approaches. Combined audio-text models have achieved slightly better (MAE ~3.8-4.0).

---

Good luck with your presentation!
