# Test Metric Discrepancy Analysis

## Summary
The two evaluation scripts calculate **different metrics** on **different data**, which explains the discrepancy.

---

## Key Differences

### 1. **What They Predict**

| Script | Predicts | Data Type |
|--------|----------|-----------|
| **backend/evaluate_models.py** | **Total PHQ score** (single value) | 1D - sum of 8 items |
| **test_model.py** | **8 individual PHQ items** (8 values) | 2D - per-item predictions |

### 2. **MAE Calculation**

#### Backend Script (`evaluate_models.py`)
```python
# Line 164-169: Predicts TOTAL score directly
pred_binary = model.clf_binary(final_hidden_binary)
symptom_scores_tensor = pred_binary
total_score_tensor = torch.sum(symptom_scores_tensor, dim=1)
final_score = total_score_tensor.squeeze().cpu().item()

# Line 250: MAE on TOTAL scores
mae = mean_absolute_error(pred_tensor, true_labels_tensor).item()
```
- Uses `clf_binary` head to get 8 symptom scores
- **Sums them** to get total score
- Calculates MAE on the **total score**

#### Test Script (`test_model.py`)
```python
# Line 254: MAE on SUMMED predictions
mae_micro_sum = mean_absolute_error(torch.sum(preds, dim=1), sum_labels)
```
- Model predictions are already the 8-item array
- **Sums predictions** to get total
- Calculates MAE on the **summed total**

---

## Why Results Differ

### Issue 1: **Different Binary Thresholds**
- **Backend**: `pred_bin = (pred_tensor >= 10)` (line 256)
  - Uses threshold of **10**
- **Test script**: `pred_bin = (pred_total > 9)` (line 343, test_model.py)
  - Uses threshold of **9**

### Issue 2: **Different Label Loading**

#### Backend Script
```python
# Line 189: Loads TOTAL score only
all_true_scores.append(data['labels'][0])
```
- Takes `labels[0]` which is the **total PHQ score**
- 1D tensor of shape `[batch_size]`

#### Test Script
```python
# Line 182-183: Loads ALL 8 items
labels_tensor = torch.tensor(labels_list)
# Shape: [batch_size, 8]
```
- Loads the full 8-item PHQ labels
- 2D tensor of shape `[batch_size, 8]`

---

## The Core Problem

### Backend Script
1. Predicts 8 symptoms → sums them
2. Compares to **total score** from `labels[0]`
3. MAE = |sum(predicted_8_items) - total_score|

### Test Script  
1. Predicts 8 symptoms (multi-label regression)
2. Compares to **8 individual items**
3. Calculates per-item MAE, then sums predictions for total MAE
4. MAE = |sum(predicted_8_items) - sum(true_8_items)|

**These are mathematically identical IF:**
- `labels[0]` == `sum(true_8_items)`

But the **JSON data structure** might not guarantee this!

---

## Data Structure Difference

### test.jsonl (Backend)
```json
{
  "turns": ["...", "..."],
  "labels": [12]  // Just the TOTAL score
}
```

### Test CSV (test_model.py)
```csv
participant_id, score1, score2, ..., score8
303, 1, 1, 2, 1, 0, 1, 0, 0  // Individual items
```

---

## Summary

| Aspect | Backend Script | Test Script |
|--------|----------------|-------------|
| **Predictions** | Sum of 8 symptoms | 8 individual symptoms |
| **Labels** | Total score (`labels[0]`) | 8 item scores |
| **MAE** | On total vs total | On sum(preds) vs sum(labels) |
| **Binary threshold** | ≥ 10 | > 9 |
| **Data source** | test.jsonl | CSV file |

**The metrics differ because:**
1. Different data formats (JSONL vs CSV)
2. Different label structures (total vs per-item)
3. Different binary thresholds
4. Potentially different test sets or preprocessing

The **test_model.py** approach is more accurate for the multi-label regression task, as it evaluates per-item predictions.
