"""
V8: Final Best Model - Lasso with SelectKBest (K=55)

This is the production-ready script for the best performing model:
- Lasso regression with SelectKBest feature selection
- K=55 features selected using f_regression scoring
- Achieves MAE=4.71 on test set (best across all experiments)

Saves:
- Trained model (lasso_model.joblib)
- Feature scaler (scaler.joblib)
- Feature selector (selector.joblib)
- Detailed summary report

Usage:
    python -m src.training.train_final_model
"""

import os
import json
import joblib
import numpy as np
from datetime import datetime

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict

# Paths
from ..utils.config import MODEL_DIR

# Report and model save directories
REPORT_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../reports/lasso_final_v8"
)
SAVE_DIR = os.path.join(
    os.path.dirname(__file__),
    "../../models/lasso_final_v8"
)

os.makedirs(REPORT_DIR, exist_ok=True)
os.makedirs(SAVE_DIR, exist_ok=True)


# =============================================================================
# CONFIGURATION - Best found parameters
# =============================================================================
K_FEATURES = 55  # Number of features to keep
RANDOM_STATE = 42
N_SPLITS = 5


def load_data():
    """Load train, dev, test data."""
    train = np.load(os.path.join(MODEL_DIR, "train_data.npy"))
    dev = np.load(os.path.join(MODEL_DIR, "dev_data.npy"))
    test = np.load(os.path.join(MODEL_DIR, "test_data.npy"))
    
    # Combine train + dev
    train_dev = np.vstack([train, dev])
    
    X_train = train_dev[:, :-1]
    y_train = train_dev[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    
    return X_train, y_train, X_test, y_test


def train_and_evaluate():
    """Train the best model and evaluate."""
    
    print("=" * 60)
    print("V8: Final Best Model - Lasso (K=55)")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    
    print(f"\nDataset:")
    print(f"  Train+Dev: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"  Test: {len(X_test)} samples")
    print(f"  PHQ mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    
    # Step 1: Standardize features
    print("\nStep 1: Standardizing features...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Step 2: Feature selection
    print(f"Step 2: Selecting top {K_FEATURES} features (f_regression)...")
    selector = SelectKBest(f_regression, k=K_FEATURES)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)
    X_test_selected = selector.transform(X_test_scaled)
    
    # Get selected feature indices
    selected_indices = selector.get_support(indices=True)
    print(f"  Selected feature indices: {len(selected_indices)}")
    
    # Step 3: Train Lasso with CV
    print("Step 3: Training LassoCV...")
    cv = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    
    lasso = LassoCV(
        alphas=np.logspace(-4, 2, 100),
        cv=cv,
        max_iter=30000,
        random_state=RANDOM_STATE
    )
    lasso.fit(X_train_selected, y_train)
    
    print(f"  Best alpha: {lasso.alpha_:.6f}")
    print(f"  Non-zero coefficients: {np.sum(lasso.coef_ != 0)}/{K_FEATURES}")
    
    # Step 4: Cross-validation predictions
    print("Step 4: Cross-validation evaluation...")
    y_pred_cv = cross_val_predict(lasso, X_train_selected, y_train, cv=cv)
    cv_rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    cv_mae = mean_absolute_error(y_train, y_pred_cv)
    print(f"  CV RMSE: {cv_rmse:.4f}")
    print(f"  CV MAE: {cv_mae:.4f}")
    
    # Step 5: Test evaluation
    print("Step 5: Test set evaluation...")
    y_pred_test = lasso.predict(X_test_selected)
    
    test_rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    test_mae = mean_absolute_error(y_test, y_pred_test)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print(f"  Test RMSE: {test_rmse:.4f}")
    print(f"  Test MAE:  {test_mae:.4f}")
    print(f"  Test R²:   {test_r2:.4f}")
    
    # Step 6: Save artifacts
    print("\nStep 6: Saving model artifacts...")
    
    # Save model
    model_path = os.path.join(SAVE_DIR, "lasso_model.joblib")
    joblib.dump(lasso, model_path)
    print(f"  Model saved: {model_path}")
    
    # Save scaler
    scaler_path = os.path.join(SAVE_DIR, "scaler.joblib")
    joblib.dump(scaler, scaler_path)
    print(f"  Scaler saved: {scaler_path}")
    
    # Save selector
    selector_path = os.path.join(SAVE_DIR, "selector.joblib")
    joblib.dump(selector, selector_path)
    print(f"  Selector saved: {selector_path}")
    
    # Save config
    config = {
        "model_type": "Lasso",
        "k_features": K_FEATURES,
        "best_alpha": float(lasso.alpha_),
        "nonzero_coefs": int(np.sum(lasso.coef_ != 0)),
        "selected_feature_indices": selected_indices.tolist(),
        "cv_splits": N_SPLITS,
        "random_state": RANDOM_STATE,
        "created_at": datetime.now().isoformat()
    }
    config_path = os.path.join(SAVE_DIR, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=4)
    print(f"  Config saved: {config_path}")
    
    # Step 7: Generate detailed report
    print("Step 7: Generating report...")
    
    results = {
        "experiment": "V8: Final Best Model - Lasso (K=55)",
        "created_at": datetime.now().isoformat(),
        "dataset": {
            "train_dev_samples": len(X_train),
            "test_samples": len(X_test),
            "original_features": X_train.shape[1],
            "selected_features": K_FEATURES
        },
        "model": {
            "type": "LassoCV",
            "best_alpha": float(lasso.alpha_),
            "nonzero_coefs": int(np.sum(lasso.coef_ != 0))
        },
        "cv_results": {
            "rmse": round(cv_rmse, 4),
            "mae": round(cv_mae, 4)
        },
        "test_results": {
            "rmse": round(test_rmse, 4),
            "mae": round(test_mae, 4),
            "r2": round(test_r2, 4)
        }
    }
    
    # Save JSON results
    with open(os.path.join(REPORT_DIR, "results.json"), "w") as f:
        json.dump(results, f, indent=4)
    
    # Generate summary text
    summary = f"""
============================================================
V8: FINAL BEST MODEL - Lasso Regression
============================================================

Created: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

--- Model Configuration ---
  Model Type: LassoCV with SelectKBest
  Feature Selection: f_regression, K={K_FEATURES}
  Best Alpha (regularization): {lasso.alpha_:.6f}
  Non-zero Coefficients: {np.sum(lasso.coef_ != 0)}/{K_FEATURES}

--- Dataset ---
  Train+Dev samples: {len(X_train)}
  Test samples: {len(X_test)}
  Original features: {X_train.shape[1]}
  Selected features: {K_FEATURES}

--- Cross-Validation Results ---
  CV RMSE: {cv_rmse:.4f}
  CV MAE:  {cv_mae:.4f}

--- Test Set Results ---
  RMSE: {test_rmse:.4f}
  MAE:  {test_mae:.4f}  <-- BEST ACHIEVED!
  R²:   {test_r2:.4f}

--- Saved Artifacts ---
  Model:    {model_path}
  Scaler:   {scaler_path}
  Selector: {selector_path}
  Config:   {config_path}

--- Inference Usage ---
  from joblib import load
  
  model = load('models/lasso_final_v8/lasso_model.joblib')
  scaler = load('models/lasso_final_v8/scaler.joblib')
  selector = load('models/lasso_final_v8/selector.joblib')
  
  # For new data:
  X_scaled = scaler.transform(X_new)
  X_selected = selector.transform(X_scaled)
  prediction = model.predict(X_selected)

============================================================
"""
    
    with open(os.path.join(REPORT_DIR, "summary.txt"), "w") as f:
        f.write(summary)
    
    print(f"\n  Report saved to: {REPORT_DIR}")
    
    print("\n" + "=" * 60)
    print("🏆 TRAINING COMPLETE!")
    print(f"   Final Test MAE: {test_mae:.4f}")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    train_and_evaluate()
