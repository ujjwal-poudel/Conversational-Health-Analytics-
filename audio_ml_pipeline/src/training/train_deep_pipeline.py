"""
Training pipeline for Wav2Vec2 + Prosody deep learning features.

Uses the same model wrappers, tuners, and evaluation code as the original
pipeline, but loads .npy files and saves to _wav2vec_prosody directories.

Usage:
    python -m src.training.train_deep_pipeline
"""

import os
import json
import joblib
import numpy as np

from ..config.paths_deep import (
    LGBM_REPORT_DIR_DEEP,
    XGB_REPORT_DIR_DEEP,
    MODELS_DIR_DEEP,
)
from ..datautils.deep_dataset_loader import (
    load_deep_datasets,
    prepare_deep_training_data,
    prepare_deep_test_data,
)

try:
    from ..modeling.lgbm_model import LightGBMModel
    from ..tuning.lgbm_tuner import tune_lgbm
    LGBM_AVAILABLE = True
except (ImportError, OSError) as e:
    LGBM_AVAILABLE = False
    print(f"[WARN] LightGBM not available: {e}")
    print("[INFO] Will train XGBoost only. To enable LightGBM on macOS, run: brew install libomp")

from ..modeling.xgb_model import XGBoostModel
from ..tuning.xgb_tuner import tune_xgb
from ..evaluation.evaluate import (
    evaluate_predictions,
    compute_residuals,
    fold_summary,
)
from ..evaluation.plots import (
    plot_fold_rmse,
    plot_optuna_trials,
    plot_predictions_vs_actual,
    plot_residual_histogram,
    plot_residuals_vs_predicted,
    plot_feature_importance,
    plot_fold_prediction_distribution,
)

from sklearn.model_selection import KFold


N_TRIALS = 20
N_SPLITS = 5
RANDOM_STATE = 42


def _get_lgbm_fold_predictions(X, y, best_params):
    """
    Compute fold-wise predictions using best LightGBM params.
    """
    from sklearn.metrics import root_mean_squared_error
    import lightgbm as lgb

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_predictions = []
    fold_rmse = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = lgb.LGBMRegressor(**best_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_valid)
        fold_predictions.append(preds)
        rmse = root_mean_squared_error(y_valid, preds)
        fold_rmse.append(rmse)

    return fold_predictions, fold_rmse


def _get_xgb_fold_predictions(X, y, best_params):
    """
    Compute fold-wise predictions using best XGBoost params.
    """
    from sklearn.metrics import root_mean_squared_error
    import xgboost as xgb

    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    fold_predictions = []
    fold_rmse = []

    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        model = xgb.XGBRegressor(**best_params)
        model.fit(X_train, y_train)

        preds = model.predict(X_valid)
        fold_predictions.append(preds)
        rmse = root_mean_squared_error(y_valid, preds)
        fold_rmse.append(rmse)

    return fold_predictions, fold_rmse


def _save_json(obj, path):
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def _save_summary_text(path, model_name, cv_info, test_metrics, feature_info):
    """
    Saves detailed summary with experiment info, feature breakdown, and metrics.
    
    Parameters
    ----------
    feature_info : dict
        Contains: experiment_version, n_features, n_train, n_test,
        feature_breakdown (detailed description of processing steps)
    """
    lines = []
    lines.append("=" * 60)
    lines.append(f"EXPERIMENT: {feature_info.get('experiment_version', 'Unknown')}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Model: {model_name}")
    lines.append("")
    lines.append("--- Feature Engineering Pipeline ---")
    lines.append(feature_info.get('feature_breakdown', 'N/A'))
    lines.append("")
    lines.append("--- Dataset Info ---")
    lines.append(f"  Train+Dev samples: {feature_info.get('n_train', 'N/A')}")
    lines.append(f"  Test samples: {feature_info.get('n_test', 'N/A')}")
    lines.append(f"  Total features: {feature_info.get('n_features', 'N/A')}")
    lines.append(f"  Feature/Sample ratio: {feature_info.get('n_features', 0) / max(feature_info.get('n_train', 1), 1):.2f}")
    lines.append("")
    lines.append("--- Cross-Validation (train+dev) ---")
    lines.append(f"  Mean RMSE: {cv_info['fold_rmse_mean']:.4f}")
    lines.append(f"  Std RMSE:  {cv_info['fold_rmse_std']:.4f}")
    lines.append(f"  Fold RMSEs: {[round(x, 4) for x in cv_info['fold_rmse_values']]}")
    lines.append("")
    lines.append("--- Test Set Performance ---")
    lines.append(f"  RMSE: {test_metrics['rmse']:.4f}")
    lines.append(f"  MAE:  {test_metrics['mae']:.4f}")
    lines.append(f"  R²:   {test_metrics['r2']:.4f}")
    lines.append("")
    lines.append("--- Interpretation ---")
    if test_metrics['r2'] > 0.1:
        lines.append("  Model is learning meaningful patterns.")
    elif test_metrics['r2'] > 0:
        lines.append("  Model slightly better than predicting mean.")
    else:
        lines.append("  Model worse than baseline (predicting mean).")
    lines.append("=" * 60)

    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_lgbm_deep_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols, feature_info):
    """
    Run tuning, training, evaluation, and reporting for LightGBM.
    """
    # Tuning
    best_params, best_rmse, fold_scores, trial_scores = tune_lgbm(
        X_train_full,
        y_train_full,
        n_trials=N_TRIALS,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    # Fold-wise predictions with best params
    fold_predictions, fold_rmse_best = _get_lgbm_fold_predictions(
        X_train_full, y_train_full, best_params
    )

    cv_info = fold_summary(fold_rmse_best)

    # Train final model on full train+dev
    lgbm_model = LightGBMModel(best_params)
    lgbm_model.train(X_train_full, y_train_full)

    # Test evaluation
    y_pred_test = lgbm_model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_pred_test)
    residuals = compute_residuals(y_test, y_pred_test)

    # Feature importances
    importance = lgbm_model.get_feature_importance(feature_cols)

    # Save model
    model_artifact = {
        "model_type": "lightgbm",
        "features": "wav2vec2_prosody",
        "params": best_params,
        "feature_names": feature_cols,
        "model": lgbm_model.model,
    }
    model_path = os.path.join(MODELS_DIR_DEEP, "lgbm_model.joblib")
    joblib.dump(model_artifact, model_path)

    # Save JSON reports
    _save_json(
        {
            "best_rmse": best_rmse,
            "fold_scores_initial": fold_scores,
            "fold_scores_best_params": fold_rmse_best,
            "trial_scores": trial_scores,
            "n_splits": N_SPLITS,
            "n_trials": N_TRIALS,
        },
        os.path.join(LGBM_REPORT_DIR_DEEP, "cv_results.json"),
    )

    _save_json(test_metrics, os.path.join(LGBM_REPORT_DIR_DEEP, "test_results.json"))
    _save_json(best_params, os.path.join(LGBM_REPORT_DIR_DEEP, "best_params.json"))

    # Save summary text
    _save_summary_text(
        os.path.join(LGBM_REPORT_DIR_DEEP, "summary.txt"),
        "LightGBM",
        cv_info,
        test_metrics,
        feature_info,
    )

    # Plots
    plot_fold_rmse(
        fold_rmse_best,
        os.path.join(LGBM_REPORT_DIR_DEEP, "cv_fold_rmse.png"),
    )

    plot_optuna_trials(
        trial_scores,
        os.path.join(LGBM_REPORT_DIR_DEEP, "optuna_trials.png"),
    )

    plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        os.path.join(LGBM_REPORT_DIR_DEEP, "predictions_vs_actual.png"),
    )

    plot_residual_histogram(
        residuals,
        os.path.join(LGBM_REPORT_DIR_DEEP, "residuals_histogram.png"),
    )

    plot_residuals_vs_predicted(
        y_pred_test,
        residuals,
        os.path.join(LGBM_REPORT_DIR_DEEP, "residuals_vs_predicted.png"),
    )

    plot_feature_importance(
        importance,
        os.path.join(LGBM_REPORT_DIR_DEEP, "feature_importance.png"),
        top_n=20,
    )

    plot_fold_prediction_distribution(
        fold_predictions,
        os.path.join(LGBM_REPORT_DIR_DEEP, "fold_prediction_distribution.png"),
    )


def run_xgb_deep_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols, feature_info):
    """
    Run tuning, training, evaluation, and reporting for XGBoost.
    """
    best_params, best_rmse, fold_scores, trial_scores = tune_xgb(
        X_train_full,
        y_train_full,
        n_trials=N_TRIALS,
        n_splits=N_SPLITS,
        random_state=RANDOM_STATE,
    )

    # Fold-wise predictions with best params
    fold_predictions, fold_rmse_best = _get_xgb_fold_predictions(
        X_train_full, y_train_full, best_params
    )

    cv_info = fold_summary(fold_rmse_best)

    # Train final model on full train+dev
    xgb_model = XGBoostModel(best_params)
    xgb_model.train(X_train_full, y_train_full)

    # Test evaluation
    y_pred_test = xgb_model.predict(X_test)
    test_metrics = evaluate_predictions(y_test, y_pred_test)
    residuals = compute_residuals(y_test, y_pred_test)

    # Feature importances
    importance = xgb_model.get_feature_importance(feature_cols)

    # Save model
    model_artifact = {
        "model_type": "xgboost",
        "features": "wav2vec2_prosody",
        "params": best_params,
        "feature_names": feature_cols,
        "model": xgb_model.model,
    }
    model_path = os.path.join(MODELS_DIR_DEEP, "xgb_model.joblib")
    joblib.dump(model_artifact, model_path)

    # Save JSON reports
    _save_json(
        {
            "best_rmse": best_rmse,
            "fold_scores_initial": fold_scores,
            "fold_scores_best_params": fold_rmse_best,
            "trial_scores": trial_scores,
            "n_splits": N_SPLITS,
            "n_trials": N_TRIALS,
        },
        os.path.join(XGB_REPORT_DIR_DEEP, "cv_results.json"),
    )

    _save_json(test_metrics, os.path.join(XGB_REPORT_DIR_DEEP, "test_results.json"))
    _save_json(best_params, os.path.join(XGB_REPORT_DIR_DEEP, "best_params.json"))

    # Save summary text
    _save_summary_text(
        os.path.join(XGB_REPORT_DIR_DEEP, "summary.txt"),
        "XGBoost",
        cv_info,
        test_metrics,
        feature_info,
    )

    # Plots
    plot_fold_rmse(
        fold_rmse_best,
        os.path.join(XGB_REPORT_DIR_DEEP, "cv_fold_rmse.png"),
    )

    plot_optuna_trials(
        trial_scores,
        os.path.join(XGB_REPORT_DIR_DEEP, "optuna_trials.png"),
    )

    plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        os.path.join(XGB_REPORT_DIR_DEEP, "predictions_vs_actual.png"),
    )

    plot_residual_histogram(
        residuals,
        os.path.join(XGB_REPORT_DIR_DEEP, "residuals_histogram.png"),
    )

    plot_residuals_vs_predicted(
        y_pred_test,
        residuals,
        os.path.join(XGB_REPORT_DIR_DEEP, "residuals_vs_predicted.png"),
    )

    plot_feature_importance(
        importance,
        os.path.join(XGB_REPORT_DIR_DEEP, "feature_importance.png"),
        top_n=20,
    )

    plot_fold_prediction_distribution(
        fold_predictions,
        os.path.join(XGB_REPORT_DIR_DEEP, "fold_prediction_distribution.png"),
    )


def main():
    """
    Entry point for training with deep learning features.
    V6: PCA (768→100) + Segment Pooling (3 segments)
    """
    print("\n" + "=" * 60)
    print("Training Pipeline: V6 PCA + Segment Pooling Hybrid")
    print("=" * 60)
    
    train_df, dev_df, test_df = load_deep_datasets()

    X_train_full, y_train_full, X_dev, y_dev, full_train_df = prepare_deep_training_data(
        train_df, dev_df
    )
    X_test, y_test, feature_cols = prepare_deep_test_data(test_df)

    n_features = len(feature_cols)
    n_train = len(X_train_full)
    n_test = len(X_test)

    print(f"Train+Dev samples: {n_train}")
    print(f"Test samples: {n_test}")
    print(f"Feature dimensions: {n_features}")

    # Build feature breakdown description
    feature_breakdown = """
  Step 1: Wav2Vec2 embeddings (768 dims per frame)
          -> Model: superb/wav2vec2-base-superb-er (emotion-tuned)
          -> PCA reduction (768 → 100 dims)
          -> Segment pooling (3 segments: Begin/Mid/End)
          -> 4 stats per segment (mean, std, min, max)
          -> 100 × 4 × 3 = 1,200 features

  Step 2: Prosody features (13 dims per frame)
          -> F0, RMS, ZCR, spectral centroid/bandwidth/rolloff, contrast
          -> Segment pooling (3 segments: Begin/Mid/End)
          -> 4 stats per segment (mean, std, min, max)
          -> 13 × 4 × 3 = 156 features

  Step 3: Concatenate
          -> 1,200 + 156 = 1,356 total features
"""

    feature_info = {
        "experiment_version": "V6: PCA (768→100) + Segment Pooling (3 segments)",
        "n_features": n_features,
        "n_train": n_train,
        "n_test": n_test,
        "feature_breakdown": feature_breakdown,
    }

    # Run LightGBM if available
    if LGBM_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training LightGBM...")
        print("=" * 60)
        run_lgbm_deep_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols, feature_info)
    else:
        print("\n[SKIP] LightGBM training (library not available)")

    # Run XGBoost
    print("\n" + "=" * 60)
    print("Training XGBoost...")
    print("=" * 60)
    run_xgb_deep_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols, feature_info)

    print("\n" + "=" * 60)
    print("Training complete!")
    print(f"Reports saved to: {LGBM_REPORT_DIR_DEEP}")
    print(f"                  {XGB_REPORT_DIR_DEEP}")
    print(f"Models saved to:  {MODELS_DIR_DEEP}")
    print("=" * 60)


if __name__ == "__main__":
    main()
