import os
import json
import joblib
import numpy as np

from ..config.paths import BASE_DIR, LGBM_REPORT_DIR, XGB_REPORT_DIR
from ..datautils.dataset_loader import (
    load_datasets,
    prepare_training_data,
    prepare_test_data,
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

# Directory to save trained models
MODELS_DIR = os.path.join(BASE_DIR, "audio_ml_pipeline", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def _get_lgbm_fold_predictions(X, y, best_params):
    """
    Compute fold-wise predictions using best LightGBM params.
    Returns:
        fold_predictions: list of np.ndarray (per-fold predictions)
        fold_rmse: list of RMSE per fold
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
    Returns:
        fold_predictions: list of np.ndarray (per-fold predictions)
        fold_rmse: list of RMSE per fold
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


def _save_summary_text(path, model_name, cv_info, test_metrics):
    lines = []
    lines.append(f"Model: {model_name}")
    lines.append("")
    lines.append("Cross-Validation (train+dev):")
    lines.append(f"  Mean RMSE: {cv_info['fold_rmse_mean']:.4f}")
    lines.append(f"  Std RMSE:  {cv_info['fold_rmse_std']:.4f}")
    lines.append(f"  Fold RMSEs: {cv_info['fold_rmse_values']}")
    lines.append("")
    lines.append("Test Set Performance:")
    lines.append(f"  RMSE: {test_metrics['rmse']:.4f}")
    lines.append(f"  MAE:  {test_metrics['mae']:.4f}")
    lines.append(f"  R²:   {test_metrics['r2']:.4f}")

    with open(path, "w") as f:
        f.write("\n".join(lines))


def run_lgbm_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols):
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

    # Fold-wise predictions with best params (for distribution plot)
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
        "params": best_params,
        "feature_names": feature_cols,
        "model": lgbm_model.model,
    }
    model_path = os.path.join(MODELS_DIR, "lgbm_model.joblib")
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
        os.path.join(LGBM_REPORT_DIR, "cv_results.json"),
    )

    _save_json(test_metrics, os.path.join(LGBM_REPORT_DIR, "test_results.json"))
    _save_json(best_params, os.path.join(LGBM_REPORT_DIR, "best_params.json"))

    # Save summary text
    _save_summary_text(
        os.path.join(LGBM_REPORT_DIR, "summary.txt"),
        "LightGBM",
        cv_info,
        test_metrics,
    )

    # Plots
    plot_fold_rmse(
        fold_rmse_best,
        os.path.join(LGBM_REPORT_DIR, "cv_fold_rmse.png"),
    )

    plot_optuna_trials(
        trial_scores,
        os.path.join(LGBM_REPORT_DIR, "optuna_trials.png"),
    )

    plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        os.path.join(LGBM_REPORT_DIR, "predictions_vs_actual.png"),
    )

    plot_residual_histogram(
        residuals,
        os.path.join(LGBM_REPORT_DIR, "residuals_histogram.png"),
    )

    plot_residuals_vs_predicted(
        y_pred_test,
        residuals,
        os.path.join(LGBM_REPORT_DIR, "residuals_vs_predicted.png"),
    )

    plot_feature_importance(
        importance,
        os.path.join(LGBM_REPORT_DIR, "feature_importance.png"),
        top_n=20,
    )

    plot_fold_prediction_distribution(
        fold_predictions,
        os.path.join(LGBM_REPORT_DIR, "fold_prediction_distribution.png"),
    )


def run_xgb_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols):
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
        "params": best_params,
        "feature_names": feature_cols,
        "model": xgb_model.model,
    }
    model_path = os.path.join(MODELS_DIR, "xgb_model.joblib")
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
        os.path.join(XGB_REPORT_DIR, "cv_results.json"),
    )

    _save_json(test_metrics, os.path.join(XGB_REPORT_DIR, "test_results.json"))
    _save_json(best_params, os.path.join(XGB_REPORT_DIR, "best_params.json"))

    # Save summary text
    _save_summary_text(
        os.path.join(XGB_REPORT_DIR, "summary.txt"),
        "XGBoost",
        cv_info,
        test_metrics,
    )

    # Plots
    plot_fold_rmse(
        fold_rmse_best,
        os.path.join(XGB_REPORT_DIR, "cv_fold_rmse.png"),
    )

    plot_optuna_trials(
        trial_scores,
        os.path.join(XGB_REPORT_DIR, "optuna_trials.png"),
    )

    plot_predictions_vs_actual(
        y_test,
        y_pred_test,
        os.path.join(XGB_REPORT_DIR, "predictions_vs_actual.png"),
    )

    plot_residual_histogram(
        residuals,
        os.path.join(XGB_REPORT_DIR, "residuals_histogram.png"),
    )

    plot_residuals_vs_predicted(
        y_pred_test,
        residuals,
        os.path.join(XGB_REPORT_DIR, "residuals_vs_predicted.png"),
    )

    plot_feature_importance(
        importance,
        os.path.join(XGB_REPORT_DIR, "feature_importance.png"),
        top_n=20,
    )

    plot_fold_prediction_distribution(
        fold_predictions,
        os.path.join(XGB_REPORT_DIR, "fold_prediction_distribution.png"),
    )


def main():
    """
    Entry point for running the full training pipeline:
    - Load datasets
    - Prepare train/dev/test
    - Run LightGBM and XGBoost pipelines
    """
    train_df, dev_df, test_df = load_datasets()

    X_train_full, y_train_full, X_dev, y_dev, full_train_df = prepare_training_data(
        train_df, dev_df
    )
    X_test, y_test, feature_cols = prepare_test_data(test_df)

    # Run LightGBM if available
    if LGBM_AVAILABLE:
        print("\n" + "=" * 60)
        print("Training LightGBM...")
        print("=" * 60)
        run_lgbm_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols)
    else:
        print("\n[SKIP] LightGBM training (library not available)")

    # Run XGBoost
    print("\n" + "=" * 60)
    print("Training XGBoost...")
    print("=" * 60)
    run_xgb_pipeline(X_train_full, y_train_full, X_test, y_test, feature_cols)

    print("\n" + "=" * 60)
    print("Training complete!")
    print("=" * 60)