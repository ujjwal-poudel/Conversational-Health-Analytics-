import os
import numpy as np
import matplotlib.pyplot as plt


def plot_fold_rmse(fold_scores, save_path):
    """
    Plot RMSE per fold for CV results.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(fold_scores) + 1), fold_scores, marker="o")
    plt.xlabel("Fold")
    plt.ylabel("RMSE")
    plt.title("Cross-Validation RMSE by Fold")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_optuna_trials(trial_scores, save_path):
    """
    Plot RMSE vs Optuna trial number.
    """
    plt.figure(figsize=(6, 4))
    plt.plot(range(1, len(trial_scores) + 1), trial_scores, marker="o")
    plt.xlabel("Trial")
    plt.ylabel("RMSE")
    plt.title("Optuna Trial Performance")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_predictions_vs_actual(y_true, y_pred, save_path):
    """
    Scatter plot of predictions vs actual PHQ values.
    """
    plt.figure(figsize=(6, 6))
    plt.scatter(y_true, y_pred, alpha=0.7)
    plt.xlabel("Actual PHQ Score")
    plt.ylabel("Predicted PHQ Score")
    plt.title("Predictions vs Actual")
    min_val = min(min(y_true), min(y_pred))
    max_val = max(max(y_true), max(y_pred))
    plt.plot([min_val, max_val], [min_val, max_val], linestyle="--", color="gray")  # identity line
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residual_histogram(residuals, save_path):
    """
    Histogram of residual values.
    """
    plt.figure(figsize=(6, 4))
    plt.hist(residuals, bins=20, edgecolor="black")
    plt.xlabel("Residual (y_true - y_pred)")
    plt.ylabel("Count")
    plt.title("Residual Distribution")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_residuals_vs_predicted(y_pred, residuals, save_path):
    """
    Scatter of residuals vs predicted values.
    """
    plt.figure(figsize=(6, 4))
    plt.scatter(y_pred, residuals, alpha=0.7)
    plt.xlabel("Predicted PHQ Score")
    plt.ylabel("Residual")
    plt.title("Residuals vs Predicted")
    plt.axhline(0, color="gray", linestyle="--")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_feature_importance(importance_dict, save_path, top_n=20):
    """
    Bar chart of the top N most important features.
    """
    items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = items[:top_n]
    names = [x[0] for x in top_features]
    scores = [x[1] for x in top_features]

    plt.figure(figsize=(8, 6))
    plt.barh(names[::-1], scores[::-1])
    plt.xlabel("Importance Score")
    plt.title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_fold_prediction_distribution(fold_predictions, save_path):
    """
    Plot the distribution of predictions for each fold.
    
    Args:
        fold_predictions: List of arrays; each array contains model predictions for a fold.
        save_path: Path to save the plot.
    """
    plt.figure(figsize=(8, 6))

    for i, preds in enumerate(fold_predictions):
        plt.hist(
            preds,
            bins=20,
            alpha=0.4,
            label=f"Fold {i + 1}",
            edgecolor="black"
        )

    plt.xlabel("Predicted PHQ Score")
    plt.ylabel("Count")
    plt.title("Fold-wise Prediction Distribution")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()