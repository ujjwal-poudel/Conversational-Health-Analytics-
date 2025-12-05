import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error, r2_score


def compute_rmse(y_true, y_pred):
    """
    Compute Root Mean Squared Error.
    """
    return root_mean_squared_error(y_true, y_pred)


def compute_mae(y_true, y_pred):
    """
    Compute Mean Absolute Error.
    """
    return mean_absolute_error(y_true, y_pred)


def compute_r2(y_true, y_pred):
    """
    Compute R² score.
    """
    return r2_score(y_true, y_pred)


def evaluate_predictions(y_true, y_pred):
    """
    Compute all key metrics and return a dictionary.
    """
    rmse = compute_rmse(y_true, y_pred)
    mae = compute_mae(y_true, y_pred)
    r2 = compute_r2(y_true, y_pred)

    return {
        "rmse": rmse,
        "mae": mae,
        "r2": r2
    }


def compute_residuals(y_true, y_pred):
    """
    Compute residuals (y_true - y_pred).
    """
    return np.array(y_true) - np.array(y_pred)


def fold_summary(fold_scores):
    """
    Convert fold-wise list of RMSE scores into a summary dictionary.
    """
    fold_scores = np.array(fold_scores)
    return {
        "fold_rmse_mean": float(np.mean(fold_scores)),
        "fold_rmse_std": float(np.std(fold_scores)),
        "fold_rmse_values": fold_scores.tolist()
    }