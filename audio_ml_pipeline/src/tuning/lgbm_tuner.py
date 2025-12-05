import numpy as np
import optuna
import lightgbm as lgb
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error
from typing import Dict, Any, Tuple, List


def tune_lgbm(
    X,
    y,
    n_trials: int = 20,
    n_splits: int = 5,
    random_state: int = 42
) -> Tuple[Dict[str, Any], float, List[float], List[float]]:
    """
    Hyperparameter tuning for LightGBM using Optuna with K-Fold CV.

    Args:
        X: Training feature matrix
        y: Training labels
        n_trials: Number of Optuna trials (default: 20)
        n_splits: Number of CV folds (default: 5)
        random_state: Reproducibility for KFold

    Returns:
        best_params: Best-performing hyperparameters from Optuna
        best_rmse: RMSE of best trial
        fold_scores: List of fold-wise RMSE values for best params
        trial_scores: List of trial RMSE values across all trials
    """

    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    trial_scores = []

    def objective(trial):
        params = {
            "objective": "regression",
            "metric": "rmse",
            "verbosity": -1,
            "boosting_type": "gbdt",
            "num_leaves": trial.suggest_int("num_leaves", 20, 200),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 2.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 0.0, 2.0)
        }

        fold_rmse = []

        for train_idx, valid_idx in kf.split(X):
            X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
            y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)

            preds = model.predict(X_valid)
            rmse = root_mean_squared_error(y_valid, preds)
            fold_rmse.append(rmse)

        avg_rmse = float(np.mean(fold_rmse))
        trial_scores.append(avg_rmse)
        return avg_rmse

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)

    best_params = study.best_params
    best_rmse = study.best_value

    # Compute fold-wise predictions for the best parameters
    fold_scores = []
    for train_idx, valid_idx in kf.split(X):
        X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
        y_train, y_valid = y.iloc[train_idx], y.iloc[valid_idx]

        best_model = lgb.LGBMRegressor(**best_params)
        best_model.fit(X_train, y_train)

        preds = best_model.predict(X_valid)
        rmse = root_mean_squared_error(y_valid, preds)
        fold_scores.append(rmse)

    return best_params, best_rmse, fold_scores, trial_scores