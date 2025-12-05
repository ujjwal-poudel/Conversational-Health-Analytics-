"""
V7: Ridge / ElasticNet Linear Models with Optional Feature Selection

This is a standalone script that tests linear models on the existing
dataset. Does not modify any existing scripts.

Linear models often work better than tree models on small datasets
due to their regularization properties.

Usage:
    python -m src.training.train_linear_models
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, ElasticNetCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_predict, KFold

# Load data from the same location as other experiments
from ..utils.config import MODEL_DIR

# Report directory
REPORT_DIR = os.path.join(
    os.path.dirname(__file__), 
    "../../reports/linear_models_v7"
)
os.makedirs(REPORT_DIR, exist_ok=True)


def load_data():
    """Load train, dev, test .npy files."""
    train = np.load(os.path.join(MODEL_DIR, "train_data.npy"))
    dev = np.load(os.path.join(MODEL_DIR, "dev_data.npy"))
    test = np.load(os.path.join(MODEL_DIR, "test_data.npy"))
    
    # Combine train + dev for CV
    train_dev = np.vstack([train, dev])
    
    X_train = train_dev[:, :-1]
    y_train = train_dev[:, -1]
    X_test = test[:, :-1]
    y_test = test[:, -1]
    
    return X_train, y_train, X_test, y_test


def evaluate(y_true, y_pred, model_name):
    """Compute metrics."""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    
    return {
        "model": model_name,
        "rmse": round(rmse, 4),
        "mae": round(mae, 4),
        "r2": round(r2, 4)
    }


def run_experiment(X_train, y_train, X_test, y_test, use_feature_selection=False, k_features=500):
    """Run Ridge, Lasso, ElasticNet experiments."""
    
    results = []
    
    # Step 1: Standardize features (important for linear models!)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"Original features: {X_train_scaled.shape[1]}")
    
    # Step 2: Optional feature selection
    if use_feature_selection:
        k = min(k_features, X_train_scaled.shape[1])
        selector = SelectKBest(f_regression, k=k)
        X_train_scaled = selector.fit_transform(X_train_scaled, y_train)
        X_test_scaled = selector.transform(X_test_scaled)
        print(f"After feature selection: {X_train_scaled.shape[1]} features")
    
    # Step 3: Cross-validation setup
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    # Ridge Regression
    print("\n--- Ridge Regression ---")
    ridge = RidgeCV(alphas=np.logspace(-3, 3, 50), cv=cv)
    ridge.fit(X_train_scaled, y_train)
    
    y_pred_cv = cross_val_predict(ridge, X_train_scaled, y_train, cv=cv)
    cv_rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"Best alpha: {ridge.alpha_:.4f}")
    
    y_pred_test = ridge.predict(X_test_scaled)
    ridge_result = evaluate(y_test, y_pred_test, "Ridge")
    ridge_result["cv_rmse"] = round(cv_rmse, 4)
    ridge_result["best_alpha"] = round(ridge.alpha_, 4)
    results.append(ridge_result)
    print(f"Test: RMSE={ridge_result['rmse']}, MAE={ridge_result['mae']}, R²={ridge_result['r2']}")
    
    # Lasso Regression
    print("\n--- Lasso Regression ---")
    lasso = LassoCV(alphas=np.logspace(-3, 1, 50), cv=cv, max_iter=10000)
    lasso.fit(X_train_scaled, y_train)
    
    y_pred_cv = cross_val_predict(lasso, X_train_scaled, y_train, cv=cv)
    cv_rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"Best alpha: {lasso.alpha_:.4f}")
    print(f"Non-zero coefficients: {np.sum(lasso.coef_ != 0)}/{len(lasso.coef_)}")
    
    y_pred_test = lasso.predict(X_test_scaled)
    lasso_result = evaluate(y_test, y_pred_test, "Lasso")
    lasso_result["cv_rmse"] = round(cv_rmse, 4)
    lasso_result["best_alpha"] = round(lasso.alpha_, 4)
    lasso_result["nonzero_coefs"] = int(np.sum(lasso.coef_ != 0))
    results.append(lasso_result)
    print(f"Test: RMSE={lasso_result['rmse']}, MAE={lasso_result['mae']}, R²={lasso_result['r2']}")
    
    # ElasticNet
    print("\n--- ElasticNet ---")
    elastic = ElasticNetCV(
        l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9],
        alphas=np.logspace(-3, 1, 30),
        cv=cv,
        max_iter=10000
    )
    elastic.fit(X_train_scaled, y_train)
    
    y_pred_cv = cross_val_predict(elastic, X_train_scaled, y_train, cv=cv)
    cv_rmse = np.sqrt(mean_squared_error(y_train, y_pred_cv))
    print(f"CV RMSE: {cv_rmse:.4f}")
    print(f"Best alpha: {elastic.alpha_:.4f}, Best l1_ratio: {elastic.l1_ratio_:.2f}")
    print(f"Non-zero coefficients: {np.sum(elastic.coef_ != 0)}/{len(elastic.coef_)}")
    
    y_pred_test = elastic.predict(X_test_scaled)
    elastic_result = evaluate(y_test, y_pred_test, "ElasticNet")
    elastic_result["cv_rmse"] = round(cv_rmse, 4)
    elastic_result["best_alpha"] = round(elastic.alpha_, 4)
    elastic_result["best_l1_ratio"] = round(elastic.l1_ratio_, 2)
    elastic_result["nonzero_coefs"] = int(np.sum(elastic.coef_ != 0))
    results.append(elastic_result)
    print(f"Test: RMSE={elastic_result['rmse']}, MAE={elastic_result['mae']}, R²={elastic_result['r2']}")
    
    return results


def save_results(results, use_feature_selection, k_features):
    """Save results to JSON and summary text."""
    
    # Find best model
    best = min(results, key=lambda x: x["mae"])
    
    # Save JSON
    output = {
        "experiment": "V7: Linear Models",
        "feature_selection": use_feature_selection,
        "k_features": k_features if use_feature_selection else "N/A",
        "results": results,
        "best_model": best["model"],
        "best_mae": best["mae"]
    }
    
    with open(os.path.join(REPORT_DIR, "results.json"), "w") as f:
        json.dump(output, f, indent=4)
    
    # Save summary
    lines = [
        "=" * 60,
        "EXPERIMENT: V7 - Linear Models (Ridge/Lasso/ElasticNet)",
        "=" * 60,
        "",
        f"Feature Selection: {'Yes (' + str(k_features) + ' features)' if use_feature_selection else 'No'}",
        "",
        "--- Results ---",
    ]
    
    for r in results:
        lines.append(f"\n{r['model']}:")
        lines.append(f"  CV RMSE:   {r['cv_rmse']}")
        lines.append(f"  Test RMSE: {r['rmse']}")
        lines.append(f"  Test MAE:  {r['mae']}")
        lines.append(f"  Test R²:   {r['r2']}")
    
    lines.append("")
    lines.append(f"--- Best Model: {best['model']} (MAE: {best['mae']}) ---")
    lines.append("=" * 60)
    
    with open(os.path.join(REPORT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nResults saved to: {REPORT_DIR}")


def main():
    """Run V7 linear models experiment with multiple configurations."""
    
    print("=" * 60)
    print("V7: Linear Models (Ridge / Lasso / ElasticNet)")
    print("=" * 60)
    
    # Load data
    X_train, y_train, X_test, y_test = load_data()
    print(f"\nTrain+Dev: {len(X_train)} samples, {X_train.shape[1]} features")
    print(f"Test: {len(X_test)} samples")
    print(f"PHQ mean: {y_train.mean():.2f}, std: {y_train.std():.2f}")
    
    # Test multiple configurations
    all_results = []
    
    # Configuration 1: No feature selection
    print("\n" + "=" * 60)
    print("CONFIG: No Feature Selection")
    print("=" * 60)
    results = run_experiment(X_train, y_train, X_test, y_test, use_feature_selection=False)
    for r in results:
        r["config"] = "No FS"
    all_results.extend(results)
    
    # Configuration 2-5: Different K values
    for k in [100, 200, 300, 500, 1000]:
        if k <= X_train.shape[1]:
            print("\n" + "=" * 60)
            print(f"CONFIG: Feature Selection (K={k})")
            print("=" * 60)
            results = run_experiment(X_train, y_train, X_test, y_test, use_feature_selection=True, k_features=k)
            for r in results:
                r["config"] = f"K={k}"
            all_results.extend(results)
    
    # Find overall best
    best = min(all_results, key=lambda x: x["mae"])
    
    # Save comprehensive results
    print("\n" + "=" * 60)
    print("SUMMARY OF ALL CONFIGURATIONS")
    print("=" * 60)
    
    # Group by config
    configs = {}
    for r in all_results:
        cfg = r["config"]
        if cfg not in configs:
            configs[cfg] = []
        configs[cfg].append(r)
    
    for cfg, results in configs.items():
        print(f"\n{cfg}:")
        for r in results:
            print(f"  {r['model']}: MAE={r['mae']}, RMSE={r['rmse']}, R²={r['r2']}")
    
    print(f"\n🏆 BEST: {best['model']} ({best['config']}) - MAE: {best['mae']}")
    
    # Save to file
    with open(os.path.join(REPORT_DIR, "all_results.json"), "w") as f:
        json.dump({"all_results": all_results, "best": best}, f, indent=4)
    
    # Save summary
    lines = [
        "=" * 60,
        "EXPERIMENT: V7 - Linear Models (Multiple Configurations)",
        "=" * 60,
        "",
    ]
    
    for cfg, results in configs.items():
        lines.append(f"\n--- {cfg} ---")
        for r in results:
            lines.append(f"{r['model']}: MAE={r['mae']}, RMSE={r['rmse']}, R²={r['r2']}")
    
    lines.append("")
    lines.append(f"🏆 BEST: {best['model']} ({best['config']}) - MAE: {best['mae']}")
    lines.append("=" * 60)
    
    with open(os.path.join(REPORT_DIR, "summary.txt"), "w") as f:
        f.write("\n".join(lines))
    
    print(f"\nResults saved to: {REPORT_DIR}")
    print("\n" + "=" * 60)
    print("Experiment complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
