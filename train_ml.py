"""
train_ml.py -- Train ML model with proper cross-validation, hyperparameter tuning,
              and comprehensive evaluation.

Usage:
    python train_ml.py

Trains both RandomForest and XGBoost, picks the best, and saves it to models/ml_model.joblib.
"""

import os
import json
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    HAS_XGB = True
except ImportError:
    HAS_XGB = False
    print("WARN: xgboost not installed -- skipping XGBoost. Install with: pip install xgboost")

# -- Config ---------------------------------------------------------------
DATA_FILE = "data/cleaned_data/merged_training_v4.csv"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "ml_model.joblib")
SCALER_FILE = os.path.join(MODEL_DIR, "ml_scaler.joblib")
METRICS_FILE = os.path.join(MODEL_DIR, "ml_metrics.json")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")

FEATURES = [
    "skill_overlap",
    "missing_skills",
    "bert_similarity",
    "resume_len",
    "jd_len",
    "skill_density",
    "jaccard_similarity",
    "tfidf_cosine",
    "resume_skill_count",
    "jd_skill_count",
]
TARGET = "match_score"
RANDOM_STATE = 42


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in dataset: {missing}")
    print(f"Loaded {len(df)} rows, {len(FEATURES)} features")
    return df


def evaluate_model(name: str, model, X_test, y_test) -> dict:
    """Compute and print evaluation metrics."""
    y_pred = model.predict(X_test)
    y_pred = np.clip(y_pred, 0, 100)

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    print(f"\n{'-' * 40}")
    print(f"  {name}")
    print(f"{'-' * 40}")
    print(f"  MSE:  {mse:.3f}")
    print(f"  RMSE: {rmse:.3f}")
    print(f"  MAE:  {mae:.3f}")
    print(f"  R2:   {r2:.4f}")

    return {"model_name": name, "mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def plot_predictions(y_test, y_pred, name: str, save_path: str):
    """Scatter plot of predicted vs actual."""
    plt.figure(figsize=(7, 7))
    plt.scatter(y_test, y_pred, alpha=0.4, s=10)
    plt.plot([0, 100], [0, 100], "r--", lw=1.5, label="Perfect")
    plt.xlabel("Actual Match Score")
    plt.ylabel("Predicted Match Score")
    plt.title(f"{name} - Predicted vs Actual")
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Plot saved -> {save_path}")


def plot_feature_importance(model, feature_names: list, name: str, save_path: str):
    """Bar chart of feature importances."""
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
    else:
        return
    idx = np.argsort(importances)[::-1]
    plt.figure(figsize=(8, 5))
    plt.bar(range(len(idx)), importances[idx], align="center")
    plt.xticks(range(len(idx)), [feature_names[i] for i in idx], rotation=45, ha="right")
    plt.title(f"{name} - Feature Importance")
    plt.tight_layout()
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Feature importance plot saved -> {save_path}")


def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # -- Load --------------------------------------------------------------
    df = load_data(DATA_FILE)
    X = df[FEATURES].values
    y = df[TARGET].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )
    print(f"Train: {len(X_train)}, Test: {len(X_test)}")

    # -- Scale features ----------------------------------------------------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -- Candidate 1: Random Forest with GridSearch ------------------------
    print("\nTraining Random Forest with hyperparameter tuning...")
    rf_param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [10, 20, None],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }
    rf_grid = GridSearchCV(
        RandomForestRegressor(random_state=RANDOM_STATE),
        rf_param_grid,
        cv=5,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=0,
    )
    rf_grid.fit(X_train_scaled, y_train)
    rf_best = rf_grid.best_estimator_
    print(f"  Best RF params: {rf_grid.best_params_}")
    rf_metrics = evaluate_model("RandomForest", rf_best, X_test_scaled, y_test)

    # -- Candidate 2: Gradient Boosting ------------------------------------
    print("\nTraining Gradient Boosting...")
    gb = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        random_state=RANDOM_STATE,
    )
    gb.fit(X_train_scaled, y_train)
    gb_metrics = evaluate_model("GradientBoosting", gb, X_test_scaled, y_test)

    # -- Candidate 3: XGBoost (if available) -------------------------------
    xgb_metrics = None
    xgb_model = None
    if HAS_XGB:
        print("\nTraining XGBoost with hyperparameter tuning...")
        xgb_param_grid = {
            "n_estimators": [200, 400],
            "max_depth": [4, 6, 8],
            "learning_rate": [0.03, 0.05, 0.1],
            "subsample": [0.8, 1.0],
        }
        xgb_grid = GridSearchCV(
            XGBRegressor(random_state=RANDOM_STATE, verbosity=0),
            xgb_param_grid,
            cv=5,
            scoring="neg_mean_squared_error",
            n_jobs=-1,
            verbose=0,
        )
        xgb_grid.fit(X_train_scaled, y_train)
        xgb_model = xgb_grid.best_estimator_
        print(f"  Best XGB params: {xgb_grid.best_params_}")
        xgb_metrics = evaluate_model("XGBoost", xgb_model, X_test_scaled, y_test)

    # -- Pick best model ---------------------------------------------------
    candidates = [
        ("RandomForest", rf_best, rf_metrics),
        ("GradientBoosting", gb, gb_metrics),
    ]
    if xgb_metrics:
        candidates.append(("XGBoost", xgb_model, xgb_metrics))

    best_name, best_model, best_metrics = min(candidates, key=lambda x: x[2]["rmse"])
    print(f"\nBest model: {best_name}  (RMSE={best_metrics['rmse']:.3f})")

    # -- Cross-validation on best model ------------------------------------
    cv_scores = cross_val_score(
        best_model, X_train_scaled, y_train, cv=5, scoring="neg_mean_squared_error"
    )
    cv_rmse = np.sqrt(-cv_scores)
    print(f"  5-Fold CV RMSE: {cv_rmse.mean():.3f} +/- {cv_rmse.std():.3f}")
    best_metrics["cv_rmse_mean"] = float(cv_rmse.mean())
    best_metrics["cv_rmse_std"] = float(cv_rmse.std())

    # -- Save --------------------------------------------------------------
    joblib.dump(best_model, MODEL_FILE)
    joblib.dump(scaler, SCALER_FILE)
    print(f"\nModel saved -> {MODEL_FILE}")
    print(f"Scaler saved -> {SCALER_FILE}")

    with open(METRICS_FILE, "w") as f:
        json.dump(best_metrics, f, indent=2)
    print(f"Metrics saved -> {METRICS_FILE}")

    # -- Plots -------------------------------------------------------------
    y_pred_best = np.clip(best_model.predict(X_test_scaled), 0, 100)
    plot_predictions(y_test, y_pred_best, best_name, os.path.join(PLOTS_DIR, "pred_vs_actual.png"))
    plot_feature_importance(best_model, FEATURES, best_name, os.path.join(PLOTS_DIR, "feature_importance.png"))

    print("\nML training complete!")


if __name__ == "__main__":
    main()
