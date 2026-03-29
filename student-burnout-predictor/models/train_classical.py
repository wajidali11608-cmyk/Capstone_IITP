"""
Classical Machine Learning Model Training
Trains Random Forest and Gradient Boosting classifiers for burnout prediction.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    f1_score,
)
import joblib

# Add project root to path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.preprocessing import load_data, clean_data, normalize_features
from pipeline.sentiment import add_sentiment_features
from pipeline.feature_engineering import engineer_features, get_feature_columns


LABEL_NAMES = {0: "Low", 1: "Medium", 2: "High"}
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")


def prepare_data(csv_path):
    """Load, preprocess, and engineer features from raw CSV.

    Returns:
        Tuple of (X, y, feature_columns, scaler, full_df)
    """
    df = load_data(csv_path)
    df = clean_data(df)
    df = add_sentiment_features(df)
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    X = df[available_cols].values
    y = df["burnout_label"].values

    return X, y, available_cols, df


def train_random_forest(X_train, y_train):
    """Train Random Forest with hyperparameter tuning."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [8, 12, 16],
        "min_samples_split": [5, 10],
        "min_samples_leaf": [2, 4],
    }

    rf = RandomForestClassifier(random_state=42, class_weight="balanced")

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best RF params: {grid_search.best_params_}")
    print(f"  Best CV F1 (macro): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def train_gradient_boosting(X_train, y_train):
    """Train Gradient Boosting Classifier."""
    param_grid = {
        "n_estimators": [100, 200],
        "max_depth": [4, 6, 8],
        "learning_rate": [0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    gbm = GradientBoostingClassifier(random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        gbm, param_grid, cv=cv, scoring="f1_macro", n_jobs=-1, verbose=0
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best GBM params: {grid_search.best_params_}")
    print(f"  Best CV F1 (macro): {grid_search.best_score_:.4f}")

    return grid_search.best_estimator_


def evaluate_model(model, X_test, y_test, model_name):
    """Evaluate a model and print metrics."""
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="macro")

    print(f"\n{'='*50}")
    print(f"  {model_name} Results")
    print(f"{'='*50}")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (macro): {f1:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))
    print(f"  Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    return acc, f1, y_pred


def run_training():
    """Full training pipeline."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "student_survey.csv")

    if not os.path.exists(csv_path):
        print("Dataset not found. Generating synthetic data...")
        from data.generate_data import generate_student_data
        df = generate_student_data()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)
        print(f"Generated {len(df)} records.\n")

    print("=" * 50)
    print("  STUDENT BURNOUT PREDICTION — MODEL TRAINING")
    print("=" * 50)

    print("\n[1/5] Loading and preprocessing data...")
    X, y, feature_cols, full_df = prepare_data(csv_path)
    print(f"  Dataset shape: {X.shape}")
    print(f"  Features: {len(feature_cols)}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("\n[2/5] Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("\n[3/5] Training Random Forest...")
    rf_model = train_random_forest(X_train, y_train)
    rf_acc, rf_f1, rf_preds = evaluate_model(rf_model, X_test, y_test, "Random Forest")

    print("\n[4/5] Training Gradient Boosting...")
    gbm_model = train_gradient_boosting(X_train, y_train)
    gbm_acc, gbm_f1, gbm_preds = evaluate_model(gbm_model, X_test, y_test, "Gradient Boosting")

    # Select best model
    best_name = "Random Forest" if rf_f1 >= gbm_f1 else "Gradient Boosting"
    best_model = rf_model if rf_f1 >= gbm_f1 else gbm_model
    best_acc = rf_acc if rf_f1 >= gbm_f1 else gbm_acc

    print(f"\n[5/5] Saving best model ({best_name})...")
    os.makedirs(MODEL_DIR, exist_ok=True)

    joblib.dump(best_model, os.path.join(MODEL_DIR, "best_model.pkl"))
    joblib.dump(rf_model, os.path.join(MODEL_DIR, "rf_model.pkl"))
    joblib.dump(gbm_model, os.path.join(MODEL_DIR, "gbm_model.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR, "feature_columns.pkl"))

    print(f"  Models saved to: {MODEL_DIR}")
    print(f"\n  BEST MODEL: {best_name} (Accuracy: {best_acc:.4f})")

    return best_model, feature_cols, X_test, y_test


if __name__ == "__main__":
    run_training()
