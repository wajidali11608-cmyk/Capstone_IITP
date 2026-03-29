"""
Explainable AI Module
Uses SHAP for global and per-student explanations of burnout predictions.
"""

import os
import sys
import numpy as np
import pandas as pd
import shap
import joblib
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.preprocessing import load_data, clean_data
from pipeline.sentiment import add_sentiment_features
from pipeline.feature_engineering import engineer_features, get_feature_columns


LABEL_NAMES = {0: "Low", 1: "Medium", 2: "High"}
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "models", "explanations")


def load_model_and_data():
    """Load the RF model (supports multi-class SHAP) and prepare data."""
    # Use RF model for SHAP — TreeExplainer doesn't support multi-class GBM
    rf_path = os.path.join(MODEL_DIR, "rf_model.pkl")
    best_path = os.path.join(MODEL_DIR, "best_model.pkl")
    model = joblib.load(rf_path if os.path.exists(rf_path) else best_path)
    feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

    csv_path = os.path.join(PROJECT_ROOT, "data", "student_survey.csv")
    df = load_data(csv_path)
    df = clean_data(df)
    df = add_sentiment_features(df)
    df = engineer_features(df)

    available_cols = [c for c in feature_cols if c in df.columns]
    X = df[available_cols]

    return model, X, available_cols, df


def compute_shap_values(model, X, feature_names):
    """Compute SHAP values using TreeExplainer."""
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    return explainer, shap_values


def get_global_feature_importance(shap_values, feature_names):
    """Compute global feature importance from SHAP values.

    Returns:
        List of dicts sorted by importance (descending).
    """
    # For multi-class: shap_values is a list of arrays, one per class
    # Average absolute SHAP across all classes
    if isinstance(shap_values, list):
        mean_abs_shap = np.mean(
            [np.abs(sv).mean(axis=0) for sv in shap_values], axis=0
        )
    elif len(shap_values.shape) == 3:
        # Shape: (n_samples, n_features, n_classes)
        # Average over samples (axis=0), then average over classes (axis=1)
        mean_abs_shap = np.abs(shap_values).mean(axis=0).mean(axis=1)
    else:
        # Shape: (n_samples, n_features)
        mean_abs_shap = np.abs(shap_values).mean(axis=0)

    importance = sorted(
        zip(feature_names, mean_abs_shap),
        key=lambda x: x[1],
        reverse=True,
    )

    return [{"feature": name, "importance": float(round(val, 4))} for name, val in importance]


def explain_student(model, explainer, X_row, feature_names, shap_values_row):
    """Generate textual explanation for a single student prediction.

    Args:
        model: Trained model.
        explainer: SHAP explainer.
        X_row: Feature values for this student (1D array or Series).
        feature_names: List of feature names.
        shap_values_row: SHAP values for this student.

    Returns:
        Dict with prediction, risk level, and top contributing factors.
    """
    prediction = model.predict(X_row.values.reshape(1, -1))[0]
    risk_level = LABEL_NAMES[prediction]

    # Get SHAP values for the predicted class
    if isinstance(shap_values_row, list):
        sv = shap_values_row[prediction]
    elif len(shap_values_row.shape) == 2:
        # Shape (n_features, n_classes) -> extract column for predicted class
        sv = shap_values_row[:, prediction]
    else:
        # Shape (n_features,)
        sv = shap_values_row

    # Top contributing features (by absolute SHAP value)
    feature_contributions = sorted(
        zip(feature_names, sv, X_row.values),
        key=lambda x: abs(x[1]),
        reverse=True,
    )

    top_factors = []
    for name, shap_val, raw_val in feature_contributions[:5]:
        direction = "increases" if shap_val > 0 else "decreases"
        top_factors.append({
            "feature": name,
            "value": float(round(raw_val, 2)),
            "shap_value": float(round(shap_val, 4)),
            "direction": direction,
            "description": _feature_description(name, raw_val, shap_val),
        })

    return {
        "prediction": int(prediction),
        "risk_level": risk_level,
        "top_factors": top_factors,
    }


def _feature_description(name, value, shap_val):
    """Generate human-readable description of a feature contribution."""
    direction = "increases" if shap_val > 0 else "decreases"

    descriptions = {
        "stress_level": f"Stress level of {value:.0f}/10 {direction} burnout risk",
        "sleep_hours": f"Sleep of {value:.1f} hrs/night {direction} burnout risk",
        "sleep_irregularity": f"Sleep irregularity (σ={value:.2f}) {direction} burnout risk",
        "late_submissions": f"{value:.0f} late submissions {direction} burnout risk",
        "study_hours": f"Studying {value:.1f} hrs/day {direction} burnout risk",
        "social_activity_freq": f"Social activity {value:.0f} days/week {direction} burnout risk",
        "social_isolation_score": f"Social isolation score of {value:.2f} {direction} burnout risk",
        "sentiment_polarity": f"Emotional sentiment ({value:.2f}) {direction} burnout risk",
        "has_negative_emotion": f"Negative emotions detected — {direction} burnout risk",
        "procrastination_score": f"Procrastination score ({value:.2f}) {direction} burnout risk",
        "stress_sleep_interaction": f"Stress × sleep deficit ({value:.1f}) {direction} burnout risk",
        "screen_study_ratio": f"Screen/study ratio ({value:.1f}) {direction} burnout risk",
        "study_overload": f"Study overload {'detected' if value > 0 else 'not detected'} — {direction} burnout risk",
        "exercise_hours": f"Exercise {value:.1f} hrs/week {direction} burnout risk",
        "screen_time": f"Screen time {value:.1f} hrs/day {direction} burnout risk",
        "negative_sentiment_trend": f"Negative sentiment trend ({value:.2f}) {direction} burnout risk",
        "sentiment_subjectivity": f"Emotional subjectivity ({value:.2f}) {direction} burnout risk",
        "has_positive_emotion": f"Positive emotions {'detected' if value > 0 else 'absent'} — {direction} burnout risk",
    }

    return descriptions.get(name, f"{name} = {value:.2f} {direction} burnout risk")


def generate_intervention(risk_level, top_factors):
    """Generate adaptive intervention recommendations."""
    interventions = []

    factor_names = [f["feature"] for f in top_factors]

    if risk_level == "High":
        interventions.append("🚨 Immediate: Consider speaking with a counselor or academic advisor.")

    if "stress_level" in factor_names or "stress_sleep_interaction" in factor_names:
        interventions.append("🧘 Practice stress management: try deep breathing, meditation, or short walks between study sessions.")

    if "sleep_hours" in factor_names or "sleep_irregularity" in factor_names:
        interventions.append("😴 Improve sleep hygiene: aim for 7-8 hours, maintain consistent sleep/wake times.")

    if "social_isolation_score" in factor_names or "social_activity_freq" in factor_names:
        interventions.append("👥 Increase social engagement: join a study group, attend campus events, or schedule time with friends.")

    if "late_submissions" in factor_names or "procrastination_score" in factor_names:
        interventions.append("📋 Combat procrastination: break tasks into smaller chunks, use time-blocking, set earlier personal deadlines.")

    if "study_overload" in factor_names or "study_hours" in factor_names:
        interventions.append("⏰ Adjust workload: take regular breaks (Pomodoro technique), limit study sessions to 2-hour blocks.")

    if "exercise_hours" in factor_names:
        interventions.append("🏃 Increase physical activity: even 30 minutes of exercise can significantly reduce stress.")

    if "has_negative_emotion" in factor_names or "sentiment_polarity" in factor_names:
        interventions.append("💭 Emotional wellbeing: journal your thoughts, practice gratitude, or talk to a trusted person.")

    if "screen_time" in factor_names or "screen_study_ratio" in factor_names:
        interventions.append("📵 Reduce screen time: set device-free periods, use app timers, take screen breaks every 30 mins.")

    if risk_level == "Low" and not interventions:
        interventions.append("✅ Keep up the good work! Maintain your current healthy habits.")

    return interventions


def run_explanations():
    """Generate and save all explanations."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 50)
    print("  GENERATING BURNOUT EXPLANATIONS (SHAP)")
    print("=" * 50)

    print("\n[1/4] Loading model and data...")
    model, X, feature_cols, full_df = load_model_and_data()
    print(f"  Data shape: {X.shape}")

    print("\n[2/4] Computing SHAP values...")
    explainer, shap_values = compute_shap_values(model, X, feature_cols)

    print("\n[3/4] Computing global feature importance...")
    global_importance = get_global_feature_importance(shap_values, feature_cols)
    for item in global_importance[:10]:
        print(f"  {item['feature']}: {item['importance']:.4f}")

    with open(os.path.join(OUTPUT_DIR, "global_importance.json"), "w") as f:
        json.dump(global_importance, f, indent=2)

    print("\n[4/4] Generating per-student explanations...")
    student_ids = full_df["student_id"].unique()
    explanations = {}

    for student_id in student_ids:
        student_mask = full_df["student_id"] == student_id
        student_indices = full_df[student_mask].index

        # Use the last week record for each student
        last_idx = student_indices[-1]
        X_row = X.iloc[last_idx]

        if isinstance(shap_values, list):
            sv_row = [sv[last_idx] for sv in shap_values]
        else:
            sv_row = shap_values[last_idx]

        explanation = explain_student(model, explainer, X_row, feature_cols, sv_row)
        explanation["interventions"] = generate_intervention(
            explanation["risk_level"], explanation["top_factors"]
        )
        explanation["student_id"] = student_id
        explanations[student_id] = explanation

    with open(os.path.join(OUTPUT_DIR, "student_explanations.json"), "w") as f:
        json.dump(explanations, f, indent=2, default=str)

    print(f"  Saved explanations for {len(explanations)} students to {OUTPUT_DIR}")

    # Stats
    risk_dist = {}
    for exp in explanations.values():
        rl = exp["risk_level"]
        risk_dist[rl] = risk_dist.get(rl, 0) + 1
    print(f"  Risk distribution: {risk_dist}")

    return global_importance, explanations


if __name__ == "__main__":
    run_explanations()
