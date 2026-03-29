"""
Flask Dashboard Application
Serves the burnout prediction dashboard and API endpoints.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
import joblib
from flask import Flask, render_template, jsonify, request
from flask_cors import CORS

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.preprocessing import load_data, clean_data
from pipeline.sentiment import add_sentiment_features, extract_sentiment, detect_emotions
from pipeline.feature_engineering import engineer_features, get_feature_columns

app = Flask(__name__)
CORS(app)

LABEL_NAMES = {0: "Low", 1: "Medium", 2: "High"}
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "saved")
EXPLANATION_DIR = os.path.join(PROJECT_ROOT, "models", "explanations")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "student_survey.csv")

_best_model, _rf_model, _explainer, _feature_cols = None, None, None, None

def _load_all():
    global _best_model, _rf_model, _explainer, _feature_cols
    if _best_model is None:
        import shap
        _best_model = joblib.load(os.path.join(MODEL_DIR, "best_model.pkl"))
        _rf_model = joblib.load(os.path.join(MODEL_DIR, "rf_model.pkl"))
        _explainer = shap.TreeExplainer(_rf_model)
        _feature_cols = joblib.load(os.path.join(MODEL_DIR, "feature_columns.pkl"))

def get_model():
    """Load the best model."""
    _load_all()
    return _best_model

def get_feature_cols():
    """Load saved feature column names."""
    _load_all()
    return _feature_cols

def get_explainer():
    _load_all()
    return _explainer

def get_rf_model():
    _load_all()
    return _rf_model

def get_processed_data():
    """Load and process the full dataset."""
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = add_sentiment_features(df)
    df = engineer_features(df)
    return df


@app.route("/")
def index():
    """Serve admin dashboard."""
    return render_template("admin.html")


@app.route("/student")
def student_portal():
    """Serve student portal."""
    return render_template("student.html")


@app.route("/api/overview")
def api_overview():
    """Return aggregated risk distribution."""
    try:
        explanations_path = os.path.join(EXPLANATION_DIR, "student_explanations.json")
        with open(explanations_path) as f:
            explanations = json.load(f)

        risk_counts = {"Low": 0, "Medium": 0, "High": 0}
        for exp in explanations.values():
            risk_counts[exp["risk_level"]] += 1

        total = sum(risk_counts.values())
        return jsonify({
            "risk_counts": risk_counts,
            "total_students": total,
            "percentages": {k: round(v / total * 100, 1) for k, v in risk_counts.items()},
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/student/<student_id>")
def api_student(student_id):
    """Return individual student risk and explanation."""
    try:
        explanations_path = os.path.join(EXPLANATION_DIR, "student_explanations.json")
        with open(explanations_path) as f:
            explanations = json.load(f)

        if student_id not in explanations:
            return jsonify({"error": f"Student {student_id} not found"}), 404

        return jsonify(explanations[student_id])
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/students")
def api_students():
    """Return list of all student IDs with their risk levels."""
    try:
        explanations_path = os.path.join(EXPLANATION_DIR, "student_explanations.json")
        with open(explanations_path) as f:
            explanations = json.load(f)

        students = [
            {"student_id": sid, "risk_level": exp["risk_level"]}
            for sid, exp in explanations.items()
        ]
        students.sort(key=lambda x: {"High": 0, "Medium": 1, "Low": 2}[x["risk_level"]])

        return jsonify({"students": students})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/trends")
def api_trends():
    """Return longitudinal risk trends across weeks."""
    try:
        df = get_processed_data()
        model = get_model()
        feature_cols = get_feature_cols()
        available_cols = [c for c in feature_cols if c in df.columns]

        weekly_trends = {}
        for week in sorted(df["week"].unique()):
            week_data = df[df["week"] == week]
            X = week_data[available_cols].values
            preds = model.predict(X)

            counts = {0: 0, 1: 0, 2: 0}
            for p in preds:
                counts[p] += 1
            total = len(preds)

            weekly_trends[int(week)] = {
                "Low": round(counts[0] / total * 100, 1),
                "Medium": round(counts[1] / total * 100, 1),
                "High": round(counts[2] / total * 100, 1),
            }

        return jsonify({"trends": weekly_trends})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/features")
def api_features():
    """Return global SHAP feature importance."""
    try:
        importance_path = os.path.join(EXPLANATION_DIR, "global_importance.json")
        with open(importance_path) as f:
            importance = json.load(f)

        return jsonify({"features": importance})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/api/predict", methods=["POST"])
def api_predict():
    """Predict burnout from new survey input."""
    try:
        data = request.json

        model = get_model()
        feature_cols = get_feature_cols()

        # Build a single-row DataFrame from input
        row = {
            "sleep_hours": float(data.get("sleep_hours", 7)),
            "study_hours": float(data.get("study_hours", 5)),
            "late_submissions": int(data.get("late_submissions", 0)),
            "stress_level": int(data.get("stress_level", 5)),
            "social_activity_freq": int(data.get("social_activity_freq", 3)),
            "exercise_hours": float(data.get("exercise_hours", 2)),
            "screen_time": float(data.get("screen_time", 6)),
            "emotional_state": data.get("emotional_state", "Feeling okay"),
        }

        df_row = pd.DataFrame([row])

        # Add sentiment features
        df_row = add_sentiment_features(df_row)

        # Add placeholder engineered features (single-row defaults)
        df_row["sleep_irregularity"] = 0.0
        df_row["procrastination_score"] = row["late_submissions"] / 8
        df_row["negative_sentiment_trend"] = -df_row["sentiment_polarity"].values[0]
        df_row["stress_sleep_interaction"] = row["stress_level"] * max(0, 8 - row["sleep_hours"])
        df_row["social_isolation_score"] = (7 - row["social_activity_freq"]) / 7
        df_row["study_overload"] = int(row["study_hours"] > 10)
        study_safe = max(row["study_hours"], 0.5)
        df_row["screen_study_ratio"] = row["screen_time"] / study_safe

        available_cols = [c for c in feature_cols if c in df_row.columns]
        X = df_row[available_cols].values

        model = get_model()
        prediction = int(model.predict(X)[0])
        risk_level = LABEL_NAMES[prediction]

        # Use SHAP explanation
        from models.explainability import explain_student
        explainer = get_explainer()
        
        # We need to get SHAP values for this single row.
        # TreeExplainer.shap_values(X) returns a shape of (1, n_features, n_classes) for RF multi-class.
        shap_values = explainer.shap_values(X)
        shap_values_row = shap_values[0] # The single row array of shape (n_features, n_classes)

        explanation = explain_student(get_rf_model(), explainer, df_row[available_cols].iloc[0], available_cols, shap_values_row)
        top_factors = explanation["top_factors"]

        # Generate interventions
        from models.explainability import generate_intervention
        interventions = generate_intervention(risk_level, top_factors)

        return jsonify({
            "prediction": prediction,
            "risk_level": risk_level,
            "top_factors": top_factors,
            "interventions": interventions,
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
