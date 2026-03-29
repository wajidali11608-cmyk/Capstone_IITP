#!/bin/bash
# Student Burnout Prediction System — Launcher
# Generates data, trains models, generates SHAP explanations, and starts dashboard.

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

# Activate virtual environment
if [ -d ".venv" ]; then
    source .venv/bin/activate
fi

echo "=============================================="
echo "  Student Burnout Prediction System"
echo "=============================================="

# Step 1: Generate synthetic data
echo ""
echo "[1/4] Generating synthetic student data..."
python data/generate_data.py

# Step 2: Train classical models
echo ""
echo "[2/4] Training ML models (Random Forest + Gradient Boosting)..."
python models/train_classical.py

# Step 3: Generate SHAP explanations
echo ""
echo "[3/4] Generating SHAP explanations..."
python models/explainability.py

# Step 4: Launch dashboard
echo ""
echo "[4/4] Starting dashboard server..."
echo "  → Admin Dashboard:   http://localhost:5000/"
echo "  → Student Portal:    http://localhost:5000/student"
echo "=============================================="
python dashboard/app.py
