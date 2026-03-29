#  Student Burnout Prediction System

AI-powered predictive pipeline that uses micro-behavioural data to detect and prevent student burnout.

## Features

- **Multi-model ML predictions** — Random Forest, Gradient Boosting, and LSTM
- **NLP sentiment analysis** — TextBlob-based emotion detection from free-text responses
- **18 engineered features** — Sleep irregularity, procrastination score, stress-sleep interaction, and more
- **SHAP explainability** — Transparent per-student explanations for every prediction
- **Interactive dashboard** — Premium dark-theme UI with Chart.js visualizations
- **Adaptive interventions** — Context-aware recommendations based on individual risk factors

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Run the full pipeline (generate data → train → explain → dashboard)
bash run.sh
```

Then open **http://localhost:5000** in your browser.

## Project Structure

```
student-burnout-predictor/
├── data/                    # Data generation and storage
├── pipeline/                # Preprocessing, NLP, feature engineering
├── models/                  # ML training, LSTM, SHAP explainability
├── dashboard/               # Flask app + HTML/CSS/JS
├── tests/                   # Automated test suite
├── docs/                    # Methodology & ethics documentation
├── run.sh                   # One-click launcher
└── requirements.txt         # Python dependencies
```

## Models

| Model | Type | Purpose |
|---|---|---|
| Random Forest | Classical ML | Primary predictor, interpretable |
| Gradient Boosting | Classical ML | Alternative predictor, often highest accuracy |
| LSTM | Deep Learning | Captures temporal patterns across weeks |

## Running Tests

```bash
python -m pytest tests/test_pipeline.py -v
```

## Documentation

See [docs/methodology.md](docs/methodology.md) for full methodology, evaluation metrics, and ethical considerations.
