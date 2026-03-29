# Student Burnout Prediction System — Methodology & Ethics

## Overview

This system predicts student burnout risk using micro-behavioural indicators collected from self-reported surveys. It combines classical machine learning with NLP sentiment analysis and explainable AI (SHAP) to provide transparent, actionable insights.

## Data Methodology

### Data Collection
- **Source**: Self-reported student surveys collected weekly over 12 weeks.
- **Variables**: Sleep hours, study hours, late submissions, stress level (1-10), social activity frequency, emotional state (free text), exercise hours, screen time.
- **Frequency**: Weekly longitudinal data enabling trend analysis.

### Feature Engineering
18 features are derived from raw survey responses:

| Category | Features |
|---|---|
| **Raw Inputs** | sleep_hours, study_hours, late_submissions, stress_level, social_activity_freq, exercise_hours, screen_time |
| **NLP-Derived** | sentiment_polarity, sentiment_subjectivity, has_negative_emotion, has_positive_emotion |
| **Engineered** | sleep_irregularity, procrastination_score, negative_sentiment_trend, stress_sleep_interaction, social_isolation_score, study_overload, screen_study_ratio |

### Burnout Classification
Three-tier risk classification:
- **Low (0)**: Healthy behaviours, good stress management
- **Medium (1)**: Some concerning patterns, early intervention recommended
- **High (2)**: Significant burnout indicators, immediate support needed

## Model Architecture

### Classical Models (Primary)
- **Random Forest**: Ensemble of decision trees with hyperparameter tuning via GridSearchCV. Provides robust predictions and native feature importance.
- **Gradient Boosting**: Sequential ensemble method that corrects errors iteratively. Often achieves highest accuracy on tabular data.

Both models use:
- 5-fold stratified cross-validation
- Balanced class weights to handle class imbalance
- Macro-averaged F1-score as primary optimization metric

### Sequential Model (LSTM)
- 2-layer LSTM network processes 12-week behavioural sequences
- Captures temporal patterns (e.g., declining sleep over weeks)
- Early stopping with patience=10 to prevent overfitting

## Explainability (SHAP)

We use **SHAP (SHapley Additive exPlanations)** to ensure every prediction is transparent:
- **Global importance**: Which features matter most across all students
- **Per-student explanations**: Which specific behaviours drive each individual's risk score
- **Human-readable descriptions**: Each factor includes a plain-language explanation

## Evaluation Metrics

| Metric | Purpose |
|---|---|
| **Accuracy** | Overall correctness (target: >80%) |
| **Precision** | How many predicted "High Risk" are actually high risk |
| **Recall** | How many actual "High Risk" students are caught |
| **F1-Score (Macro)** | Balanced measure across all three risk classes |
| **Confusion Matrix** | Detailed error analysis per class |

## Ethical Considerations

### Privacy & Anonymization
- All student identifiers are anonymized UUIDs (e.g., `STU-A1B2C3D4`)
- No personally identifiable information (PII) is stored or processed
- The system is designed for **privacy-by-design** — data minimization principles apply

### Consent & Transparency
- Students must opt-in to data collection
- All predictions include explanations — no "black box" decisions
- Students can access their own risk assessments and understand contributing factors

### Bias Mitigation
- Balanced class weights prevent models from ignoring minority risk classes
- Stratified splits ensure fair representation in training/testing
- Features are behaviour-based, not demographic, reducing demographic bias

### Responsible Use
- Predictions are **advisory**, not deterministic — no automated punitive actions
- System recommends supportive interventions (counseling, study strategies), not penalties
- Risk scores should be validated by qualified counselors before intervention

### Data Retention
- Survey data should be retained only for the active academic term
- Longitudinal data beyond one semester requires renewed consent
- Students have the right to request data deletion

## Limitations

1. **Self-reported data**: Responses may be subject to social desirability bias
2. **Synthetic training data**: The current model is trained on synthetic data; real-world validation is needed
3. **Text analysis**: TextBlob sentiment analysis is basic; a fine-tuned model would improve accuracy on student-specific language
4. **Cultural variation**: Burnout indicators may vary across cultural contexts
5. **Temporal lag**: Weekly surveys may miss acute episodes between collection points

## Future Work

- Integration with LMS data (assignment submissions, login patterns) for passive monitoring
- Fine-tuned language model for student emotional text analysis
- Multi-institutional validation study
- Mobile app for real-time micro-check-ins
- Integration with campus counseling referral systems
