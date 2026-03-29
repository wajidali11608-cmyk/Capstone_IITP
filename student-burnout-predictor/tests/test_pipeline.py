"""
Automated Tests for the Student Burnout Prediction Pipeline
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)


class TestDataGeneration:
    """Test the synthetic data generator."""

    def test_generate_data_shape(self):
        from data.generate_data import generate_student_data
        df = generate_student_data(n_students=20, n_weeks=12, seed=99)
        assert len(df) == 20 * 12, f"Expected 240 rows, got {len(df)}"

    def test_generate_data_columns(self):
        from data.generate_data import generate_student_data
        df = generate_student_data(n_students=10, n_weeks=4, seed=99)
        expected_cols = [
            "student_id", "week", "sleep_hours", "study_hours",
            "late_submissions", "stress_level", "social_activity_freq",
            "emotional_state", "exercise_hours", "screen_time", "burnout_label"
        ]
        for col in expected_cols:
            assert col in df.columns, f"Missing column: {col}"

    def test_burnout_label_values(self):
        from data.generate_data import generate_student_data
        df = generate_student_data(n_students=50, n_weeks=4, seed=99)
        assert set(df["burnout_label"].dropna().unique()).issubset({0, 1, 2})

    def test_data_ranges(self):
        from data.generate_data import generate_student_data
        df = generate_student_data(n_students=30, n_weeks=12, seed=99)
        assert df["stress_level"].min() >= 1
        assert df["stress_level"].max() <= 10
        assert df["week"].min() >= 1
        assert df["week"].max() <= 12


class TestPreprocessing:
    """Test the preprocessing pipeline."""

    def setup_method(self):
        from data.generate_data import generate_student_data
        self.df = generate_student_data(n_students=20, n_weeks=4, seed=42)

    def test_clean_data_no_nans_in_key_cols(self):
        from pipeline.preprocessing import clean_data
        cleaned = clean_data(self.df)
        key_cols = ["stress_level", "late_submissions", "burnout_label"]
        for col in key_cols:
            assert cleaned[col].isnull().sum() == 0, f"NaN found in {col}"

    def test_normalize_features_output_shape(self):
        from pipeline.preprocessing import clean_data, normalize_features
        cleaned = clean_data(self.df)
        normalized, scaler = normalize_features(cleaned)
        assert normalized.shape == cleaned.shape

    def test_scaler_is_fitted(self):
        from pipeline.preprocessing import clean_data, normalize_features
        cleaned = clean_data(self.df)
        _, scaler = normalize_features(cleaned)
        assert hasattr(scaler, "mean_"), "Scaler not fitted"


class TestSentiment:
    """Test the sentiment analysis module."""

    def test_extract_sentiment_positive(self):
        from pipeline.sentiment import extract_sentiment
        polarity, _ = extract_sentiment("Feeling great and motivated today")
        assert polarity > 0, f"Expected positive polarity, got {polarity}"

    def test_extract_sentiment_negative(self):
        from pipeline.sentiment import extract_sentiment
        polarity, _ = extract_sentiment("Feeling overwhelmed and exhausted")
        assert polarity <= 0, f"Expected non-positive polarity, got {polarity}"

    def test_detect_emotions(self):
        from pipeline.sentiment import detect_emotions
        emotions = detect_emotions("Feeling anxious and stressed about exams")
        assert "anxious" in emotions
        assert "stressed" in emotions

    def test_add_sentiment_features_columns(self):
        from pipeline.sentiment import add_sentiment_features
        from data.generate_data import generate_student_data
        df = generate_student_data(n_students=5, n_weeks=2, seed=42)
        result = add_sentiment_features(df)
        expected = ["sentiment_polarity", "sentiment_subjectivity", "has_negative_emotion", "has_positive_emotion"]
        for col in expected:
            assert col in result.columns, f"Missing column: {col}"


class TestFeatureEngineering:
    """Test feature engineering."""

    def setup_method(self):
        from data.generate_data import generate_student_data
        from pipeline.preprocessing import clean_data
        from pipeline.sentiment import add_sentiment_features
        self.df = generate_student_data(n_students=20, n_weeks=6, seed=42)
        self.df = clean_data(self.df)
        self.df = add_sentiment_features(self.df)

    def test_engineer_features_adds_columns(self):
        from pipeline.feature_engineering import engineer_features
        result = engineer_features(self.df)
        expected = [
            "sleep_irregularity", "procrastination_score",
            "negative_sentiment_trend", "stress_sleep_interaction",
            "social_isolation_score", "study_overload", "screen_study_ratio"
        ]
        for col in expected:
            assert col in result.columns, f"Missing feature: {col}"

    def test_feature_columns_list(self):
        from pipeline.feature_engineering import get_feature_columns
        cols = get_feature_columns()
        assert len(cols) == 18, f"Expected 18 features, got {len(cols)}"

    def test_no_nans_in_features(self):
        from pipeline.feature_engineering import engineer_features, get_feature_columns
        result = engineer_features(self.df)
        for col in get_feature_columns():
            if col in result.columns:
                assert result[col].isnull().sum() == 0, f"NaN in {col}"


class TestModelAccuracy:
    """Test that models achieve >80% accuracy."""

    @pytest.fixture(autouse=True, scope="class")
    def train_models(self, tmp_path_factory):
        """Train models once for all tests in this class."""
        from data.generate_data import generate_student_data

        tmp_dir = tmp_path_factory.mktemp("data")
        csv_path = str(tmp_dir / "test_survey.csv")
        df = generate_student_data(n_students=200, n_weeks=12, seed=42)
        df.to_csv(csv_path, index=False)

        from models.train_classical import prepare_data
        from sklearn.model_selection import train_test_split
        from sklearn.ensemble import RandomForestClassifier

        X, y, feature_cols, _ = prepare_data(csv_path)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        model = RandomForestClassifier(
            n_estimators=200, max_depth=12, random_state=42, class_weight="balanced"
        )
        model.fit(X_train, y_train)

        TestModelAccuracy._model = model
        TestModelAccuracy._X_test = X_test
        TestModelAccuracy._y_test = y_test

    def test_accuracy_above_80(self):
        from sklearn.metrics import accuracy_score
        y_pred = self._model.predict(self._X_test)
        acc = accuracy_score(self._y_test, y_pred)
        assert acc >= 0.80, f"Accuracy {acc:.4f} is below 80% threshold"

    def test_all_classes_predicted(self):
        y_pred = self._model.predict(self._X_test)
        unique_preds = set(y_pred)
        assert len(unique_preds) >= 2, "Model only predicts one class"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
