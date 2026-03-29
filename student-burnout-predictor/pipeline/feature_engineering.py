"""
Feature Engineering Module
Creates micro-behavioural features from raw and preprocessed student data.
"""

import pandas as pd
import numpy as np


def compute_sleep_irregularity(df):
    """Compute per-student sleep irregularity as std deviation over weeks."""
    sleep_std = df.groupby("student_id")["sleep_hours"].transform("std").fillna(0)
    return sleep_std


def compute_procrastination_score(df):
    """Compute procrastination score from late submissions and study behaviour.

    Higher score = more procrastination.
    """
    # Normalize late_submissions to 0-1 range
    max_late = df["late_submissions"].max()
    if max_late == 0:
        max_late = 1
    norm_late = df["late_submissions"] / max_late

    # Inverse study efficiency (low study + high late = procrastination)
    study_inv = 1 - (df["study_hours"] / df["study_hours"].max()).clip(0, 1)

    return (norm_late * 0.6 + study_inv * 0.4).round(3)


def compute_negative_sentiment_trend(df):
    """Compute rolling average of negative sentiment per student over weeks."""
    df = df.sort_values(["student_id", "week"])

    # Use sentiment_polarity: lower = more negative
    # Invert so negative sentiment gives positive trend values
    df["neg_sentiment"] = -df["sentiment_polarity"]

    rolling_neg = (
        df.groupby("student_id")["neg_sentiment"]
        .transform(lambda x: x.rolling(window=3, min_periods=1).mean())
    )

    return rolling_neg.round(3)


def compute_interaction_features(df):
    """Compute interaction-based features."""
    features = pd.DataFrame(index=df.index)

    # Stress × sleep deficit interaction
    features["stress_sleep_interaction"] = (
        df["stress_level"] * np.maximum(0, 8 - df["sleep_hours"])
    ).round(2)

    # Social isolation score (inverse of social frequency)
    isolation = (7 - df["social_activity_freq"]) / 7
    features["social_isolation_score"] = np.round(isolation, 3)

    # Study overload flag
    features["study_overload"] = (df["study_hours"] > 10).astype(int)

    # Screen to study ratio
    study_safe = df["study_hours"].replace(0, 0.5)
    features["screen_study_ratio"] = (df["screen_time"] / study_safe).round(2)

    return features


def engineer_features(df):
    """Run the full feature engineering pipeline.

    Expects a DataFrame that already has sentiment features
    (sentiment_polarity, has_negative_emotion, etc.).

    Returns:
        DataFrame with all engineered features added.
    """
    df = df.copy()

    df["sleep_irregularity"] = compute_sleep_irregularity(df)
    df["procrastination_score"] = compute_procrastination_score(df)
    df["negative_sentiment_trend"] = compute_negative_sentiment_trend(df)

    interaction_feats = compute_interaction_features(df)
    for col in interaction_feats.columns:
        df[col] = interaction_feats[col]

    # Drop intermediate columns
    if "neg_sentiment" in df.columns:
        df.drop(columns=["neg_sentiment"], inplace=True)

    return df


def get_feature_columns():
    """Return the list of feature column names used for model training."""
    return [
        "sleep_hours",
        "study_hours",
        "late_submissions",
        "stress_level",
        "social_activity_freq",
        "exercise_hours",
        "screen_time",
        "sentiment_polarity",
        "sentiment_subjectivity",
        "has_negative_emotion",
        "has_positive_emotion",
        "sleep_irregularity",
        "procrastination_score",
        "negative_sentiment_trend",
        "stress_sleep_interaction",
        "social_isolation_score",
        "study_overload",
        "screen_study_ratio",
    ]
