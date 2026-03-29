"""
Data Preprocessing Pipeline
Handles cleaning, imputation, normalization, and encoding.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler


def load_data(filepath):
    """Load CSV data and return a DataFrame."""
    df = pd.read_csv(filepath)
    return df


def clean_data(df):
    """Handle missing values and validate data types."""
    df = df.copy()

    # Impute numeric missing values with column median
    numeric_cols = ["sleep_hours", "study_hours", "exercise_hours", "screen_time"]
    for col in numeric_cols:
        if col in df.columns and df[col].isnull().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)

    # Ensure integer columns
    int_cols = ["late_submissions", "stress_level", "social_activity_freq", "burnout_label", "week"]
    for col in int_cols:
        if col in df.columns:
            df[col] = df[col].astype(int)

    # Fill missing emotional state with neutral placeholder
    if "emotional_state" in df.columns:
        df["emotional_state"] = df["emotional_state"].fillna("Feeling okay, nothing special")

    return df


def normalize_features(df, feature_cols=None, scaler=None):
    """Normalize numerical features using StandardScaler.

    Args:
        df: DataFrame with features.
        feature_cols: List of columns to normalize.
        scaler: Pre-fitted scaler (for inference). If None, fits a new one.

    Returns:
        Tuple of (normalized DataFrame, fitted scaler).
    """
    df = df.copy()

    if feature_cols is None:
        feature_cols = [
            "sleep_hours",
            "study_hours",
            "late_submissions",
            "stress_level",
            "social_activity_freq",
            "exercise_hours",
            "screen_time",
        ]

    # Only include columns that exist in the DataFrame
    feature_cols = [c for c in feature_cols if c in df.columns]

    if scaler is None:
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])
    else:
        df[feature_cols] = scaler.transform(df[feature_cols])

    return df, scaler


def preprocess_pipeline(filepath):
    """Run the full preprocessing pipeline.

    Returns:
        Tuple of (cleaned DataFrame, fitted scaler).
    """
    df = load_data(filepath)
    df = clean_data(df)
    df, scaler = normalize_features(df)
    return df, scaler
