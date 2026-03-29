"""
LSTM Model Training for Sequential Behaviour Data
Trains an LSTM network on per-student weekly sequences to predict burnout.
"""

import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from pipeline.preprocessing import load_data, clean_data
from pipeline.sentiment import add_sentiment_features
from pipeline.feature_engineering import engineer_features, get_feature_columns


def prepare_sequences(csv_path, seq_length=12):
    """Reshape student data into sequences for LSTM.

    Each student gets a sequence of `seq_length` weekly records.

    Returns:
        Tuple of (X_sequences, y_labels, feature_columns)
        X shape: (n_students, seq_length, n_features)
        y shape: (n_students,) — label is the LAST week's burnout label
    """
    df = load_data(csv_path)
    df = clean_data(df)
    df = add_sentiment_features(df)
    df = engineer_features(df)

    feature_cols = get_feature_columns()
    available_cols = [c for c in feature_cols if c in df.columns]

    # Normalize features
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    df[available_cols] = scaler.fit_transform(df[available_cols])

    sequences = []
    labels = []

    for student_id, group in df.groupby("student_id"):
        group = group.sort_values("week")

        if len(group) < seq_length:
            # Pad with repeat of first row
            pad_rows = seq_length - len(group)
            padding = pd.concat([group.iloc[:1]] * pad_rows, ignore_index=True)
            group = pd.concat([padding, group], ignore_index=True)

        seq = group[available_cols].values[-seq_length:]
        label = group["burnout_label"].values[-1]

        sequences.append(seq)
        labels.append(label)

    X = np.array(sequences)
    y = np.array(labels)

    return X, y, available_cols


def build_lstm_model(seq_length, n_features, n_classes=3):
    """Build a 2-layer LSTM model."""
    import tensorflow as tf
    from tensorflow.keras import layers, models

    model = models.Sequential([
        layers.LSTM(64, return_sequences=True, input_shape=(seq_length, n_features)),
        layers.Dropout(0.3),
        layers.LSTM(32),
        layers.Dropout(0.3),
        layers.Dense(32, activation="relu"),
        layers.Dense(n_classes, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def train_lstm():
    """Full LSTM training pipeline."""
    csv_path = os.path.join(PROJECT_ROOT, "data", "student_survey.csv")

    if not os.path.exists(csv_path):
        print("Dataset not found. Generating synthetic data...")
        from data.generate_data import generate_student_data
        df = generate_student_data()
        os.makedirs(os.path.dirname(csv_path), exist_ok=True)
        df.to_csv(csv_path, index=False)

    print("=" * 50)
    print("  STUDENT BURNOUT — LSTM TRAINING")
    print("=" * 50)

    print("\n[1/4] Preparing sequences...")
    X, y, feature_cols = prepare_sequences(csv_path, seq_length=12)
    print(f"  Sequences shape: {X.shape}")
    print(f"  Labels shape: {y.shape}")
    print(f"  Class distribution: {dict(zip(*np.unique(y, return_counts=True)))}")

    print("\n[2/4] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"  Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")

    print("\n[3/4] Building and training LSTM...")
    import tensorflow as tf
    tf.random.set_seed(42)

    model = build_lstm_model(
        seq_length=X.shape[1],
        n_features=X.shape[2],
        n_classes=3
    )
    model.summary()

    callbacks = [
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=5
        ),
    ]

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1,
    )

    print("\n[4/4] Evaluating LSTM...")
    y_pred = np.argmax(model.predict(X_test), axis=1)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n  LSTM Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, target_names=["Low", "Medium", "High"]))

    # Save model
    model_dir = os.path.join(PROJECT_ROOT, "models", "saved")
    os.makedirs(model_dir, exist_ok=True)
    model.save(os.path.join(model_dir, "lstm_model.keras"))
    print(f"\n  LSTM model saved to: {model_dir}/lstm_model.keras")

    return model, history, acc


if __name__ == "__main__":
    train_lstm()
