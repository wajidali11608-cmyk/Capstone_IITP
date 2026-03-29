"""
NLP Sentiment Analysis for Emotional State Text Responses
Uses TextBlob for sentiment polarity and subjectivity extraction,
plus keyword-based emotion tagging.
"""

from textblob import TextBlob
import pandas as pd
import re


# Emotion keyword dictionaries
EMOTION_KEYWORDS = {
    "anxious": ["anxious", "anxiety", "nervous", "worried", "panic", "dread"],
    "exhausted": ["exhausted", "tired", "fatigue", "burned out", "burnt out", "drained", "numb"],
    "stressed": ["stressed", "stress", "pressure", "overwhelmed", "overloaded"],
    "lonely": ["lonely", "isolated", "alone", "disconnected", "no friends"],
    "hopeless": ["hopeless", "hopeless", "giving up", "can't handle", "breaking down", "lost"],
    "motivated": ["motivated", "excited", "energized", "productive", "accomplished", "confident"],
    "calm": ["calm", "relaxed", "peaceful", "balanced", "focused", "in control"],
    "happy": ["happy", "great", "good", "enjoying", "looking forward"],
}


def extract_sentiment(text):
    """Extract sentiment polarity and subjectivity from text.

    Returns:
        Tuple of (polarity, subjectivity).
        polarity: -1 (negative) to 1 (positive)
        subjectivity: 0 (objective) to 1 (subjective)
    """
    if not isinstance(text, str) or text.strip() == "":
        return 0.0, 0.0

    blob = TextBlob(text)
    return blob.sentiment.polarity, blob.sentiment.subjectivity


def detect_emotions(text):
    """Detect emotion tags from text using keyword matching.

    Returns:
        List of detected emotion tags.
    """
    if not isinstance(text, str):
        return []

    text_lower = text.lower()
    detected = []

    for emotion, keywords in EMOTION_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                detected.append(emotion)
                break

    return detected


def add_sentiment_features(df):
    """Add sentiment and emotion features to the DataFrame.

    Adds columns:
        - sentiment_polarity: float [-1, 1]
        - sentiment_subjectivity: float [0, 1]
        - emotion_tags: list of detected emotions
        - has_negative_emotion: binary flag
        - has_positive_emotion: binary flag
        - emotion_tag_count: number of detected emotions

    Args:
        df: DataFrame with 'emotional_state' column.

    Returns:
        DataFrame with added sentiment features.
    """
    df = df.copy()

    sentiments = df["emotional_state"].apply(extract_sentiment)
    df["sentiment_polarity"] = sentiments.apply(lambda x: x[0])
    df["sentiment_subjectivity"] = sentiments.apply(lambda x: x[1])

    df["emotion_tags"] = df["emotional_state"].apply(detect_emotions)

    negative_emotions = {"anxious", "exhausted", "stressed", "lonely", "hopeless"}
    positive_emotions = {"motivated", "calm", "happy"}

    df["has_negative_emotion"] = df["emotion_tags"].apply(
        lambda tags: int(any(t in negative_emotions for t in tags))
    )
    df["has_positive_emotion"] = df["emotion_tags"].apply(
        lambda tags: int(any(t in positive_emotions for t in tags))
    )
    df["emotion_tag_count"] = df["emotion_tags"].apply(len)

    return df
