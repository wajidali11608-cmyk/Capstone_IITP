"""
Synthetic Student Survey Data Generator
Generates realistic longitudinal data for 200 students over 12 weeks.
"""

import pandas as pd
import numpy as np
import uuid
import os
import random

EMOTIONAL_STATES = {
    "positive": [
        "Feeling great and motivated today",
        "Had a productive day, feeling accomplished",
        "Excited about upcoming projects",
        "Feeling calm and focused",
        "Good energy, looking forward to classes",
        "Feeling confident about exams",
        "Happy and well-rested",
        "Motivated to study and stay on track",
        "Enjoying the learning process",
        "Feeling balanced and in control",
    ],
    "neutral": [
        "Feeling okay, nothing special",
        "Average day, going through the motions",
        "Neither good nor bad, just normal",
        "Doing alright, could be better",
        "Feeling indifferent about things",
        "Just getting by day to day",
        "Things are manageable but not exciting",
        "Routine day, nothing remarkable",
    ],
    "negative": [
        "Feeling overwhelmed with assignments",
        "Very stressed about deadlines",
        "Can't sleep well, too much anxiety",
        "Exhausted and burned out from studying",
        "Feeling isolated and lonely",
        "Struggling to concentrate on anything",
        "Feeling hopeless about my grades",
        "Constantly tired no matter how much I sleep",
        "Dreading going to class",
        "Feeling anxious about everything",
        "Too much pressure from all sides",
        "Breaking down emotionally, can't handle it",
        "Lost all motivation to study",
        "Feeling numb and disconnected",
    ],
}


def generate_student_data(n_students=200, n_weeks=12, seed=42):
    """Generate synthetic student survey data with realistic burnout patterns."""
    np.random.seed(seed)
    random.seed(seed)

    records = []

    for i in range(n_students):
        hex_id = uuid.uuid4().hex
        student_id = f"STU-{hex_id[:8].upper()}"

        # Assign a burnout trajectory type
        trajectory = np.random.choice(
            ["stable_low", "stable_medium", "escalating", "high_burnout", "recovering"],
            p=[0.25, 0.20, 0.25, 0.20, 0.10],
        )

        # Base parameters per trajectory
        if trajectory == "stable_low":
            base_sleep = np.random.uniform(7, 9)
            base_study = np.random.uniform(3, 6)
            base_stress = np.random.uniform(1, 4)
            base_social = np.random.randint(4, 7)
            base_late = np.random.uniform(0, 1)
            base_exercise = np.random.uniform(3, 6)
            base_screen = np.random.uniform(3, 7)
            emotion_pool = "positive"
        elif trajectory == "stable_medium":
            base_sleep = np.random.uniform(5.5, 7.5)
            base_study = np.random.uniform(5, 9)
            base_stress = np.random.uniform(4, 6.5)
            base_social = np.random.randint(2, 5)
            base_late = np.random.uniform(1, 3)
            base_exercise = np.random.uniform(1.5, 4)
            base_screen = np.random.uniform(6, 10)
            emotion_pool = "neutral"
        elif trajectory == "escalating":
            base_sleep = np.random.uniform(6, 8)
            base_study = np.random.uniform(4, 7)
            base_stress = np.random.uniform(3, 5)
            base_social = np.random.randint(3, 6)
            base_late = np.random.uniform(0.5, 2)
            base_exercise = np.random.uniform(2, 5)
            base_screen = np.random.uniform(5, 8)
            emotion_pool = "neutral"
        elif trajectory == "high_burnout":
            base_sleep = np.random.uniform(3.5, 5.5)
            base_study = np.random.uniform(8, 14)
            base_stress = np.random.uniform(7, 10)
            base_social = np.random.randint(0, 2)
            base_late = np.random.uniform(3, 7)
            base_exercise = np.random.uniform(0, 1.5)
            base_screen = np.random.uniform(10, 15)
            emotion_pool = "negative"
        else:  # recovering
            base_sleep = np.random.uniform(4.5, 6)
            base_study = np.random.uniform(7, 11)
            base_stress = np.random.uniform(6, 8.5)
            base_social = np.random.randint(1, 3)
            base_late = np.random.uniform(2, 5)
            base_exercise = np.random.uniform(0.5, 2)
            base_screen = np.random.uniform(8, 12)
            emotion_pool = "negative"

        for week in range(1, n_weeks + 1):
            progress = week / n_weeks

            # Apply trajectory-specific modifications
            if trajectory == "escalating":
                stress_mod = progress * 4
                sleep_mod = -progress * 2.5
                late_mod = progress * 3
                social_mod = -progress * 3
                exercise_mod = -progress * 2
                screen_mod = progress * 3
                if progress > 0.5:
                    emotion_pool = "negative"
            elif trajectory == "recovering":
                stress_mod = -progress * 3
                sleep_mod = progress * 2
                late_mod = -progress * 2
                social_mod = progress * 2
                exercise_mod = progress * 2
                screen_mod = -progress * 2
                if progress > 0.5:
                    emotion_pool = "neutral"
                if progress > 0.75:
                    emotion_pool = "positive"
            else:
                stress_mod = 0
                sleep_mod = 0
                late_mod = 0
                social_mod = 0
                exercise_mod = 0
                screen_mod = 0

            # Add weekly noise
            noise = lambda scale=0.5: np.random.normal(0, scale)

            sleep_hours = np.clip(base_sleep + sleep_mod + noise(0.6), 3, 10)
            study_hours = np.clip(base_study + noise(1.0), 0, 14)
            stress_level = int(np.clip(base_stress + stress_mod + noise(0.8), 1, 10))
            social_activity = int(np.clip(base_social + social_mod + noise(0.7), 0, 7))
            late_submissions = int(np.clip(base_late + late_mod + noise(0.5), 0, 8))
            exercise_hours = np.clip(base_exercise + exercise_mod + noise(0.4), 0, 7)
            screen_time = np.clip(base_screen + screen_mod + noise(0.8), 2, 16)

            # Select emotional state text
            if emotion_pool == "negative":
                text = random.choice(EMOTIONAL_STATES["negative"])
            elif emotion_pool == "positive":
                text = random.choice(EMOTIONAL_STATES["positive"])
            else:
                pool = random.choices(["positive", "neutral", "negative"], weights=[0.2, 0.6, 0.2])[0]
                text = random.choice(EMOTIONAL_STATES[pool])

            # Compute burnout score
            burnout_score = (
                stress_level * 0.25
                + max(0, 8 - sleep_hours) * 0.15
                + late_submissions * 0.15
                + max(0, 5 - social_activity) * 0.10
                + max(0, study_hours - 8) * 0.10
                + max(0, 3 - exercise_hours) * 0.10
                + max(0, screen_time - 8) * 0.05
                + (1 if emotion_pool == "negative" else 0) * 0.10
            )

            if burnout_score < 1.8:
                burnout_label = 0  # Low
            elif burnout_score < 3.0:
                burnout_label = 1  # Medium
            else:
                burnout_label = 2  # High

            # Randomly introduce ~3% missing values
            if np.random.random() < 0.03:
                sleep_hours = np.nan
            if np.random.random() < 0.03:
                exercise_hours = np.nan

            records.append(
                {
                    "student_id": student_id,
                    "week": week,
                    "sleep_hours": round(sleep_hours, 1) if not np.isnan(sleep_hours) else np.nan,
                    "study_hours": round(study_hours, 1),
                    "late_submissions": late_submissions,
                    "stress_level": stress_level,
                    "social_activity_freq": social_activity,
                    "emotional_state": text,
                    "exercise_hours": round(exercise_hours, 1) if not np.isnan(exercise_hours) else np.nan,
                    "screen_time": round(screen_time, 1),
                    "burnout_label": burnout_label,
                }
            )

    df = pd.DataFrame(records)
    return df


if __name__ == "__main__":
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "student_survey.csv")

    df = generate_student_data(n_students=200, n_weeks=12)
    df.to_csv(output_path, index=False)

    print(f"Generated {len(df)} records for {df['student_id'].nunique()} students")
    print(f"Saved to: {output_path}")
    print(f"\nBurnout distribution:")
    labels = {0: "Low", 1: "Medium", 2: "High"}
    for label, count in df["burnout_label"].value_counts().sort_index().items():
        pct = count / len(df) * 100
        print(f"  {labels[label]}: {count} ({pct:.1f}%)")
