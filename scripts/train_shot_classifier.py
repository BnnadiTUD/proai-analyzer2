import pandas as pd
from pathlib import Path
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score


# -----------------------------
# Config
# -----------------------------
DATA_PATH = Path("features_filled.csv")
MODEL_PATH = Path("shot_classifier.pkl")

FEATURE_COLUMNS = [
    "plant_to_ball_norm",
    "trunk_lean_deg",
    "hip_facing_deg",
    "follow_through_arc_deg",
    "lock_angle_deg",
]

LABEL_COLUMN = "label"



df = pd.read_csv(DATA_PATH)

print("Loaded rows:", len(df))
print(df.head())


def label_shot(row) -> str:
    score = 0

    # Ideal ranges based on my ProAI metric bands
    if 10 <= row["plant_to_ball_norm"] <= 30:
        score += 1

    if -20 <= row["trunk_lean_deg"] <= -5:
        score += 1

    if 60 <= row["hip_facing_deg"] <= 120:
        score += 1

    if 80 <= row["follow_through_arc_deg"] <= 250:
        score += 1

    if 100 <= row["lock_angle_deg"] <= 135:
        score += 1

    return "good" if score >= 3 else "bad"


# Remove rows missing feature values first
df = df.dropna(subset=FEATURE_COLUMNS).copy()

# Create label column automatically
df[LABEL_COLUMN] = df.apply(label_shot, axis=1)

print("\nLabel counts:")
print(df[LABEL_COLUMN].value_counts())


X = df[FEATURE_COLUMNS]
y = df[LABEL_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTrain size:", len(X_train))
print("Test size:", len(X_test))



model = RandomForestClassifier(
    n_estimators=200,
    max_depth=6,
    random_state=42
)

model.fit(X_train, y_train)


y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
print("\nAccuracy:", round(acc, 4))

print("\nClassification report:")
print(classification_report(y_test, y_pred))

print("\nConfusion matrix:")
print(confusion_matrix(y_test, y_pred))



print("\nFeature importances:")
for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
    print(f"{feature}: {importance:.4f}")



joblib.dump(model, MODEL_PATH)
print(f"\nSaved model to: {MODEL_PATH.resolve()}")