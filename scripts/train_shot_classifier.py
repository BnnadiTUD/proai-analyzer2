import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURE_COLUMNS = [
    "plant_to_ball_norm",
    "trunk_lean_deg",
    "hip_facing_deg",
    "follow_through_arc_deg",
    "lock_angle_deg",
]

MANUAL_METRIC_COLUMNS = [
    "human_plant_to_ball_norm",
    "human_trunk_lean_deg",
    "human_hip_facing_deg",
    "human_follow_through_arc_deg",
    "human_lock_angle_deg",
]

DEFAULT_DATASET_PATH = Path("training_data_calibrated.csv")
DEFAULT_MODEL_PATH = Path("shot_classifier.pkl")
DEFAULT_METADATA_PATH = Path("shot_classifier_metadata.json")

SHOT_HEADER_RE = re.compile(r"^##\s*(shot_\d+)\b(.*)$", re.IGNORECASE)
RATING_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the shot classifier from calibrated analysis outputs."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        required=True,
        help="Path to results JSON for the shots you want to train on.",
    )
    parser.add_argument(
        "--notes-path",
        type=Path,
        help="Optional path to manual-rating notes Markdown.",
    )
    parser.add_argument(
        "--dataset-out",
        type=Path,
        default=DEFAULT_DATASET_PATH,
        help="Where to save the assembled training dataset CSV.",
    )
    parser.add_argument(
        "--model-out",
        type=Path,
        default=DEFAULT_MODEL_PATH,
        help="Where to save the trained classifier.",
    )
    parser.add_argument(
        "--metadata-out",
        type=Path,
        default=DEFAULT_METADATA_PATH,
        help="Where to save training metadata JSON.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=70.0,
        help="Fallback overall-score threshold for good/bad labels when no manual notes exist.",
    )
    parser.add_argument(
        "--manual-threshold",
        type=float,
        default=7.0,
        help="Average manual rating threshold for good/bad labels.",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Held-out test split ratio when there is enough data.",
    )
    return parser.parse_args()


def parse_manual_notes(notes_path: Path):
    if not notes_path or not notes_path.exists():
        return {}

    lines = notes_path.read_text(encoding="utf-8").splitlines()
    parsed = {}

    index = 0
    while index < len(lines):
        header_match = SHOT_HEADER_RE.match(lines[index].strip())
        if not header_match:
            index += 1
            continue

        shot_id = header_match.group(1)
        search_lines = [header_match.group(2).strip()]

        look_ahead = index + 1
        while look_ahead < len(lines):
            stripped = lines[look_ahead].strip()
            if stripped.startswith("## "):
                break
            if stripped:
                search_lines.append(stripped)
            look_ahead += 1

        rating_line = None
        rating_values = None
        for candidate in search_lines:
            matches = [float(value) for value in RATING_RE.findall(candidate)]
            if len(matches) >= 5:
                rating_line = candidate
                rating_values = matches[:5]
                break

        if rating_values:
            summary = ""
            if rating_line:
                trailing = RATING_RE.sub("", rating_line, count=5)
                summary = trailing.replace(",", " ").strip(" -,:")

            if not summary:
                summary_lines = []
                for candidate in search_lines:
                    if candidate == rating_line:
                        continue
                    if len(RATING_RE.findall(candidate)) >= 5:
                        continue
                    if candidate.startswith("- Video:") or candidate.startswith("- Overall score:"):
                        continue
                    if candidate.startswith("- Weakest metric:") or candidate == "- Metrics:":
                        continue
                    if candidate.startswith("- Summary:") or candidate.startswith("- "):
                        continue
                    summary_lines.append(candidate)
                summary = " ".join(summary_lines).strip()

            parsed[shot_id] = {
                "ratings": rating_values,
                "summary": summary,
            }

        index = look_ahead

    return parsed


def load_results(results_path: Path):
    data = json.loads(results_path.read_text(encoding="utf-8"))
    if not isinstance(data, list):
        raise ValueError("Results JSON must contain a list of analyzed shots.")
    return data


def classify_overall_score(overall_score, threshold):
    if overall_score is None:
        return None
    return "good" if float(overall_score) >= threshold else "bad"


def classify_manual_average(avg_rating, threshold):
    return "good" if float(avg_rating) >= threshold else "bad"


def build_dataset(results, manual_notes, manual_threshold, score_threshold):
    rows = []

    for item in results:
        shot_id = item.get("shot_id")
        metrics = item.get("metrics") or {}
        scores = item.get("scores") or {}

        row = {
            "shot_id": shot_id,
            "video_path": item.get("video_path"),
            "score_overall": scores.get("overall"),
            "label_source": "calibrated_score",
            "manual_summary": None,
        }

        for feature in FEATURE_COLUMNS:
            row[feature] = metrics.get(feature)

        manual_entry = manual_notes.get(shot_id)
        if manual_entry:
            ratings = manual_entry["ratings"]
            for column, rating in zip(MANUAL_METRIC_COLUMNS, ratings):
                row[column] = rating
            row["manual_avg_rating"] = round(sum(ratings) / len(ratings), 4)
            row["label"] = classify_manual_average(row["manual_avg_rating"], manual_threshold)
            row["label_source"] = "manual_notes"
            row["manual_summary"] = manual_entry.get("summary") or None
        else:
            for column in MANUAL_METRIC_COLUMNS:
                row[column] = None
            row["manual_avg_rating"] = None
            row["label"] = classify_overall_score(row["score_overall"], score_threshold)

        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
    return df


def choose_test_split(df, requested_test_size):
    label_counts = df["label"].value_counts()
    if len(label_counts) < 2:
        return None

    min_class_count = int(label_counts.min())
    if min_class_count < 2 or len(df) < 10:
        return None

    max_test_size = max(1, min_class_count - 1) / len(df)
    safe_test_size = min(requested_test_size, max_test_size)
    safe_test_size = max(safe_test_size, 1 / len(df))
    if safe_test_size >= 1.0:
        return None

    return safe_test_size


def train_model(df, test_size):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    )

    actual_test_size = choose_test_split(df, test_size)
    evaluation = None

    if actual_test_size is None:
        model.fit(X, y)
        return model, evaluation

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=actual_test_size,
        random_state=42,
        stratify=y,
    )

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    evaluation = {
        "test_size": actual_test_size,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }
    return model, evaluation


def save_metadata(df, evaluation, metadata_path):
    metadata = {
        "feature_columns": FEATURE_COLUMNS,
        "rows_used": int(len(df)),
        "label_counts": {key: int(value) for key, value in df["label"].value_counts().items()},
        "label_source_counts": {
            key: int(value) for key, value in df["label_source"].value_counts().items()
        },
        "manual_note_rows": int((df["label_source"] == "manual_notes").sum()),
        "fallback_rows": int((df["label_source"] == "calibrated_score").sum()),
        "evaluation": evaluation,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main():
    args = parse_args()

    if not args.results_path.exists():
        raise FileNotFoundError(f"Results file not found: {args.results_path}")

    manual_notes = parse_manual_notes(args.notes_path)
    results = load_results(args.results_path)
    df = build_dataset(
        results=results,
        manual_notes=manual_notes,
        manual_threshold=args.manual_threshold,
        score_threshold=args.score_threshold,
    )

    if df.empty:
        raise ValueError(
            "No trainable rows were found. Check results.json and ensure shots have metrics."
        )

    if df["label"].nunique() < 2:
        raise ValueError(
            "Training needs at least two classes. Add more varied shots or lower the threshold."
        )

    args.dataset_out.parent.mkdir(parents=True, exist_ok=True)
    args.model_out.parent.mkdir(parents=True, exist_ok=True)
    args.metadata_out.parent.mkdir(parents=True, exist_ok=True)

    df.sort_values("shot_id").to_csv(args.dataset_out, index=False)

    model, evaluation = train_model(df, args.test_size)
    joblib.dump(model, args.model_out)
    save_metadata(df, evaluation, args.metadata_out)

    print(f"Rows used: {len(df)}")
    print(f"Manual-note rows: {(df['label_source'] == 'manual_notes').sum()}")
    print(f"Fallback score rows: {(df['label_source'] == 'calibrated_score').sum()}")
    print("\nLabel counts:")
    print(df["label"].value_counts())

    if evaluation:
        print(f"\nAccuracy: {evaluation['accuracy']:.4f}")
        print("\nClassification report:")
        print(evaluation["classification_report"])
        print("Confusion matrix:")
        print(evaluation["confusion_matrix"])
    else:
        print("\nHeld-out evaluation skipped because the dataset is too small for a stable split.")

    print("\nFeature importances:")
    for feature, importance in zip(FEATURE_COLUMNS, model.feature_importances_):
        print(f"{feature}: {importance:.4f}")

    print(f"\nSaved dataset to: {args.dataset_out.resolve()}")
    print(f"Saved model to: {args.model_out.resolve()}")
    print(f"Saved metadata to: {args.metadata_out.resolve()}")


if __name__ == "__main__":
    main()
