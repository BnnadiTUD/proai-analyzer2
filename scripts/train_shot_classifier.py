import argparse
import json
import re
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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

MODEL_FACTORIES = {
    "random_forest": lambda: RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=42,
    ),
    "logistic_regression": lambda: Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    max_iter=2000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    ),
    "gradient_boosting": lambda: GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
}

SHOT_HEADER_RE = re.compile(r"^##\s*(shot_\d+)\b(.*)$", re.IGNORECASE)
RATING_RE = re.compile(r"(\d+(?:\.\d+)?)\s*/\s*10")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train the shot classifier from calibrated analysis outputs."
    )
    parser.add_argument(
        "--dataset-path",
        type=Path,
        help="Optional path to an already assembled training CSV.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
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
    parser.add_argument(
        "--model-type",
        choices=sorted(MODEL_FACTORIES),
        default="random_forest",
        help="Which trained model to save as the app classifier.",
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


def build_model(model_type):
    return MODEL_FACTORIES[model_type]()


def choose_cv_folds(df):
    label_counts = df["label"].value_counts()
    if len(label_counts) < 2:
        return None
    min_class_count = int(label_counts.min())
    if min_class_count < 3 or len(df) < 12:
        return None
    return min(5, min_class_count)


def evaluate_cross_validation(model, X, y, cv_folds):
    if cv_folds is None:
        return None

    cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    accuracy_scores = cross_val_score(model, X, y, cv=cv, scoring="accuracy")
    f1_scores = cross_val_score(model, X, y, cv=cv, scoring="f1_macro")
    return {
        "folds": cv_folds,
        "accuracy_mean": float(accuracy_scores.mean()),
        "accuracy_std": float(accuracy_scores.std()),
        "f1_macro_mean": float(f1_scores.mean()),
        "f1_macro_std": float(f1_scores.std()),
    }


def evaluate_holdout(model, X_train, X_test, y_train, y_test, test_size):
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return {
        "test_size": test_size,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "classification_report": classification_report(y_test, y_pred),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def compare_models(df, test_size):
    X = df[FEATURE_COLUMNS]
    y = df["label"]

    actual_test_size = choose_test_split(df, test_size)
    cv_folds = choose_cv_folds(df)
    comparison = {}

    if actual_test_size is None:
        for model_type in MODEL_FACTORIES:
            model = build_model(model_type)
            comparison[model_type] = {
                "cross_validation": evaluate_cross_validation(model, X, y, cv_folds),
                "holdout": None,
            }
        return comparison, actual_test_size

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=actual_test_size,
        random_state=42,
        stratify=y,
    )

    for model_type in MODEL_FACTORIES:
        model = build_model(model_type)
        comparison[model_type] = {
            "cross_validation": evaluate_cross_validation(model, X, y, cv_folds),
            "holdout": evaluate_holdout(
                model=build_model(model_type),
                X_train=X_train,
                X_test=X_test,
                y_train=y_train,
                y_test=y_test,
                test_size=actual_test_size,
            ),
        }
    return comparison, actual_test_size


def train_selected_model(df, model_type):
    model = build_model(model_type)
    X = df[FEATURE_COLUMNS]
    y = df["label"]
    model.fit(X, y)
    return model


def build_feature_summary(model):
    estimator = model
    if isinstance(model, Pipeline):
        estimator = model.named_steps.get("classifier", model)

    if hasattr(estimator, "feature_importances_"):
        return {
            feature: float(importance)
            for feature, importance in zip(FEATURE_COLUMNS, estimator.feature_importances_)
        }

    if hasattr(estimator, "coef_"):
        coefficients = estimator.coef_
        if len(coefficients.shape) == 2 and coefficients.shape[0] >= 1:
            values = coefficients[0]
            return {feature: float(value) for feature, value in zip(FEATURE_COLUMNS, values)}

    return None


def save_metadata(df, model_type, comparison, feature_summary, metadata_path):
    metadata = {
        "selected_model_type": model_type,
        "candidate_models": sorted(MODEL_FACTORIES),
        "feature_columns": FEATURE_COLUMNS,
        "rows_used": int(len(df)),
        "label_counts": {key: int(value) for key, value in df["label"].value_counts().items()},
        "label_source_counts": {
            key: int(value) for key, value in df["label_source"].value_counts().items()
        },
        "manual_note_rows": int((df["label_source"] == "manual_notes").sum()),
        "fallback_rows": int((df["label_source"] == "calibrated_score").sum()),
        "model_comparison": comparison,
        "selected_model_feature_summary": feature_summary,
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def print_model_comparison(comparison):
    ranked = []
    for model_type, metrics in comparison.items():
        holdout = metrics.get("holdout")
        cv = metrics.get("cross_validation")
        ranked.append(
            (
                holdout["accuracy"] if holdout else float("-inf"),
                cv["accuracy_mean"] if cv else float("-inf"),
                model_type,
                metrics,
            )
        )

    ranked.sort(reverse=True)

    print("\nModel comparison:")
    for _, _, model_type, metrics in ranked:
        print(f"\n[{model_type}]")
        holdout = metrics.get("holdout")
        cv = metrics.get("cross_validation")
        if holdout:
            print(
                f"Holdout accuracy: {holdout['accuracy']:.4f} | "
                f"holdout macro F1: {holdout['f1_macro']:.4f}"
            )
            print("Classification report:")
            print(holdout["classification_report"])
            print("Confusion matrix:")
            print(holdout["confusion_matrix"])
        else:
            print("Holdout evaluation skipped because the dataset is too small for a stable split.")

        if cv:
            print(
                f"Cross-val accuracy: {cv['accuracy_mean']:.4f} +/- {cv['accuracy_std']:.4f} | "
                f"cross-val macro F1: {cv['f1_macro_mean']:.4f} +/- {cv['f1_macro_std']:.4f}"
            )
        else:
            print("Cross-validation skipped because the dataset is too small for stable folds.")


def main():
    args = parse_args()

    if args.dataset_path:
        if not args.dataset_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {args.dataset_path}")
        df = pd.read_csv(args.dataset_path)
        df = df.dropna(subset=FEATURE_COLUMNS + ["label"]).copy()
    else:
        if not args.results_path:
            raise ValueError("Provide either --dataset-path or --results-path.")
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

    comparison, _ = compare_models(df, args.test_size)
    model = train_selected_model(df, args.model_type)
    joblib.dump(model, args.model_out)
    feature_summary = build_feature_summary(model)
    save_metadata(df, args.model_type, comparison, feature_summary, args.metadata_out)

    print(f"Rows used: {len(df)}")
    print(f"Manual-note rows: {(df['label_source'] == 'manual_notes').sum()}")
    print(f"Fallback score rows: {(df['label_source'] == 'calibrated_score').sum()}")
    print("\nLabel counts:")
    print(df["label"].value_counts())
    print_model_comparison(comparison)
    print(f"\nSaved model type: {args.model_type}")

    if feature_summary:
        print("\nSelected model feature summary:")
        for feature, value in feature_summary.items():
            print(f"{feature}: {value:.4f}")

    print(f"\nSaved dataset to: {args.dataset_out.resolve()}")
    print(f"Saved model to: {args.model_out.resolve()}")
    print(f"Saved metadata to: {args.metadata_out.resolve()}")


if __name__ == "__main__":
    main()
