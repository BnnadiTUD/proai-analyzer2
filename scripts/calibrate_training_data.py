import argparse
import csv
from pathlib import Path

from score_technique import score_shot_technique


FEATURE_COLUMNS = [
    "plant_to_ball_norm",
    "trunk_lean_deg",
    "hip_facing_deg",
    "follow_through_arc_deg",
    "lock_angle_deg",
]


DEFAULT_INPUT = Path("training_data.csv")
DEFAULT_OUTPUT = Path("training_data_calibrated.csv")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rebuild classifier-ready training data using the current scoring ranges."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=DEFAULT_INPUT,
        help="Raw metrics CSV exported from the app/Firestore.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Classifier-ready calibrated CSV to write.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=70.0,
        help="Overall score required for the good label.",
    )
    return parser.parse_args()


def blank_to_none(value):
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def main():
    args = parse_args()

    if not args.input.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.input}")

    with args.input.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    output_fields = [
        "shot_id",
        "video_path",
        "score_overall",
        "label_source",
        "manual_summary",
        *FEATURE_COLUMNS,
        "human_plant_to_ball_norm",
        "human_trunk_lean_deg",
        "human_hip_facing_deg",
        "human_follow_through_arc_deg",
        "human_lock_angle_deg",
        "manual_avg_rating",
        "label",
    ]

    calibrated = []
    skipped = 0
    for row in rows:
        metrics = {feature: blank_to_none(row.get(feature)) for feature in FEATURE_COLUMNS}
        if any(metrics[feature] is None for feature in FEATURE_COLUMNS):
            skipped += 1
            continue

        scores = score_shot_technique(metrics)
        overall = scores.get("overall")
        if overall is None:
            skipped += 1
            continue

        calibrated.append(
            {
                "shot_id": row.get("shot_id"),
                "video_path": row.get("video_path") or row.get("file_path") or row.get("uri"),
                "score_overall": overall,
                "label_source": "calibrated_score",
                "manual_summary": None,
                **metrics,
                "human_plant_to_ball_norm": None,
                "human_trunk_lean_deg": None,
                "human_hip_facing_deg": None,
                "human_follow_through_arc_deg": None,
                "human_lock_angle_deg": None,
                "manual_avg_rating": None,
                "label": "good" if overall >= args.score_threshold else "bad",
            }
        )

    calibrated.sort(key=lambda item: str(item.get("shot_id") or ""))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(calibrated)

    print(f"Read rows: {len(rows)}")
    print(f"Wrote calibrated rows: {len(calibrated)}")
    print(f"Skipped incomplete rows: {skipped}")
    print(f"Output: {args.output.resolve()}")


if __name__ == "__main__":
    main()
