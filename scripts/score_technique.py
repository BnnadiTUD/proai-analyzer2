import csv
import math
import os

# PATHS
ROOT_DIR = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2"
IN_CSV = os.path.join(ROOT_DIR, "features.csv")
OUT_CSV = os.path.join(ROOT_DIR, "scored_features.csv")

# FYP2-calibrated bands
BANDS = {
    "plant_to_ball_norm": {
        "ideal_min": 5.0,
        "ideal_max": 12.5,
        "lo_min": 2.5,
        "lo_max": 28.0,
    },
    "trunk_lean_deg": {
        "ideal_min": -6.0,
        "ideal_max": 8.0,
        "lo_min": -20.0,
        "lo_max": 28.0,
    },
    "hip_facing_deg": {
        "ideal_min": 0.0,
        "ideal_max": 70.0,
        "lo_min": 0.0,
        "lo_max": 110.0,
    },
    "follow_through_arc_deg": {
        "ideal_min": 60.0,
        "ideal_max": 190.0,
        "lo_min": 20.0,
        "lo_max": 240.0,
    },
    "lock_angle_deg": {
        "ideal_min": 85.0,
        "ideal_max": 165.0,
        "lo_min": 70.0,
        "lo_max": 180.0,
    },
}

IDEAL_FLOOR = 90.0
IDEAL_CENTER_BONUS = 10.0
EDGE_PENALTY_EXPONENT = 1.35
RANGE_PENALTY_EXPONENT = 1.6
WORST_METRIC_WEIGHT = 0.2


def to_float(x):
    if x is None:
        return None
    if isinstance(x, (int, float)):
        return float(x)
    x = str(x).strip()
    if not x:
        return None
    try:
        v = float(x)
        if math.isfinite(v):
            return v
        return None
    except ValueError:
        return None


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def normalize_metric_value(name, value):
    v = to_float(value)
    if v is None:
        return None

    if name == "trunk_lean_deg":
        while v > 180.0:
            v -= 360.0
        while v < -180.0:
            v += 360.0
        if v > 90.0:
            v = 180.0 - v
        elif v < -90.0:
            v = -180.0 - v
        return v

    if name == "hip_facing_deg":
        v = abs(v) % 360.0
        if v > 180.0:
            v = 360.0 - v
        return min(v, 180.0 - v)

    return v


def curved_score(value, ideal_min, ideal_max, lo_min, lo_max):
    """
    Match the FYP2 scoring curve:
    - 90-100 inside the ideal band, with the center scoring highest
    - decays non-linearly toward 0 as the value moves toward the loose bounds
    - 0 outside the loose bounds
    """
    v = to_float(value)
    if v is None:
        return None

    ideal_center = (ideal_min + ideal_max) / 2.0
    ideal_half_width = max((ideal_max - ideal_min) / 2.0, 1e-6)

    if ideal_min <= v <= ideal_max:
        normalized_offset = abs(v - ideal_center) / ideal_half_width
        edge_penalty = 1.0 - normalized_offset ** EDGE_PENALTY_EXPONENT
        return clamp(IDEAL_FLOOR + IDEAL_CENTER_BONUS * edge_penalty, IDEAL_FLOOR, 100.0)

    if v <= lo_min or v >= lo_max:
        return 0.0

    if v < ideal_min:
        distance_from_ideal = ideal_min - v
        penalty_range = ideal_min - lo_min
    else:
        distance_from_ideal = v - ideal_max
        penalty_range = lo_max - ideal_max

    normalized_distance = clamp(distance_from_ideal / max(penalty_range, 1e-6), 0.0, 1.0)
    score = IDEAL_FLOOR * (1.0 - normalized_distance ** RANGE_PENALTY_EXPONENT)
    return clamp(score, 0.0, IDEAL_FLOOR)


def score_metric(name, value):
    cfg = BANDS.get(name)
    if not cfg:
        return None
    normalized_value = normalize_metric_value(name, value)
    return round(
        curved_score(
            normalized_value,
            cfg["ideal_min"],
            cfg["ideal_max"],
            cfg["lo_min"],
            cfg["lo_max"],
        ),
        2,
    ) if value not in ("", None) else None


def overall_score(scores):
    vals = [s for s in scores if s is not None]
    if not vals:
        return None
    average = sum(vals) / len(vals)
    worst = min(vals)
    combined = average * (1.0 - WORST_METRIC_WEIGHT) + worst * WORST_METRIC_WEIGHT
    return round(combined, 2)


def warnings_for_row(row):
    """Flags for obviously broken detections."""
    warns = []

    ptbn = to_float(row.get("plant_to_ball_norm"))
    if ptbn is not None and (ptbn < 0.2 or ptbn > 3.0):
        warns.append("suspect_plant_to_ball")

    trunk = to_float(row.get("trunk_lean_deg"))
    if trunk is not None and abs(trunk) > 25:
        warns.append("suspect_trunk_lean")

    ft = to_float(row.get("follow_through_arc_deg"))
    if ft is not None and (ft < 5 or ft > 360):
        warns.append("suspect_follow_through")

    la = to_float(row.get("lock_angle_deg"))
    if la is not None and (la < 50 or la > 180):
        warns.append("suspect_lock_angle")

    missing_core = sum(
        1
        for k in [
            "plant_to_ball_norm",
            "trunk_lean_deg",
            "hip_facing_deg",
            "follow_through_arc_deg",
            "lock_angle_deg",
        ]
        if to_float(row.get(k)) is None
    )
    if missing_core >= 4:
        warns.append("weak_pose_data")

    return ";".join(warns) if warns else ""


def score_shot_technique(metrics: dict) -> dict:
    """
    Take a metrics dict for one shot and return the same 0-100
    scores used in FYP2.
    """
    s_plant = score_metric("plant_to_ball_norm", metrics.get("plant_to_ball_norm"))
    s_trunk = score_metric("trunk_lean_deg", metrics.get("trunk_lean_deg"))
    s_hip = score_metric("hip_facing_deg", metrics.get("hip_facing_deg"))
    s_ft = score_metric("follow_through_arc_deg", metrics.get("follow_through_arc_deg"))
    s_lock = score_metric("lock_angle_deg", metrics.get("lock_angle_deg"))

    scores = {
        "plant_to_ball_norm": s_plant,
        "trunk_lean_deg": s_trunk,
        "hip_facing_deg": s_hip,
        "follow_through_arc_deg": s_ft,
        "lock_angle_deg": s_lock,
    }

    overall = overall_score([s_plant, s_trunk, s_hip, s_ft, s_lock])
    if overall is not None:
        scores["overall"] = overall

    return scores


def score_features_file(in_csv=IN_CSV, out_csv=OUT_CSV):
    if not os.path.exists(in_csv):
        print(f"Input features.csv not found at: {in_csv}")
        return

    with open(in_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        rows = list(reader)

    if not rows:
        print("No rows in features.csv")
        return

    out_fields = list(reader.fieldnames) + [
        "score_plant_to_ball",
        "score_trunk_lean",
        "score_hip_facing",
        "score_follow_through",
        "score_lock_angle",
        "overall_score",
        "warnings",
    ]

    scored = []
    for r in rows:
        s_plant = score_metric("plant_to_ball_norm", r.get("plant_to_ball_norm"))
        s_trunk = score_metric("trunk_lean_deg", r.get("trunk_lean_deg"))
        s_hip = score_metric("hip_facing_deg", r.get("hip_facing_deg"))
        s_ft = score_metric("follow_through_arc_deg", r.get("follow_through_arc_deg"))
        s_lock = score_metric("lock_angle_deg", r.get("lock_angle_deg"))

        ovr = overall_score([s_plant, s_trunk, s_hip, s_ft, s_lock])
        warn = warnings_for_row(r)

        r_out = dict(r)
        r_out.update(
            score_plant_to_ball=s_plant,
            score_trunk_lean=s_trunk,
            score_hip_facing=s_hip,
            score_follow_through=s_ft,
            score_lock_angle=s_lock,
            overall_score=ovr,
            warnings=warn,
        )
        scored.append(r_out)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in scored:
            w.writerow(r)

    print(f"Wrote {len(scored)} scored rows -> {out_csv}")


if __name__ == "__main__":
    score_features_file()
