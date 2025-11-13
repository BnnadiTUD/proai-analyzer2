import os
import csv
import math

# PATHS
ROOT_DIR = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2"
IN_CSV   = os.path.join(ROOT_DIR, "features.csv")
OUT_CSV  = os.path.join(ROOT_DIR, "scored_features.csv")

# Dataset-specific good shots (tuned from your sample
BANDS = {
    "plant_to_ball_norm": {
        "ideal_min": 10.0,  # typical good clips cluster in here
        "ideal_max": 30.0,
        "lo_min": 3.0,      # outside this strongly penalize
        "lo_max": 45.0,
    },
    "trunk_lean_deg": {
        # from my data most good looking clips land 5-20 in the coordinate system
        "ideal_min": 5.0,
        "ideal_max": 20.0,
        "lo_min": 0.0,
        "lo_max": 30.0,
    },
    "hip_facing_deg": {
        # 60-120 degrees = reasonable hip orientation in diagonal/side views
        "ideal_min": 60.0,
        "ideal_max": 120.0,
        "lo_min": 20.0,
        "lo_max": 170.0,
    },
    "follow_through_arc_deg": {
        # 80-250 degrees = confident follow-through; below 40 often tracking/short clip
        "ideal_min": 80.0,
        "ideal_max": 250.0,
        "lo_min": 30.0,
        "lo_max": 320.0,
    },
    "lock_angle_deg": {
        # 100-135 degrees = solid locked ankle; <80 or >160 looks off / wrong frame
        "ideal_min": 100.0,
        "ideal_max": 135.0,
        "lo_min": 70.0,
        "lo_max": 170.0,
    },
}

# ---------- helpers ----------

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

def linear_score(value, ideal_min, ideal_max, lo_min, lo_max):
    """
    Generic band-based score:
    - 5.0 inside [ideal_min, ideal_max]
    - linearly down towards 1.0 as you move to [lo_min, lo_max] bounds
    - <lo_min or >lo_max → close to 1.0
    """
    v = to_float(value)
    if v is None:
        return None

    # inside ideal band
    if ideal_min <= v <= ideal_max:
        return 5.0

    # below ideal
    if v < ideal_min:
        if v <= lo_min:
            return 1.0
        # map [lo_min -> ideal_min] to [1 -> 5]
        t = (v - lo_min) / (ideal_min - lo_min)
        return clamp(1.0 + 4.0 * t, 1.0, 5.0)

    # above ideal
    if v > ideal_max:
        if v >= lo_max:
            return 1.0
        # map [ideal_max -> lo_max] to [5 -> 1]
        t = (v - ideal_max) / (lo_max - ideal_max)
        return clamp(5.0 - 4.0 * t, 1.0, 5.0)

    return None  # fallback (shouldn't hit)

def score_metric(name, value):
    cfg = BANDS.get(name)
    if not cfg:
        return None
    return round(
        linear_score(
            value,
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
    return round(sum(vals) / len(vals), 2)

def warnings_for_row(row):
    """ flags for obviously broken detections."""
    warns = []

    ptbn = to_float(row.get("plant_to_ball_norm"))
    if ptbn is not None and (ptbn < 0.5 or ptbn > 60):
        warns.append("suspect_plant_to_ball")

    ft = to_float(row.get("follow_through_arc_deg"))
    if ft is not None and (ft < 5 or ft > 360):
        warns.append("suspect_follow_through")

    la = to_float(row.get("lock_angle_deg"))
    if la is not None and (la < 50 or la > 180):
        warns.append("suspect_lock_angle")

    # If almost everything is missing, flag it
    missing_core = sum(
        1
        for k in ["plant_to_ball_norm", "trunk_lean_deg", "hip_facing_deg",
                  "follow_through_arc_deg", "lock_angle_deg"]
        if not to_float(row.get(k))
    )
    if missing_core >= 4:
        warns.append("weak_pose_data")

    return ";".join(warns) if warns else ""

# ---------- main pipeline ----------

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
        s_trunk = score_metric("trunk_lean_deg",      r.get("trunk_lean_deg"))
        s_hip   = score_metric("hip_facing_deg",      r.get("hip_facing_deg"))
        s_ft    = score_metric("follow_through_arc_deg", r.get("follow_through_arc_deg"))
        s_lock  = score_metric("lock_angle_deg",      r.get("lock_angle_deg"))

        ovr = overall_score([s_plant, s_trunk, s_hip, s_ft, s_lock])
        warn = warnings_for_row(r)

        r_out = dict(r)
        r_out.update(
            score_plant_to_ball = s_plant,
            score_trunk_lean    = s_trunk,
            score_hip_facing    = s_hip,
            score_follow_through= s_ft,
            score_lock_angle    = s_lock,
            overall_score       = ovr,
            warnings            = warn,
        )
        scored.append(r_out)

    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=out_fields)
        w.writeheader()
        for r in scored:
            w.writerow(r)

    print(f" Wrote {len(scored)} scored rows → {out_csv}")

if __name__ == "__main__":
    score_features_file()
