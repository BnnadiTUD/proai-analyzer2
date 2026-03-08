import csv
from pathlib import Path
import firebase_admin
from firebase_admin import credentials, firestore


SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
OUTPUT_CSV = Path("training_data.csv")
cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
firebase_admin.initialize_app(cred)

db = firestore.client()

rows = []

users_ref = db.collection("users").stream()

for user_doc in users_ref:
    user_id = user_doc.id
    videos_ref = db.collection("users").document(user_id).collection("videos").stream()

    for shot_doc in videos_ref:
        data = shot_doc.to_dict()

        metrics = data.get("metrics", {}) or {}
        scores = data.get("scores", {}) or {}

        row = {
            "user_id": user_id,
            "shot_id": data.get("shotId", shot_doc.id),
            "plant_to_ball_norm": metrics.get("plant_to_ball_norm") or metrics.get("plantToBallNorm"),
            "trunk_lean_deg": metrics.get("trunk_lean_deg") or metrics.get("trunkLeanDeg"),
            "hip_facing_deg": metrics.get("hip_facing_deg") or metrics.get("hipFacingDeg"),
            "follow_through_arc_deg": metrics.get("follow_through_arc_deg") or metrics.get("followThroughArcDeg"),
            "lock_angle_deg": metrics.get("lock_angle_deg") or metrics.get("lockAngleDeg"),
            "overall_score": data.get("overallScore"),
            "created_at": data.get("createdAt"),
        }

        rows.append(row)

# Save
fieldnames = [
    "user_id",
    "shot_id",
    "plant_to_ball_norm",
    "trunk_lean_deg",
    "hip_facing_deg",
    "follow_through_arc_deg",
    "lock_angle_deg",
    "overall_score",
    "created_at",
]

with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)

print(f"Exported {len(rows)} shots to {OUTPUT_CSV.resolve()}")