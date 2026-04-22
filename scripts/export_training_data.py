import csv
from pathlib import Path
import pandas as pd
import firebase_admin
from firebase_admin import credentials, firestore

SERVICE_ACCOUNT_PATH = "serviceAccountKey.json"
OUTPUT_CSV = Path("training_data.csv")

# -----------------------------
# Init Firebase Admin
if not firebase_admin._apps:
    cred = credentials.Certificate(SERVICE_ACCOUNT_PATH)
    firebase_admin.initialize_app(cred)

db = firestore.client()

firestore_rows = []

# Read Firestore shots
users_ref = db.collection("users").stream()

for user_doc in users_ref:
    user_id = user_doc.id
    videos_ref = db.collection("users").document(user_id).collection("videos").stream()

    for shot_doc in videos_ref:
        data = shot_doc.to_dict()
        metrics = data.get("metrics", {}) or {}

        firestore_rows.append({
            "user_id": user_id,
            "shot_id": data.get("shotId", shot_doc.id),
            "plant_to_ball_norm": metrics.get("plant_to_ball_norm") or metrics.get("plantToBallNorm"),
            "trunk_lean_deg": metrics.get("trunk_lean_deg") or metrics.get("trunkLeanDeg"),
            "hip_facing_deg": metrics.get("hip_facing_deg") or metrics.get("hipFacingDeg"),
            "follow_through_arc_deg": metrics.get("follow_through_arc_deg") or metrics.get("followThroughArcDeg"),
            "lock_angle_deg": metrics.get("lock_angle_deg") or metrics.get("lockAngleDeg"),
            "overall_score": data.get("overallScore"),
            "created_at": data.get("createdAt"),
        })

firestore_df = pd.DataFrame(firestore_rows)

# -----------------------------
# Read existing CSV
if OUTPUT_CSV.exists():
    existing_df = pd.read_csv(OUTPUT_CSV)
    print(f"Loaded existing CSV rows: {len(existing_df)}")
else:
    existing_df = pd.DataFrame()
    print("No existing training_data.csv found, creating a new one.")

# Merge both datasets
combined_df = pd.concat([existing_df, firestore_df], ignore_index=True)

# Remove duplicates by shot_id, keeping the newest occurrence
if "shot_id" in combined_df.columns:
    combined_df = combined_df.drop_duplicates(subset=["shot_id"], keep="last")

# Optional: sort by shot_id
combined_df = combined_df.sort_values(by="shot_id", na_position="last")

# Save merged CSV
combined_df.to_csv(OUTPUT_CSV, index=False)

print(f"Firestore rows fetched: {len(firestore_df)}")
print(f"Final merged rows saved: {len(combined_df)}")
print(f"Saved merged dataset to: {OUTPUT_CSV.resolve()}")