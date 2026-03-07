import pandas as pd
import numpy as np
import random

random.seed(42)

# Read the actual CSV file from disk
df = pd.read_csv("features.csv")

metric_cols = [
    "plant_to_ball_norm",
    "trunk_lean_deg",
    "hip_facing_deg",
    "follow_through_arc_deg",
    "lock_angle_deg"
]

# simple fps grouping so 30fps shots borrow from other 30fps shots, etc.
df["fps_bin"] = np.where(df["fps"] < 40, "30ish", "high")

original_missing_mask = df[metric_cols].isna().any(axis=1)

for idx, row in df.iterrows():
    subgroup = df[
        (df["fps_bin"] == row["fps_bin"]) &
        (df["striking_foot"] == row["striking_foot"])
    ]

    for col in metric_cols:
        if pd.isna(row[col]):
            vals = subgroup[col].dropna().tolist()

            # fallback if subgroup has no values
            if not vals:
                vals = df[col].dropna().tolist()

            # skip if still no values at all
            if not vals:
                continue

            base = random.choice(vals)
            std = np.std(vals) if len(vals) > 1 else 1.0

            # add tiny jitter so values are not exact duplicates
            jitter = random.uniform(-0.08, 0.08) * std
            value = base + jitter

            # clamp to reasonable bounds from  dataset
            lo = df[col].quantile(0.05)
            hi = df[col].quantile(0.95)
            value = min(max(value, lo), hi)

            df.at[idx, col] = round(value, 2)

df = df.drop(columns=["fps_bin"])

print("Filled rows only:\n")
print(df.loc[original_missing_mask].to_csv(index=False))

print("\nFull completed CSV:\n")
print(df.to_csv(index=False))

df.to_csv("features_filled.csv", index=False)
print("\nSaved completed file as features_filled.csv")