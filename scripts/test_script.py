import joblib
import pandas as pd

model = joblib.load("shot_classifier.pkl")

sample = pd.DataFrame([{
    "plant_to_ball_norm": 21.5,
    "trunk_lean_deg": -12.0,
    "hip_facing_deg": 95.0,
    "follow_through_arc_deg": 160.0,
    "lock_angle_deg": 118.0,
}])

prediction = model.predict(sample)[0]
probabilities = model.predict_proba(sample)[0]

print("Prediction:", prediction)
print("Probabilities:", probabilities)
print("Classes:", model.classes_)