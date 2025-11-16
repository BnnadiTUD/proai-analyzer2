# main.py
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware

from pathlib import Path
import shutil
import uuid
import os

from extract_pose import extract_pose_for_video        
from analyze_pose_features import compute_metrics_for_shot  
from score_technique import score_shot_technique        

app = FastAPI(
    title="ProAI Analyzer API",
    description="Pose to Metrics to Technique scoring for football shots",
    version="0.1.0",
)

# Allowing my Android app to call this API from emulator/device
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# temp storage for uploaded videos & pose data
BASE_UPLOAD_DIR = Path("uploads")
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


@app.post("/analyze_shot")
async def analyze_shot(file: UploadFile = File(...)):
    """
    Android uploads a single shot video (2–3 sec).
    Pipeline:
      video to extract_pose to analyze_pose_features to score_technique to feedback
    Returns:
      {
        "shot_id": "...",
        "metrics": {...},
        "scores": {...},
        "feedback": {...}
      }
    """
    # --- 1) Basic validation -------------------------------------------------
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_types = {"video/mp4", "video/quicktime", "video/x-matroska"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload an MP4 or compatible video.",
        )

    # --- 2) Create unique shot_id & dirs -----------------------------------
    original_name = Path(file.filename).stem
    random_suffix = uuid.uuid4().hex[:8]
    shot_id = f"{original_name}_{random_suffix}"

    shot_dir = BASE_UPLOAD_DIR / shot_id
    shot_dir.mkdir(parents=True, exist_ok=True)

    video_path = shot_dir / f"{shot_id}.mp4"

    # --- 3) Save uploaded video to disk --------------------------------------
    try:
        with open(video_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {e}")
    finally:
        file.file.close()

    # --- 4) Run pipeline: extract_pose -> analyze_pose_features -> score_technique
    try:
        #Extract pose **only for this video**
        pose_data_path = extract_pose_for_video(
            video_path=str(video_path),
            output_dir=str(shot_dir),
            shot_id=shot_id,
        )

        #Compute metrics for that shot

        metrics = compute_metrics_for_shot(
            pose_data_path=str(pose_data_path),
            shot_id=shot_id,
        )

        if not isinstance(metrics, dict):
            raise RuntimeError("analyze_pose_features did not return a dict of metrics")

        # Apply calibrated bands to get numeric scores

        scores = score_shot_technique(metrics)

        if not isinstance(scores, dict):
            raise RuntimeError("score_technique did not return a dict of scores")

        #Generate interpretable feedback sentences for the player
        feedback = generate_feedback(metrics, scores)

    except HTTPException:
        # Pass through HTTPExceptions unchanged
        raise
    except Exception as e:
        # Catch any pipeline error and expose a clean message to the client
        raise HTTPException(
            status_code=500,
            detail=f"Error during analysis pipeline: {e}",
        )

    # --- 5) Return JSON in the exact shape specified ---------------------
    result = {
        "shot_id": shot_id,
        "metrics": metrics,
        "scores": scores,
        "feedback": feedback,
    }

    return JSONResponse(content=result)


# Feedback gen

def generate_feedback(metrics: dict, scores: dict) -> dict:
    """
    Turn numeric metrics + scores into human-friendly coaching feedback.

    IMPORTANT:
    - Tune the thresholds/messages to match the calibrated bands
      you used in score_technique.py so everything is consistent.
    """

    fb = {}

    # Helper to convert a 0–100 score into a generic phrase
    def score_bucket_text(score: float) -> str:
        if score >= 85:
            return "excellent"
        elif score >= 70:
            return "good"
        elif score >= 50:
            return "ok"
        else:
            return "needs improvement"

    # --- plant_to_ball_norm --------------------------------------------------
    plant = metrics.get("plant_to_ball_norm")
    plant_score = scores.get("plant_to_ball_norm")

    if plant is not None and plant_score is not None:
        # TODO: plug in real bands instead of these demo ones
        if plant < 10:
            msg = "Your plant foot is very close to the ball. Take a slightly bigger step for more power and balance."
        elif plant > 35:
            msg = "Your plant foot is quite far from the ball. Step closer so you can strike through the middle of the ball."
        else:
            msg = "Nice plant-foot distance — you're setting up a solid base for your shot."
        fb["plant_to_ball_norm"] = {
            "score_comment": f"Plant-to-ball distance is {score_bucket_text(plant_score)}.",
            "technical_tip": msg,
        }

    # --- trunk_lean_deg ------------------------------------------------------
    trunk = metrics.get("trunk_lean_deg")
    trunk_score = scores.get("trunk_lean_deg")

    if trunk is not None and trunk_score is not None:
        # negative = leaning back, positive = over the ball
        if trunk < -5:
            msg = "You're leaning back at impact. Try leaning your chest slightly over the ball to keep shots down."
        elif trunk > 25:
            msg = "You are leaning a lot over the ball. That’s safe, but you might be losing power — find a comfortable forward lean."
        else:
            msg = "Good trunk angle — you’re staying over the ball and controlling height."
        fb["trunk_lean_deg"] = {
            "score_comment": f"Upper-body lean is {score_bucket_text(trunk_score)}.",
            "technical_tip": msg,
        }

    # --- hip_facing_deg ------------------------------------------------------
    hip = metrics.get("hip_facing_deg")
    hip_score = scores.get("hip_facing_deg")

    if hip is not None and hip_score is not None:
        # Example ranges – adjust to your bands
        if hip < 40:
            msg = "Your hips stay quite closed at impact. Try opening them a bit more towards the target for a cleaner strike."
        elif hip > 120:
            msg = "Your hips are very open. This can cause slicing or pulling shots — aim for a more controlled rotation."
        else:
            msg = "Nice hip rotation — you’re generating power while still staying in control."
        fb["hip_facing_deg"] = {
            "score_comment": f"Hip rotation is {score_bucket_text(hip_score)}.",
            "technical_tip": msg,
        }

    # --- follow_through_arc_deg ----------------------------------------------
    arc = metrics.get("follow_through_arc_deg")
    arc_score = scores.get("follow_through_arc_deg")

    if arc is not None and arc_score is not None:
        if arc < 40:
            msg = "Your follow-through is quite short. Swing your leg through the ball more to generate power and consistency."
        elif arc > 200:
            msg = "Your follow-through is very big. Make sure you're balanced and not over-rotating after contact."
        else:
            msg = "Balanced follow-through — you're striking through the ball nicely."
        fb["follow_through_arc_deg"] = {
            "score_comment": f"Follow-through arc is {score_bucket_text(arc_score)}.",
            "technical_tip": msg,
        }

    # --- lock_angle_deg ------------------------------------------------------
    lock = metrics.get("lock_angle_deg")
    lock_score = scores.get("lock_angle_deg")

    if lock is not None and lock_score is not None:
        if lock < 90:
            msg = "Your kicking leg is quite bent at impact. Try locking the knee a bit more for a cleaner strike."
        elif lock > 160:
            msg = "Your leg is very straight. That's powerful, but be sure you're not over-extending and risking discomfort."
        else:
            msg = "Solid leg extension — you're transferring power efficiently into the ball."
        fb["lock_angle_deg"] = {
            "score_comment": f"Leg lock at impact is {score_bucket_text(lock_score)}.",
            "technical_tip": msg,
        }

    # --- overall comment -----------------------------------------------------
    overall = scores.get("overall")
    if overall is not None:
        overall_text = score_bucket_text(overall)
        if overall >= 85:
            summary = "Overall, this is a very strong shooting technique. Keep repeating this pattern and only tweak small details."
        elif overall >= 70:
            summary = "Overall, your technique is good. A few targeted tweaks will make your shots more consistent and powerful."
        elif overall >= 50:
            summary = "Your technique is developing well. Focus on the key tips above and you should see quick improvements."
        else:
            summary = "You’ve got the basics, but several pieces need work. Use the tips above and re-record your shot after some focused practice."
        fb["overall"] = {
            "score_comment": f"Overall shooting technique is {overall_text} ({overall:.0f}/100).",
            "summary": summary,
        }

    return fb
