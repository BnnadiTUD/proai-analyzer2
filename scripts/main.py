from pathlib import Path
import shutil
import uuid

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from analyze_pose_features import compute_metrics_for_shot
from extract_pose import extract_pose_for_video
from score_technique import score_shot_technique


app = FastAPI(
    title="ProAI Analyzer API",
    description="Pose to Metrics to Technique scoring for football shots",
    version="0.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_UPLOAD_DIR = Path("uploads")
BASE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def score_bucket_text(score: float | None) -> str:
    if score is None:
        return "unknown"
    if score >= 92:
        return "excellent"
    if score >= 78:
        return "good"
    if score >= 60:
        return "ok"
    return "needs improvement"


def generate_feedback(metrics: dict, scores: dict) -> dict:
    metric_feedback = {}

    plant = metrics.get("plant_to_ball_norm")
    plant_score = scores.get("plant_to_ball_norm")
    if plant is not None and plant_score is not None:
        if plant < 0.9:
            msg = "Your plant foot is too close to the ball. Give yourself a little more space so you can strike cleanly and stay balanced."
        elif plant > 1.15:
            msg = "Your plant foot is too far from the ball. Step in slightly closer to improve control and contact quality."
        else:
            msg = "Your plant-foot distance is well set up for balance, control, and a clean strike."
        metric_feedback["plant_to_ball_norm"] = {
            "score_comment": f"Plant-to-ball distance is {score_bucket_text(plant_score)}.",
            "technical_tip": msg,
        }

    trunk = metrics.get("trunk_lean_deg")
    trunk_score = scores.get("trunk_lean_deg")
    if trunk is not None and trunk_score is not None:
        if trunk < -3.0:
            msg = "You're leaning back at contact. Bring your chest a little more over the ball to keep the shot down and improve control."
        elif trunk > 3.0:
            msg = "You're leaning too far over the ball. Stay slightly forward, but not so much that you lose power and freedom through the strike."
        else:
            msg = "Your trunk position is controlled and balanced through contact."
        metric_feedback["trunk_lean_deg"] = {
            "score_comment": f"Upper-body lean is {score_bucket_text(trunk_score)}.",
            "technical_tip": msg,
        }

    hip = metrics.get("hip_facing_deg")
    hip_score = scores.get("hip_facing_deg")
    if hip is not None and hip_score is not None:
        if hip < 72.0:
            msg = "Your hips are staying too closed at impact. Rotate through the target a bit more to improve power and direction."
        elif hip > 108.0:
            msg = "Your hips are opening too much at impact. Control the rotation more so the strike stays stable and on line."
        else:
            msg = "Your hip rotation is well timed and gives you a good balance of power and control."
        metric_feedback["hip_facing_deg"] = {
            "score_comment": f"Hip rotation is {score_bucket_text(hip_score)}.",
            "technical_tip": msg,
        }

    arc = metrics.get("follow_through_arc_deg")
    arc_score = scores.get("follow_through_arc_deg")
    if arc is not None and arc_score is not None:
        if arc < 110.0:
            msg = "Your follow-through is too short. Let the kicking leg travel through the ball more to improve power and consistency."
        elif arc > 220.0:
            msg = "Your follow-through is too large. Stay more compact after contact so you keep balance and don't over-rotate."
        else:
            msg = "Your follow-through is controlled and complete, which supports a cleaner strike."
        metric_feedback["follow_through_arc_deg"] = {
            "score_comment": f"Follow-through arc is {score_bucket_text(arc_score)}.",
            "technical_tip": msg,
        }

    lock = metrics.get("lock_angle_deg")
    lock_score = scores.get("lock_angle_deg")
    if lock is not None and lock_score is not None:
        if lock < 108.0:
            msg = "Your striking leg is too bent at contact. Lock the leg a little more so the ball is struck with a firmer shape."
        elif lock > 128.0:
            msg = "Your leg is over-extending at contact. Stay firm, but avoid forcing it completely straight so you keep control."
        else:
            msg = "Your leg extension is strong and controlled through the strike."
        metric_feedback["lock_angle_deg"] = {
            "score_comment": f"Leg lock at impact is {score_bucket_text(lock_score)}.",
            "technical_tip": msg,
        }

    overall = scores.get("overall")
    overall_feedback = None
    if overall is not None:
        if overall >= 92:
            summary = "Overall, this is a very strong shooting technique. Keep repeating this pattern and refine only the small details."
        elif overall >= 78:
            summary = "Overall, your technique is solid, but there are still a few areas stopping this from being consistently high quality."
        elif overall >= 60:
            summary = "There are some good parts in the strike, but a few technical issues are still limiting consistency and power."
        else:
            summary = "Several parts of the technique need work right now. Focus on the weakest metrics first, then re-record after targeted practice."
        overall_feedback = {
            "score_comment": f"Overall shooting technique is {score_bucket_text(overall)} ({int(overall)}/100).",
            "summary": summary,
        }

    return {
        "metric_feedback": metric_feedback,
        "overall": overall_feedback,
    }


def analyze_saved_video(video_path: Path, shot_id: str, shot_dir: Path) -> dict:
    extract_pose_for_video(
        video_path=str(video_path),
        output_dir=str(shot_dir),
        shot_id=shot_id,
    )
    pose_json_path = shot_dir / f"{shot_id}.json"
    metrics = compute_metrics_for_shot(
        video_path=str(video_path),
        pose_json_path=str(pose_json_path),
    )
    scores = score_shot_technique(metrics)
    feedback = generate_feedback(metrics, scores)
    return {
        "shot_id": shot_id,
        "metrics": metrics,
        "scores": scores,
        "feedback": feedback,
    }


@app.post("/analyze_shot")
async def analyze_shot(file: UploadFile = File(...)):
    """
    Android uploads a single shot video.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file uploaded")

    allowed_types = {"video/mp4", "video/quicktime", "video/x-matroska"}
    if file.content_type not in allowed_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {file.content_type}. Please upload an MP4 or compatible video.",
        )

    original_name = Path(file.filename).stem
    random_suffix = uuid.uuid4().hex[:8]
    shot_id = f"{original_name}_{random_suffix}"

    shot_dir = BASE_UPLOAD_DIR / shot_id
    shot_dir.mkdir(parents=True, exist_ok=True)
    video_path = shot_dir / f"{shot_id}.mp4"

    try:
        with open(video_path, "wb") as out_file:
            shutil.copyfileobj(file.file, out_file)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Failed to save video: {exc}") from exc
    finally:
        file.file.close()

    try:
        result = analyze_saved_video(video_path=video_path, shot_id=shot_id, shot_dir=shot_dir)
    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Error during analysis pipeline: {exc}",
        ) from exc

    return JSONResponse(
        content={
            "shot_id": result["shot_id"],
            "metrics": result["metrics"],
            "scores": result["scores"],
            "feedback": result["feedback"],
        }
    )
