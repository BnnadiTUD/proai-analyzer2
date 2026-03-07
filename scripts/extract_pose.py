import os
import json
from pathlib import Path

import cv2
import mediapipe as mp
from tqdm import tqdm

# paths for offline processing
VIDEOS_DIR = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2\trimmed_videos"
OUTPUT_DIR = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2\poses_blazepose"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Pose config
mp_pose = mp.solutions.pose

def run_pose_on_video(video_path: str):
    """
    Run BlazePose over a single video; return (fps, frames_list)
    where frames_list is the same structure used in your JSON:
      [{"frame": idx, "landmarks": {name: [x,y,z,visibility], ...}}, ...]
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    frames = []
    frame_idx = 0

    with mp_pose.Pose(
        static_image_mode=True,
        model_complexity=2,
        enable_segmentation=False,
        min_detection_confidence=0.3,
        min_tracking_confidence=0.3,
    ) as pose:

        with tqdm(total=total if total > 0 else None,
                  desc=os.path.basename(video_path),
                  unit="f") as pbar:
            while True:
                ok, frame = cap.read()
                if not ok:
                    break

                image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = pose.process(image_rgb)

                frame_rec = {"frame": frame_idx, "landmarks": {}}
                if res.pose_landmarks:
                    lmks = res.pose_landmarks.landmark
                    for i, l in enumerate(lmks):
                        name = mp_pose.PoseLandmark(i).name.lower()
                        frame_rec["landmarks"][name] = [
                            round(l.x, 6),
                            round(l.y, 6),
                            round(l.z, 6),
                            round(l.visibility, 6),
                        ]

                frames.append(frame_rec)
                frame_idx += 1
                pbar.update(1)

    cap.release()
    return float(fps), frames


def extract_pose_for_video(video_path: str, output_dir: str, shot_id: str) -> str:
    """
    Run BlazePose pose extraction for ONE video.

    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    out_file = output_path / f"{shot_id}.json"

    fps, frames = run_pose_on_video(video_path)

    payload = {
        "shot_id": shot_id,
        "fps": fps,
        "frames": frames,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return str(out_file)


def process_all():
    """
    Offline batch: run pose on all videos in VIDEOS_DIR and write JSON to OUTPUT_DIR.
    """
    SUPPORTED = (".mp4", ".mov", ".m4v", ".avi", ".MP4", ".MOV", ".M4V", ".AVI")
    files = [f for f in os.listdir(VIDEOS_DIR) if f.endswith(SUPPORTED)]
    files.sort()

    if not files:
        print(f"WARNING No video files found in: {VIDEOS_DIR}")
        return

    for fname in files:
        in_path = os.path.join(VIDEOS_DIR, fname)
        shot_id = os.path.splitext(fname)[0]
        out_path = os.path.join(OUTPUT_DIR, f"{shot_id}.json")

        if os.path.exists(out_path):
            print(f"[SKIP] {fname} -> already exists")
            continue

        print(f"\n[PROCESS] {fname}")
        fps, frames = run_pose_on_video(in_path)

        payload = {"shot_id": shot_id, "fps": fps, "frames": frames}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[DONE] Saved → {out_path}")


if __name__ == "__main__":
    process_all()
    print("\n Finished all videos.")

