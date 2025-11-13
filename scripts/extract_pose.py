import os
import json #filing
import cv2 # readinf frames
import mediapipe as mp # BlazePose model (33 landmarks)
from tqdm import tqdm #progress bar

# paths
VIDEOS_DIR = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2\trimmed_videos"
OUTPUT_DIR = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2\poses_blazepose"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# MediaPipe Pose config
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    static_image_mode=True,        # force detection each frame
    model_complexity=2,
    enable_segmentation=False,
    min_detection_confidence=0.3,  # lower thresholds
    min_tracking_confidence=0.3,
)

def extract_one(video_path: str):
    """Run BlazePose over a single video; return (fps, frames_list)."""
    cap = cv2.VideoCapture(video_path) #open video file
    if not cap.isOpened():
        raise RuntimeError(f"Could not open: {video_path}")

#get fps and total frames
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    frame_idx = 0

    with tqdm(total=total if total > 0 else None,
              desc=os.path.basename(video_path),
              unit="f") as pbar:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
# Convert BGR to RGB
            image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = pose.process(image_rgb)

            frame_rec = {"frame": frame_idx, "landmarks": {}}

            if res.pose_landmarks:
                lmks = res.pose_landmarks.landmark
                for i, l in enumerate(lmks):
                    name = mp_pose.PoseLandmark(i).name.lower()
                    frame_rec["landmarks"][name] = [
                        round(l.x, 6), round(l.y, 6), round(l.z, 6), round(l.visibility, 6)
                    ]

            frames.append(frame_rec)


            frame_idx += 1
            pbar.update(1)

    cap.release()
    return float(fps), frames

def process_all():
    SUPPORTED = (".mp4", ".mov", ".m4v", ".avi", ".MP4", ".MOV", ".M4V", ".AVI")
    # walk the folder
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
        fps, frames = extract_one(in_path)

        payload = {"shot_id": shot_id, "fps": fps, "frames": frames}
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

        print(f"[DONE] Saved → {out_path}")

if __name__ == "__main__":
    process_all()
    print("\n Finished all videos.")

