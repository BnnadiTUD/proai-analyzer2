import os
import json
import math
from typing import Dict, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm

ROOT       = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2"
VIDEOS_DIR = os.path.join(ROOT, "trimmed_videos")
POSES_DIR  = os.path.join(ROOT, "poses_blazepose")
OUT_DIR    = os.path.join(ROOT, "debug_out_contact")
os.makedirs(OUT_DIR, exist_ok=True)

MIN_VIS   = 0.3      # visibility threshold for drawing
SMOOTH_N  = 3        # smoothing for ankles to detect contact
WINDOW    = 5        # neighborhood used when guessing contact by foot speed

# BlazePose landmark names (33 total, we use a subset)
LANDMARKS = [
    "nose",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_foot_index","right_foot_index",
]

# Skeleton edges for drawing
EDGES = [
    ("left_shoulder","right_shoulder"),
    ("left_hip","right_hip"),
    ("left_shoulder","left_elbow"), ("left_elbow","left_wrist"),
    ("right_shoulder","right_elbow"), ("right_elbow","right_wrist"),
    ("left_hip","left_knee"), ("left_knee","left_ankle"),
    ("right_hip","right_knee"), ("right_knee","right_ankle"),
    ("left_ankle","left_heel"), ("left_heel","left_foot_index"),
    ("right_ankle","right_heel"), ("right_heel","right_foot_index"),
    ("left_shoulder","left_hip"), ("right_shoulder","right_hip"),
]

def moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1 or x.size == 0:
        return x
    y = np.convolve(x, np.ones(n)/n, mode="same")
    for i in range(n//2):
        y[i] = np.mean(x[:i+1])
        y[-i-1] = np.mean(x[-(i+1):])
    return y

def build_series(frames, joint_name: str) -> np.ndarray:
    coords = []
    for fr in frames:
        lm = fr.get("landmarks", {}).get(joint_name)
        if not lm:
            coords.append([np.nan, np.nan])
        else:
            x, y, z, vis = lm
            if vis < MIN_VIS:
                coords.append([np.nan, np.nan])
            else:
                coords.append([float(x), float(y)])
    if not coords:
        return np.empty((0, 2), dtype=float)
    arr = np.array(coords, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 2)
    return arr

def contact_from_foot_speed(la: np.ndarray, ra: np.ndarray) -> int:
    """Guess contact frame as where combined ankle speed is highest."""
    T = max(la.shape[0], ra.shape[0])
    if T == 0:
        return 0

    def speed(a):
        if a.shape[0] < 2:
            return np.zeros(a.shape[0])
        v = np.linalg.norm(np.diff(a, axis=0), axis=1)
        v = np.r_[0, v]
        return moving_avg(v, SMOOTH_N)

    vL = speed(la)
    vR = speed(ra)
    # pad to same length
    if vL.size < T: vL = np.pad(vL, (0, T-vL.size))
    if vR.size < T: vR = np.pad(vR, (0, T-vR.size))
    score = vL + vR
    # if all zero/NaN, just pick middle frame
    if not np.isfinite(score).any():
        return T // 2
    return int(np.nanargmax(score))

def draw_skeleton(frame, frame_landmarks: Dict[str, list]):
    h, w = frame.shape[:2]
    pts = {}

    for name in LANDMARKS:
        lm = frame_landmarks.get(name)
        if not lm:
            continue
        x, y, z, vis = lm
        if vis < MIN_VIS:
            continue
        px, py = int(x * w), int(y * h)
        pts[name] = (px, py)
        cv2.circle(frame, (px, py), 3, (0, 255, 0), -1)

    for a, b in EDGES:
        if a in pts and b in pts:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)

def process_one(shot_base: str):
    # find video path
    video_path = None
    for ext in (".mp4",".mov",".m4v",".avi",".MP4",".MOV",".M4V",".AVI"):
        cand = os.path.join(VIDEOS_DIR, shot_base + ext)
        if os.path.exists(cand):
            video_path = cand
            break
    pose_path = os.path.join(POSES_DIR, shot_base + ".json")

    if not video_path or not os.path.exists(pose_path):
        print(f"[MISS] {shot_base}: missing video or pose json")
        return

    # load pose
    with open(pose_path, "r", encoding="utf-8") as f:
        d = json.load(f)
    frames = d.get("frames", [])
    T = len(frames)
    if T == 0:
        print(f"[SKIP] {shot_base}: no pose frames")
        return

    # build ankle series
    la = build_series(frames, "left_ankle")
    ra = build_series(frames, "right_ankle")

    if la.size == 0 and ra.size == 0:
        print(f"[SKIP] {shot_base}: no ankle data")
        return

    # smooth
    if la.ndim == 2 and la.shape[0] > 0:
        la[:,0] = moving_avg(la[:,0], SMOOTH_N)
        la[:,1] = moving_avg(la[:,1], SMOOTH_N)
    if ra.ndim == 2 and ra.shape[0] > 0:
        ra[:,0] = moving_avg(ra[:,0], SMOOTH_N)
        ra[:,1] = moving_avg(ra[:,1], SMOOTH_N)

    # guess contact frame
    k = contact_from_foot_speed(la, ra)
    k = max(0, min(T-1, k))

    # grab that frame from video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[SKIP] {shot_base}: cannot open video")
        return
    cap.set(cv2.CAP_PROP_POS_FRAMES, k)
    ok, frame = cap.read()
    cap.release()
    if not ok:
        print(f"[SKIP] {shot_base}: cannot read frame {k}")
        return

    # draw skeleton on contact frame
    frame_lm = frames[k].get("landmarks", {})
    draw_skeleton(frame, frame_lm)

    # label HUD
    cv2.putText(frame, f"{shot_base}  contact_frame={k}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

    out_path = os.path.join(OUT_DIR, f"{shot_base}_contact.png")
    cv2.imwrite(out_path, frame)
    print(f"[OK] {shot_base} -> {out_path}")

def main():
    files = [f for f in os.listdir(VIDEOS_DIR)
             if f.lower().endswith((".mp4",".mov",".m4v",".avi"))]
    if not files:
        print(f"[WARN] No videos in {VIDEOS_DIR}")
        return

    # turn into unique basenames (shot_1, shot_10, etc.)
    bases = sorted({os.path.splitext(f)[0] for f in files})

    for b in tqdm(bases, desc="render_all"):
        process_one(b)

if __name__ == "__main__":
    main()
