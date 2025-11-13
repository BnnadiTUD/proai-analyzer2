import os, json, math
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
from tqdm import tqdm

# ==== YOUR PATHS ====
ROOT       = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2"
VIDEOS_DIR = os.path.join(ROOT, "trimmed_videos")
POSES_DIR  = os.path.join(ROOT, "poses_blazepose")
OUT_DIR    = os.path.join(ROOT, "debug_out")
os.makedirs(OUT_DIR, exist_ok=True)

# ==== VISUAL WINDOW / PARAMS ====
WINDOW_PRE  = 10
WINDOW_POST = 18
SMOOTH_N    = 3
MIN_VIS     = 0.3

# Ball detect params (same idea as before)
BALL_SEARCH_PAD = 200
BALL_MIN_RAD = 4
BALL_MAX_RAD = 30

# ---- BlazePose 33 landmark names (subset used for skeleton) ----
J = [
    "nose",
    "left_eye_inner","left_eye","left_eye_outer",
    "right_eye_inner","right_eye","right_eye_outer",
    "left_ear","right_ear",
    "mouth_left","mouth_right",
    "left_shoulder","right_shoulder",
    "left_elbow","right_elbow",
    "left_wrist","right_wrist",
    "left_pinky","right_pinky",
    "left_index","right_index",
    "left_thumb","right_thumb",
    "left_hip","right_hip",
    "left_knee","right_knee",
    "left_ankle","right_ankle",
    "left_heel","right_heel",
    "left_foot_index","right_foot_index"
]

# Simple skeleton connections (good enough for technique viz)
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

# ---------- utils ----------
def moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1: return x
    y = np.convolve(x, np.ones(n)/n, mode="same")
    # fix border shrink
    for i in range(n//2):
        y[i] = np.mean(x[:i+1]); y[-i-1] = np.mean(x[-(i+1):])
    return y

def get_xy(frame_rec: Dict, name: str):
    lm = frame_rec.get("landmarks", {}).get(name)
    if lm is None:
        return None
    x, y, z, vis = lm
    if vis < MIN_VIS:
        return None
    return (float(x), float(y), float(vis))

def contact_from_foot_speed(la: np.ndarray, ra: np.ndarray) -> int:
    vL = np.r_[0, np.linalg.norm(np.diff(la,axis=0),axis=1)] if la.size else np.zeros(1)
    vR = np.r_[0, np.linalg.norm(np.diff(ra,axis=0),axis=1)] if ra.size else np.zeros(1)
    vL = moving_avg(vL, SMOOTH_N); vR = moving_avg(vR, SMOOTH_N)
    S = (vL - np.nanmean(vL))/(np.nanstd(vL)+1e-6) + (vR - np.nanmean(vR))/(np.nanstd(vR)+1e-6)
    return int(np.nanargmax(S))

def hough_ball_positions(video_path: str, guess_k: int, feet_px: List[Tuple[int,int]], H:int, W:int,
                         pre:int, post:int) -> np.ndarray:
    start = max(0, guess_k - pre)
    end   = guess_k + post
    T = end - start + 1
    centers = np.full((T,2), np.nan, dtype=float)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return centers

    prev = None
    for idx, f in enumerate(range(start, end+1)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, f)
        ok, frame = cap.read()
        if not ok: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 1.5)

        mask = np.zeros((H,W), dtype=np.uint8)
        for (fx,fy) in feet_px:
            if fx is None: continue
            cv2.circle(mask, (int(fx), int(fy)), BALL_SEARCH_PAD, 255, -1)
        roi = cv2.bitwise_and(gray, gray, mask=mask)

        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=120, param2=15, minRadius=BALL_MIN_RAD, maxRadius=BALL_MAX_RAD
        )
        cand = None
        if circles is not None:
            c = np.uint16(np.around(circles))[0, :]
            if prev is not None:
                dists = [np.hypot(cx-prev[0], cy-prev[1]) for (cx,cy,_) in c]
                j = int(np.argmin(dists))
                cand = (float(c[j][0]), float(c[j][1]))
            else:
                j = int(np.argmin([(abs(cx-W/2)+abs(cy-H/2)) for (cx,cy,_) in c]))
                cand = (float(c[j][0]), float(c[j][1]))
        if cand is not None:
            centers[idx] = cand
            prev = cand

    cap.release()
    return centers  # NaN rows on miss

def draw_skeleton(frame, pts_px: Dict[str,Tuple[int,int]], color=(0,255,0)):
    # joints
    for name, (x,y) in pts_px.items():
        if x is None or y is None: continue
        cv2.circle(frame, (int(x),int(y)), 3, color, -1)
    # edges
    for a,b in EDGES:
        pa = pts_px.get(a); pb = pts_px.get(b)
        if not pa or not pb: continue
        if None in pa or None in pb: continue
        cv2.line(frame, (int(pa[0]),int(pa[1])), (int(pb[0]),int(pb[1])), color, 2)

def to_px(xy_norm, W, H):
    if xy_norm is None: return None
    x,y,_ = xy_norm
    if np.isnan(x) or np.isnan(y): return None
    return (int(x*W), int(y*H))

def visualize_one(shot_base: str, write_video=True) -> Optional[str]:
    video_path = os.path.join(VIDEOS_DIR, f"{shot_base}.mp4")
    if not os.path.exists(video_path):
        # try other extensions
        for ext in (".mov",".m4v",".avi",".MP4",".MOV",".M4V",".AVI"):
            p2 = os.path.join(VIDEOS_DIR, f"{shot_base}{ext}")
            if os.path.exists(p2):
                video_path = p2; break
    pose_path  = os.path.join(POSES_DIR,  f"{shot_base}.json")
    if not os.path.exists(video_path) or not os.path.exists(pose_path):
        print(f"[MISS] {shot_base}: video or pose json missing")
        return None

    with open(pose_path,"r",encoding="utf-8") as f:
        d = json.load(f)
    frames = d["frames"]; fps = float(d.get("fps",30.0))
    T = len(frames)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"[MISS] cannot open {video_path}")
        return None
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)); W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # ---- build ankle time series safely ----
    def build_series(frames_list, joint_name: str) -> np.ndarray:
        coords = []
        for fr in frames_list:
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
            # if something went weird, force shape (N,2)
            arr = arr.reshape(-1, 2)
        return arr

    la = build_series(frames, "left_ankle")
    ra = build_series(frames, "right_ankle")

    # if no valid frames, bail for this shot
    if la.size == 0 or ra.size == 0:
        cap.release()
        print(f"[SKIP] {shot_base}: no ankle data")
        return None

    # smooth (only if correct shape)
    if la.ndim == 2 and la.shape[0] > 0:
        la[:, 0] = moving_avg(la[:, 0], SMOOTH_N)
        la[:, 1] = moving_avg(la[:, 1], SMOOTH_N)
    if ra.ndim == 2 and ra.shape[0] > 0:
        ra[:, 0] = moving_avg(ra[:, 0], SMOOTH_N)
        ra[:, 1] = moving_avg(ra[:, 1], SMOOTH_N)


    def build_series(frames, joint_name):
        coords = []
        for fr in frames:
            xy = get_xy(fr, joint_name)
            if xy is None:
                coords.append([np.nan, np.nan])
            else:
                x, y, _ = xy
                coords.append([x, y])
        return np.array(coords, dtype=float)

    # collect per-frame ankle coords (normalized)
    la = build_series(frames, "left_ankle")
    ra = build_series(frames, "right_ankle")
    la[:,0] = moving_avg(la[:,0], SMOOTH_N); la[:,1] = moving_avg(la[:,1], SMOOTH_N)
    ra[:,0] = moving_avg(ra[:,0], SMOOTH_N); ra[:,1] = moving_avg(ra[:,1], SMOOTH_N)

    # contact guess and window
    k_guess = contact_from_foot_speed(la, ra)
    k0 = max(0, k_guess - WINDOW_PRE)
    k1 = min(T-1, k_guess + WINDOW_POST)

    # feet px around guess for ROI
    feet_px = []
    for t in range(k0, min(T-1, k_guess+2)):
        if not np.isnan(la[t]).any(): feet_px.append((int(la[t,0]*W), int(la[t,1]*H)))
        if not np.isnan(ra[t]).any(): feet_px.append((int(ra[t,0]*W), int(ra[t,1]*H)))
    if not feet_px: feet_px = [(W//2,H//2)]

    # detect ball centers in window (pixels)
    ball_px_window = hough_ball_positions(video_path, k_guess, feet_px, H, W, WINDOW_PRE, WINDOW_POST)

    # Prepare writer
    out_path = os.path.join(OUT_DIR, f"{shot_base}_debug.mp4")
    if write_video:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W,H))
    else:
        writer = None

    # Draw loop over full video but annotate window clearly
    for t in tqdm(range(T), desc=f"render {shot_base}"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, t)
        ok, frame = cap.read()
        if not ok: break

        # build landmarks dict -> px
        fr = frames[t]
        pts_px = {}
        for name in J:
            xy = get_xy(fr, name)
            pts_px[name] = to_px(xy, W, H) if xy else (None,None)

        # draw skeleton
        draw_skeleton(frame, pts_px, color=(0,255,0))

        # overlay ball (only within window we searched)
        if k0 <= t <= k1:
            idx = t - k0
            bx, by = ball_px_window[idx] if idx < len(ball_px_window) else (np.nan,np.nan)
            if not np.isnan(bx):
                cv2.circle(frame, (int(bx),int(by)), 6, (0,165,255), -1)  # orange ball
                # trajectory tail
                for j in range(max(0,idx-6), idx):
                    pbx,pby = ball_px_window[j]
                    if not (np.isnan(pbx) or np.isnan(pby)):
                        cv2.circle(frame, (int(pbx),int(pby)), 3, (0,140,255), -1)

        # mark contact guess frame with a red vertical bar/top text
        if t == k_guess:
            cv2.putText(frame, "CONTACT (guess)", (20,40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)
            cv2.rectangle(frame, (0,0), (W,5), (0,0,255), -1)

        # draw window bounds
        if t == k0:
            cv2.rectangle(frame, (0,H-6), (W,H-1), (0,255,255), -1)  # start (yellow)
        if t == k1:
            cv2.rectangle(frame, (0,H-12), (W,H-7), (0,255,255), -1) # end (yellow)

        # HUD
        cv2.putText(frame, f"{shot_base}  t={t}  k0={k0}  k*~{k_guess}  k1={k1}", (20,H-20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 2)

        if writer: writer.write(frame)
        else:
            cv2.imshow("debug", frame)
            if cv2.waitKey(int(1000//fps)) & 0xFF == 27: break

    cap.release()
    if writer: writer.release()
    if not write_video: cv2.destroyAllWindows()
    return out_path if write_video else None


if __name__ == "__main__":
    # EXAMPLES:
    # - Single clip by base name (without extension)
    #   shot = "p01_s001"
    # - Or iterate all mp4s
    shot = None

    if shot:
        out = visualize_one(shot, write_video=True)
        print(f"\nSaved: {out}")
    else:
        files = [os.path.splitext(f)[0] for f in os.listdir(VIDEOS_DIR)
                 if f.lower().endswith((".mp4",".mov",".m4v",".avi"))]
        files.sort()
        for base in files[:5]:   # render first 5 for sanity — change/remove slice as you like
            out = visualize_one(base, write_video=True)
            print(f"Saved: {out}")
