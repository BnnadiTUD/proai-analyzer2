import os, json, math, csv
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2
from tqdm import tqdm

#shot	striking_foot	plant_to_ball_norm	trunk_lean	hip_facing	follow_arc	lock_angle	Interpretation
# my paths
VIDEOS_DIR = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2\trimmed_videos"
POSES_DIR  = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2\poses_blazepose"
OUT_CSV    = r"C:\Users\pc\OneDrive - Technological University Dublin\Pro-AI Analyzer 2\features.csv"

# Analysis parameters
WINDOW_PRE  = 10         # frames before contact for analysis window
WINDOW_POST = 18         # frames after contact
FOLLOW_MS   = 300        # follow-through duration (ms)
SMOOTH_N    = 3          # moving-average smoothing window (frames)
MIN_VIS     = 0.3      # ignore landmarks below this visibility
BALL_SEARCH_PAD = 200    # px radius ROI around feet for HoughCircles
BALL_MIN_RAD   = 4       # px, adjust to your footage
BALL_MAX_RAD   = 30      # px, adjust to your footage

# POSE LANDMARK KEYS (MediaPipe BlazePose)
L = {
    "left_hip":"left_hip", "right_hip":"right_hip",
    "left_knee":"left_knee", "right_knee":"right_knee",
    "left_ankle":"left_ankle","right_ankle":"right_ankle",
    "left_shoulder":"left_shoulder","right_shoulder":"right_shoulder",
    "left_foot_index":"left_foot_index","right_foot_index":"right_foot_index"
}

#utils
def moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1: return x
    k = n
    c = np.convolve(x, np.ones(k)/k, mode='same')
    # fix boundary shrink
    for i in range(k//2):
        c[i] = np.mean(x[:i+1])
        c[-i-1] = np.mean(x[-(i+1):])
    return c

#used for angle calculations for locking ankle
def ang_deg(p, q, r) -> float:
    """Angle at q formed by p-q-r (internal angle, degrees)."""
    v1 = np.array(p) - np.array(q)
    v2 = np.array(r) - np.array(q)
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6: return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))

#for trunk lean and hip facing
def line_angle_deg(v, axis="vertical") -> float:
    """Angle of vector v against vertical or horizontal."""
    vx, vy = v[0], v[1]
    if axis == "vertical":
        # angle to (0,-1) up; positive forward lean = torso ahead (vy negative)
        ref = np.array([0.0, -1.0])
    else:
        ref = np.array([1.0, 0.0])
    n1 = np.linalg.norm([vx, vy]); n2 = np.linalg.norm(ref)
    if n1 < 1e-6: return float("nan")
    cos = np.clip((vx*ref[0]+vy*ref[1])/(n1*n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))

# hip width for normalization
def hip_width(pose_frame: Dict) -> Optional[float]:
    try:
        lh = pose_frame["landmarks"][L["left_hip"]]
        rh = pose_frame["landmarks"][L["right_hip"]]
        return float(np.linalg.norm(np.array(lh[:2])-np.array(rh[:2])))
    except: return None

def get_xy(pose_frame: Dict, name: str) -> Optional[Tuple[float,float]]:
    if name not in pose_frame["landmarks"]: return None
    x,y,z,vis = pose_frame["landmarks"][name]
    if vis < MIN_VIS: return None
    return (x,y)

def zscore(x):
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x); m = np.nanmean(x)
    return (x - m) / (s + 1e-6)

# Contact detection helpers
def contact_from_ball_and_feet(ball_xy: np.ndarray, la_xy: np.ndarray, ra_xy: np.ndarray, fallback_idx: int) -> int:
    """contact = frame with minimal distance between ball and either ankle, else fallback."""
    if ball_xy is None or len(ball_xy) == 0:
        return fallback_idx

    dL = np.linalg.norm(ball_xy - la_xy, axis=1) if la_xy is not None else np.full(len(ball_xy), np.inf)
    dR = np.linalg.norm(ball_xy - ra_xy, axis=1) if ra_xy is not None else np.full(len(ball_xy), np.inf)
    d  = np.minimum(dL, dR)

    # keep only finite values
    valid = np.isfinite(d)
    if not np.any(valid):
        return fallback_idx  # no usable ball-foot distances

    # assume ball_xy is already aligned to the full timeline window
    best_rel = np.argmin(d[valid])
    best_idx = np.arange(len(d))[valid][best_rel]
    return int(best_idx)


def contact_from_foot_speed(la_xy: np.ndarray, ra_xy: np.ndarray) -> int:
    vL = np.r_[0, np.linalg.norm(np.diff(la_xy,axis=0),axis=1)] if la_xy is not None else np.zeros(1)
    vR = np.r_[0, np.linalg.norm(np.diff(ra_xy,axis=0),axis=1)] if ra_xy is not None else np.zeros(1)
    S  = zscore(moving_avg(vL,SMOOTH_N)) + zscore(moving_avg(vR,SMOOTH_N))
    return int(np.nanargmax(S))

#  Ball detection around contact 
def hough_ball_positions(video_path: str, guess_k: int, feet_px: List[Tuple[int,int]], frame_shape, fps: float) -> Tuple[np.ndarray, List[bool]]:
    """
    Detect ball center (in pixel coords) around guess_k±WINDOW using HoughCircles.
    Restrict search to ROI around the feet to reduce false positives.
    Returns (T,2) array with NaNs for misses, and list of found flags.
    """
    h, w = frame_shape[:2]
    start = max(0, guess_k - WINDOW_PRE)
    end   = guess_k + WINDOW_POST
    T = end - start + 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    # Seek helper
    def read_frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        return frm if ok else None

    centers = np.full((T,2), np.nan, dtype=float)
    found   = [False]*T
    prev_c  = None

    for idx, f in enumerate(range(start, end+1)):
        frame = read_frame(f)
        if frame is None: continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7,7), 1.5)

        # Build ROI around feet
        mask = np.zeros((h,w), dtype=np.uint8)
        for (fx,fy) in feet_px:
            if fx is None: continue
            cv2.circle(mask, (int(fx), int(fy)), BALL_SEARCH_PAD, 255, -1)
        roi = cv2.bitwise_and(gray, gray, mask=mask)

        # HoughCircles (with tuned param2 incase theres too many/few)
        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=120, param2=15, minRadius=BALL_MIN_RAD, maxRadius=BALL_MAX_RAD
        )
        candidate = None
        if circles is not None:
            c = np.uint16(np.around(circles))[0, :]
            # choose closest to previous, else any
            if prev_c is not None:
                dists = [np.hypot(cx-prev_c[0], cy-prev_c[1]) for (cx,cy,_) in c]
                j = int(np.argmin(dists))
                candidate = (float(c[j][0]), float(c[j][1]))
            else:
                # pick the most central among detected
                j = int(np.argmin([(abs(cx-w/2)+abs(cy-h/2)) for (cx,cy,_) in c]))
                candidate = (float(c[j][0]), float(c[j][1]))

        if candidate is not None:
            centers[idx] = candidate
            prev_c = candidate
            found[idx] = True

    cap.release()
    return centers, found

#  Core per-clip analysis 
def analyze_clip(video_path: str, pose_json_path: str) -> Optional[Dict]:
    # Load pose
    if not os.path.exists(pose_json_path): return None
    with open(pose_json_path, "r", encoding="utf-8") as f:
        pose = json.load(f)

    fps = float(pose.get("fps", 30.0))
    frames = pose["frames"]
    T_pose = len(frames)
    if T_pose < 3: return None

    # Build per-frame XY for key joints 
    def series(name):
        arr = []
        for fr in frames:
            xy = get_xy(fr, name)
            arr.append([np.nan, np.nan] if xy is None else [xy[0], xy[1]])
        return np.array(arr, dtype=float)

    la = series(L["left_ankle"])
    ra = series(L["right_ankle"])
    lk = series(L["left_knee"])
    rk = series(L["right_knee"])
    lfi= series(L["left_foot_index"])
    rfi= series(L["right_foot_index"])
    lhip= series(L["left_hip"])
    rhip= series(L["right_hip"])
    lsh = series(L["left_shoulder"])
    rsh = series(L["right_shoulder"])

    # Smooth ankle traces
    for arr in (la, ra, lhip, rhip, lsh, rsh, lfi, rfi, lk, rk):
        for c in (0,1):
            arr[:,c] = moving_avg(arr[:,c], SMOOTH_N)

    # First estimate contact by foot speed peak (seconds)
    k_guess = contact_from_foot_speed(la, ra)

    # Map normalized foot coords to pixel coords for ball search actual frame size to convert to px
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened(): return None
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    def to_px(xy):
        return None if (xy is None or np.isnan(xy[0]) or np.isnan(xy[1])) else (int(xy[0]*W), int(xy[1]*H))

    feet_px = []
    for f in range(max(0,k_guess-2), min(T_pose-1,k_guess+2)):
        lpx = to_px(la[f]) if not np.isnan(la[f]).any() else None
        rpx = to_px(ra[f]) if not np.isnan(ra[f]).any() else None
        if lpx: feet_px.append(lpx)
        if rpx: feet_px.append(rpx)
    if not feet_px:
        # fallback
        feet_px = [(W//2, H//2)]

    # Detect ball centers around k_guess window in pixels
    ball_px, ok = hough_ball_positions(video_path, k_guess, feet_px, (H,W,3), fps=fps)

    # Convert detected ball_px in the window back to normalized, aligned to full timeline
    start = max(0, k_guess - WINDOW_PRE)
    end   = k_guess + WINDOW_POST
    bp = np.full((T_pose,2), np.nan, dtype=float)
    for i, t in enumerate(range(start, end+1)):
        if 0 <= t < T_pose and not np.isnan(ball_px[i,0]):
            bp[t,0] = ball_px[i,0] / W
            bp[t,1] = ball_px[i,1] / H

    # If ball found on enough frames, recompute contact as foot-ball min
    valid = np.where(~np.isnan(bp[:,0]))[0]
    if len(valid) >= 3:
        k_contact = contact_from_ball_and_feet(bp, la, ra, fallback_idx=k_guess)
    else:
        k_contact = k_guess  # fallback

    # Safety clamp for windows
    k0 = max(0, k_contact - WINDOW_PRE)
    k1 = min(T_pose-1, k_contact + WINDOW_POST)

    # Determine striking vs plant foot at contact (closest ankle to ball)
    if not np.isnan(bp[k_contact,0]):
        dL = np.linalg.norm(bp[k_contact]-la[k_contact])
        dR = np.linalg.norm(bp[k_contact]-ra[k_contact])
        striking = "left" if dL <= dR else "right"
    else:
        # fallback: which foot is moving faster at k_contact
        vL = 0.0 if k_contact==0 else np.linalg.norm(la[k_contact]-la[k_contact-1])
        vR = 0.0 if k_contact==0 else np.linalg.norm(ra[k_contact]-ra[k_contact-1])
        striking = "left" if vL >= vR else "right"

    # Build mid-hip and mid-shoulder
    mid_hip = (lhip + rhip) / 2.0
    mid_sh  = (lsh  + rsh ) / 2.0

    #  Metrics 
    # Hip width for normalization
    hw = []
    for t in range(k0, k1+1):
        if not (np.isnan(lhip[t]).any() or np.isnan(rhip[t]).any()):
            hw.append(np.linalg.norm(lhip[t]-rhip[t]))
    norm_hip = float(np.nanmedian(hw)) if len(hw)>0 else np.nan

    # 1) Standing foot ball distance (normalized by hip width) at impact
    if not np.isnan(bp[k_contact,0]):
        if striking == "left":
            plant_xy = ra[k_contact]
        else:
            plant_xy = la[k_contact]
        plant2ball = float(np.linalg.norm(plant_xy - bp[k_contact])) if not np.isnan(plant_xy).any() else np.nan
        plant2ball_norm = plant2ball / (norm_hip + 1e-6) if not math.isnan(plant2ball) else np.nan
    else:
        plant2ball_norm = np.nan

    # 2) Body angle/position at impact
    # Trunk lean: angle between vector (mid_hip - mid_shoulder) and vertical
    if not (np.isnan(mid_hip[k_contact]).any() or np.isnan(mid_sh[k_contact]).any()):
        torso_vec = mid_sh[k_contact] - mid_hip[k_contact]
        trunk_lean_deg = line_angle_deg(torso_vec, axis="vertical")
        # make "forward lean" negative if shoulders ahead (smaller y)
        if torso_vec[1] < 0:
            trunk_lean_deg = -abs(trunk_lean_deg)
        else:
            trunk_lean_deg = abs(trunk_lean_deg)
    else:
        trunk_lean_deg = np.nan

    # Hip facing: angle of left_hip -> right_hip against horizontal
    if not (np.isnan(lhip[k_contact]).any() or np.isnan(rhip[k_contact]).any()):
        hip_vec = rhip[k_contact] - lhip[k_contact]
        hip_facing_deg = line_angle_deg(hip_vec, axis="horizontal")
    else:
        hip_facing_deg = np.nan

    # 3) Follow-through arc: hip rotation change over FOLLOW_MS after contact
    frames_follow = int(round((FOLLOW_MS/1000.0) * fps))
    end_ft = min(T_pose-1, k_contact + frames_follow)
    hip_angles = []
    for t in range(k_contact, end_ft+1):
        if not (np.isnan(lhip[t]).any() or np.isnan(rhip[t]).any()):
            hv = rhip[t] - lhip[t]
            hip_angles.append(math.degrees(math.atan2(hv[1], hv[0])))
    follow_arc = float(abs((hip_angles[-1] - hip_angles[0]))) if len(hip_angles)>=2 else np.nan

    # 4) Lock angle (knee-ankle-toe) at impact (striking foot)
    if striking == "left":
        knee = lk[k_contact]; ankle = la[k_contact]; toe = lfi[k_contact]
    else:
        knee = rk[k_contact]; ankle = ra[k_contact]; toe = rfi[k_contact]
    if not (np.isnan(knee).any() or np.isnan(ankle).any() or np.isnan(toe).any()):
        lock_angle = float(ang_deg(knee, ankle, toe))  # smaller angle ⇒ more locked
    else:
        lock_angle = np.nan

    # Package
    return {
        "shot_id": os.path.splitext(os.path.basename(video_path))[0],
        "fps": fps,
        "contact_frame": int(k_contact),
        "striking_foot": striking,
        "plant_to_ball_norm": round(plant2ball_norm, 4) if not math.isnan(plant2ball_norm) else "",
        "trunk_lean_deg": round(trunk_lean_deg, 2) if not math.isnan(trunk_lean_deg) else "",
        "hip_facing_deg": round(hip_facing_deg, 2) if not math.isnan(hip_facing_deg) else "",
        "follow_through_arc_deg": round(follow_arc, 2) if not math.isnan(follow_arc) else "",
        "lock_angle_deg": round(lock_angle, 2) if not math.isnan(lock_angle) else ""
    }

def main():
    files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4",".mov",".m4v",".avi"))]
    files.sort()
    if not files:
        print(f"[WARN] No videos found in {VIDEOS_DIR}")
        return

    rows = []
    for fname in files:
        video_path = os.path.join(VIDEOS_DIR, fname)
        shot_id = os.path.splitext(fname)[0]
        pose_json_path = os.path.join(POSES_DIR, f"{shot_id}.json")
        try:
            res = analyze_clip(video_path, pose_json_path)
            if res: rows.append(res)
            else: print(f"[SKIP] {fname} (pose or frames missing)")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    if rows:
        keys = ["shot_id","fps","contact_frame","striking_foot",
                "plant_to_ball_norm","trunk_lean_deg","hip_facing_deg",
                "follow_through_arc_deg","lock_angle_deg"]
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n Wrote {len(rows)} rows → {OUT_CSV}")
    else:
        print("[WARN] No results written.")

if __name__ == "__main__":
    main()
