import os, json, math, csv
from typing import Dict, Tuple, List, Optional
import numpy as np
import cv2

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
MIN_VIS     = 0.3        # ignore landmarks below this visibility
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


def moving_avg(x: np.ndarray, n: int) -> np.ndarray:
    if n <= 1:
        return x
    k = n
    c = np.convolve(x, np.ones(k) / k, mode='same')
    for i in range(k // 2):
        c[i] = np.mean(x[:i + 1])
        c[-i - 1] = np.mean(x[-(i + 1):])
    return c


def ang_deg(p, q, r) -> float:
    """Angle at q formed by p-q-r (internal angle, degrees)."""
    v1 = np.array(p) - np.array(q)
    v2 = np.array(r) - np.array(q)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def angle_at(p, q, r) -> float:
    """Angle at q formed by p-q-r (internal angle, degrees)."""
    v1 = np.array(p, dtype=float) - np.array(q, dtype=float)
    v2 = np.array(r, dtype=float) - np.array(q, dtype=float)
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-6 or n2 < 1e-6:
        return float("nan")
    cos = np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def normalize_signed_angle(angle: float) -> float:
    """Keep direction but fold wrapped body angles into a stable [-90, 90] style range."""
    a = ((angle + 180.0) % 360.0) - 180.0
    if a > 90.0:
        a -= 180.0
    elif a < -90.0:
        a += 180.0
    return a


def normalize_unsigned_angle(angle: float) -> float:
    """Fold to the smaller equivalent angle in [0, 180]."""
    a = abs(angle) % 360.0
    if a > 180.0:
        a = 360.0 - a
    return a


def line_angle_deg(v, axis="vertical") -> float:
    """Angle of vector v against vertical or horizontal."""
    vx, vy = v[0], v[1]
    if axis == "vertical":
        ref = np.array([0.0, -1.0])
    else:
        ref = np.array([1.0, 0.0])
    n1 = np.linalg.norm([vx, vy])
    n2 = np.linalg.norm(ref)
    if n1 < 1e-6:
        return float("nan")
    cos = np.clip((vx * ref[0] + vy * ref[1]) / (n1 * n2), -1.0, 1.0)
    return math.degrees(math.acos(cos))


def hip_width(pose_frame: Dict) -> Optional[float]:
    try:
        lh = pose_frame["landmarks"][L["left_hip"]]
        rh = pose_frame["landmarks"][L["right_hip"]]
        return float(np.linalg.norm(np.array(lh[:2]) - np.array(rh[:2])))
    except Exception:
        return None


def get_xy(pose_frame: Dict, name: str) -> Optional[Tuple[float, float]]:
    if name not in pose_frame["landmarks"]:
        return None
    x, y, z, vis = pose_frame["landmarks"][name]
    if vis < MIN_VIS:
        return None
    return (x, y)


def zscore(x):
    x = np.asarray(x, dtype=float)
    s = np.nanstd(x)
    m = np.nanmean(x)
    return (x - m) / (s + 1e-6)


def contact_from_ball_and_feet(ball_xy: np.ndarray, la_xy: np.ndarray, ra_xy: np.ndarray, fallback_idx: int) -> int:
    """contact = frame with minimal distance between ball and either ankle, else fallback."""
    if ball_xy is None or len(ball_xy) == 0:
        return fallback_idx

    dL = np.linalg.norm(ball_xy - la_xy, axis=1) if la_xy is not None else np.full(len(ball_xy), np.inf)
    dR = np.linalg.norm(ball_xy - ra_xy, axis=1) if ra_xy is not None else np.full(len(ball_xy), np.inf)
    d = np.minimum(dL, dR)

    valid = np.isfinite(d)
    if not np.any(valid):
        return fallback_idx

    best_rel = np.argmin(d[valid])
    best_idx = np.arange(len(d))[valid][best_rel]
    return int(best_idx)


def contact_from_foot_speed(la_xy: np.ndarray, ra_xy: np.ndarray) -> int:
    vL = np.r_[0, np.linalg.norm(np.diff(la_xy, axis=0), axis=1)] if la_xy is not None else np.zeros(1)
    vR = np.r_[0, np.linalg.norm(np.diff(ra_xy, axis=0), axis=1)] if ra_xy is not None else np.zeros(1)
    S = zscore(moving_avg(vL, SMOOTH_N)) + zscore(moving_avg(vR, SMOOTH_N))
    return int(np.nanargmax(S))


def hough_ball_positions(video_path: str, guess_k: int, feet_px: List[Tuple[int, int]], frame_shape, fps: float) -> Tuple[np.ndarray, List[bool]]:
    """
    Detect ball center (in pixel coords) around guess_k +/- WINDOW using HoughCircles.
    Restrict search to ROI around the feet to reduce false positives.
    Returns (T,2) array with NaNs for misses, and list of found flags.
    """
    h, w = frame_shape[:2]
    start = max(0, guess_k - WINDOW_PRE)
    end = guess_k + WINDOW_POST
    T = end - start + 1

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    def read_frame(i):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ok, frm = cap.read()
        return frm if ok else None

    centers = np.full((T, 2), np.nan, dtype=float)
    found = [False] * T
    prev_c = None

    for idx, f in enumerate(range(start, end + 1)):
        frame = read_frame(f)
        if frame is None:
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (7, 7), 1.5)

        mask = np.zeros((h, w), dtype=np.uint8)
        for (fx, fy) in feet_px:
            if fx is None:
                continue
            cv2.circle(mask, (int(fx), int(fy)), BALL_SEARCH_PAD, 255, -1)
        roi = cv2.bitwise_and(gray, gray, mask=mask)

        circles = cv2.HoughCircles(
            roi, cv2.HOUGH_GRADIENT, dp=1.2, minDist=20,
            param1=120, param2=15, minRadius=BALL_MIN_RAD, maxRadius=BALL_MAX_RAD
        )
        candidate = None
        if circles is not None:
            c = np.uint16(np.around(circles))[0, :]
            if prev_c is not None:
                dists = [np.hypot(cx - prev_c[0], cy - prev_c[1]) for (cx, cy, _) in c]
                j = int(np.argmin(dists))
                candidate = (float(c[j][0]), float(c[j][1]))
            else:
                j = int(np.argmin([(abs(cx - w / 2) + abs(cy - h / 2)) for (cx, cy, _) in c]))
                candidate = (float(c[j][0]), float(c[j][1]))

        if candidate is not None:
            centers[idx] = candidate
            prev_c = candidate
            found[idx] = True

    cap.release()
    return centers, found


def compute_metrics_for_shot(video_path: str, pose_json_path: str) -> dict:
    """
    Used for FastAPI / single-shot pipeline.
    """
    res = analyze_clip(video_path, pose_json_path)
    if not res:
        raise RuntimeError(f"Could not analyze clip for {video_path}")

    def safe_float(v):
        try:
            if v == "" or v is None:
                return None
            return float(v)
        except (TypeError, ValueError):
            return None

    metrics = {
        "plant_to_ball_norm": safe_float(res.get("plant_to_ball_norm")),
        "trunk_lean_deg": safe_float(res.get("trunk_lean_deg")),
        "hip_facing_deg": safe_float(res.get("hip_facing_deg")),
        "follow_through_arc_deg": safe_float(res.get("follow_through_arc_deg")),
        "lock_angle_deg": safe_float(res.get("lock_angle_deg")),
    }

    return metrics


def analyze_clip(video_path: str, pose_json_path: str) -> Optional[Dict]:
    if not os.path.exists(pose_json_path):
        return None
    with open(pose_json_path, "r", encoding="utf-8") as f:
        pose = json.load(f)

    fps = float(pose.get("fps", 30.0))
    frames = pose["frames"]
    T_pose = len(frames)
    if T_pose < 3:
        return None

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
    lfi = series(L["left_foot_index"])
    rfi = series(L["right_foot_index"])
    lhip = series(L["left_hip"])
    rhip = series(L["right_hip"])
    lsh = series(L["left_shoulder"])
    rsh = series(L["right_shoulder"])

    for arr in (la, ra, lhip, rhip, lsh, rsh, lfi, rfi, lk, rk):
        for c in (0, 1):
            arr[:, c] = moving_avg(arr[:, c], SMOOTH_N)

    k_guess = contact_from_foot_speed(la, ra)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap.release()

    def to_px(xy):
        return None if (xy is None or np.isnan(xy[0]) or np.isnan(xy[1])) else (int(xy[0] * W), int(xy[1] * H))

    feet_px = []
    for f in range(max(0, k_guess - 2), min(T_pose - 1, k_guess + 2)):
        lpx = to_px(la[f]) if not np.isnan(la[f]).any() else None
        rpx = to_px(ra[f]) if not np.isnan(ra[f]).any() else None
        if lpx:
            feet_px.append(lpx)
        if rpx:
            feet_px.append(rpx)
    if not feet_px:
        feet_px = [(W // 2, H // 2)]

    ball_px, ok = hough_ball_positions(video_path, k_guess, feet_px, (H, W, 3), fps=fps)

    start = max(0, k_guess - WINDOW_PRE)
    end = k_guess + WINDOW_POST
    bp = np.full((T_pose, 2), np.nan, dtype=float)
    for i, t in enumerate(range(start, end + 1)):
        if 0 <= t < T_pose and not np.isnan(ball_px[i, 0]):
            bp[t, 0] = ball_px[i, 0] / W
            bp[t, 1] = ball_px[i, 1] / H

    valid = np.where(~np.isnan(bp[:, 0]))[0]
    if len(valid) >= 3:
        k_contact = contact_from_ball_and_feet(bp, la, ra, fallback_idx=k_guess)
    else:
        k_contact = k_guess

    k0 = max(0, k_contact - WINDOW_PRE)
    k1 = min(T_pose - 1, k_contact + WINDOW_POST)

    if not np.isnan(bp[k_contact, 0]):
        dL = np.linalg.norm(bp[k_contact] - la[k_contact])
        dR = np.linalg.norm(bp[k_contact] - ra[k_contact])
        striking = "left" if dL <= dR else "right"
    else:
        vL = 0.0 if k_contact == 0 else np.linalg.norm(la[k_contact] - la[k_contact - 1])
        vR = 0.0 if k_contact == 0 else np.linalg.norm(ra[k_contact] - ra[k_contact - 1])
        striking = "left" if vL >= vR else "right"

    mid_hip = (lhip + rhip) / 2.0
    mid_sh = (lsh + rsh) / 2.0

    hw = []
    for t in range(k0, k1 + 1):
        if not (np.isnan(lhip[t]).any() or np.isnan(rhip[t]).any()):
            hw.append(np.linalg.norm(lhip[t] - rhip[t]))
    norm_hip = float(np.nanmedian(hw)) if len(hw) > 0 else np.nan

    if not np.isnan(bp[k_contact, 0]):
        if striking == "left":
            plant_xy = ra[k_contact]
        else:
            plant_xy = la[k_contact]
        plant2ball = float(np.linalg.norm(plant_xy - bp[k_contact])) if not np.isnan(plant_xy).any() else np.nan
        plant2ball_norm = plant2ball / (norm_hip + 1e-6) if not math.isnan(plant2ball) else np.nan
    else:
        plant2ball_norm = np.nan

    trunk_samples = []
    for t in range(k0, k_contact + 1):
        if not (np.isnan(mid_hip[t]).any() or np.isnan(mid_sh[t]).any()):
            dx = float(mid_sh[t][0] - mid_hip[t][0])
            dy = float(mid_sh[t][1] - mid_hip[t][1])
            trunk_samples.append(normalize_signed_angle(math.degrees(math.atan2(dx, -dy))))
    trunk_lean_deg = float(np.mean(trunk_samples)) if trunk_samples else np.nan

    if striking == "left":
        stand_ankle = ra
        kick_knee = lk
        kick_ankle = la
    else:
        stand_ankle = la
        kick_knee = rk
        kick_ankle = ra

    hip_samples = []
    for t in range(max(0, k_contact - 3), min(T_pose - 1, k_contact + 3) + 1):
        if not (np.isnan(mid_hip[t]).any() or np.isnan(stand_ankle[t]).any()):
            dx = float(stand_ankle[t][0] - mid_hip[t][0])
            dy = float(stand_ankle[t][1] - mid_hip[t][1])
            hip_samples.append(normalize_unsigned_angle(math.degrees(math.atan2(dy, dx))))
    hip_facing_deg = float(np.mean(hip_samples)) if hip_samples else np.nan

    frames_follow = int(round((FOLLOW_MS / 1000.0) * fps))
    end_ft = min(T_pose - 1, k_contact + frames_follow)
    follow_angles = []
    for t in range(0, end_ft + 1):
        if not (np.isnan(kick_knee[t]).any() or np.isnan(kick_ankle[t]).any() or np.isnan(stand_ankle[t]).any()):
            angle = angle_at(kick_ankle[t], kick_knee[t], stand_ankle[t])
            if not math.isnan(angle):
                follow_angles.append(angle)
    follow_arc = float(max(follow_angles) - min(follow_angles)) if len(follow_angles) >= 3 else np.nan

    lock_start = max(0, k_contact - int(round(0.25 * fps)))
    lock_angles = []
    for t in range(lock_start, k_contact + 1):
        if not (np.isnan(mid_hip[t]).any() or np.isnan(kick_knee[t]).any() or np.isnan(kick_ankle[t]).any()):
            angle = angle_at(mid_hip[t], kick_knee[t], kick_ankle[t])
            if not math.isnan(angle):
                lock_angles.append(angle)
    lock_angle = float(min(lock_angles)) if lock_angles else np.nan

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
    files = [f for f in os.listdir(VIDEOS_DIR) if f.lower().endswith((".mp4", ".mov", ".m4v", ".avi"))]
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
            if res:
                rows.append(res)
            else:
                print(f"[SKIP] {fname} (pose or frames missing)")
        except Exception as e:
            print(f"[ERROR] {fname}: {e}")

    if rows:
        keys = ["shot_id", "fps", "contact_frame", "striking_foot",
                "plant_to_ball_norm", "trunk_lean_deg", "hip_facing_deg",
                "follow_through_arc_deg", "lock_angle_deg"]
        with open(OUT_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            for r in rows:
                w.writerow(r)
        print(f"\n Wrote {len(rows)} rows -> {OUT_CSV}")
    else:
        print("[WARN] No results written.")


if __name__ == "__main__":
    main()
