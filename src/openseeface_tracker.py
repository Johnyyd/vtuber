"""
OpenSeeFace ONNX Tracker Integration for VTuber 3D Avatar.

Provides real-time facial landmark detection, pupil gaze tracking, and 3D head pose
using the specialized MobileNetV3 ONNX models from OpenSeeFace.
"""

import os
import sys
import urllib.request
from typing import Dict, Tuple, Optional, Any
import numpy as np

try:
    from .openseeface.tracker import Tracker
except ImportError:
    from openseeface.tracker import Tracker

MODEL_FILES = {
    "mnv3_detection_opt.onnx": "https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/models/mnv3_detection_opt.onnx",
    "lm_model3_opt.onnx": "https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/models/lm_model3_opt.onnx",
    "mnv3_gaze32_split_opt.onnx": "https://raw.githubusercontent.com/emilianavt/OpenSeeFace/master/models/mnv3_gaze32_split_opt.onnx",
}

if getattr(sys, "frozen", False):
    _bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    _exe_dir = os.path.dirname(sys.executable)
    _cand = os.path.join(_exe_dir, "models", "openseeface")
    DEFAULT_MODEL_DIR = _cand if os.path.exists(_cand) else os.path.join(_bundle_dir, "models", "openseeface")
else:
    DEFAULT_MODEL_DIR = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "models", "openseeface")
    )


def _clamp(val: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to range [min_val, max_val]."""
    return max(min_val, min(max_val, float(val)))


def ensure_openseeface_models_downloaded(model_dir: Optional[str] = None) -> str:
    """Download OpenSeeFace ONNX model files if not already present."""
    target_dir = model_dir or DEFAULT_MODEL_DIR
    os.makedirs(target_dir, exist_ok=True)

    for filename, url in MODEL_FILES.items():
        filepath = os.path.join(target_dir, filename)
        if not os.path.exists(filepath) or os.path.getsize(filepath) < 1000:
            print(f"[OpenSeeFace] Downloading {filename} ...")
            req = urllib.request.Request(url, headers={"User-Agent": "VTuberAvatar/1.0"})
            with urllib.request.urlopen(req) as resp, open(filepath, "wb") as f:
                f.write(resp.read())
            print(f"[OpenSeeFace] Saved to {filepath}")

    return target_dir


def map_openseeface_to_vrm(face: Any, config: Optional[dict] = None) -> Dict[str, float]:
    """
    Map OpenSeeFace facial features and landmarks to standard VRM 0.x blendshapes.
    Supports speech vowel synthesis (a, i, u, e, o), natural blinks, and emotions.
    """
    cfg = config or {}
    features = getattr(face, "current_features", {}) or {}

    mouth_open_deadzone = float(cfg.get("mouth_open_deadzone", 0.06))
    mouth_open_gain = float(cfg.get("mouth_open_gain", 1.8))
    smile_deadzone = float(cfg.get("smile_deadzone", 0.10))
    smile_gain = float(cfg.get("smile_gain", 2.8))
    brow_raise_deadzone = float(cfg.get("brow_raise_deadzone", 0.08))
    brow_raise_gain = float(cfg.get("brow_raise_gain", 2.4))
    frown_deadzone = float(cfg.get("frown_deadzone", 0.04))
    frown_gain = float(cfg.get("frown_gain", 3.2))
    u_max_clamp = float(cfg.get("u_max_clamp", 0.58))
    o_gain = float(cfg.get("o_gain", 2.4))
    o_max_clamp = float(cfg.get("o_max_clamp", 0.55))
    blink_deadzone = float(cfg.get("blink_deadzone", 0.13))
    blink_snap_thresh = float(cfg.get("blink_snap_thresh", 0.25))

    # Eyes & Blinking
    eye_blink = getattr(face, "eye_blink", None)
    if eye_blink is not None and len(eye_blink) >= 2:
        # OpenSeeFace eye_blink is 1.0 for open, 0.0 for closed
        r_open = float(eye_blink[0])
        l_open = float(eye_blink[1])
        blink_r = _clamp(1.0 - r_open)
        blink_l = _clamp(1.0 - l_open)
    else:
        blink_r = 0.0
        blink_l = 0.0

    # Apply snap threshold for clear, crisp blinking
    blink_r = 0.0 if blink_r < blink_deadzone else (_clamp((blink_r - blink_deadzone) / (1.0 - blink_deadzone)) if blink_r < blink_snap_thresh else 1.0)
    blink_l = 0.0 if blink_l < blink_deadzone else (_clamp((blink_l - blink_deadzone) / (1.0 - blink_deadzone)) if blink_l < blink_snap_thresh else 1.0)
    blink = max(blink_l, blink_r)

    # Mouth & Speech Vowels
    mouth_open_raw = float(features.get("mouth_open", 0.0))
    mouth_wide_raw = float(features.get("mouth_wide", 0.0))

    # Physical Ground Truth: if landmarks are present, verify inner lip separation
    lms = getattr(face, "lms", None)
    if lms is not None and len(lms) >= 66 and float(lms[30, 0]) > 10.0 and float(lms[51, 0]) > 10.0:
        nose_h = max(1.0, float(lms[30, 0] - lms[27, 0]))
        upper_inner_y = float(np.mean(lms[[59, 60, 61], 0]))
        lower_inner_y = float(np.mean(lms[[63, 64, 65], 0]))
        lip_gap = lower_inner_y - upper_inner_y
        inner_mar = lip_gap / nose_h
        # If inner lips are touching (inner_mar <= 0.06), mouth is physically closed
        if inner_mar <= 0.06:
            mouth_open_raw = 0.0

    if mouth_open_raw < mouth_open_deadzone:
        # Resting mouth stability: neutral expression, no random vowels
        v_a = 0.0
        v_i = 0.0
        v_u = 0.0
        v_e = 0.0
        v_o = 0.0
    else:
        open_val = _clamp((mouth_open_raw - mouth_open_deadzone) * mouth_open_gain)

        # Deadzone on mouth_wide_raw to prevent ordinary speech width from false-triggering I & E
        # In OpenSeeFace, mouth_wide_raw is normalized offset from median (resting width).
        # Deliberate horizontal vowel stretch requires mouth_wide_raw > 0.20
        wide_deadzone = 0.20
        if mouth_wide_raw > wide_deadzone:
            wide_active = _clamp((mouth_wide_raw - wide_deadzone) * 2.5)
        else:
            wide_active = 0.0

        # Transition between I and E based on mouth vertical opening:
        # Minimal open (open_val <= 0.12) -> pure 'I'
        # Moderately open (open_val >= 0.28) -> pure 'E'
        # Intermediate -> smooth transition
        if open_val <= 0.12:
            i_ratio = 1.0
            e_ratio = 0.0
        elif open_val >= 0.28:
            i_ratio = 0.0
            e_ratio = 1.0
        else:
            t = (open_val - 0.12) / 0.16
            i_ratio = 1.0 - t
            e_ratio = t

        # VOWEL PRIORITY CASCADE: A > E > I > U > O
        # 1. A (Top priority - Vertical mouth opening)
        cand_a = open_val

        # 2. E (2nd priority - Open lateral stretch)
        cand_e = _clamp(open_val * 1.6 * wide_active * e_ratio)

        # 3. I (3rd priority - Close lateral stretch, only when jaw is relatively closed)
        cand_i = _clamp(wide_active * i_ratio)

        # 4. U (4th priority - Puckered lips)
        cand_u = _clamp((open_val * 0.65) * (1.0 - max(0.0, mouth_wide_raw) * 0.85), 0.0, u_max_clamp)

        # 5. O (5th priority - Rounded mouth)
        cand_o = _clamp(open_val * (1.0 - max(0.0, mouth_wide_raw) * 0.75) * (o_gain / 1.8), 0.0, o_max_clamp)

        # Suppress I if jaw is wide open speaking A or E (enforces E > I and A > I)
        v_e = cand_e
        v_i = _clamp(max(0.0, cand_i - (open_val * 1.2 if open_val > 0.15 else 0.0) - v_e * 1.5))

        # A is attenuated when distinct vowels (E, I) are formed
        v_a = _clamp(max(0.0, cand_a - v_e * 0.35 - v_i * 0.5))

        # U has priority over O (enforces U > O)
        v_u = cand_u
        v_o = cand_o
        if v_u > 0.20 and open_val < 0.25:
            v_o = _clamp(max(0.0, v_o - v_u * 0.5))

    # Emotions
    # Physical Landmark Ground Truth for Smile (Joy):
    # In OpenSeeFace 68 landmarks:
    # 48 is right mouth corner, 54 is left mouth corner
    # 51 is upper lip center, 57 is lower lip center
    # In image coordinates, Y points downwards.
    # When smiling, corners elevate UPWARDS (smaller Y) relative to lip center.
    if lms is not None and len(lms) >= 66 and float(lms[30, 0]) > 10.0 and float(lms[51, 0]) > 10.0:
        nose_h = max(1.0, float(lms[30, 0] - lms[27, 0]))
        corner_y = float(lms[48, 0] + lms[54, 0]) * 0.5
        center_y = float(lms[51, 0] + lms[57, 0]) * 0.5
        smile_lift = (center_y - corner_y) / nose_h

        # Resting face has corners level or slightly below center (smile_lift <= 0.015)
        # Genuine deliberate smile lifts corners upwards above center (smile_lift > 0.015)
        if smile_lift > 0.015:
            # Scale proportionally: 0.015 to 0.08 maps to 0.0 to 1.0
            joy = _clamp((smile_lift - 0.015) / 0.065 * (smile_gain / 2.0))
        else:
            joy = 0.0

        if smile_lift < -0.06:
            frown_val = _clamp((-0.06 - smile_lift) / 0.06)
        else:
            frown_val = 0.0
    else:
        smile_l = float(features.get("mouth_corner_updown_l", 0.0))
        smile_r = float(features.get("mouth_corner_updown_r", 0.0))
        avg_smile = max(0.0, (smile_l + smile_r) / 2.0)
        joy = _clamp((avg_smile - smile_deadzone) * smile_gain) if avg_smile > smile_deadzone else 0.0
        frown_val = max(0.0, -(smile_l + smile_r) / 2.0)
    steepness = (float(features.get("eyebrow_steepness_l", 0.0)) + float(features.get("eyebrow_steepness_r", 0.0))) / 2.0
    sorrow = _clamp((frown_val - frown_deadzone) * frown_gain + max(0.0, steepness) * 0.6)

    brow_up = (float(features.get("eyebrow_updown_l", 0.0)) + float(features.get("eyebrow_updown_r", 0.0))) / 2.0
    surprised = _clamp(max(0.0, brow_up - brow_raise_deadzone) * brow_raise_gain)

    angry = _clamp(max(0.0, -steepness) * 1.5 + frown_val * 0.5)
    fun = _clamp(joy * 0.85)

    # Neutral weight decreases when active expressions or mouth opening engage
    active_sum = blink + v_a + v_i + v_u + v_e + v_o + joy + sorrow + surprised
    neutral = _clamp(1.0 - active_sum * 0.5)

    return {
        "neutral": neutral,
        "a": v_a,
        "i": v_i,
        "u": v_u,
        "e": v_e,
        "o": v_o,
        "blink": blink,
        "blink_l": blink_l,
        "blink_r": blink_r,
        "joy": joy,
        "angry": angry,
        "sorrow": sorrow,
        "fun": fun,
        "surprised": surprised,
        "lookup": 0.0,
        "lookdown": 0.0,
        "lookleft": 0.0,
        "lookright": 0.0,
    }


def map_openseeface_to_vrc(face: Any) -> Dict[str, float]:
    """
    Map OpenSeeFace features to 16 standard VRChat viseme morph targets.
    """
    features = getattr(face, "current_features", {}) or {}
    eye_blink = getattr(face, "eye_blink", None)
    if eye_blink is not None and len(eye_blink) >= 2:
        blink = _clamp(1.0 - min(float(eye_blink[0]), float(eye_blink[1])))
    else:
        blink = 0.0

    mouth_open = float(features.get("mouth_open", 0.0))
    mouth_wide = float(features.get("mouth_wide", 0.0))

    if mouth_open < 0.06:
        sil = 1.0
        aa = oh = ou = ee = ih = 0.0
    else:
        sil = 0.0
        aa = _clamp(mouth_open * (1.0 - mouth_wide * 0.4))
        oh = _clamp(mouth_open * (1.0 - mouth_wide * 0.7) * 0.8)
        ou = _clamp(mouth_open * 0.5 * (1.0 - mouth_wide))
        ee = _clamp(mouth_open * mouth_wide * 1.2)
        ih = _clamp(mouth_wide * 0.6)

    return {
        "vrc_blink": blink,
        "vrc_v_sil": sil,
        "vrc_v_aa": aa,
        "vrc_v_oh": oh,
        "vrc_v_ou": ou,
        "vrc_v_ee": ee,
        "vrc_v_ih": ih,
        "vrc_v_ch": 0.0,
        "vrc_v_dd": 0.0,
        "vrc_v_ff": 0.0,
        "vrc_v_kk": 0.0,
        "vrc_v_nn": 0.0,
        "vrc_v_pp": 0.0,
        "vrc_v_rr": 0.0,
        "vrc_v_ss": 0.0,
        "vrc_v_th": 0.0,
    }


def extract_head_pose(face: Any, pitch_offset_deg: float = 18.0) -> Dict[str, float]:
    """
    Extract 3D head rotation (Pitch, Yaw, Roll in radians) for the 3D VRM skeletal pipeline.
    Prefers robust geometric landmark triangulation from 68 anatomical landmarks;
    falls back to Euler angles if landmarks are absent.
    """
    lms = getattr(face, "lms", None)
    if (
        lms is not None
        and len(lms) >= 68
        and (abs(float(lms[30, 0])) > 1.0 or abs(float(lms[30, 1])) > 1.0)
    ):
        # OpenSeeFace landmark convention:
        # lms[:, 0] = Y (vertical pixel coordinates, pointing down)
        # lms[:, 1] = X (horizontal pixel coordinates, pointing right)
        # Right eye (landmarks 36..41), Left eye (landmarks 42..47)
        x_r = float((lms[36, 1] + lms[39, 1]) * 0.5)
        y_r = float((lms[36, 0] + lms[39, 0]) * 0.5)
        x_l = float((lms[42, 1] + lms[45, 1]) * 0.5)
        y_l = float((lms[42, 0] + lms[45, 0]) * 0.5)

        dx = x_l - x_r
        dy = y_l - y_r
        eye_dist = float(np.sqrt(dx * dx + dy * dy))
        # 1. Roll: Angle of eye line relative to horizontal
        roll = float(np.arctan2(dy, dx))

        # 2. Yaw: Nose tip horizontal offset from eye midpoint
        x_eye = (x_r + x_l) * 0.5
        x_nose = float(lms[30, 1])
        yaw_offset = (x_nose - x_eye) / max(eye_dist, 1.0)
        yaw = float(yaw_offset * 1.5)

        # 3. Pitch: Vertical facial proportion of bridge-to-nose vs nose-to-chin
        y_nose = float(lms[30, 0])
        y_bridge = float(lms[27, 0])
        y_chin = float(lms[8, 0])
        top_dist = max(y_nose - y_bridge, 1.0)
        bottom_dist = max(y_chin - y_nose, 1.0)
        ratio = bottom_dist / top_dist
        # Baseline vertical ratio for neutral upright head is ~1.40
        raw_pitch = float((1.40 - ratio) * 0.8)
        # Compensate for camera pitch relative to default 18.0 deg
        pitch = raw_pitch + float(np.radians(pitch_offset_deg - 18.0))

        return {
            "pitch": float(np.clip(pitch, -0.6, 0.6)),
            "yaw": float(np.clip(yaw, -0.7, 0.7)),
            "roll": float(np.clip(roll, -0.5, 0.5)),
        }

    # Fallback to Euler angles if geometric landmarks are not available
    euler = getattr(face, "euler", None)
    if euler is None or len(euler) < 3:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    deg2rad = float(np.pi / 180.0)
    raw_pitch = float(euler[0])
    raw_yaw = float(euler[1])
    raw_roll = float(euler[2])

    pitch = (raw_pitch + pitch_offset_deg) * deg2rad
    yaw = -raw_yaw * deg2rad
    roll = raw_roll * deg2rad

    return {
        "pitch": float(np.clip(pitch, -0.6, 0.6)),
        "yaw": float(np.clip(yaw, -0.7, 0.7)),
        "roll": float(np.clip(roll, -0.5, 0.5)),
    }


def extract_iris_gaze(face: Any) -> Dict[str, float]:
    """
    Extract normalized iris gaze vector (-1.0 to 1.0) from OpenSeeFace pupil landmarks.
    x: -1.0 (looking right) to +1.0 (looking left)
    y: -1.0 (looking up) to +1.0 (looking down)
    """
    lms = getattr(face, "lms", None)
    if lms is None or len(lms) < 68:
        return {"x": 0.0, "y": 0.0}

    try:
        # Check if blinking/closing eyes
        eye_blink = getattr(face, "eye_blink", None)
        if eye_blink is not None and len(eye_blink) >= 2:
            if float(eye_blink[0]) < 0.35 or float(eye_blink[1]) < 0.35:
                return {"x": 0.0, "y": 0.0}

        r_pupil_conf = float(lms[66][2])
        l_pupil_conf = float(lms[67][2])

        gaze_x_list = []
        gaze_y_list = []

        # OpenSeeFace landmark convention:
        # lms[:, 0] is Y (vertical row, 0 at top, increases downwards)
        # lms[:, 1] is X (horizontal column, 0 at left, increases rightwards)
        # Right Eye: outer 36, inner 39, top 37/38, bot 40/41, pupil 66
        if r_pupil_conf >= 0.35:
            r_width = abs(float(lms[39, 1]) - float(lms[36, 1]))
            r_center_x = (float(lms[36, 1]) + float(lms[39, 1])) * 0.5
            r_top_y = (float(lms[37, 0]) + float(lms[38, 0])) * 0.5
            r_bot_y = (float(lms[40, 0]) + float(lms[41, 0])) * 0.5
            r_height = abs(r_bot_y - r_top_y)
            r_center_y = (r_top_y + r_bot_y) * 0.5

            if r_width >= 8.0:
                dx = float(lms[66, 1]) - r_center_x
                gaze_x_list.append(dx / (r_width * 0.25))

            if r_height >= 3.0:
                dy = float(lms[66, 0]) - r_center_y
                gaze_y_list.append(dy / (r_height * 0.30))

        # Left Eye: inner 42, outer 45, top 43/44, bot 46/47, pupil 67
        if l_pupil_conf >= 0.35:
            l_width = abs(float(lms[45, 1]) - float(lms[42, 1]))
            l_center_x = (float(lms[42, 1]) + float(lms[45, 1])) * 0.5
            l_top_y = (float(lms[43, 0]) + float(lms[44, 0])) * 0.5
            l_bot_y = (float(lms[46, 0]) + float(lms[47, 0])) * 0.5
            l_height = abs(l_bot_y - l_top_y)
            l_center_y = (l_top_y + l_bot_y) * 0.5

            if l_width >= 8.0:
                dx = float(lms[67, 1]) - l_center_x
                gaze_x_list.append(dx / (l_width * 0.25))

            if l_height >= 3.0:
                dy = float(lms[67, 0]) - l_center_y
                gaze_y_list.append(dy / (l_height * 0.30))

        if not gaze_x_list and not gaze_y_list:
            return {"x": 0.0, "y": 0.0}

        def _dz(val: float, deadzone: float = 0.12) -> float:
            if abs(val) <= deadzone:
                return 0.0
            sign = 1.0 if val > 0 else -1.0
            return float(sign * min(1.0, (abs(val) - deadzone) / (1.0 - deadzone)))

        gx = _dz(float(np.mean(gaze_x_list))) if gaze_x_list else 0.0
        gy = _dz(float(np.mean(gaze_y_list))) if gaze_y_list else 0.0

        return {"x": float(np.clip(gx, -1.0, 1.0)), "y": float(np.clip(gy, -1.0, 1.0))}

    except Exception:
        pass

    return {"x": 0.0, "y": 0.0}


class OpenSeeFaceTracker:
    """
    Wrapper around OpenSeeFace Tracker for seamless integration with VTuber motion pipeline.
    """

    def __init__(
        self,
        width: int = 1280,
        height: int = 720,
        model_dir: Optional[str] = None,
        max_threads: int = 2,
    ):
        self.width = width
        self.height = height
        self.model_dir = ensure_openseeface_models_downloaded(model_dir)

        # Initialize OpenSeeFace ONNX Tracker
        self.tracker = Tracker(
            width=self.width,
            height=self.height,
            model_type=3,  # Standard quality landmark model (lm_model3_opt.onnx)
            threshold=None,
            max_faces=1,
            discard_after=5,
            scan_every=3,
            max_threads=max_threads,
            silent=True,
            model_dir=self.model_dir,
            no_gaze=False,
            use_retinaface=False,
            feature_level=2,
        )

    def process_frame(
        self, frame: np.ndarray, config: Optional[dict] = None
    ) -> Optional[Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]]:
        """
        Process a BGR video frame and extract face tracking data.

        Returns:
            (vrm_shapes, vrc_targets, raw_rotation, raw_gaze) if a face is detected,
            or None if no face is present.
        """
        if frame is None or frame.size == 0:
            return None

        # Predict faces using OpenSeeFace
        faces = self.tracker.predict(frame)
        if not faces:
            return None

        face = faces[0]
        vrm_shapes = map_openseeface_to_vrm(face, config=config)
        vrc_targets = map_openseeface_to_vrc(face)
        pitch_offset = float(config.get("pitch_offset_deg", 18.0)) if config else 18.0
        raw_rotation = extract_head_pose(face, pitch_offset_deg=pitch_offset)
        raw_gaze = extract_iris_gaze(face)

        return vrm_shapes, vrc_targets, raw_rotation, raw_gaze
