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

        # Natural vowel mapping:
        # 'a': tall opening, normal width (attenuated only during deliberate wide stretch)
        v_a = _clamp(open_val * (1.0 - wide_active * 0.40))
        # 'i': deliberate horizontal stretch with low opening
        v_i = _clamp(wide_active * i_ratio)
        # 'u': round forward pucker (low width, moderate opening)
        v_u = _clamp((open_val * 0.65) * (1.0 - max(0.0, mouth_wide_raw) * 0.85), 0.0, u_max_clamp)
        # 'e': wide grin-like opening (both open and deliberately wide)
        v_e = _clamp(open_val * 1.6 * wide_active * e_ratio)
        # 'o': round mouth opening with narrow corners
        v_o = _clamp(open_val * (1.0 - max(0.0, mouth_wide_raw) * 0.75) * (o_gain / 1.8), 0.0, o_max_clamp)

    # Emotions
    smile_l = float(features.get("mouth_corner_updown_l", 0.0))
    smile_r = float(features.get("mouth_corner_updown_r", 0.0))
    avg_smile = max(0.0, (smile_l + smile_r) / 2.0)
    joy = _clamp((avg_smile - smile_deadzone) * smile_gain) if avg_smile > smile_deadzone else 0.0

    frown_val = max(0.0, -(smile_l + smile_r) / 2.0)
    steepness = (float(features.get("eyebrow_steepness_l", 0.0)) + float(features.get("eyebrow_steepness_r", 0.0))) / 2.0
    sorrow = _clamp((frown_val - frown_deadzone) * frown_gain + max(0.0, steepness) * 0.6)

    brow_up = (float(features.get("eyebrow_updown_l", 0.0)) + float(features.get("eyebrow_updown_r", 0.0))) / 2.0
    surprised = _clamp(max(0.0, brow_up - brow_raise_deadzone) * brow_raise_gain * 0.6 + (v_a * 0.5))

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
    eye_blink = getattr(face, "eye_blink", [1.0, 1.0])
    blink = _clamp(1.0 - min(float(eye_blink[0]), float(eye_blink[1])))

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
    Extract 3D head rotation (Pitch, Yaw, Roll) from OpenSeeFace Euler angles in degrees.
    """
    euler = getattr(face, "euler", None)
    if euler is None or len(euler) < 3:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    # OpenSeeFace Euler angles: euler[0] = Pitch, euler[1] = Yaw, euler[2] = Roll
    raw_pitch = float(euler[0])
    raw_yaw = float(euler[1])
    raw_roll = float(euler[2])

    # Apply webcam pitch compensation offset
    pitch = raw_pitch + pitch_offset_deg
    # Mirror yaw for intuitive camera mirror interaction
    yaw = -raw_yaw
    roll = raw_roll

    return {
        "pitch": float(pitch),
        "yaw": float(yaw),
        "roll": float(roll),
    }


def extract_iris_gaze(face: Any) -> Dict[str, float]:
    """
    Extract normalized iris gaze vector (-1.0 to 1.0) from OpenSeeFace pupil landmarks.
    """
    lms = getattr(face, "lms", None)
    if lms is None or len(lms) < 68:
        return {"x": 0.0, "y": 0.0}

    try:
        # Landmarks: Right Eye (36-41), Pupil (66); Left Eye (42-47), Pupil (67)
        # Check confidences
        r_pupil_conf = lms[66][2]
        l_pupil_conf = lms[67][2]

        gaze_x_list = []
        gaze_y_list = []

        if r_pupil_conf > 0.15:
            r_outer = lms[36][:2]
            r_inner = lms[39][:2]
            r_center = (r_outer + r_inner) / 2.0
            r_width = max(1.0, abs(r_inner[0] - r_outer[0]))
            r_pupil = lms[66][:2]
            # Offset from eye center normalized by eye half-width
            gaze_x_list.append((r_pupil[0] - r_center[0]) / (r_width * 0.5))
            gaze_y_list.append((r_pupil[1] - r_center[1]) / (r_width * 0.4))

        if l_pupil_conf > 0.15:
            l_inner = lms[42][:2]
            l_outer = lms[45][:2]
            l_center = (l_outer + l_inner) / 2.0
            l_width = max(1.0, abs(l_outer[0] - l_inner[0]))
            l_pupil = lms[67][:2]
            gaze_x_list.append((l_pupil[0] - l_center[0]) / (l_width * 0.5))
            gaze_y_list.append((l_pupil[1] - l_center[1]) / (l_width * 0.4))

        if gaze_x_list:
            gx = _clamp(float(np.mean(gaze_x_list)), -1.0, 1.0)
            gy = _clamp(float(np.mean(gaze_y_list)), -1.0, 1.0)
            return {"x": gx, "y": gy}

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
