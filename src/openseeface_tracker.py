"""
OpenSeeFace ONNX Tracker Integration for VTuber 3D Avatar.

Provides real-time facial landmark detection, pupil gaze tracking, and 3D head pose
using the specialized MobileNetV3 ONNX models from OpenSeeFace.
"""

import os
import sys
import time
import urllib.request
from typing import Dict, Tuple, Optional, Any
import numpy as np
import cv2

try:
    from .openseeface.tracker import Tracker
except ImportError:
    from openseeface.tracker import Tracker

try:
    from .landmark_mapping import OneEuroFilter
except ImportError:
    from landmark_mapping import OneEuroFilter

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


def _smoothstep(edge0: float, edge1: float, x: float) -> float:
    """
    Cubic Hermite Smoothstep S-curve mapping to [0.0, 1.0].
    Provides C1 smooth continuity with zero first derivatives at endpoints,
    eliminating sharp threshold elbows, boundary popping, and sensor micro-flicker.
    """
    if x <= edge0:
        return 0.0
    if x >= edge1:
        return 1.0
    t = (x - edge0) / (edge1 - edge0)
    return float(t * t * (3.0 - 2.0 * t))


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


def map_openseeface_to_vrm(
    face: Any,
    config: Optional[dict] = None,
    tracker_instance: Optional[Any] = None,
    timestamp: Optional[float] = None,
) -> Dict[str, float]:
    """
    Map OpenSeeFace facial features and landmarks to standard VRM 0.x blendshapes.
    Supports speech vowel synthesis (a, i, u, e, o), natural blinks, and emotions.
    """
    cfg = config or {}
    features = getattr(face, "current_features", {}) or {}
    now_t = float(timestamp) if timestamp is not None else time.time()

    mouth_open_deadzone = float(cfg.get("mouth_open_deadzone", 0.06))
    mouth_open_gain = float(cfg.get("mouth_open_gain", 1.8))
    smile_deadzone = float(cfg.get("smile_deadzone", 0.10))
    smile_gain = float(cfg.get("smile_gain", 2.8))
    brow_raise_deadzone = float(cfg.get("brow_raise_deadzone", 0.08))
    brow_raise_gain = float(cfg.get("brow_raise_gain", 2.4))
    frown_deadzone = float(cfg.get("frown_deadzone", 0.04))
    frown_gain = float(cfg.get("frown_gain", 3.2))
    u_gain = float(cfg.get("u_gain", 1.8))
    u_max_clamp = float(cfg.get("u_max_clamp", 0.58))
    o_gain = float(cfg.get("o_gain", 2.4))
    o_max_clamp = float(cfg.get("o_max_clamp", 0.55))
    blink_deadzone = float(cfg.get("blink_deadzone", 0.13))
    blink_snap_thresh = float(cfg.get("blink_snap_thresh", 0.25))

    lms = getattr(face, "lms", None)

    # Mouth & Speech Vowels
    mouth_open_raw = float(features.get("mouth_open", 0.0))
    mouth_wide_raw = float(features.get("mouth_wide", 0.0))
    if tracker_instance is not None and hasattr(tracker_instance, "filter_mouth_open"):
        mouth_open_raw = tracker_instance.filter_mouth_open.filter(mouth_open_raw, now_t)
        mouth_wide_raw = tracker_instance.filter_mouth_wide.filter(mouth_wide_raw, now_t)

    iod = float(np.linalg.norm(lms[36, :2] - lms[45, :2])) if (lms is not None and len(lms) >= 46) else 0.0

    # Physical Ground Truth from 2D landmarks:
    width_delta = 0.0
    if lms is not None and len(lms) >= 66 and float(lms[30, 0]) > 10.0 and float(lms[51, 0]) > 10.0:
        nose_h = max(1.0, float(lms[30, 0] - lms[27, 0]))
        upper_inner_y = float(np.mean(lms[[59, 60, 61], 0]))
        lower_inner_y = float(np.mean(lms[[63, 64, 65], 0]))
        lip_gap = lower_inner_y - upper_inner_y
        inner_mar = lip_gap / nose_h
        # If inner lips are touching (inner_mar <= 0.07), mouth is physically closed
        if inner_mar <= 0.07:
            mouth_open_raw = 0.0

        # Geometric mouth width tracking:
        m_width = float(np.linalg.norm(lms[54, :2] - lms[48, :2]))
        w_ref = iod if iod >= 10.0 else nose_h * 2.0
        w_ratio = m_width / max(1.0, w_ref)

        width_baseline = getattr(tracker_instance, "mouth_width_baseline", None) if tracker_instance else getattr(face, "_mouth_width_baseline", None)
        if width_baseline is None:
            width_baseline = float(w_ratio)
        else:
            width_baseline = width_baseline * 0.98 + w_ratio * 0.02
        if tracker_instance is not None:
            tracker_instance.mouth_width_baseline = float(width_baseline)
        if hasattr(face, "__dict__"):
            setattr(face, "_mouth_width_baseline", float(width_baseline))
        width_delta = float(w_ratio - width_baseline)

    wide_active = 0.0
    pucker_active = 0.0

    # Combined horizontal stretch metric:
    # mouth_wide_raw is normalized offset; width_delta captures physical lateral retraction
    geo_wide = width_delta * 6.0
    combined_wide = mouth_wide_raw * 0.6 + geo_wide * 0.4 if width_delta != 0.0 else mouth_wide_raw

    # Puckered lips for U (< -0.04)
    if combined_wide < -0.04:
        pucker_active = _clamp((-0.04 - combined_wide) * 2.5)
    else:
        pucker_active = 0.0

    if mouth_open_raw < mouth_open_deadzone:
        # Resting mouth stability: neutral expression, no random vowels
        v_a = 0.0
        v_i = 0.0
        v_e = 0.0
        v_o = 0.0
        open_val = 0.0
        # If lips are explicitly puckered, allow subtle U even with small aperture
        v_u = _clamp(pucker_active * 0.5 * (u_gain / 1.5), 0.0, u_max_clamp) if pucker_active > 0.18 else 0.0
    else:
        open_val = _clamp((mouth_open_raw - mouth_open_deadzone) * mouth_open_gain)

        # Deadzone on combined width for lateral stretch vowels (E, I):
        # Resting mouth width natural baseline is around 0.08..0.14.
        # Intentional lateral stretch for 'I' or 'E' starts above 0.15.
        wide_deadzone = 0.15
        if combined_wide > wide_deadzone:
            wide_active = _clamp((combined_wide - wide_deadzone) * 2.5)
        else:
            wide_active = 0.0

        # Vertical mouth opening curve for I vs E:
        # Pronouncing 'I' requires teeth slightly apart (open_val roughly 0.03 to 0.18).
        # When open_val < 0.03, lips are at rest: I and E MUST BE 0.0!
        # When open_val >= 0.26, mouth is opened wide for E or A, not I.
        if open_val < 0.025:
            i_ratio = 0.0
            e_ratio = 0.0
        elif open_val <= 0.14:
            i_ratio = _clamp((open_val - 0.025) / 0.035)
            e_ratio = 0.0
        elif open_val >= 0.26:
            i_ratio = 0.0
            e_ratio = 1.0
        else:
            t = (open_val - 0.14) / 0.12
            i_ratio = 1.0 - t
            e_ratio = t

        # CANDIDATE VOWELS:
        # 1. U: Lips puckered forward with narrow-to-moderate jaw (chu môi)
        cand_u = _clamp(pucker_active * (open_val * 0.5 + 0.65) * (u_gain / 1.5) * 1.5, 0.0, u_max_clamp) if (pucker_active > 0.0 and open_val < 0.35) else 0.0

        # 2. A: Vertical jaw opening
        cand_a = open_val

        # 3. I: Lateral stretch with slight opening (teeth visible)
        cand_i = _clamp(wide_active * i_ratio * 1.4)

        # 4. E: Lateral stretch with medium jaw opening
        cand_e = _clamp(open_val * 1.8 * wide_active * e_ratio)

        # 5. O: Rounded lips with open jaw
        if combined_wide < 0.04 and open_val > 0.15:
            cand_o = _clamp((open_val * 1.1) * (1.0 - max(0.0, combined_wide + 0.04) * 5.0) * (o_gain / 2.0), 0.0, o_max_clamp)
        else:
            cand_o = 0.0

        # ==============================================================================
        # VOWEL PRIORITY CASCADE: U > A > I > E > O
        # ==============================================================================

        # Priority 1: U (highest)
        # When U is actively puckered, it suppresses all other vowels
        v_u = cand_u
        if v_u > 0.08:
            cand_a = _clamp(max(0.0, cand_a - v_u * 2.0))
            cand_i = _clamp(max(0.0, cand_i - v_u * 2.0))
            cand_e = _clamp(max(0.0, cand_e - v_u * 2.0))
            cand_o = _clamp(max(0.0, cand_o - v_u * 1.5))

        # Priority 2: A
        # Vertical jaw open dominates over horizontal stretch when jaw is open wide without lateral stretch.
        v_a = cand_a
        if v_a > 0.20:
            cand_i = _clamp(max(0.0, cand_i - (v_a - 0.20) * 2.2))
            cand_o = _clamp(max(0.0, cand_o - (v_a - 0.25) * 1.0))
        if v_a > 0.55:
            cand_e = _clamp(max(0.0, cand_e - (v_a - 0.55) * 1.5))

        # Priority 3: I
        # I requires lateral retraction with narrow gap. When active, suppresses E and O.
        v_i = cand_i
        if v_i > 0.08:
            cand_e = _clamp(max(0.0, cand_e - v_i * 0.8))
            cand_o = _clamp(max(0.0, cand_o - v_i * 1.2))

        # Priority 4: E
        # Medium jaw with lateral stretch. Suppresses O.
        v_e = cand_e
        if v_e > 0.08:
            cand_o = _clamp(max(0.0, cand_o - v_e * 1.2))

        # Attenuate A when lateral vowels (I, E) are strongly formed with lateral stretch
        v_a = _clamp(max(0.0, cand_a - v_i * 0.3 - v_e * 0.35))

        # Priority 5: O (lowest)
        v_o = cand_o

    # Emotions: Joy, Frown, Eyebrows
    joy = 0.0
    frown_val = 0.0
    smile_delta = 0.0
    surprised_geo = 0.0
    brow_furrow = 0.0

    if lms is not None and len(lms) >= 66 and float(lms[30, 0]) > 10.0 and float(lms[51, 0]) > 10.0:
        nose_h = max(1.0, float(lms[30, 0] - lms[27, 0]))
        corner_y = float(lms[48, 0] + lms[54, 0]) * 0.5
        # Anchor to rigid upper lip 51 (attached to maxilla) so opening jaw/mouth never creates false smile lift
        upper_lip_y = float(lms[51, 0])
        smile_lift_raw = (upper_lip_y - corner_y) / nose_h
        if tracker_instance is not None and hasattr(tracker_instance, "filter_smile"):
            smile_lift = tracker_instance.filter_smile.filter(smile_lift_raw, now_t)
        else:
            smile_lift = smile_lift_raw
        setattr(face, "smile_lift", float(smile_lift))

        # Adaptive baseline for smile corners to handle individual resting mouth shapes:
        smile_base = getattr(tracker_instance, "smile_baseline", None) if tracker_instance else getattr(face, "_smile_baseline", None)
        prev_s_delta = getattr(tracker_instance, "_prev_smile_delta", 0.0) if tracker_instance else getattr(face, "_prev_smile_delta", 0.0)
        smile_dz = max(0.060, smile_deadzone * 0.70)

        if smile_base is None:
            smile_base = float(smile_lift)
        else:
            s_delta = float(smile_lift - smile_base)
            # If not in deliberate smile (<= smile_dz), or if mouth corners are releasing downward, adapt fast to rest:
            if s_delta <= smile_dz or s_delta < (prev_s_delta - 0.003):
                s_adapt = 0.04
            else:
                s_adapt = 0.002
            smile_base = smile_base * (1.0 - s_adapt) + smile_lift * s_adapt

        if tracker_instance is not None:
            tracker_instance.smile_baseline = float(smile_base)
            tracker_instance._prev_smile_delta = float(smile_lift - smile_base)
        if hasattr(face, "__dict__"):
            setattr(face, "_smile_baseline", float(smile_base))
            setattr(face, "_prev_smile_delta", float(smile_lift - smile_base))

        smile_delta = float(smile_lift - smile_base)

        # Hermite Smoothstep S-curve mapping: zero initial derivative eliminates threshold pop-in and micro-flicker
        if smile_delta > smile_dz:
            joy = _clamp(_smoothstep(smile_dz, smile_dz + 0.11, smile_delta) * (smile_gain / 2.0))
        else:
            joy = 0.0

        # Mouth frown: droop below resting baseline (attenuated when speaking with open jaw)
        frown_delta = max(0.0, -smile_delta)
        if frown_delta > 0.025:
            frown_val = _clamp(_smoothstep(0.025, 0.060, frown_delta))
            if open_val > 0.12:
                frown_val = _clamp(frown_val * max(0.0, 1.0 - (open_val - 0.12) * 5.0))
        else:
            frown_val = 0.0

        # Eyebrow elevation (Surprised):
        # Anchor eye reference to the fixed eye corners (36, 39, 42, 45) which DO NOT move downward when eyelids close!
        if float(lms[36, 0]) > 10.0 and float(lms[39, 0]) > 10.0:
            r_eye_y = float(lms[36, 0] + lms[39, 0]) * 0.5
        else:
            r_eye_y = float(lms[37, 0] + lms[38, 0]) * 0.5
        r_brow_y = float(lms[18, 0] + lms[19, 0] + lms[20, 0]) / 3.0
        r_dist = r_eye_y - r_brow_y

        if float(lms[42, 0]) > 10.0 and float(lms[45, 0]) > 10.0:
            l_eye_y = float(lms[42, 0] + lms[45, 0]) * 0.5
        else:
            l_eye_y = float(lms[43, 0] + lms[44, 0]) * 0.5
        l_brow_y = float(lms[23, 0] + lms[24, 0] + lms[25, 0]) / 3.0
        l_dist = l_eye_y - l_brow_y

        avg_brow_dist = (r_dist + l_dist) * 0.5
        ref_scale = (iod * 0.6 + nose_h * 0.4) if iod >= 10.0 else nose_h

        # Compensate for perspective foreshortening when head pitches up/down:
        if len(lms) >= 9 and float(lms[8, 0]) > 10.0:
            y_nose = float(lms[30, 0])
            y_bridge = float(lms[27, 0])
            y_chin = float(lms[8, 0])
            top_dist = max(y_nose - y_bridge, 1.0)
            bottom_dist = max(y_chin - y_nose, 1.0)
            pitch_ratio = bottom_dist / top_dist
            raw_pitch = float((1.40 - pitch_ratio) * 0.8)
            pitch_penalty = _clamp(max(0.0, -raw_pitch) * 0.16, 0.0, 0.25)
        else:
            pitch_penalty = 0.0

        brow_ratio_raw = max(0.0, (avg_brow_dist / max(1.0, ref_scale)) - pitch_penalty)
        if tracker_instance is not None and hasattr(tracker_instance, "filter_brow"):
            brow_ratio = tracker_instance.filter_brow.filter(brow_ratio_raw, now_t)
        else:
            brow_ratio = brow_ratio_raw

        # Retrieve prior baseline from tracker instance or face cache
        baseline = None
        if tracker_instance is not None and hasattr(tracker_instance, "brow_baseline"):
            baseline = tracker_instance.brow_baseline
        if baseline is None and hasattr(face, "_brow_baseline"):
            baseline = getattr(face, "_brow_baseline", None)

        thresh = max(0.038, brow_raise_deadzone * 0.55)

        if baseline is None:
            baseline = float(brow_ratio)
        else:
            delta = float(brow_ratio - baseline)
            # If actively raising eyebrows (delta > thresh), adapt very slowly so expression doesn't wash out.
            # Otherwise (neutral face, breathing, posture drift), adapt smoothly in both directions!
            adapt_rate = 0.003 if delta > thresh else 0.025
            baseline = baseline * (1.0 - adapt_rate) + brow_ratio * adapt_rate

        if tracker_instance is not None:
            tracker_instance.brow_baseline = float(baseline)
        if hasattr(face, "__dict__"):
            setattr(face, "_brow_baseline", float(baseline))

        delta = float(brow_ratio - baseline)
        setattr(face, "brow_ratio", float(brow_ratio))
        setattr(face, "brow_baseline", float(baseline))
        setattr(face, "brow_delta", float(delta))

        if delta > thresh:
            surprised_geo = _clamp(_smoothstep(thresh, thresh + 0.050, delta) * (brow_raise_gain / 2.0))
        else:
            surprised_geo = 0.0

        # Eyebrow furrowing (Angry):
        furrow_thresh = 0.020
        if delta < -furrow_thresh:
            brow_furrow = _clamp(_smoothstep(furrow_thresh, furrow_thresh + 0.035, -delta))
        else:
            brow_furrow = 0.0
    else:
        smile_l = float(features.get("mouth_corner_updown_l", 0.0))
        smile_r = float(features.get("mouth_corner_updown_r", 0.0))
        avg_smile = max(0.0, (smile_l + smile_r) / 2.0)
        setattr(face, "smile_lift", float(avg_smile))
        joy = _clamp(_smoothstep(smile_deadzone, smile_deadzone + 0.12, avg_smile) * smile_gain) if avg_smile > smile_deadzone else 0.0
        frown_val = max(0.0, -(smile_l + smile_r) / 2.0)
        surprised_geo = 0.0
        brow_furrow = 0.0
        setattr(face, "brow_ratio", 0.35)
        setattr(face, "brow_baseline", 0.35)
        setattr(face, "brow_delta", 0.0)

    # Eyes & Blinking
    eye_blink = getattr(face, "eye_blink", None)
    if eye_blink is not None and len(eye_blink) >= 2:
        # OpenSeeFace eye_blink is 1.0 for open, 0.0 for closed
        r_open = float(eye_blink[0])
        l_open = float(eye_blink[1])
        raw_blink_r = _clamp(1.0 - r_open)
        raw_blink_l = _clamp(1.0 - l_open)
    else:
        raw_blink_r = 0.0
        raw_blink_l = 0.0

    # Smile & laughter squint suppression:
    # Smiling (raised corners, wide mouth, or laugh) contracts orbicularis oculi ("Duchenne marker").
    # This narrows the eye aperture naturally. We raise the blink deadzone and bilateral threshold
    # so smiling with open eyes NEVER triggers accidental blinking.
    is_smiling = (joy > 0.05) or (smile_delta > 0.02) or (wide_active > 0.08) or (mouth_wide_raw > 0.08 and mouth_open_raw > 0.06)
    effective_blink_dz = blink_deadzone + (0.14 if is_smiling else 0.0)

    # Apply snap threshold for clear, crisp blinking:
    snap_closure = max(0.50, 1.0 - blink_snap_thresh)
    bl_r = 0.0 if raw_blink_r < effective_blink_dz else (_clamp((raw_blink_r - effective_blink_dz) / (snap_closure - effective_blink_dz)) if raw_blink_r < snap_closure else 1.0)
    bl_l = 0.0 if raw_blink_l < effective_blink_dz else (_clamp((raw_blink_l - effective_blink_dz) / (snap_closure - effective_blink_dz)) if raw_blink_l < snap_closure else 1.0)

    # Distinguish natural synchronized blink vs deliberate single-eye wink:
    diff_l = bl_l - bl_r
    diff_r = bl_r - bl_l

    # Single-eye wink conditions (one eye firmly closed, other eye open):
    is_wink_l = bl_l >= 0.55 and bl_r < 0.20 and diff_l >= 0.35
    is_wink_r = bl_r >= 0.55 and bl_l < 0.20 and diff_r >= 0.35

    # Bilateral blink: requires BOTH eyes to close synchronously
    # (average closure exceeds threshold AND both eyes have significant closure)
    bl_avg = (bl_l + bl_r) * 0.5
    min_bl = min(bl_l, bl_r)
    bilateral_thresh = 0.52 if is_smiling else 0.42
    is_bilateral = bl_avg >= bilateral_thresh and min_bl >= 0.25

    if is_wink_l:
        vrm_blink = 0.0
        vrm_blink_l = 1.0
        vrm_blink_r = 0.0
    elif is_wink_r:
        vrm_blink = 0.0
        vrm_blink_l = 0.0
        vrm_blink_r = 1.0
    elif is_bilateral:
        vrm_blink = _smoothstep(bilateral_thresh, 0.75, bl_avg)
        vrm_blink_l = 0.0
        vrm_blink_r = 0.0
    else:
        vrm_blink = 0.0
        vrm_blink_l = 0.0
        vrm_blink_r = 0.0

    # Eyes closing/closed flag: prevents eyebrow distance calculation or Surprised from triggering when eyelids drop
    is_closing_eyes = max(raw_blink_l, raw_blink_r) >= 0.35 or vrm_blink > 0.0 or vrm_blink_l > 0.0 or vrm_blink_r > 0.0

    # Surprised expression gating: if eyelids are closing, Surprised must be 0.0
    if lms is not None and len(lms) >= 66 and float(lms[30, 0]) > 10.0 and float(lms[51, 0]) > 10.0:
        surprised = 0.0 if is_closing_eyes else surprised_geo
    else:
        brow_up_feat = (float(features.get("eyebrow_updown_l", 0.0)) + float(features.get("eyebrow_updown_r", 0.0))) / 2.0
        surprised = _clamp(_smoothstep(brow_raise_deadzone, brow_raise_deadzone + 0.10, brow_up_feat) * brow_raise_gain) if (brow_up_feat > brow_raise_deadzone and not is_closing_eyes) else 0.0

    # Sorrow: mouth droop below baseline
    if frown_val > frown_deadzone:
        sorrow = _clamp(_smoothstep(frown_deadzone, frown_deadzone + 0.08, frown_val) * (frown_gain / 1.8))
    else:
        sorrow = 0.0

    # Angry: deliberate brow furrowing
    if brow_furrow > 0.04:
        angry = _clamp(_smoothstep(0.04, 0.12, brow_furrow) * 2.2 + frown_val * 0.4)
    else:
        angry = 0.0

    # Mutual exclusivity and speech distinction:
    # If articulating horizontal speech stretch (E/I) without genuine smile, Joy is suppressed
    horiz_stretch_only = max(0.0, wide_active - joy * 1.2)
    if horiz_stretch_only > 0.10 and joy < 0.50:
        joy = 0.0
    elif open_val > 0.18 and joy < 0.45:
        joy = 0.0

    # STRICT MUTUAL EXCLUSIVITY FOR FACIAL EXPRESSIONS:
    # In VRM avatars, expressions (Joy, Surprised, Angry, Sorrow) alter both eyes and mouth.
    # Combining two expressions at once causes severe mesh collision and distortion.
    # Therefore, ONLY THE SINGLE HIGHEST (DOMINANT) EMOTION is active at any time.
    emo_candidates = {
        "joy": joy if joy > 0.18 else 0.0,
        "surprised": surprised if surprised > 0.18 else 0.0,
        "angry": angry if angry > 0.16 else 0.0,
        "sorrow": sorrow if sorrow > 0.18 else 0.0,
    }
    dominant_emo, max_val = max(emo_candidates.items(), key=lambda kv: kv[1])
    final_joy = max_val if dominant_emo == "joy" and max_val > 0.0 else 0.0
    final_surprised = max_val if dominant_emo == "surprised" and max_val > 0.0 else 0.0
    final_angry = max_val if dominant_emo == "angry" and max_val > 0.0 else 0.0
    final_sorrow = max_val if dominant_emo == "sorrow" and max_val > 0.0 else 0.0

    fun = _clamp(final_joy * 0.85)

    # Neutral weight decreases when active expressions or mouth opening engage
    active_sum = vrm_blink + v_a + v_i + v_u + v_e + v_o + final_joy + final_sorrow + final_surprised + final_angry
    neutral = _clamp(1.0 - active_sum * 0.5)

    return {
        "neutral": neutral,
        "a": v_a,
        "i": v_i,
        "u": v_u,
        "e": v_e,
        "o": v_o,
        "blink": vrm_blink,
        "blink_l": vrm_blink_l,
        "blink_r": vrm_blink_r,
        "joy": final_joy,
        "angry": final_angry,
        "sorrow": final_sorrow,
        "fun": fun,
        "surprised": final_surprised,
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


def extract_head_pose(
    face: Any,
    pitch_offset_deg: float = 18.0,
    tracker_instance: Optional[Any] = None,
) -> Dict[str, float]:
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

        # Compensate for jaw opening so talking does not cause head pitch tilt
        features = getattr(face, "current_features", {}) or {}
        mouth_open = max(0.0, float(features.get("mouth_open", 0.0)))
        top_dist = max(y_nose - y_bridge, 1.0)
        chin_drop_comp = mouth_open * top_dist * 0.40
        bottom_dist = max(y_chin - chin_drop_comp - y_nose, 1.0)
        ratio = bottom_dist / top_dist

        # Adaptive baseline for pitch ratio to account for individual anatomy and webcam angle:
        baseline = None
        if tracker_instance is not None:
            baseline = getattr(tracker_instance, "pitch_ratio_baseline", None)
        elif hasattr(face, "_pitch_baseline"):
            baseline = getattr(face, "_pitch_baseline", None)

        if baseline is None:
            baseline = float(ratio)
        else:
            # Only drift adapt baseline when near neutral; when tilted up or down, preserve baseline
            d_ratio = abs(ratio - baseline)
            if d_ratio < 0.20:
                adapt_rate = 0.003
                baseline = baseline * (1.0 - adapt_rate) + ratio * adapt_rate

        if tracker_instance is not None:
            tracker_instance.pitch_ratio_baseline = float(baseline)
        if hasattr(face, "__dict__"):
            setattr(face, "_pitch_baseline", float(baseline))

        raw_pitch = float((baseline - ratio) * 0.8)
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


def draw_openseeface_debug(
    frame: np.ndarray,
    face: Any,
    vrm_shapes: Optional[Dict[str, float]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Render 68 facial landmarks, contour meshes, pupil gaze, and HUD telemetry on frame.
    Returns a new annotated BGR image.
    """
    if frame is None or frame.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    canvas = frame.copy()
    h, w = canvas.shape[:2]
    vrm = vrm_shapes or {}
    telem = telemetry or {}

    JAW_LINE = list(range(0, 17))
    RIGHT_EYEBROW = list(range(17, 22))
    LEFT_EYEBROW = list(range(22, 27))
    NOSE_BRIDGE = list(range(27, 31))
    NOSE_BASE = list(range(31, 36))
    RIGHT_EYE = list(range(36, 42)) + [36]
    LEFT_EYE = list(range(42, 48)) + [42]
    OUTER_LIPS = list(range(48, 60)) + [48]
    INNER_LIPS = list(range(60, 68)) + [60]

    lms = getattr(face, "lms", None) if face is not None else None
    if lms is not None and len(lms) >= 68:
        pts = {}
        for idx in range(len(lms)):
            px = int(round(float(lms[idx, 1])))
            py = int(round(float(lms[idx, 0])))
            pts[idx] = (px, py)

        def _draw_contour(indices, color, thickness=1):
            for i in range(len(indices) - 1):
                p1 = pts.get(indices[i])
                p2 = pts.get(indices[i + 1])
                if p1 and p2:
                    cv2.line(canvas, p1, p2, color, thickness, cv2.LINE_AA)

        # 1. Draw contour segments
        _draw_contour(JAW_LINE, (255, 200, 0), 1)
        _draw_contour(RIGHT_EYEBROW, (0, 255, 128), 2)
        _draw_contour(LEFT_EYEBROW, (0, 255, 128), 2)
        _draw_contour(NOSE_BRIDGE, (0, 255, 255), 1)
        _draw_contour(NOSE_BASE, (0, 255, 255), 1)
        _draw_contour(RIGHT_EYE, (255, 255, 0), 1)
        _draw_contour(LEFT_EYE, (255, 255, 0), 1)
        _draw_contour(OUTER_LIPS, (200, 100, 255), 2)
        _draw_contour(INNER_LIPS, (150, 50, 255), 1)

        # 2. Draw all landmark points as small dots
        for idx, (px, py) in pts.items():
            if 0 <= px < w and 0 <= py < h:
                cv2.circle(canvas, (px, py), 2, (0, 255, 0), -1, cv2.LINE_AA)

        # 3. Highlight key tracking anchors
        joy_val = float(vrm.get("joy", 0.0))
        corner_color = (0, 0, 255) if joy_val > 0.05 else (0, 165, 255)
        for c_idx in (48, 54):
            if c_idx in pts:
                cv2.circle(canvas, pts[c_idx], 4, corner_color, -1, cv2.LINE_AA)
        for m_idx in (51, 57):
            if m_idx in pts:
                cv2.circle(canvas, pts[m_idx], 3, (255, 255, 0), -1, cv2.LINE_AA)

        # 4. Highlight pupils (66: right, 67: left) & Gaze vector
        for pupil_idx in (66, 67):
            if pupil_idx in pts:
                cv2.circle(canvas, pts[pupil_idx], 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, pts[pupil_idx], 2, (0, 255, 255), -1, cv2.LINE_AA)

        gaze = telem.get("gaze", {})
        gx = float(gaze.get("x", 0.0))
        gy = float(gaze.get("y", 0.0))
        if 66 in pts:
            cv2.line(canvas, pts[66], (int(pts[66][0] + gx * 25), int(pts[66][1] + gy * 25)), (0, 0, 255), 2, cv2.LINE_AA)
        if 67 in pts:
            cv2.line(canvas, pts[67], (int(pts[67][0] + gx * 25), int(pts[67][1] + gy * 25)), (0, 0, 255), 2, cv2.LINE_AA)

    else:
        cv2.putText(
            canvas,
            "KHONG PHAT HIEN KHUON MAT (NO FACE)",
            (30, h // 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 0, 255),
            2,
            cv2.LINE_AA,
        )

    # 5. Draw telemetry HUD overlay on canvas
    hud_h = 95
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

    rot = telem.get("rotation", {})
    pitch = float(rot.get("pitch", 0.0)) * (180.0 / np.pi)
    yaw = float(rot.get("yaw", 0.0)) * (180.0 / np.pi)
    roll = float(rot.get("roll", 0.0)) * (180.0 / np.pi)
    smile_lift = float(telem.get("smile_lift", 0.0))
    joy = float(vrm.get("joy", 0.0))
    mouth_a = float(vrm.get("a", 0.0))
    mouth_u = float(vrm.get("u", 0.0))
    mouth_o = float(vrm.get("o", 0.0))
    blink_both = float(vrm.get("blink", 0.0))
    blink_l = float(vrm.get("blink_l", 0.0))
    blink_r = float(vrm.get("blink_r", 0.0))
    brow_ratio = float(telem.get("brow_ratio", getattr(face, "brow_ratio", 0.0) if face else 0.0))
    brow_base = float(telem.get("brow_baseline", getattr(face, "brow_baseline", 0.0) if face else 0.0))
    brow_delta = float(telem.get("brow_delta", getattr(face, "brow_delta", 0.0) if face else 0.0))
    surprised = float(vrm.get("surprised", 0.0))
    angry = float(vrm.get("angry", 0.0))
    sorrow = float(vrm.get("sorrow", 0.0))

    line1 = f"Backend: OpenSeeFace ONNX | Head: Pitch:{pitch:+.1f}*  Yaw:{yaw:+.1f}*  Roll:{roll:+.1f}*"
    line2 = f"SmileLift: {smile_lift:+.3f} -> JOY: {joy:.2f} | Brow: {brow_ratio:.2f} (Base:{brow_base:.2f} D:{brow_delta:+.2f}) -> SURP: {surprised:.2f} ANGRY: {angry:.2f}"
    line3 = f"Mouth: A={mouth_a:.2f} U={mouth_u:.2f} O={mouth_o:.2f} | Sorrow: {sorrow:.2f} | Blink: Both={blink_both:.2f} L={blink_l:.2f} R={blink_r:.2f} | Gaze: X={telem.get('gaze', {}).get('x', 0.0):+.2f} Y={telem.get('gaze', {}).get('y', 0.0):+.2f}"

    cv2.putText(canvas, line1, (12, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    info_col = (0, 0, 255) if angry > 0.05 else ((0, 165, 255) if surprised > 0.05 else ((0, 100, 255) if joy > 0.05 else (0, 255, 128)))
    cv2.putText(canvas, line2, (12, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, info_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)

    return canvas


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
        self.brow_baseline = None
        self.smile_baseline = None
        self.mouth_width_baseline = None
        self.pitch_ratio_baseline = None
        self.last_face = None
        self.active_emotion = None

        # OneEuroFilters for facial feature metrics:
        # Eliminates webcam CMOS sensor micro-jitter at rest, 0ms lag during fast movement
        self.filter_smile = OneEuroFilter(min_cutoff=0.9, beta=0.015)
        self.filter_brow = OneEuroFilter(min_cutoff=0.8, beta=0.015)
        self.filter_mouth_open = OneEuroFilter(min_cutoff=1.0, beta=0.02)
        self.filter_mouth_wide = OneEuroFilter(min_cutoff=0.8, beta=0.015)

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

    def reset_baselines(self):
        """Reset all adaptive baselines and metric filters to neutral calibration state."""
        self.brow_baseline = None
        self.smile_baseline = None
        self._prev_smile_delta = 0.0
        self.mouth_width_baseline = None
        self.pitch_ratio_baseline = None
        self.active_emotion = None
        self.filter_smile.reset()
        self.filter_brow.reset()
        self.filter_mouth_open.reset()
        self.filter_mouth_wide.reset()
        attrs = ["_pitch_baseline", "_brow_baseline", "_smile_baseline", "_mouth_width_baseline", "_prev_smile_delta", "_active_emotion"]
        if self.last_face is not None:
            for attr in attrs:
                if hasattr(self.last_face, attr):
                    try:
                        delattr(self.last_face, attr)
                    except AttributeError:
                        pass
        if hasattr(self.tracker, "faces") and self.tracker.faces:
            for f in self.tracker.faces:
                for attr in attrs:
                    if hasattr(f, attr):
                        try:
                            delattr(f, attr)
                        except AttributeError:
                            pass

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
            self.last_face = None
            return None

        # Predict faces using OpenSeeFace
        faces = self.tracker.predict(frame)
        if not faces:
            self.last_face = None
            return None

        face = faces[0]
        self.last_face = face
        vrm_shapes = map_openseeface_to_vrm(face, config=config, tracker_instance=self)
        vrc_targets = map_openseeface_to_vrc(face)
        pitch_offset = float(config.get("pitch_offset_deg", 18.0)) if config else 18.0
        raw_rotation = extract_head_pose(face, pitch_offset_deg=pitch_offset, tracker_instance=self)
        raw_gaze = extract_iris_gaze(face)

        return vrm_shapes, vrc_targets, raw_rotation, raw_gaze
