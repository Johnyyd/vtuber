"""
MediaPipe FaceLandmarker -> VRM & VRChat Morph Target Mapping & Head Pose Computation

Maps MediaPipe ARKit blendshapes to:
1. 16 VRChat morph targets (matching character.vrm's Mesh_InteriorMouth2)
2. 17 VRM standard preset blendshapes (for universal VRM compatibility)
3. 3D Head Pose (Pitch, Yaw, Roll) via cv2.solvePnP
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional, Any


# 16 Morph Targets present in character.vrm Mesh_InteriorMouth2
VRC_TARGETS = [
    "vrc_blink",
    "vrc_v_aa",
    "vrc_v_oh",
    "vrc_v_ou",
    "vrc_v_ee",
    "vrc_v_ih",
    "vrc_v_sil",
    "vrc_v_ch",
    "vrc_v_dd",
    "vrc_v_ff",
    "vrc_v_kk",
    "vrc_v_nn",
    "vrc_v_pp",
    "vrc_v_rr",
    "vrc_v_ss",
    "vrc_v_th",
]

# VRM 0.x standard blendshape preset names
VRM_BLENDSHAPES = [
    "neutral",
    "a",
    "i",
    "u",
    "e",
    "o",
    "blink",
    "joy",
    "angry",
    "sorrow",
    "fun",
    "surprised",
    "lookup",
    "lookdown",
    "lookleft",
    "lookright",
    "blink_l",
    "blink_r",
]


def _clamp(value: float, min_val: float = 0.0, max_val: float = 1.0) -> float:
    """Clamp value to [min_val, max_val]."""
    return max(min_val, min(max_val, float(value)))


def map_mediapipe_to_vrc(mp_blendshapes: Dict[str, float]) -> Dict[str, float]:
    """
    Map MediaPipe ARKit blendshapes to the 16 VRChat morph targets of character.vrm.

    Args:
        mp_blendshapes: Dict of MediaPipe blendshape name -> score (0-1)

    Returns:
        Dict of VRC morph target name -> weight (0-1)
    """
    def g(name: str) -> float:
        return mp_blendshapes.get(name, 0.0)

    vrc: Dict[str, float] = {name: 0.0 for name in VRC_TARGETS}

    # 1. Eye Blink (both eyes)
    blink_val = max(g("eyeBlinkLeft"), g("eyeBlinkRight"))
    vrc["vrc_blink"] = _calibrate_blink(blink_val, deadzone=0.15, snap_thresh=0.35)

    # 2. Visemes
    # "aa" - wide open mouth (jawOpen)
    vrc["vrc_v_aa"] = _clamp(
        g("jawOpen") * 1.2
        + g("mouthLowerDownLeft") * 0.3
        + g("mouthLowerDownRight") * 0.3
        - g("mouthClose") * 0.5
    )

    # "ee" - wide horizontal mouth stretch / smile
    vrc["vrc_v_ee"] = _clamp(
        (g("mouthStretchLeft") + g("mouthStretchRight")) * 0.5
        + (g("mouthSmileLeft") + g("mouthSmileRight")) * 0.4
        - g("mouthPucker") * 0.4
    )

    # "ih" - teeth together, slight smile/press
    vrc["vrc_v_ih"] = _clamp(
        (g("mouthPressLeft") + g("mouthPressRight")) * 0.5
        + (g("mouthSmileLeft") + g("mouthSmileRight")) * 0.3
    )

    # "oh" - rounded, open mouth (requires open jaw)
    if g("jawOpen") > 0.10:
        vrc["vrc_v_oh"] = _clamp(
            g("jawOpen") * 0.6
            + g("mouthFunnel") * 0.6
            - g("mouthClose") * 0.5
        )
    else:
        vrc["vrc_v_oh"] = 0.0

    # "ou" - puckered / protruded lips
    vrc["vrc_v_ou"] = _clamp(
        g("mouthPucker") * 1.1
        + g("mouthFunnel") * 0.7
        - (g("mouthStretchLeft") + g("mouthStretchRight")) * 0.3
    )

    # Consonants / intermediate shapes
    vrc["vrc_v_ch"] = _clamp((g("mouthPressLeft") + g("mouthPressRight")) * 0.3)
    vrc["vrc_v_dd"] = _clamp(g("mouthDimpleLeft") * 0.3 + g("mouthDimpleRight") * 0.3)
    vrc["vrc_v_ff"] = _clamp(g("mouthRollLower") * 0.6)
    vrc["vrc_v_kk"] = _clamp(g("mouthUpperUpLeft") * 0.3 + g("mouthUpperUpRight") * 0.3)
    vrc["vrc_v_nn"] = _clamp(g("mouthClose") * 0.4)
    vrc["vrc_v_pp"] = _clamp(g("mouthPucker") * 0.5 + g("mouthClose") * 0.4)
    vrc["vrc_v_rr"] = _clamp(g("mouthRollUpper") * 0.4 + g("mouthRollLower") * 0.4)
    vrc["vrc_v_ss"] = _clamp((g("mouthStretchLeft") + g("mouthStretchRight")) * 0.3)
    vrc["vrc_v_th"] = _clamp(g("mouthFunnel") * 0.4)

    # "sil" - silence / neutral rest
    active_mouth = (
        vrc["vrc_v_aa"]
        + vrc["vrc_v_ee"]
        + vrc["vrc_v_ih"]
        + vrc["vrc_v_oh"]
        + vrc["vrc_v_ou"]
    )
    vrc["vrc_v_sil"] = _clamp(1.0 - active_mouth + g("mouthClose") * 0.3)

    return vrc


def _calibrate_blink(raw: float, deadzone: float = 0.15, snap_thresh: float = 0.30) -> float:
    """
    Calibrate raw MediaPipe blink value:
    - Below deadzone: 0.0 (prevents sleepy/half-closed eyes when open)
    - Above snap_thresh: 1.0 (snappy, complete blink closure)
    - In between: smooth Hermite curve
    """
    if raw <= deadzone:
        return 0.0
    if raw >= snap_thresh:
        return 1.0
    t = (raw - deadzone) / (snap_thresh - deadzone)
    return float(t * t * (3.0 - 2.0 * t))


class OneEuroFilter:
    """
    1€ Filter: Adaptive low-pass filter minimizing jitter at low speeds
    while eliminating latency during fast movements.
    """
    def __init__(self, min_cutoff: float = 0.8, beta: float = 0.015, d_cutoff: float = 1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None

    def _alpha(self, cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * np.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def filter(self, x: float, t: float) -> float:
        if self.t_prev is None:
            self.t_prev = t
            self.x_prev = x
            return float(x)

        dt = max(t - self.t_prev, 1e-4)
        self.t_prev = t
        dx = (x - self.x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self.dx_prev
        self.dx_prev = dx_hat

        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self.x_prev
        self.x_prev = x_hat
        return float(x_hat)

    def reset(self):
        self.x_prev = None
        self.dx_prev = 0.0
        self.t_prev = None


def compute_ear(landmarks: Optional[Dict[str, Tuple[float, float, float]]]) -> Tuple[float, float]:
    """
    Compute Eye Aspect Ratio (EAR) for both eyes using 3D biomechanical geometry.

    Args:
        landmarks: Dict of landmark name -> (x, y, z)

    Returns:
        Tuple of (ear_left, ear_right) in normalized coordinate space.
        Default is (0.30, 0.30) if landmarks are unavailable.
    """
    if not landmarks:
        return (0.30, 0.30)

    req_r = ["lm33", "lm133", "lm160", "lm144", "lm158", "lm153"]
    req_l = ["lm263", "lm362", "lm385", "lm380", "lm387", "lm373"]

    def _dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return float(np.sqrt(dx * dx + dy * dy))

    ear_r = 0.30
    if all(k in landmarks for k in req_r):
        v1 = _dist(landmarks["lm160"], landmarks["lm144"])
        v2 = _dist(landmarks["lm158"], landmarks["lm153"])
        h = _dist(landmarks["lm33"], landmarks["lm133"])
        if h > 1e-6:
            ear_r = (v1 + v2) / (2.0 * h)

    ear_l = 0.30
    if all(k in landmarks for k in req_l):
        v1 = _dist(landmarks["lm385"], landmarks["lm380"])
        v2 = _dist(landmarks["lm387"], landmarks["lm373"])
        h = _dist(landmarks["lm263"], landmarks["lm362"])
        if h > 1e-6:
            ear_l = (v1 + v2) / (2.0 * h)

    return (float(ear_l), float(ear_r))


def ear_to_blink(ear: float, ear_open: float = 0.24, ear_closed: float = 0.16) -> float:
    """
    Convert EAR to blink weight [0.0, 1.0].
    - ear >= ear_open: 0.0 (wide open, resting eyes)
    - ear <= ear_closed: 1.0 (fully shut)
    - smooth Hermite interpolation between them
    """
    if ear >= ear_open:
        return 0.0
    if ear <= ear_closed:
        return 1.0
    t = (ear_open - ear) / (ear_open - ear_closed)
    return float(t * t * (3.0 - 2.0 * t))


def compute_mar(landmarks: Optional[Dict[str, Tuple[float, float, float]]]) -> float:
    """
    Compute Mouth Aspect Ratio (MAR) for mouth opening aperture.

    Args:
        landmarks: Dict of landmark name -> (x, y, z)

    Returns:
        MAR ratio (vertical inner lip separation / horizontal mouth width).
    """
    req = ["lm13", "lm14", "lm61", "lm291"]
    if not landmarks or not all(k in landmarks for k in req):
        return 0.0

    def _dist(p1, p2):
        dx = p1[0] - p2[0]
        dy = p1[1] - p2[1]
        return float(np.sqrt(dx * dx + dy * dy))

    v_center = _dist(landmarks["lm13"], landmarks["lm14"])
    h_width = _dist(landmarks["lm61"], landmarks["lm291"])

    v_extra = 0.0
    extra_count = 0
    if "lm82" in landmarks and "lm87" in landmarks:
        v_extra += _dist(landmarks["lm82"], landmarks["lm87"])
        extra_count += 1
    if "lm312" in landmarks and "lm317" in landmarks:
        v_extra += _dist(landmarks["lm312"], landmarks["lm317"])
        extra_count += 1

    v_total = (v_center + (v_extra / extra_count)) * 0.5 if extra_count > 0 else v_center

    if h_width > 1e-6:
        return float(v_total / h_width)
    return 0.0


def mar_to_mouth_open(mar: float, mar_rest: float = 0.20, mar_max: float = 0.42) -> float:
    """
    Convert MAR to mouth open weight [0.0, 1.0].
    Natural closed mouth MAR is ~0.10 - 0.16 due to 2D lip thickness and head tilt.
    """
    if mar <= mar_rest:
        return 0.0
    return float(_clamp((mar - mar_rest) / (mar_max - mar_rest)))


def compute_iris_gaze(
    landmarks: Optional[Dict[str, Tuple[float, float, float]]] = None,
    mp_blendshapes: Optional[Dict[str, float]] = None,
) -> Dict[str, float]:
    """
    Compute 2D normalized iris gaze coordinates (x, y in range [-1.0, 1.0]).
    Uses dense iris landmarks (468, 473) if available, falling back to ARKit gaze blendshapes.
    When eyes are blinking or closed, returns neutral gaze (0.0, 0.0) to prevent iris jumping.
    x: -1.0 (looking full right in webcam/mirror) to +1.0 (looking full left)
    y: -1.0 (looking full up) to +1.0 (looking full down)
    """
    # Guard: if blinking or closing eyes, iris is occluded -> do not calculate gaze
    if mp_blendshapes:
        bl_l = mp_blendshapes.get("eyeBlinkLeft", 0.0)
        bl_r = mp_blendshapes.get("eyeBlinkRight", 0.0)
        if bl_l >= 0.25 or bl_r >= 0.25:
            return {"x": 0.0, "y": 0.0}

    landmarks = landmarks or {}
    has_right_iris = "lm468" in landmarks and "lm33" in landmarks and "lm133" in landmarks
    has_left_iris = "lm473" in landmarks and "lm263" in landmarks and "lm362" in landmarks

    if has_right_iris or has_left_iris:
        gaze_x_list = []
        gaze_y_list = []

        if has_right_iris:
            p_iris = landmarks["lm468"]
            p_outer = landmarks["lm33"]
            p_inner = landmarks["lm133"]
            dx = p_inner[0] - p_outer[0]
            if abs(dx) > 0.015:
                ratio_x = (p_iris[0] - p_outer[0]) / dx
                gaze_x_list.append((ratio_x - 0.5) * 3.5)

            if "lm159" in landmarks and "lm145" in landmarks:
                p_top = landmarks["lm159"]
                p_bot = landmarks["lm145"]
                dy = p_bot[1] - p_top[1]
                # Eyelids must be open wide enough (>= 0.020) to calculate valid vertical gaze
                if dy >= 0.020:
                    ratio_y = (p_iris[1] - p_top[1]) / dy
                    gaze_y_list.append((ratio_y - 0.5) * 3.5)

        if has_left_iris:
            p_iris = landmarks["lm473"]
            p_inner = landmarks["lm362"]
            p_outer = landmarks["lm263"]
            dx = p_outer[0] - p_inner[0]
            if abs(dx) > 0.015:
                ratio_x = (p_iris[0] - p_inner[0]) / dx
                gaze_x_list.append((ratio_x - 0.5) * 3.5)

            if "lm386" in landmarks and "lm374" in landmarks:
                p_top = landmarks["lm386"]
                p_bot = landmarks["lm374"]
                dy = p_bot[1] - p_top[1]
                if dy >= 0.020:
                    ratio_y = (p_iris[1] - p_top[1]) / dy
                    gaze_y_list.append((ratio_y - 0.5) * 3.5)

        gx = float(np.mean(gaze_x_list)) if gaze_x_list else 0.0
        gy = float(np.mean(gaze_y_list)) if gaze_y_list else 0.0
        return {
            "x": float(np.clip(gx, -1.0, 1.0)),
            "y": float(np.clip(gy, -1.0, 1.0)),
        }

    if mp_blendshapes:
        def g(k: str) -> float:
            return mp_blendshapes.get(k, 0.0)
        gx = (g("eyeLookOutLeft") + g("eyeLookInRight")) - (g("eyeLookInLeft") + g("eyeLookOutRight"))
        gy = (g("eyeLookDownLeft") + g("eyeLookDownRight")) - (g("eyeLookUpLeft") + g("eyeLookUpRight"))
        return {
            "x": float(np.clip(gx * 1.5, -1.0, 1.0)),
            "y": float(np.clip(gy * 1.5, -1.0, 1.0)),
        }

    return {"x": 0.0, "y": 0.0}


def map_mediapipe_to_vrm(
    mp_blendshapes: Dict[str, float],
    landmarks: Optional[Dict[str, Tuple[float, float, float]]] = None,
    pitch: float = 0.0,
    config: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """
    Map MediaPipe ARKit blendshapes & dense 3D biomechanical landmarks to VRM 0.x standard blendshapes.
    Fuses physical EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) for sub-millimeter precision.
    Supports optional config parameter dictionary for custom deadzones and gains.
    """
    cfg = config or {}
    smile_dz = float(cfg.get("smile_deadzone", 0.08))
    smile_gain = float(cfg.get("smile_gain", 2.5))
    mouth_dz = float(cfg.get("mouth_open_deadzone", 0.06))
    mouth_gain = float(cfg.get("mouth_open_gain", 1.5))
    u_max = float(cfg.get("u_max_clamp", 0.60))
    blink_dz = float(cfg.get("blink_deadzone", 0.10))
    blink_snap = float(cfg.get("blink_snap_thresh", 0.22))
    brow_dz = float(cfg.get("brow_raise_deadzone", 0.08))
    brow_gain = float(cfg.get("brow_raise_gain", 2.4))
    frown_dz = float(cfg.get("frown_deadzone", 0.04))
    frown_gain = float(cfg.get("frown_gain", 3.2))
    o_dz = float(cfg.get("o_deadzone", 0.09))
    o_gain = float(cfg.get("o_gain", 2.6))
    o_max = float(cfg.get("o_max_clamp", 0.55))

    def g(name: str) -> float:
        return mp_blendshapes.get(name, 0.0)

    vrm = {name: 0.0 for name in VRM_BLENDSHAPES}

    # 1. Mouth / Visemes - High sensitivity speech tracking with head-tilt protection
    jaw = g("jawOpen")
    mouth_close = g("mouthClose")
    
    # If the user is looking down (pitch > 0), the 2D tracking distances compress 
    # and artificially inflate jawOpen and MAR. Apply a proportional penalty.
    tilt_penalty = max(0.0, pitch * 0.15) if pitch > 0 else 0.0
    jaw = max(0.0, jaw - tilt_penalty)

    # Check if mouth is occluded or out-of-frame
    mouth_occluded = False
    if landmarks:
        if "lm14" in landmarks and "lm152" in landmarks:
            if landmarks["lm14"][1] > 0.96 or landmarks["lm152"][1] > 0.98:
                mouth_occluded = True
        if "lm61" in landmarks and "lm291" in landmarks:
            w_mouth = np.linalg.norm(np.array(landmarks["lm61"][:2]) - np.array(landmarks["lm291"][:2]))
            if w_mouth < 0.015:
                mouth_occluded = True

    if jaw > mouth_dz:
        jaw_active = _clamp((jaw - mouth_dz) / max(0.04, 0.18 - mouth_dz))
    else:
        jaw_active = 0.0
    mar_active = 0.0
    if mouth_occluded:
        raw_speech = 0.0
    elif landmarks:
        mar = compute_mar(landmarks)
        # Inner lips are touching/closed when MAR <= 0.025
        # Normal speaking MAR is 0.035..0.20
        if mar > 0.025:
            mar_active = _clamp((mar - 0.025) / 0.09)
        else:
            mar_active = 0.0

        # Physical Ground Truth: if inner lips are touching (mar_active == 0.0),
        # mouth is physically closed. Blendshape jawOpen noise or resting offset MUST NOT force mouth open!
        # When lips separate (mar_active > 0.0), mouth opens instantly with zero lag!
        if mar_active == 0.0 or mouth_close > 0.40:
            raw_speech = 0.0
        else:
            raw_speech = max(mar_active, jaw_active)
    else:
        # Fallback when landmarks are not provided
        raw_speech = _clamp(jaw_active - mouth_close * 0.3)

    # Dynamic speech boost curve: opens immediately on natural speech
    if raw_speech > 0.0:
        raw_a = _clamp(pow(raw_speech, 0.65) * 1.35)
    else:
        raw_a = 0.0

    # 1. Mouth Visemes / Vowels (A, I, U, E, O)
    stretch = (g("mouthStretchLeft") + g("mouthStretchRight")) * 0.5
    smile = (g("mouthSmileLeft") + g("mouthSmileRight")) * 0.5
    pucker = g("mouthPucker")
    funnel = g("mouthFunnel")

    def _dz(val: float, deadzone: float = 0.02) -> float:
        return max(0.0, val - deadzone)

    # Physical landmark mouth dimensions
    mouth_ratio = 0.33
    lm_horiz = 0.0
    if landmarks and "lm61" in landmarks and "lm291" in landmarks and "lm234" in landmarks and "lm454" in landmarks:
        w_mouth = np.linalg.norm(np.array(landmarks["lm61"][:2]) - np.array(landmarks["lm291"][:2]))
        w_face = np.linalg.norm(np.array(landmarks["lm234"][:2]) - np.array(landmarks["lm454"][:2]))
        mouth_ratio = float(w_mouth / w_face) if w_face > 1e-5 else 0.33
        if mouth_ratio > 0.34:
            lm_horiz = _clamp((mouth_ratio - 0.34) / 0.07)

    # VOWEL U (Lips puckered forward / chu môi):
    # Puckering physically contracts mouth corners horizontally (mouth_ratio < 0.32).
    # When resting, lips are flat/uncontracted (mouth_ratio >= 0.32); when puckered, ratio contracts.
    if mouth_occluded:
        u_val = 0.0
    elif landmarks and "lm61" in landmarks and "lm291" in landmarks and "lm234" in landmarks and "lm454" in landmarks:
        if mouth_ratio < 0.32 and smile < 0.18 and raw_speech < 0.55:
            ratio_act = _clamp((0.32 - mouth_ratio) / 0.07)
            pucker_act = _clamp((pucker - 0.08) * 2.8) if pucker > 0.08 else 0.0
            u_cand = max(ratio_act, pucker_act) * 0.7 + min(ratio_act, pucker_act) * 0.5
            u_val = min(u_max, u_cand * 1.3)
        else:
            u_val = 0.0
    else:
        # Fallback without landmarks (unit tests / synthetic calls)
        if raw_speech < 0.55 and smile < 0.18:
            u_val = min(u_max, _dz(pucker, 0.06) * 3.5)
        else:
            u_val = 0.0

    # VOWEL O (Lips rounded/funneled with open mouth / tròn miệng mở):
    # Physically requires an open mouth:
    # If landmarks are present and inner lips are closed (mar_active == 0.0) or occluded: O is strictly 0.0!
    if (landmarks and mar_active == 0.0) or mouth_occluded:
        o_val = 0.0
    else:
        funnel_act = _dz(funnel, o_dz) * o_gain
        pucker_open_act = _dz(pucker, o_dz) * (o_gain * 0.92) if raw_speech > 0.08 else 0.0
        cand_o = max(funnel_act, pucker_open_act)
        # Suppress O if mouth is stretched wide horizontally (user is saying A, E, or smiling)
        if landmarks and mouth_ratio > 0.34:
            cand_o *= max(0.0, 1.0 - (mouth_ratio - 0.34) / 0.04)
        # Apply o_max_clamp so O morph never opens excessively wide
        o_val = min(o_max, _clamp(cand_o))

    # VOWELS I & E (Horizontal mouth shapes):
    # Stretching mouth sideways (detected via mouthStretch)
    horiz_stretch = max(stretch, (g("mouthStretchLeft") + g("mouthStretchRight")) * 0.5)

    # 1. Smile vs Vowel Separation:
    # A smile elevates mouth corners upwards (Joy). It is NOT vowel I or E!
    # A closed-mouth smile (jaw <= 0.05 and raw_speech <= 0.05) must NEVER trigger I or E.
    # True I/E requires lateral retraction (mouthStretch > 0.04) or landmark width expansion (lm_horiz > 0.04).
    # During open-mouth speech (open_metric > 0.05), mouthSmile can contribute to speech width.
    horiz_active = 0.0
    open_metric = max(raw_speech, jaw)

    if mouth_occluded or (mouth_close > 0.35 and horiz_stretch < 0.15):
        i_val = 0.0
        e_val = 0.0
    else:
        stretch_comp = (horiz_stretch - 0.04) * 2.8 if horiz_stretch > 0.04 else 0.0
        # Only allow smile to contribute if mouth is actively speaking or lateral stretch is already engaged
        if (open_metric > 0.05 or horiz_stretch > 0.05) and smile > 0.04:
            smile_comp = (smile - 0.04) * 2.0
            blend_horiz = max(stretch_comp, stretch_comp * 0.4 + smile_comp * 0.7)
        else:
            blend_horiz = stretch_comp

        lm_horiz_act = _clamp((lm_horiz - 0.04) * 2.5) if lm_horiz > 0.04 else 0.0
        horiz_active = max(blend_horiz, lm_horiz_act)

        # If smile is dominant over stretch (e.g. wide grin / laugh), Joy takes priority over I/E
        if smile > 0.30 and smile > horiz_stretch * 2.2 and open_metric < 0.15:
            horiz_active = 0.0

        if horiz_active > 0.04:
            horiz_mag = _clamp(_dz(horiz_active, 0.02) * 2.6)
            # Speech biomechanics:
            # Close front unrounded vowel 'I' has teeth close together (open_metric <= 0.10)
            # Open-mid front unrounded vowel 'E' has jaw noticeably open (open_metric >= 0.20)
            if open_metric <= 0.10:
                i_val = horiz_mag
                e_val = 0.0
            elif open_metric >= 0.20:
                e_val = horiz_mag
                i_val = 0.0
            else:
                t_e = (open_metric - 0.10) / 0.10
                e_val = horiz_mag * t_e
                i_val = horiz_mag * (1.0 - t_e)
        else:
            i_val = 0.0
            e_val = 0.0

    # VOWEL PRIORITY CASCADE: A > E > I > U > O
    # 1. E and I separation (E has higher priority than I: E > I):
    # During open mouth speech (open_metric > 0.12), horizontal stretch is E, NEVER I.
    # Close front vowel I requires narrow jaw (open_metric <= 0.10).
    cand_e = _clamp(e_val)
    # I is suppressed when jaw is open (A or E active)
    cand_i = _clamp(max(0.0, i_val - (raw_a * 1.2 if raw_speech > 0.12 else 0.0) - cand_e * 1.5))

    # 2. U and O separation (U has higher priority than O: U > O):
    # Puckered mouth with small/moderate jaw is U; O is funneled/rounded open mouth.
    cand_u = min(u_max, _clamp(u_val))
    cand_o = min(o_max, _clamp(o_val))

    # U has higher priority than O (suppress O if U is active with small jaw)
    if cand_u > 0.25 and raw_speech < 0.25:
        cand_o = max(0.0, cand_o - cand_u * 0.5)

    # 3. A (Top priority - Vertical Jaw Opening):
    # Attenuated gently when U, O, I, E are strongly formed
    raw_a_clean = _dz(raw_a, 0.02) * mouth_gain
    cand_a = _clamp(max(0.0, raw_a_clean - cand_u * 0.7 - cand_o * 0.5 - cand_i * 0.5 - cand_e * 0.35))

    vrm["a"] = cand_a
    vrm["e"] = cand_e
    vrm["i"] = cand_i
    vrm["u"] = cand_u
    vrm["o"] = cand_o

    vrm["neutral"] = _clamp(1.0 - (vrm["a"] + vrm["i"] + vrm["u"] + vrm["e"] + vrm["o"]))

    # 2. Eye Blinks & Independent Winking - Robust differential classification
    raw_l = g("eyeBlinkLeft")
    raw_r = g("eyeBlinkRight")

    bl_l = _calibrate_blink(raw_l, deadzone=blink_dz, snap_thresh=blink_snap)
    bl_r = _calibrate_blink(raw_r, deadzone=blink_dz, snap_thresh=blink_snap)

    if landmarks:
        ear_l, ear_r = compute_ear(landmarks)
        ear_bl_l = ear_to_blink(ear_l)
        ear_bl_r = ear_to_blink(ear_r)

        # Biomechanical fusion:
        # 1. Physical eyelid contact -> snap closure
        if ear_bl_l >= 0.5:
            bl_l = max(bl_l, ear_bl_l)
        elif ear_bl_l == 0.0 and raw_l < 0.30:
            # 2. Resting wide open eyes (and raw is noise < 0.30, common with eyeglasses) -> suppress false droop
            bl_l = 0.0

        if ear_bl_r >= 0.5:
            bl_r = max(bl_r, ear_bl_r)
        elif ear_bl_r == 0.0 and raw_r < 0.30:
            bl_r = 0.0

    # Distinguish natural synchronized blink vs deliberate single-eye wink:
    diff_l = bl_l - bl_r
    diff_r = bl_r - bl_l

    # A single-eye wink requires:
    # 1. The winking eye is firmly closed (bl >= 0.40)
    # 2. The other eye is genuinely resting/open (bl_other < 0.20 and raw_other < 0.28)
    # 3. The difference between eyes is distinct (diff >= 0.25)
    is_wink_l = bl_l >= 0.40 and bl_r < 0.20 and raw_r < 0.28 and diff_l >= 0.25
    is_wink_r = bl_r >= 0.40 and bl_l < 0.20 and raw_l < 0.28 and diff_r >= 0.25

    if is_wink_l:
        # Deliberate left eye wink -> full 1.0 closure
        vrm["blink"] = 0.0
        vrm["blink_l"] = 1.0
        vrm["blink_r"] = 0.0
    elif is_wink_r:
        # Deliberate right eye wink -> full 1.0 closure
        vrm["blink"] = 0.0
        vrm["blink_l"] = 0.0
        vrm["blink_r"] = 1.0
    elif max(bl_l, bl_r) >= 0.18:
        # Natural synchronized blink of both eyes -> snap to 1.0 full closure
        vrm["blink"] = 1.0
        vrm["blink_l"] = 0.0
        vrm["blink_r"] = 0.0
    else:
        # Resting / eyes open
        vrm["blink"] = 0.0
        vrm["blink_l"] = 0.0
        vrm["blink_r"] = 0.0

    # 3. Facial Expressions (Strictly mutually exclusive: only 1 dominant emotion allowed at a time)
    peak_smile = max(g("mouthSmileLeft"), g("mouthSmileRight"))
    effective_smile = peak_smile * 0.6 + smile * 0.4

    if effective_smile > smile_dz:
        raw_joy = _clamp((effective_smile - smile_dz) * smile_gain)
    else:
        raw_joy = 0.0

    # Distinguish genuine Joy (mouth corners pulled upwards) from horizontal speech stretch (I/E):
    # If user is articulating speech for I or E:
    # - If horiz_active > 0.05 and effective_smile is not a full-face beaming smile (< 0.65):
    #   Joy is suppressed so it never wipes out I and E!
    if horiz_active > 0.05 and effective_smile < 0.65:
        raw_joy = 0.0
    elif stretch > 0.06 and stretch >= (smile - 0.12):
        raw_joy = 0.0

    raw_angry = _clamp((g("browDownLeft") + g("browDownRight")) * 1.5)

    # Eyebrow raising & eye widening -> Surprised (not sorrow)
    # Surprised requires deliberate eyebrow raising above brow_dz
    brow_up_l = g("browOuterUpLeft")
    brow_up_r = g("browOuterUpRight")
    brow_inner = g("browInnerUp")
    brow_raise = max(brow_inner, (brow_up_l + brow_up_r) * 0.5)
    eye_wide = (g("eyeWideLeft") + g("eyeWideRight")) * 0.5
    frown = (g("mouthFrownLeft") + g("mouthFrownRight")) * 0.5

    # Surprise detection:
    # Naturally occurs with raised eyebrows PLUS either:
    # 1. Eyes widened (eye_wide >= 0.04)
    # 2. Mouth gasped/opened (raw_speech >= 0.06 or jaw_active >= 0.06)
    # 3. Deliberate eyebrow raising (brow_raise >= brow_dz + 0.10)
    is_surprised_marker = (eye_wide >= 0.04 or raw_speech >= 0.06 or brow_raise >= (brow_dz + 0.10))
    if not mouth_occluded and brow_raise > brow_dz and frown < 0.15 and is_surprised_marker:
        brow_act = (brow_raise - brow_dz) * brow_gain
        eye_wide_boost = max(0.0, eye_wide - 0.03) * 2.5
        raw_surprised = _clamp(brow_act + eye_wide_boost)
    else:
        raw_surprised = 0.0

    # Sorrow requires mouth frown or sad knitted inner brows (inner up while outer down)
    knitted_sad_brow = max(0.0, brow_inner - (brow_up_l + brow_up_r) * 0.5)
    
    # Landmark mouth corner droop:
    # Measure corner height relative to the middle of the lips (midline of lm13 and lm14).
    # Only active when mouth is closed/resting (raw_speech < 0.12), to prevent speech mouth opening from being misclassified as sorrow.
    lm_frown = 0.0
    if raw_speech < 0.12 and landmarks and "lm61" in landmarks and "lm291" in landmarks and "lm13" in landmarks and "lm14" in landmarks:
        corner_y = (landmarks["lm61"][1] + landmarks["lm291"][1]) * 0.5
        center_y = (landmarks["lm13"][1] + landmarks["lm14"][1]) * 0.5
        droop_delta = corner_y - center_y
        if droop_delta > 0.005:
            lm_frown = _clamp((droop_delta - 0.005) / 0.015)

    effective_frown = max(frown, lm_frown * 0.25)
    if smile < 0.15 and (effective_frown > frown_dz or knitted_sad_brow > 0.05):
        frown_act = (effective_frown - frown_dz) * frown_gain if effective_frown > frown_dz else 0.0
        brow_sad_act = (knitted_sad_brow - 0.05) * 3.0 if knitted_sad_brow > 0.05 else 0.0
        raw_sorrow = _clamp(max(frown_act, brow_sad_act) + min(frown_act, brow_sad_act) * 0.5)
    else:
        raw_sorrow = 0.0

    # Winner-Takes-All Emotion Arbitration:
    # Only 1 dominant emotion can be active at any given moment to prevent mesh tearing and overlapping
    emotions = {
        "joy": raw_joy,
        "surprised": raw_surprised,
        "sorrow": raw_sorrow,
        "angry": raw_angry,
    }
    dominant_emo = max(emotions, key=emotions.get)
    max_emo_val = emotions[dominant_emo]

    # Deliberate emotion detection threshold (>= 0.22)
    has_active_emotion = max_emo_val >= 0.22

    for emo in ["joy", "surprised", "sorrow", "angry"]:
        vrm[emo] = max_emo_val if (emo == dominant_emo and has_active_emotion) else 0.0

    vrm["fun"] = 0.0

    # If an emotion is confirmed active:
    # Strong deliberate emotions (max_emo_val >= 0.60) override mouth & blinks for dedicated poses.
    # When actively speaking (raw_speech >= 0.08 or i/e/u/o active), speech vowels drive the mouth in real time!
    is_actively_speaking = raw_speech >= 0.08 or vrm["i"] >= 0.08 or vrm["e"] >= 0.08 or vrm["u"] >= 0.08 or vrm["o"] >= 0.08
    if has_active_emotion and (max_emo_val >= 0.60 or not is_actively_speaking):
        vrm["a"] = 0.0
        vrm["i"] = 0.0
        vrm["u"] = 0.0
        vrm["e"] = 0.0
        vrm["o"] = 0.0
        vrm["blink"] = 0.0
        vrm["blink_l"] = 0.0
        vrm["blink_r"] = 0.0
        vrm["neutral"] = 0.0

    # 4. Gaze Direction (VRM Blendshape fallback)
    gaze = compute_iris_gaze(landmarks, mp_blendshapes)
    gx = gaze["x"]
    gy = gaze["y"]
    vrm["lookleft"] = _clamp(-gx)
    vrm["lookright"] = _clamp(gx)
    vrm["lookup"] = _clamp(-gy)
    vrm["lookdown"] = _clamp(gy)

    return vrm


def compute_head_pose_from_matrix(
    matrix: np.ndarray,
    pitch_offset_deg: float = 0.0,
) -> Dict[str, float]:
    """
    Extract 3D head rotation angles (Pitch, Yaw, Roll in radians) from MediaPipe 4x4 matrix.
    Uses cv2.RQDecomp3x3 on the rotation submatrix with optional camera tilt compensation.
    """
    if matrix is None:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    mat = np.array(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    R = mat[:3, :3]
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    # RQDecomp3x3 returns angles in degrees: [pitch, yaw, roll]
    pitch_deg = angles[0] - pitch_offset_deg
    pitch = float(np.radians(pitch_deg))
    yaw = float(np.radians(angles[1]))
    roll = float(np.radians(angles[2]))

    return {
        "pitch": float(np.clip(pitch, -0.6, 0.6)),
        "yaw": float(np.clip(yaw, -0.7, 0.7)),
        "roll": float(np.clip(roll, -0.5, 0.5)),
    }


def compute_head_pose(
    landmarks: Dict[str, Tuple[float, float, float]],
    frame_shape: Tuple[int, int] = (480, 640),
    matrix: Optional[np.ndarray] = None,
    pitch_offset_deg: float = 0.0,
) -> Dict[str, float]:
    """
    Compute 3D head rotation angles (Pitch, Yaw, Roll in radians).
    Prefers 4x4 matrix decomposition; falls back to landmark geometry.
    """
    if matrix is not None:
        return compute_head_pose_from_matrix(matrix, pitch_offset_deg=pitch_offset_deg)

    required_keys = ["lm1", "lm33", "lm263"]
    if not landmarks or not all(k in landmarks for k in required_keys):
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    nose = np.array(landmarks["lm1"][:2], dtype=np.float64)
    eye_l = np.array(landmarks["lm33"][:2], dtype=np.float64)   # User right eye (image left)
    eye_r = np.array(landmarks["lm263"][:2], dtype=np.float64)  # User left eye (image right)

    # 1. Roll: Angle of the eye line relative to horizontal
    dx = eye_r[0] - eye_l[0]
    dy = eye_r[1] - eye_l[1]
    roll = float(np.arctan2(dy, dx))

    # 2. Yaw: Offset of nose from eye midpoint, normalized by eye distance
    eye_center = (eye_l + eye_r) * 0.5
    eye_dist = float(np.linalg.norm(eye_r - eye_l))
    if eye_dist > 1e-6:
        yaw_offset = (nose[0] - eye_center[0]) / eye_dist
        yaw = float(yaw_offset * 1.5)
    else:
        yaw = 0.0

    # 3. Pitch: Use rigid forehead-to-bridge vertical distance if available,
    # to prevent mouth opening from tilting the head
    if "lm10" in landmarks and "lm168" in landmarks:
        forehead = np.array(landmarks["lm10"][:2], dtype=np.float64)
        bridge = np.array(landmarks["lm168"][:2], dtype=np.float64)
        vert_forehead_to_bridge = bridge[1] - forehead[1]
        vert_bridge_to_nose = nose[1] - bridge[1]
        if vert_bridge_to_nose > 1e-6:
            ratio = vert_forehead_to_bridge / vert_bridge_to_nose
            pitch = float((2.2 - ratio) * 0.7)
        else:
            pitch = 0.0
    elif "lm152" in landmarks:
        chin = np.array(landmarks["lm152"][:2], dtype=np.float64)
        vert_eye_to_nose = nose[1] - eye_center[1]
        vert_nose_to_chin = chin[1] - nose[1]
        if vert_eye_to_nose > 1e-6:
            ratio = vert_nose_to_chin / vert_eye_to_nose
            pitch = float((1.2 - ratio) * 0.8)
        else:
            pitch = 0.0
    else:
        pitch = 0.0

    return {
        "pitch": float(np.clip(pitch, -0.6, 0.6)),
        "yaw": float(np.clip(yaw, -0.8, 0.8)),
        "roll": float(np.clip(roll, -0.6, 0.6)),
    }


def map_landmarks_to_blendshapes(landmarks: Dict[str, tuple]) -> Dict[str, float]:
    """Legacy function - kept for backward compatibility."""
    lm13 = landmarks.get("lm13", (0.5, 0.5, 0))
    lm14 = landmarks.get("lm14", (0.5, 0.5, 0))
    dist = float(np.linalg.norm(np.array(lm13) - np.array(lm14)))
    mouth_open = _clamp(dist * 2.5)
    return {"mouth_open": mouth_open}


def draw_mediapipe_debug(
    frame: np.ndarray,
    landmarks: Optional[Dict[str, tuple]] = None,
    vrm_shapes: Optional[Dict[str, float]] = None,
    telemetry: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """
    Render key MediaPipe facial landmarks, iris gaze, and HUD telemetry on frame.
    Returns a new annotated BGR image.
    """
    if frame is None or frame.size == 0:
        return np.zeros((480, 640, 3), dtype=np.uint8)

    canvas = frame.copy()
    h, w = canvas.shape[:2]
    vrm = vrm_shapes or {}
    telem = telemetry or {}

    if landmarks:
        for name, coords in landmarks.items():
            if len(coords) >= 2:
                cx, cy = coords[0], coords[1]
                px = int(cx * w) if cx <= 1.0 else int(cx)
                py = int(cy * h) if cy <= 1.0 else int(cy)
                if 0 <= px < w and 0 <= py < h:
                    cv2.circle(canvas, (px, py), 1, (0, 255, 0), -1, cv2.LINE_AA)

        for iris_key in ("lm468", "lm473"):
            if iris_key in landmarks:
                coords = landmarks[iris_key]
                px = int(coords[0] * w) if coords[0] <= 1.0 else int(coords[0])
                py = int(coords[1] * h) if coords[1] <= 1.0 else int(coords[1])
                cv2.circle(canvas, (px, py), 4, (0, 0, 255), -1, cv2.LINE_AA)
                cv2.circle(canvas, (px, py), 2, (0, 255, 255), -1, cv2.LINE_AA)
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

    hud_h = 95
    overlay = canvas.copy()
    cv2.rectangle(overlay, (0, h - hud_h), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.70, canvas, 0.30, 0, canvas)

    rot = telem.get("rotation", {})
    pitch = float(rot.get("pitch", 0.0)) * (180.0 / np.pi)
    yaw = float(rot.get("yaw", 0.0)) * (180.0 / np.pi)
    roll = float(rot.get("roll", 0.0)) * (180.0 / np.pi)
    joy = float(vrm.get("joy", 0.0))
    mouth_a = float(vrm.get("a", 0.0))
    blink_l = float(vrm.get("blink_l", 0.0))
    blink_r = float(vrm.get("blink_r", 0.0))

    line1 = f"Backend: Google MediaPipe | Head: Pitch:{pitch:+.1f}*  Yaw:{yaw:+.1f}*  Roll:{roll:+.1f}*"
    line2 = f"JOY: {joy:.2f} | Surprised: {vrm.get('surprised', 0.0):.2f} | Sorrow: {vrm.get('sorrow', 0.0):.2f}"
    line3 = f"Mouth: A={mouth_a:.2f} | Blink: L={blink_l:.2f} R={blink_r:.2f} | Gaze: X={telem.get('gaze', {}).get('x', 0.0):+.2f} Y={telem.get('gaze', {}).get('y', 0.0):+.2f}"

    cv2.putText(canvas, line1, (12, h - 68), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    joy_col = (0, 100, 255) if joy > 0.05 else (0, 255, 128)
    cv2.putText(canvas, line2, (12, h - 42), cv2.FONT_HERSHEY_SIMPLEX, 0.45, joy_col, 1, cv2.LINE_AA)
    cv2.putText(canvas, line3, (12, h - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 220, 255), 1, cv2.LINE_AA)

    return canvas