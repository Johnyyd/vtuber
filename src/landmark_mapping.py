"""
MediaPipe FaceLandmarker -> VRM & VRChat Morph Target Mapping & Head Pose Computation

Maps MediaPipe ARKit blendshapes to:
1. 16 VRChat morph targets (matching character.vrm's Mesh_InteriorMouth2)
2. 17 VRM standard preset blendshapes (for universal VRM compatibility)
3. 3D Head Pose (Pitch, Yaw, Roll) via cv2.solvePnP
"""

import cv2
import numpy as np
from typing import Dict, Tuple, Optional


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

    # "oh" - rounded, open mouth
    vrc["vrc_v_oh"] = _clamp(
        g("jawOpen") * 0.6
        + g("mouthPucker") * 0.7
        + g("mouthFunnel") * 0.3
    )

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
        if bl_l >= 0.30 or bl_r >= 0.30:
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
                # Eyelids must be open enough to calculate valid vertical gaze
                if dy >= 0.012:
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
                if dy >= 0.012:
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
) -> Dict[str, float]:
    """
    Map MediaPipe ARKit blendshapes & dense 3D biomechanical landmarks to VRM 0.x standard blendshapes.
    Fuses physical EAR (Eye Aspect Ratio) and MAR (Mouth Aspect Ratio) for sub-millimeter precision.
    """
    def g(name: str) -> float:
        return mp_blendshapes.get(name, 0.0)

    vrm = {name: 0.0 for name in VRM_BLENDSHAPES}

    # 1. Mouth / Visemes - Robust jawOpen with head-tilt rejection
    jaw = g("jawOpen")
    # Clean deadzone at 0.035 to guarantee closed mouth at rest or when tilting head down
    jaw_active = _clamp((jaw - 0.035) * 2.5) if jaw > 0.035 else 0.0

    if landmarks:
        mar = compute_mar(landmarks)
        # mar_rest = 0.20 accounts for natural lip thickness & head tilt foreshortening
        mar_active = mar_to_mouth_open(mar, mar_rest=0.20, mar_max=0.42)
        raw_a = _clamp(max(jaw_active, mar_active))
    else:
        raw_a = jaw_active

    stretch = (g("mouthStretchLeft") + g("mouthStretchRight")) * 0.5
    smile = (g("mouthSmileLeft") + g("mouthSmileRight")) * 0.5
    pucker = g("mouthPucker")
    funnel = g("mouthFunnel")

    # Phonetically distinct Japanese / Anime vowel classification:
    # "O": Rounded open funnel or pucker with open jaw
    is_o = funnel > 0.15 or (pucker > 0.18 and raw_a > 0.15)
    vrm["o"] = _clamp(funnel * 1.6 + (pucker * 0.8 if raw_a > 0.1 else 0.0)) if is_o else 0.0

    # "U": Protruded tight lips with minimal opening
    is_u = pucker > 0.18 and funnel < 0.25 and raw_a < 0.35 and not is_o
    vrm["u"] = _clamp(pucker * 1.5) if is_u else 0.0

    # "E": Horizontal mouth stretch with moderate/open jaw
    is_e = stretch > 0.15 and raw_a > 0.15
    vrm["e"] = _clamp(stretch * 1.3 + raw_a * 0.7) if is_e else 0.0

    # "I": Wide grin/smile with teeth close together (low jaw opening)
    is_i = (stretch > 0.15 or smile > 0.25) and raw_a <= 0.15
    vrm["i"] = _clamp(stretch * 1.6 + smile * 0.8) if is_i else 0.0

    # "A": Pure vertical jaw opening; subtract other vowel shapes so A doesn't dominate O, U, E, I
    a_suppression = max(vrm["o"] * 0.75, vrm["u"] * 0.9, vrm["i"] * 0.8, vrm["e"] * 0.65)
    vrm["a"] = _clamp(raw_a - a_suppression)

    vrm["neutral"] = _clamp(1.0 - (vrm["a"] + vrm["i"] + vrm["u"] + vrm["e"] + vrm["o"]))

    # 2. Eye Blinks & Independent Winking - Robust differential classification
    raw_l = g("eyeBlinkLeft")
    raw_r = g("eyeBlinkRight")

    bl_l = _calibrate_blink(raw_l, deadzone=0.12, snap_thresh=0.28)
    bl_r = _calibrate_blink(raw_r, deadzone=0.12, snap_thresh=0.28)

    if landmarks:
        ear_l, ear_r = compute_ear(landmarks)
        ear_bl_l = ear_to_blink(ear_l)
        ear_bl_r = ear_to_blink(ear_r)

        # Biomechanical fusion:
        # 1. Physical eyelid contact -> snap closure
        if ear_bl_l >= 0.5:
            bl_l = max(bl_l, ear_bl_l)
        elif ear_bl_l == 0.0 and raw_l < 0.25:
            # 2. Resting wide open eyes (and raw is noise < 0.25) -> suppress false droop
            bl_l = 0.0

        if ear_bl_r >= 0.5:
            bl_r = max(bl_r, ear_bl_r)
        elif ear_bl_r == 0.0 and raw_r < 0.25:
            bl_r = 0.0

    # Distinguish natural synchronized blink vs deliberate single-eye wink:
    diff_l = bl_l - bl_r
    diff_r = bl_r - bl_l

    # A single-eye wink requires:
    # 1. The winking eye is firmly closed (bl >= 0.50)
    # 2. The other eye is genuinely resting/open (bl_other < 0.20 and raw_other < 0.20)
    # 3. The difference between eyes is distinct (diff >= 0.35)
    is_wink_l = bl_l >= 0.50 and bl_r < 0.20 and raw_r < 0.20 and diff_l >= 0.35
    is_wink_r = bl_r >= 0.50 and bl_l < 0.20 and raw_l < 0.20 and diff_r >= 0.35

    if is_wink_l:
        # Deliberate left eye wink
        vrm["blink"] = 0.0
        vrm["blink_l"] = bl_l
        vrm["blink_r"] = 0.0
    elif is_wink_r:
        # Deliberate right eye wink
        vrm["blink"] = 0.0
        vrm["blink_l"] = 0.0
        vrm["blink_r"] = bl_r
    elif max(bl_l, bl_r) >= 0.25:
        # Natural synchronized blink of both eyes
        vrm["blink"] = max(bl_l, bl_r)
        vrm["blink_l"] = 0.0
        vrm["blink_r"] = 0.0
    else:
        # Resting / eyes open
        vrm["blink"] = 0.0
        vrm["blink_l"] = 0.0
        vrm["blink_r"] = 0.0

    # 3. Facial Expressions
    vrm["joy"] = _clamp((smile - 0.10) * 1.8 if smile > 0.10 else 0.0)
    vrm["angry"] = _clamp((g("browDownLeft") + g("browDownRight")) * 0.8)
    vrm["sorrow"] = _clamp(g("browInnerUp") * 0.85)
    vrm["fun"] = _clamp((g("eyeWideLeft") + g("eyeWideRight")) * 0.5 + vrm["a"] * 0.3)

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