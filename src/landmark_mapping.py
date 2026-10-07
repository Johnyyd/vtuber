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
    blink_avg = (g("eyeBlinkLeft") + g("eyeBlinkRight")) * 0.5
    vrc["vrc_blink"] = _clamp(blink_avg * 1.2)

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


def _calibrate_blink(raw: float, deadzone: float = 0.10, snap_thresh: float = 0.60) -> float:
    """
    Calibrate raw MediaPipe blink value:
    - Below deadzone: 0.0 (prevents sleepy eyes when open)
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


def map_mediapipe_to_vrm(mp_blendshapes: Dict[str, float]) -> Dict[str, float]:
    """Map MediaPipe ARKit blendshapes to standard VRM 0.x blendshape groups."""
    def g(name: str) -> float:
        return mp_blendshapes.get(name, 0.0)

    vrm = {name: 0.0 for name in VRM_BLENDSHAPES}

    # 1. Mouth Deadzone & Visemes
    jaw = g("jawOpen")
    # Deadzone at 0.05 to keep mouth cleanly shut when resting/silent
    jaw_active = max(0.0, (jaw - 0.05) / 0.95) if jaw > 0.05 else 0.0
    mouth_close = g("mouthClose")
    stretch = (g("mouthStretchLeft") + g("mouthStretchRight")) * 0.5
    smile = (g("mouthSmileLeft") + g("mouthSmileRight")) * 0.5
    pucker = g("mouthPucker")
    funnel = g("mouthFunnel")

    vrm["a"] = _clamp(jaw_active * 1.4 - mouth_close * 0.5)
    vrm["i"] = _clamp(stretch * 0.7 + smile * 0.5 - jaw_active * 0.3)
    vrm["u"] = _clamp(pucker * 1.2 - stretch * 0.4)
    vrm["e"] = _clamp(jaw_active * 0.5 + stretch * 0.6 - pucker * 0.4)
    vrm["o"] = _clamp(funnel * 0.8 + jaw_active * 0.6 + pucker * 0.3)

    vrm["neutral"] = _clamp(1.0 - (vrm["a"] + vrm["i"] + vrm["u"] + vrm["e"] + vrm["o"]))

    # 2. Eye Blinks & Independent Winking
    raw_l = g("eyeBlinkLeft")
    raw_r = g("eyeBlinkRight")
    blink_l = _calibrate_blink(raw_l)
    blink_r = _calibrate_blink(raw_r)

    # If both eyes are closing together, activate simultaneous blink
    if blink_l > 0.3 and blink_r > 0.3:
        vrm["blink"] = max(blink_l, blink_r)
        vrm["blink_l"] = blink_l
        vrm["blink_r"] = blink_r
    else:
        # Independent winking
        vrm["blink"] = 0.0
        vrm["blink_l"] = blink_l
        vrm["blink_r"] = blink_r

    # 3. Facial Expressions
    vrm["joy"] = _clamp(smile * 0.95)
    vrm["angry"] = _clamp((g("browDownLeft") + g("browDownRight")) * 0.8)
    vrm["sorrow"] = _clamp(g("browInnerUp") * 0.85)
    vrm["fun"] = _clamp((g("eyeWideLeft") + g("eyeWideRight")) * 0.5 + jaw_active * 0.5)

    # 4. Gaze Direction
    vrm["lookup"] = _clamp((g("eyeLookUpLeft") + g("eyeLookUpRight")) * 0.6)
    vrm["lookdown"] = _clamp((g("eyeLookDownLeft") + g("eyeLookDownRight")) * 0.6)
    vrm["lookleft"] = _clamp((g("eyeLookInRight") + g("eyeLookOutLeft")) * 0.6)
    vrm["lookright"] = _clamp((g("eyeLookInLeft") + g("eyeLookOutRight")) * 0.6)

    return vrm


def compute_head_pose_from_matrix(matrix: np.ndarray) -> Dict[str, float]:
    """
    Extract 3D head rotation angles (Pitch, Yaw, Roll in radians) from MediaPipe 4x4 matrix.
    Uses cv2.RQDecomp3x3 on the rotation submatrix to completely decouple head pose
    from facial movements (e.g. mouth opening).
    """
    if matrix is None:
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    mat = np.array(matrix, dtype=np.float64)
    if mat.shape != (4, 4):
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    R = mat[:3, :3]
    angles, _, _, _, _, _ = cv2.RQDecomp3x3(R)
    # RQDecomp3x3 returns angles in degrees: [pitch, yaw, roll]
    pitch = float(np.radians(angles[0]))
    yaw = float(np.radians(angles[1]))
    roll = float(np.radians(angles[2]))

    return {
        "pitch": float(np.clip(pitch, -0.6, 0.6)),
        "yaw": float(np.clip(yaw, -0.8, 0.8)),
        "roll": float(np.clip(roll, -0.6, 0.6)),
    }


def compute_head_pose(
    landmarks: Dict[str, Tuple[float, float, float]],
    frame_shape: Tuple[int, int] = (480, 640),
    matrix: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Compute 3D head rotation angles (Pitch, Yaw, Roll in radians).
    Prefers 4x4 matrix decomposition; falls back to landmark geometry.
    """
    if matrix is not None:
        return compute_head_pose_from_matrix(matrix)

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