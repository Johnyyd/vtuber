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


def map_mediapipe_to_vrm(mp_blendshapes: Dict[str, float]) -> Dict[str, float]:
    """Map MediaPipe ARKit blendshapes to standard VRM 0.x blendshape groups."""
    def g(name: str) -> float:
        return mp_blendshapes.get(name, 0.0)

    vrm = {name: 0.0 for name in VRM_BLENDSHAPES}

    # Visemes
    vrm["a"] = _clamp(g("jawOpen") * 1.2 - g("mouthClose") * 0.5)
    vrm["i"] = _clamp((g("mouthStretchLeft") + g("mouthStretchRight")) * 0.6)
    vrm["u"] = _clamp(g("mouthPucker") * 1.1)
    vrm["e"] = _clamp(g("jawOpen") * 0.4 + (g("mouthStretchLeft") + g("mouthStretchRight")) * 0.3)
    vrm["o"] = _clamp(g("mouthPucker") * 0.7 + g("jawOpen") * 0.5)

    vrm["neutral"] = _clamp(1.0 - (vrm["a"] + vrm["i"] + vrm["u"] + vrm["e"] + vrm["o"]))

    # Blinks
    blink_avg = (g("eyeBlinkLeft") + g("eyeBlinkRight")) * 0.5
    vrm["blink"] = _clamp(blink_avg * 1.2)
    vrm["blink_l"] = _clamp(g("eyeBlinkLeft") * 1.2)
    vrm["blink_r"] = _clamp(g("eyeBlinkRight") * 1.2)

    # Expressions
    vrm["joy"] = _clamp((g("mouthSmileLeft") + g("mouthSmileRight")) * 0.8)
    vrm["angry"] = _clamp((g("browDownLeft") + g("browDownRight")) * 0.8)
    vrm["sorrow"] = _clamp(g("browInnerUp") * 0.8)
    vrm["fun"] = _clamp((g("eyeWideLeft") + g("eyeWideRight")) * 0.5 + g("jawOpen") * 0.4)

    # Gaze
    vrm["lookup"] = _clamp((g("eyeLookUpLeft") + g("eyeLookUpRight")) * 0.6)
    vrm["lookdown"] = _clamp((g("eyeLookDownLeft") + g("eyeLookDownRight")) * 0.6)
    vrm["lookleft"] = _clamp((g("eyeLookInRight") + g("eyeLookOutLeft")) * 0.6)
    vrm["lookright"] = _clamp((g("eyeLookInLeft") + g("eyeLookOutRight")) * 0.6)

    return vrm


def compute_head_pose(landmarks: Dict[str, Tuple[float, float, float]], frame_shape: Tuple[int, int] = (480, 640)) -> Dict[str, float]:
    """
    Compute 3D head rotation angles (Pitch, Yaw, Roll in radians).
    Uses robust landmark vector geometry:
    - Roll: angle of the eye baseline relative to horizontal
    - Yaw: horizontal offset of nose from eye midpoint, normalized by eye distance
    - Pitch: ratio of nose-to-chin vs eye-to-nose vertical distance

    Args:
        landmarks: Dict of landmark name ("lm{i}") -> (x, y, z) normalized coords.
        frame_shape: (height, width) of the camera frame.

    Returns:
        Dict: {"pitch": float, "yaw": float, "roll": float} in radians.
    """
    required_keys = ["lm1", "lm152", "lm33", "lm263"]
    if not landmarks or not all(k in landmarks for k in required_keys):
        return {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    nose = np.array(landmarks["lm1"][:2], dtype=np.float64)
    chin = np.array(landmarks["lm152"][:2], dtype=np.float64)
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

    # 3. Pitch: Ratio of nose-to-chin vs eye-to-nose vertical distance
    vert_eye_to_nose = nose[1] - eye_center[1]
    vert_nose_to_chin = chin[1] - nose[1]
    if vert_eye_to_nose > 1e-6:
        # Standard neutral facial ratio is ~1.2
        ratio = vert_nose_to_chin / vert_eye_to_nose
        pitch = float((1.2 - ratio) * 0.8)
    else:
        pitch = 0.0

    return {
        "pitch": float(np.clip(pitch, -0.5, 0.5)),
        "yaw": float(np.clip(yaw, -0.7, 0.7)),
        "roll": float(np.clip(roll, -0.4, 0.4)),
    }

    return {
        "pitch": pitch,
        "yaw": yaw,
        "roll": roll,
    }


def map_landmarks_to_blendshapes(landmarks: Dict[str, tuple]) -> Dict[str, float]:
    """Legacy function - kept for backward compatibility."""
    lm13 = landmarks.get("lm13", (0.5, 0.5, 0))
    lm14 = landmarks.get("lm14", (0.5, 0.5, 0))
    dist = float(np.linalg.norm(np.array(lm13) - np.array(lm14)))
    mouth_open = _clamp(dist * 2.5)
    return {"mouth_open": mouth_open}