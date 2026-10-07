"""
MediaPipe (ARKit) → VRM Blendshape Mapping

Maps the 52 MediaPipe FaceLandmarker blendshapes to the 17 VRM blendshape groups:
- neutral, a, i, u, e, o (visemes)
- blink, blink_l, blink_r (eye blinks)
- joy, angry, sorrow, fun (expressions)
- lookup, lookdown, lookleft, lookright (eye gaze)
"""

import numpy as np
from typing import Dict


# VRM blendshape preset names (from your character.vrm)
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
    return max(min_val, min(max_val, value))


def map_mediapipe_to_vrm(mp_blendshapes: Dict[str, float]) -> Dict[str, float]:
    """
    Map MediaPipe ARKit blendshapes to VRM blendshape groups.

    Args:
        mp_blendshapes: Dict of MediaPipe blendshape name -> score (0-1)
        Note: MediaPipe uses camelCase ARKit names (e.g., jawOpen, mouthSmileLeft)

    Returns:
        Dict of VRM blendshape preset name -> weight (0-1)
    """
    # Helper to get value with default 0
    def g(name: str) -> float:
        return mp_blendshapes.get(name, 0.0)

    vrm = {name: 0.0 for name in VRM_BLENDSHAPES}

    # ==================== VISEMES (mouth shapes) ====================
    # VRM uses Japanese vowel system: a, i, u, e, o

    # "a" - mouth wide open, jaw down
    vrm["a"] = _clamp(
        g("jawOpen") * 1.2
        + g("mouthLowerDownLeft") * 0.3
        + g("mouthLowerDownRight") * 0.3
        - g("mouthClose") * 0.5
        - g("mouthPressLeft") * 0.2
        - g("mouthPressRight") * 0.2
    )

    # "i" - mouth stretched wide horizontally (smile-like)
    vrm["i"] = _clamp(
        g("mouthStretchLeft") * 1.0
        + g("mouthStretchRight") * 1.0
        + g("mouthSmileLeft") * 0.5
        + g("mouthSmileRight") * 0.5
        - g("mouthPucker") * 0.5
        - g("mouthFunnel") * 0.5
    )

    # "u" - mouth puckered/protruded (O shape)
    vrm["u"] = _clamp(
        g("mouthPucker") * 1.2
        + g("mouthFunnel") * 0.8
        + g("mouthRollLower") * 0.3
        + g("mouthRollUpper") * 0.3
        - g("mouthStretchLeft") * 0.3
        - g("mouthStretchRight") * 0.3
    )

    # "e" - mouth slightly open, corners back (like "eh")
    vrm["e"] = _clamp(
        g("jawOpen") * 0.5
        + g("mouthStretchLeft") * 0.4
        + g("mouthStretchRight") * 0.4
        + g("mouthLowerDownLeft") * 0.2
        + g("mouthLowerDownRight") * 0.2
        - g("mouthPucker") * 0.4
    )

    # "o" - mouth rounded, moderately open
    vrm["o"] = _clamp(
        g("mouthPucker") * 0.7
        + g("jawOpen") * 0.5
        + g("mouthRollLower") * 0.3
        + g("mouthRollUpper") * 0.3
        - g("mouthStretchLeft") * 0.3
        - g("mouthStretchRight") * 0.3
    )

    # Neutral - base face, influenced by mouth being closed
    vrm["neutral"] = _clamp(
        1.0
        - vrm["a"]
        - vrm["i"]
        - vrm["u"]
        - vrm["e"]
        - vrm["o"]
        + g("mouthClose") * 0.3
    )

    # ==================== EYE BLINKS ====================
    # Average blink for both eyes
    blink_avg = (g("eyeBlinkLeft") + g("eyeBlinkRight")) * 0.5
    vrm["blink"] = _clamp(blink_avg * 1.2)

    # Individual eye blinks
    vrm["blink_l"] = _clamp(g("eyeBlinkLeft") * 1.2)
    vrm["blink_r"] = _clamp(g("eyeBlinkRight") * 1.2)

    # ==================== EXPRESSIONS ====================

    # Joy/Happy - smile + cheek raise + eye squint
    vrm["joy"] = _clamp(
        g("mouthSmileLeft") * 1.0
        + g("mouthSmileRight") * 1.0
        + g("cheekSquintLeft") * 0.5
        + g("cheekSquintRight") * 0.5
        + g("eyeSquintLeft") * 0.3
        + g("eyeSquintRight") * 0.3
        - g("browDownLeft") * 0.3
        - g("browDownRight") * 0.3
    )

    # Angry - brow down + eye squint + mouth press/frown
    vrm["angry"] = _clamp(
        g("browDownLeft") * 1.0
        + g("browDownRight") * 1.0
        + g("eyeSquintLeft") * 0.6
        + g("eyeSquintRight") * 0.6
        + g("mouthPressLeft") * 0.4
        + g("mouthPressRight") * 0.4
        + g("mouthPressLeft") * 0.5
        + g("mouthPressRight") * 0.5
        - g("browInnerUp") * 0.4
    )

    # Sorrow/Sad - brow inner up + mouth frown + eye wide down
    vrm["sorrow"] = _clamp(
        g("browInnerUp") * 1.0
        + g("mouthPressLeft") * 0.7
        + g("mouthPressRight") * 0.7
        + g("eyeLookDownLeft") * 0.3
        + g("eyeLookDownRight") * 0.3
        - g("mouthSmileLeft") * 0.3
        - g("mouthSmileRight") * 0.3
    )

    # Fun/Surprise - brow up + eye wide + jaw open
    vrm["fun"] = _clamp(
        g("browOuterUpLeft") * 0.7
        + g("browOuterUpRight") * 0.7
        + g("browInnerUp") * 0.5
        + g("eyeWideLeft") * 0.8
        + g("eyeWideRight") * 0.8
        + g("jawOpen") * 0.6
        - g("eyeSquintLeft") * 0.3
        - g("eyeSquintRight") * 0.3
    )

    # ==================== EYE GAZE ====================

    # Look Up - eyes looking up
    vrm["lookup"] = _clamp(
        (g("eyeLookUpLeft") + g("eyeLookUpRight")) * 0.5 * 1.2
    )

    # Look Down - eyes looking down
    vrm["lookdown"] = _clamp(
        (g("eyeLookDownLeft") + g("eyeLookDownRight")) * 0.5 * 1.2
    )

    # Look Left - eyes looking left
    vrm["lookleft"] = _clamp(
        (g("eyeLookInRight") + g("eyeLookOutLeft")) * 0.5 * 1.2
    )

    # Look Right - eyes looking right
    vrm["lookright"] = _clamp(
        (g("eyeLookInLeft") + g("eyeLookOutRight")) * 0.5 * 1.2
    )

    return vrm


def map_landmarks_to_blendshapes(landmarks: Dict[str, tuple]) -> Dict[str, float]:
    """
    Legacy function - kept for backward compatibility.
    Maps raw landmarks to a simple mouth_open blendshape.
    """
    def _distance(p1, p2):
        return np.linalg.norm(np.array(p1) - np.array(p2))

    # Use mouth landmarks (13=upper lip, 14=lower lip)
    lm13 = landmarks.get("lm13", (0.5, 0.5, 0))
    lm14 = landmarks.get("lm14", (0.5, 0.5, 0))
    mouth_open = _distance(lm13, lm14)
    mouth_open = _clamp(mouth_open * 2.5)
    return {"mouth_open": mouth_open}


# For debugging - print the mapping
if __name__ == "__main__":
    # Test with sample MediaPipe blendshapes
    # Note: MediaPipe uses camelCase ARKit names
    test_mp = {
        "jawOpen": 0.8,
        "mouthSmileLeft": 0.7,
        "mouthSmileRight": 0.7,
        "eyeBlinkLeft": 0.9,
        "eyeBlinkRight": 0.9,
        "browDownLeft": 0.6,
        "browDownRight": 0.6,
        "eyeLookUpLeft": 0.5,
        "eyeLookUpRight": 0.5,
    }
    vrm_result = map_mediapipe_to_vrm(test_mp)
    print("VRM Blendshapes:")
    for name in VRM_BLENDSHAPES:
        if vrm_result[name] > 0.01:
            print(f"  {name}: {vrm_result[name]:.3f}")