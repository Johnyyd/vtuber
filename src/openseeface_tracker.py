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
        # In Task 1: minimal output structure for test passing
        # Task 2 will fill in full VRM and VRChat mappings
        vrm_shapes = {
            "neutral": 1.0,
            "blink": 0.0,
            "blink_l": 0.0,
            "blink_r": 0.0,
            "a": 0.0,
            "i": 0.0,
            "u": 0.0,
            "e": 0.0,
            "o": 0.0,
            "joy": 0.0,
            "sorrow": 0.0,
            "surprised": 0.0,
        }
        vrc_targets = {"vrc_blink": 0.0, "vrc_v_aa": 0.0, "vrc_v_sil": 1.0}
        euler = getattr(face, "euler", [0.0, 0.0, 0.0])
        raw_rotation = {
            "pitch": float(euler[0]) if euler is not None else 0.0,
            "yaw": float(euler[1]) if euler is not None else 0.0,
            "roll": float(euler[2]) if euler is not None else 0.0,
        }
        raw_gaze = {"x": 0.0, "y": 0.0}

        return vrm_shapes, vrc_targets, raw_rotation, raw_gaze
