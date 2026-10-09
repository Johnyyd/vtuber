"""
Face landmark detection using MediaPipe Tasks API (mediapipe >= 1.0).

The legacy `mp.solutions.face_mesh` API was removed in mediapipe 1.0;
this module wraps the new FaceLandmarker task.
"""

import os
import sys
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"

if getattr(sys, "frozen", False):
    _bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
    _exe_dir = os.path.dirname(sys.executable)
    _cand = os.path.join(_exe_dir, "models")
    MODEL_DIR = _cand if os.path.exists(_cand) else os.path.join(_bundle_dir, "models")
else:
    MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

MODEL_PATH = os.path.join(MODEL_DIR, "face_landmarker.task")


def ensure_model_downloaded() -> str:
    """Download the FaceLandmarker model file if not present."""
    if os.path.exists(MODEL_PATH):
        return MODEL_PATH
    os.makedirs(MODEL_DIR, exist_ok=True)
    print(f"[Model] Downloading face_landmarker.task ...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print(f"[Model] Saved to {MODEL_PATH}")
    return MODEL_PATH


def create_face_landmarker():
    """Create a FaceLandmarker in VIDEO mode for webcam streaming."""
    model_path = ensure_model_downloaded()
    base_options = mp_python.BaseOptions(model_asset_path=model_path)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
        min_face_detection_confidence=0.25,
        min_face_presence_confidence=0.25,
        min_tracking_confidence=0.25,
        output_face_blendshapes=True,
        output_facial_transformation_matrixes=True,
    )
    return vision.FaceLandmarker.create_from_options(options)


def parse_landmarks(result) -> list:
    """Convert FaceLandmarkerResult into a list of {lm{i}: (x, y, z)} dicts."""
    out = []
    for face_landmarks in result.face_landmarks:
        face = {f"lm{i}": (lm.x, lm.y, lm.z) for i, lm in enumerate(face_landmarks)}
        out.append(face)
    return out


def parse_blendshapes(result) -> dict:
    """Convert FaceLandmarkerResult blendshapes into a {name: score} dict."""
    out = {}
    if result.face_blendshapes:
        for category in result.face_blendshapes[0]:
            out[category.category_name] = category.score
    return out


def parse_transformation_matrix(result):
    """Extract first 4x4 facial transformation matrix if available."""
    if (
        hasattr(result, "facial_transformation_matrixes")
        and result.facial_transformation_matrixes
    ):
        return result.facial_transformation_matrixes[0]
    return None
