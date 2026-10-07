"""
Face landmark detection using MediaPipe Tasks API (mediapipe >= 1.0).

The legacy `mp.solutions.face_mesh` API was removed in mediapipe 1.0;
this module wraps the new FaceLandmarker task.
"""

import os
import urllib.request

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision

MODEL_URL = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task"
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
        output_face_blendshapes=True,
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
