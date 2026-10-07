"""
VTuber Desktop Window & In-Memory IPC Renderer

Uses PyQt6 and QWebEngineView to render the 3D VRM avatar inside a native desktop window.
Webcam capture and MediaPipe facial tracking run in a background QThread.
Tracking data is passed directly into the 3D scene runtime via page.runJavaScript()
(zero WebSockets, zero open network ports).
"""

import os
import sys
import json
import time
from typing import Dict, Any, Optional

import cv2
import numpy as np
import mediapipe as mp

from PyQt6.QtCore import QThread, pyqtSignal, QUrl, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from src.face_detector import create_face_landmarker, parse_landmarks, parse_blendshapes
from src.landmark_mapping import map_mediapipe_to_vrc, compute_head_pose, VRC_TARGETS


def format_motion_packet(
    vrc: Optional[Dict[str, float]] = None,
    rotation: Optional[Dict[str, float]] = None
) -> Dict[str, Any]:
    """
    Format VRC blendshape weights and 3D head rotation into a clean motion packet.
    """
    if vrc is None:
        vrc = {name: (1.0 if name == "vrc_v_sil" else 0.0) for name in VRC_TARGETS}
    if rotation is None:
        rotation = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}

    return {
        "vrc": vrc,
        "rotation": rotation,
        "timestamp": time.time(),
    }


class TrackingWorker(QThread):
    """
    Background worker thread capturing webcam frames and computing facial tracking.
    """
    motion_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)

    def __init__(self, camera_id: int = 0):
        super().__init__()
        self.camera_id = camera_id
        self._running = True

    def run(self):
        self.status_changed.emit("Initializing webcam & FaceLandmarker...")
        cap = cv2.VideoCapture(self.camera_id)

        if not cap.isOpened():
            self.status_changed.emit(f"Error: Unable to open camera {self.camera_id}")
            return

        # Optimize camera latency
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        try:
            landmarker = create_face_landmarker()
        except Exception as e:
            self.status_changed.emit(f"Error loading FaceLandmarker: {e}")
            cap.release()
            return

        self.status_changed.emit("Camera tracking active")
        timestamp_ms = 0

        while self._running:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.01)
                continue

            h, w = frame.shape[:2]
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

            timestamp_ms += 33  # ~30 FPS step
            result = landmarker.detect_for_video(mp_image, timestamp_ms)

            lm_list = parse_landmarks(result)
            mp_blendshapes = parse_blendshapes(result)

            if mp_blendshapes and lm_list:
                vrc = map_mediapipe_to_vrc(mp_blendshapes)
                rotation = compute_head_pose(lm_list[0], frame_shape=(h, w))
                packet = format_motion_packet(vrc, rotation)
                self.motion_ready.emit(packet)
            else:
                # No face detected in frame -> send decay signal
                packet = format_motion_packet(None, None)
                self.motion_ready.emit(packet)

        cap.release()
        self.status_changed.emit("Camera stopped")

    def stop(self):
        self._running = False
        self.wait(1000)


class VTuberWindow(QMainWindow):
    """
    Native Desktop Application Window hosting the 3D VRM Viewport.
    """
    def __init__(self, camera_id: int = 0, width: int = 1024, height: int = 768):
        super().__init__()
        self.setWindowTitle("VTuber 3D Avatar")
        self.resize(width, height)

        # Central widget and WebEngineView
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.web_view = QWebEngineView(self)
        layout.addWidget(self.web_view)

        # Configure WebEngine settings for local 3D rendering
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)

        # Load local viewer HTML file
        viewer_html_path = os.path.abspath(
            os.path.join(os.path.dirname(__file__), "..", "viewer", "index.html")
        )
        self.web_view.load(QUrl.fromLocalFile(viewer_html_path))

        # Start tracking thread
        self.worker = TrackingWorker(camera_id=camera_id)
        self.worker.motion_ready.connect(self.on_motion_ready)
        self.worker.start()

    def on_motion_ready(self, packet: dict):
        """Forward motion data into Javascript runtime via direct in-memory IPC."""
        json_str = json.dumps(packet)
        self.web_view.page().runJavaScript(f"window.updateMotion({json_str});")

    def keyPressEvent(self, event):
        """Allow quick exit via Esc or Q key."""
        if event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean shutdown of background tracking thread."""
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
        event.accept()


def render_vrm(blendshapes: Dict[str, float]):
    """Legacy stub function for backward compatibility."""
    for name, val in blendshapes.items():
        if val > 0.05:
            print(f"  [{name}] -> {val:.3f}")