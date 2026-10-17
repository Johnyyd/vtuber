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
import threading
from typing import Dict, Any, Optional

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
for _p in (_REPO_ROOT, _CURRENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import mediapipe as mp

from PyQt6.QtCore import QThread, pyqtSignal, QUrl, Qt
from PyQt6.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings

from face_detector import (
    create_face_landmarker,
    parse_landmarks,
    parse_blendshapes,
    parse_transformation_matrix,
)
from landmark_mapping import (
    map_mediapipe_to_vrc,
    map_mediapipe_to_vrm,
    compute_head_pose,
    compute_iris_gaze,
    draw_mediapipe_debug,
    OneEuroFilter,
    VRC_TARGETS,
    VRM_BLENDSHAPES,
)
try:
    from openseeface_tracker import OpenSeeFaceTracker, draw_openseeface_debug
    from debug_window import TrackingDebugWindow
except ImportError:
    from src.openseeface_tracker import OpenSeeFaceTracker, draw_openseeface_debug
    from src.debug_window import TrackingDebugWindow



def format_motion_packet(
    vrc: Optional[Dict[str, float]] = None,
    rotation: Optional[Dict[str, float]] = None,
    vrm: Optional[Dict[str, float]] = None,
    gaze: Optional[Dict[str, float]] = None,
    body: Optional[Dict[str, float]] = None,
) -> Dict[str, Any]:
    """
    Format VRM and VRC blendshape weights, 3D head rotation, 2D iris gaze, and body motion into a clean motion packet.
    """
    if vrc is None:
        vrc = {name: (1.0 if name == "vrc_v_sil" else 0.0) for name in VRC_TARGETS}
    if vrm is None:
        vrm = {name: (1.0 if name == "neutral" else 0.0) for name in VRM_BLENDSHAPES}
    if rotation is None:
        rotation = {"pitch": 0.0, "yaw": 0.0, "roll": 0.0}
    if gaze is None:
        gaze = {"x": 0.0, "y": 0.0}
    if body is None:
        body = {"sway_gain": 1.0, "breath_gain": 1.0}

    return {
        "vrm": vrm,
        "vrc": vrc,
        "rotation": rotation,
        "gaze": gaze,
        "body": body,
        "timestamp": time.time(),
    }


class CameraReader:
    """
    Decoupled background camera capture thread.
    Continuously drains the V4L2 camera buffer to prevent queue accumulation
    and guarantees zero-latency, real-time latest frames.
    """
    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480, fps: int = 30):
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.running = True
        self.lock = threading.Lock()
        self.frame = None
        self.ret = False
        self.thread = threading.Thread(target=self._capture_loop, daemon=True)
        self.thread.start()

    def _capture_loop(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret = ret
                    self.frame = frame
            else:
                time.sleep(0.005)

    def read_latest(self):
        with self.lock:
            if not self.ret or self.frame is None:
                return False, None
            return True, self.frame

    def is_opened(self) -> bool:
        return self.cap.isOpened()

    def release(self):
        self.running = False
        if hasattr(self, "thread"):
            self.thread.join(timeout=0.5)
        self.cap.release()


class TrackingWorker(QThread):
    """
    Background worker thread capturing webcam frames and computing facial tracking.
    """
    motion_ready = pyqtSignal(dict)
    status_changed = pyqtSignal(str)
    debug_frame_ready = pyqtSignal(np.ndarray, dict)

    def __init__(self, camera_id: int = 0, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.camera_id = camera_id
        self.config = config or {}
        self._running = True
        self.debug_mode = bool(self.config.get("debug_mode", False))
        self.fps = 0.0
        self._fps_counter = 0
        self._fps_last_t = time.time()
        # OneEuroFilters for smooth, responsive head rotation and iris gaze without micro-jitter
        cutoff = float(self.config.get("filter_min_cutoff", 0.8))
        beta = float(self.config.get("filter_beta", 0.015))
        self.filter_pitch = OneEuroFilter(min_cutoff=cutoff, beta=beta)
        self.filter_yaw = OneEuroFilter(min_cutoff=cutoff, beta=beta)
        self.filter_roll = OneEuroFilter(min_cutoff=cutoff, beta=beta)
        self.filter_gaze_x = OneEuroFilter(min_cutoff=cutoff * 1.25, beta=beta * 1.3)
        self.filter_gaze_y = OneEuroFilter(min_cutoff=cutoff * 1.25, beta=beta * 1.3)
        self.last_stable_gaze = {"x": 0.0, "y": 0.0}
        self.body_sway_gain = float(self.config.get("body_sway_gain", 1.0))
        self.idle_breath_intensity = float(self.config.get("idle_breath_intensity", 1.0))
        self.hair_gravity = float(self.config.get("hair_gravity", 1.60))
        self.hair_stiffness = float(self.config.get("hair_stiffness", 0.38))
        self.hair_drag = float(self.config.get("hair_drag", 0.58))
        self.eye_pitch_offset = float(self.config.get("eye_pitch_offset", -0.14))
        self.backend = str(self.config.get("tracker_backend", "mediapipe")).strip().lower()
        self.tracker = None

    def set_debug_mode(self, enabled: bool):
        self.debug_mode = bool(enabled)

    def recalibrate(self):
        """Reset tracking baselines for neutral posture calibration."""
        if hasattr(self, "tracker") and self.tracker is not None:
            if hasattr(self.tracker, "reset_baselines"):
                self.tracker.reset_baselines()

    def run(self):
        backend_name = "OpenSeeFace ONNX" if self.backend == "openseeface" else "Google MediaPipe"
        print("\n" + "=" * 65)
        print(f" [VTuber] MO HINH TRACKING DANG CHAY: {backend_name.upper()}")
        if self.backend == "openseeface":
            print(" [VTuber] -> He thong ONNX chuyen dung cho VTuber (68 3D landmarks)")
        else:
            print(" [VTuber] -> He thong Google MediaPipe (468 landmarks)")
        print("=" * 65 + "\n")

        if self.backend == "openseeface":
            self.status_changed.emit("Initializing webcam & OpenSeeFace ONNX...")
            reader = CameraReader(camera_id=self.camera_id, width=1280, height=720, fps=30)

            if not reader.is_opened():
                self.status_changed.emit(f"Error: Unable to open camera {self.camera_id}")
                reader.release()
                return

            try:
                self.tracker = OpenSeeFaceTracker(width=1280, height=720, max_threads=2)
                tracker = self.tracker
            except Exception as e:
                self.status_changed.emit(f"Error loading OpenSeeFace: {e}")
                reader.release()
                return

            self.status_changed.emit("Camera tracking active (OpenSeeFace)")
            wait_frames_start = time.time()
            has_received_first_frame = False

            while self._running:
                try:
                    ret, frame = reader.read_latest()
                    if not ret or frame is None:
                        if not has_received_first_frame and (time.time() - wait_frames_start > 3.0):
                            self.status_changed.emit(
                                f"Warning: Waiting for frames from camera {self.camera_id}... Check device index."
                            )
                            wait_frames_start = time.time()
                        time.sleep(0.005)
                        continue

                    has_received_first_frame = True
                    tracked = tracker.process_frame(frame, self.config)

                    if tracked is not None:
                        vrm_shapes, vrc, raw_rotation, raw_gaze = tracked

                        # Check if eyes are blinking or closing
                        is_blinking = (
                            vrm_shapes.get("blink", 0.0) > 0.40 or
                            vrm_shapes.get("blink_l", 0.0) > 0.45 or
                            vrm_shapes.get("blink_r", 0.0) > 0.45
                        )

                        # Filter rotation & iris gaze to eliminate jitter while keeping instant response
                        t_now = time.time()
                        filtered_rotation = {
                            "pitch": self.filter_pitch.filter(raw_rotation["pitch"], t_now),
                            "yaw": self.filter_yaw.filter(raw_rotation["yaw"], t_now),
                            "roll": self.filter_roll.filter(raw_rotation["roll"], t_now),
                        }
                        if is_blinking:
                            # Freeze gaze at pre-blink position so pupils never twitch or jump during blink
                            filtered_gaze = dict(self.last_stable_gaze)
                        else:
                            filtered_gaze = {
                                "x": self.filter_gaze_x.filter(raw_gaze["x"], t_now),
                                "y": self.filter_gaze_y.filter(raw_gaze["y"], t_now),
                            }
                            self.last_stable_gaze = dict(filtered_gaze)

                        body_params = {
                            "backend": "OpenSeeFace",
                            "sway_gain": self.body_sway_gain,
                            "breath_gain": self.idle_breath_intensity,
                            "hair_gravity": self.hair_gravity,
                            "hair_stiffness": self.hair_stiffness,
                            "hair_drag": self.hair_drag,
                            "eye_pitch_offset": self.eye_pitch_offset,
                        }
                        packet = format_motion_packet(
                            vrc=vrc,
                            rotation=filtered_rotation,
                            vrm=vrm_shapes,
                            gaze=filtered_gaze,
                            body=body_params,
                        )
                        self.motion_ready.emit(packet)
                    else:
                        # No face detected in frame -> send decay signal and reset filters
                        self.filter_pitch.reset()
                        self.filter_yaw.reset()
                        self.filter_roll.reset()
                        self.filter_gaze_x.reset()
                        self.filter_gaze_y.reset()
                        self.last_stable_gaze = {"x": 0.0, "y": 0.0}
                        body_params = {
                            "backend": "OpenSeeFace",
                            "sway_gain": self.body_sway_gain,
                            "breath_gain": self.idle_breath_intensity,
                            "hair_gravity": self.hair_gravity,
                            "hair_stiffness": self.hair_stiffness,
                            "hair_drag": self.hair_drag,
                        }
                        packet = format_motion_packet(None, None, None, None, body=body_params)
                        self.motion_ready.emit(packet)

                    # Calculate FPS & emit debug frame if debug mode active
                    self._fps_counter += 1
                    t_loop = time.time()
                    if t_loop - self._fps_last_t >= 1.0:
                        self.fps = self._fps_counter / (t_loop - self._fps_last_t)
                        self._fps_counter = 0
                        self._fps_last_t = t_loop

                    if self.debug_mode:
                        has_face = tracked is not None and tracker.last_face is not None
                        smile_val = getattr(tracker.last_face, "smile_lift", 0.0) if has_face else 0.0
                        brow_val = getattr(tracker.last_face, "brow_ratio", 0.0) if has_face else 0.0
                        brow_base = getattr(tracker.last_face, "brow_baseline", 0.0) if has_face else 0.0
                        brow_del = getattr(tracker.last_face, "brow_delta", 0.0) if has_face else 0.0
                        telemetry = {
                            "backend": "OpenSeeFace ONNX",
                            "fps": self.fps,
                            "rotation": filtered_rotation if tracked is not None else {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                            "gaze": filtered_gaze if tracked is not None else {"x": 0.0, "y": 0.0},
                            "smile_lift": smile_val,
                            "brow_ratio": brow_val,
                            "brow_baseline": brow_base,
                            "brow_delta": brow_del,
                            "vrm_shapes": vrm_shapes if tracked is not None else {},
                        }
                        debug_frame = draw_openseeface_debug(
                            frame,
                            tracker.last_face if tracked is not None else None,
                            vrm_shapes if tracked is not None else {},
                            telemetry,
                        )
                        self.debug_frame_ready.emit(debug_frame, telemetry)

                except Exception as e:
                    print(f"[TrackingWorker Warning] Transient frame error: {e}")
                    time.sleep(0.01)

            reader.release()
            self.status_changed.emit("Camera stopped")
            return

        # MediaPipe Backend
        self.status_changed.emit("Initializing webcam & FaceLandmarker...")
        reader = CameraReader(camera_id=self.camera_id, width=1280, height=720, fps=30)

        if not reader.is_opened():
            self.status_changed.emit(f"Error: Unable to open camera {self.camera_id}")
            reader.release()
            return

        try:
            landmarker = create_face_landmarker()
        except Exception as e:
            self.status_changed.emit(f"Error loading FaceLandmarker: {e}")
            reader.release()
            return

        self.status_changed.emit("Camera tracking active")
        last_timestamp_ms = 0
        wait_frames_start = time.time()
        has_received_first_frame = False

        while self._running:
            try:
                ret, frame = reader.read_latest()
                if not ret or frame is None:
                    if not has_received_first_frame and (time.time() - wait_frames_start > 3.0):
                        self.status_changed.emit(
                            f"Warning: Waiting for frames from camera {self.camera_id}... Check device index."
                        )
                        wait_frames_start = time.time()
                    time.sleep(0.005)
                    continue

                has_received_first_frame = True

                h, w = frame.shape[:2]
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

                # Use monotonically increasing wall-clock timestamps for accurate MediaPipe video filters
                now_ms = int(time.time() * 1000)
                if now_ms <= last_timestamp_ms:
                    now_ms = last_timestamp_ms + 1
                last_timestamp_ms = now_ms

                result = landmarker.detect_for_video(mp_image, now_ms)

                matrix = parse_transformation_matrix(result)
                lm_list = parse_landmarks(result)
                mp_blendshapes = parse_blendshapes(result)

                if mp_blendshapes and (lm_list or matrix is not None):
                    landmarks = lm_list[0] if lm_list else {}
                    pitch_offset = float(self.config.get("pitch_offset_deg", 18.0))
                    raw_rotation = compute_head_pose(
                        landmarks,
                        frame_shape=(h, w),
                        matrix=matrix,
                        pitch_offset_deg=pitch_offset,
                    )
                    vrm_shapes = map_mediapipe_to_vrm(
                        mp_blendshapes,
                        landmarks=landmarks,
                        pitch=raw_rotation["pitch"],
                        config=self.config,
                    )
                    vrc = map_mediapipe_to_vrc(mp_blendshapes)
                    raw_gaze = compute_iris_gaze(landmarks, mp_blendshapes=mp_blendshapes)

                    # Check if eyes are blinking or closing (using calibrated vrm_shapes to ignore glasses shadow noise)
                    is_blinking = (
                        vrm_shapes.get("blink", 0.0) > 0.40 or
                        vrm_shapes.get("blink_l", 0.0) > 0.45 or
                        vrm_shapes.get("blink_r", 0.0) > 0.45
                    )

                    # Filter rotation & iris gaze to eliminate jitter while keeping instant response
                    t_now = time.time()
                    filtered_rotation = {
                        "pitch": self.filter_pitch.filter(raw_rotation["pitch"], t_now),
                        "yaw": self.filter_yaw.filter(raw_rotation["yaw"], t_now),
                        "roll": self.filter_roll.filter(raw_rotation["roll"], t_now),
                    }
                    if is_blinking:
                        # Freeze gaze at pre-blink position so pupils never twitch or jump during blink
                        filtered_gaze = dict(self.last_stable_gaze)
                    else:
                        filtered_gaze = {
                            "x": self.filter_gaze_x.filter(raw_gaze["x"], t_now),
                            "y": self.filter_gaze_y.filter(raw_gaze["y"], t_now),
                        }
                        self.last_stable_gaze = dict(filtered_gaze)

                    body_params = {
                        "backend": "MediaPipe",
                        "sway_gain": self.body_sway_gain,
                        "breath_gain": self.idle_breath_intensity,
                        "hair_gravity": self.hair_gravity,
                        "hair_stiffness": self.hair_stiffness,
                        "hair_drag": self.hair_drag,
                        "eye_pitch_offset": self.eye_pitch_offset,
                    }
                    packet = format_motion_packet(
                        vrc=vrc,
                        rotation=filtered_rotation,
                        vrm=vrm_shapes,
                        gaze=filtered_gaze,
                        body=body_params,
                    )
                    self.motion_ready.emit(packet)
                else:
                    # No face detected in frame -> send decay signal and reset filters
                    self.filter_pitch.reset()
                    self.filter_yaw.reset()
                    self.filter_roll.reset()
                    self.filter_gaze_x.reset()
                    self.filter_gaze_y.reset()
                    self.last_stable_gaze = {"x": 0.0, "y": 0.0}
                    body_params = {
                        "backend": "MediaPipe",
                        "sway_gain": self.body_sway_gain,
                        "breath_gain": self.idle_breath_intensity,
                        "hair_gravity": self.hair_gravity,
                        "hair_stiffness": self.hair_stiffness,
                        "hair_drag": self.hair_drag,
                    }
                    packet = format_motion_packet(None, None, None, None, body=body_params)
                    self.motion_ready.emit(packet)

                # Calculate FPS & emit debug frame if debug mode active
                self._fps_counter += 1
                t_loop = time.time()
                if t_loop - self._fps_last_t >= 1.0:
                    self.fps = self._fps_counter / (t_loop - self._fps_last_t)
                    self._fps_counter = 0
                    self._fps_last_t = t_loop

                if self.debug_mode:
                    has_face = bool(mp_blendshapes and (lm_list or matrix is not None))
                    telemetry = {
                        "backend": "Google MediaPipe",
                        "fps": self.fps,
                        "rotation": filtered_rotation if has_face else {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
                        "gaze": filtered_gaze if has_face else {"x": 0.0, "y": 0.0},
                        "smile_lift": 0.0,
                        "brow_ratio": 0.0,
                        "brow_baseline": 0.0,
                        "brow_delta": 0.0,
                        "vrm_shapes": vrm_shapes if has_face else {},
                    }
                    debug_frame = draw_mediapipe_debug(
                        frame,
                        landmarks if has_face else None,
                        vrm_shapes if has_face else {},
                        telemetry,
                    )
                    self.debug_frame_ready.emit(debug_frame, telemetry)

            except Exception as e:
                # Catch any transient frame processing exceptions without terminating tracking thread
                print(f"[TrackingWorker Warning] Transient frame error: {e}")
                time.sleep(0.01)

        reader.release()
        self.status_changed.emit("Camera stopped")

    def stop(self):
        self._running = False
        self.wait(1000)


class VTuberWindow(QMainWindow):
    """
    Native Desktop Application Window hosting the 3D VRM Viewport.
    """
    def __init__(self, camera_id: int = 0, width: int = 1024, height: int = 768, config: Optional[Dict[str, Any]] = None):
        super().__init__()
        self.config = config or {}
        self.setWindowTitle("VTuber 3D Avatar")
        self.resize(width, height)

        # Make window completely transparent, frameless, and stay on top
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)

        # Central widget and WebEngineView
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(0, 0, 0, 0)

        self.web_view = QWebEngineView(self)
        self.web_view.page().setBackgroundColor(Qt.GlobalColor.transparent)
        layout.addWidget(self.web_view)

        # Create a transparent overlay to capture mouse drag events (leaving bottom controls clickable)
        self.drag_overlay = QWidget(self)
        self.drag_overlay.setStyleSheet("background: transparent;")
        overlay_h = max(0, height - 60)
        self.drag_overlay.setGeometry(0, 0, width, overlay_h)

        # Configure WebEngine settings for local 3D rendering
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.WebGLEnabled, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.Accelerated2dCanvasEnabled, True)

        # Connect title changes for IPC commands (e.g. window resize)
        self.web_view.titleChanged.connect(self.on_title_changed)

        # Load local viewer HTML file
        if getattr(sys, "frozen", False):
            _bundle_dir = getattr(sys, "_MEIPASS", os.path.dirname(sys.executable))
            _exe_dir = os.path.dirname(sys.executable)
            _cand_external = os.path.join(_exe_dir, "viewer", "index.html")
            viewer_html_path = _cand_external if os.path.exists(_cand_external) else os.path.join(_bundle_dir, "viewer", "index.html")
        else:
            viewer_html_path = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "..", "viewer", "index.html")
            )
        self.web_view.load(QUrl.fromLocalFile(viewer_html_path))

        # Start tracking thread
        self.worker = TrackingWorker(camera_id=camera_id, config=self.config)
        self.worker.motion_ready.connect(self.on_motion_ready)
        self.worker.start()

        # Debug Window setup
        self.debug_window: Optional[TrackingDebugWindow] = None
        self.worker.debug_frame_ready.connect(self.on_debug_frame)
        if self.config.get("debug_mode", False):
            self.toggle_debug_window()

    def on_debug_frame(self, frame: np.ndarray, telemetry: dict):
        """Receive processed debug frame and update debug window if visible."""
        if self.debug_window is not None and self.debug_window.isVisible():
            self.debug_window.update_debug_view(frame, telemetry)

    def recalibrate(self):
        """Trigger neutral baseline recalibration on tracking worker."""
        if hasattr(self, "worker") and hasattr(self.worker, "recalibrate"):
            self.worker.recalibrate()
            print("[VTuber] Da can bang lai tu the trung tinh (Neutral Pose Calibrated - Phim C).")

    def toggle_debug_window(self):
        """Toggle visibility of tracking telemetry & video debug window."""
        if self.debug_window is None:
            self.debug_window = TrackingDebugWindow()
            self.debug_window.on_recalibrate = self.recalibrate
        if self.debug_window.isVisible():
            self.debug_window.hide()
            self.worker.set_debug_mode(False)
        else:
            self.debug_window.show()
            self.debug_window.raise_()
            self.debug_window.activateWindow()
            self.worker.set_debug_mode(True)

    def resizeEvent(self, event):
        """Ensure the drag overlay covers the window except bottom controls, and notify web view."""
        new_size = event.size()
        bottom_controls_height = 60
        overlay_h = max(0, new_size.height() - bottom_controls_height)
        if hasattr(self, 'drag_overlay'):
            self.drag_overlay.setGeometry(0, 0, new_size.width(), overlay_h)
        super().resizeEvent(event)

        if hasattr(self, 'web_view') and self.web_view:
            js = f"if (typeof window.onWindowResized === 'function') {{ window.onWindowResized({new_size.width()}, {new_size.height()}); }}"
            self.web_view.page().runJavaScript(js)

    def on_title_changed(self, title: str):
        """Handle IPC commands from web view via document.title."""
        if title.startswith("vtuber:resize:"):
            try:
                parts = title.split(":")
                w = int(parts[2])
                h = int(parts[3])
                self.resize(w, h)
            except (ValueError, IndexError):
                pass

    def mousePressEvent(self, event):
        """Allow dragging the frameless window."""
        if event.button() == Qt.MouseButton.LeftButton:
            self.drag_pos = event.globalPosition().toPoint() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        """Handle window dragging."""
        if event.buttons() == Qt.MouseButton.LeftButton and hasattr(self, 'drag_pos'):
            self.move(event.globalPosition().toPoint() - self.drag_pos)
            event.accept()

    def on_motion_ready(self, packet: dict):
        """Forward motion data into Javascript runtime via direct in-memory IPC."""
        json_str = json.dumps(packet)
        js_code = f"if (typeof window.updateMotion === 'function') {{ window.updateMotion({json_str}); }}"
        self.web_view.page().runJavaScript(js_code)

    def keyPressEvent(self, event):
        """Allow quick exit via Esc or Q key, toggle debug window via D key, recalibrate via C key."""
        if event.key() == Qt.Key.Key_D:
            self.toggle_debug_window()
            event.accept()
        elif event.key() == Qt.Key.Key_C:
            self.recalibrate()
            event.accept()
        elif event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.close()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Clean shutdown of background tracking thread and child debug window."""
        if self.debug_window is not None:
            self.debug_window.close()
        if hasattr(self, "worker") and self.worker.isRunning():
            self.worker.stop()
        event.accept()


def render_vrm(blendshapes: Dict[str, float]):
    """Legacy stub function for backward compatibility."""
    for name, val in blendshapes.items():
        if val > 0.05:
            print(f"  [{name}] -> {val:.3f}")