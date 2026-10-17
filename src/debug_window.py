"""
PyQt6 Tracking Debug Window: Displays real-time camera feed with facial landmark mesh
and comprehensive telemetry HUD for diagnosing expression triggers.
"""

import sys
from typing import Optional, Dict, Any
import numpy as np

from PyQt6.QtWidgets import (
    QWidget,
    QLabel,
    QVBoxLayout,
    QHBoxLayout,
    QGroupBox,
    QGridLayout,
    QProgressBar,
    QFrame,
    QScrollArea,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap, QFont, QColor


class TelemetryBar(QWidget):
    """A metric row with label, value text, and colored progress bar."""

    def __init__(self, name: str, min_val: float = 0.0, max_val: float = 1.0, parent=None):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val

        layout = QHBoxLayout(self)
        layout.setContentsMargins(2, 2, 2, 2)
        layout.setSpacing(6)

        self.lbl_name = QLabel(name)
        self.lbl_name.setFixedWidth(85)
        self.lbl_name.setStyleSheet("color: #ccc; font-weight: bold; font-size: 11px;")

        self.bar = QProgressBar()
        self.bar.setRange(0, 100)
        self.bar.setTextVisible(False)
        self.bar.setFixedHeight(12)
        self.bar.setStyleSheet("""
            QProgressBar {
                background-color: #2a2a2a;
                border-radius: 4px;
                border: 1px solid #444;
            }
            QProgressBar::chunk {
                background-color: #3b82f6;
                border-radius: 3px;
            }
        """)

        self.lbl_val = QLabel("0.00")
        self.lbl_val.setFixedWidth(45)
        self.lbl_val.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        self.lbl_val.setStyleSheet("color: #fff; font-family: monospace; font-size: 11px;")

        layout.addWidget(self.lbl_name)
        layout.addWidget(self.bar, 1)
        layout.addWidget(self.lbl_val)

    def set_value(self, val: float, active_color: str = "#10b981", alert_threshold: Optional[float] = None):
        clamped = max(self.min_val, min(self.max_val, float(val)))
        pct = int(((clamped - self.min_val) / (self.max_val - self.min_val)) * 100)
        self.bar.setValue(pct)
        self.lbl_val.setText(f"{val:+.2f}" if self.min_val < 0 else f"{val:.2f}")

        # Highlight color if active/threshold reached
        is_alert = (alert_threshold is not None and val > alert_threshold) or (alert_threshold is None and val > 0.05)
        color = active_color if is_alert else "#4b5563"
        self.bar.setStyleSheet(f"""
            QProgressBar {{
                background-color: #222;
                border-radius: 4px;
                border: 1px solid #333;
            }}
            QProgressBar::chunk {{
                background-color: {color};
                border-radius: 3px;
            }}
        """)


class TrackingDebugWindow(QWidget):
    """
    Dedicated Tracking Telemetry & Video Debug Window.
    Allows real-time inspection of 2D/3D landmarks, pupil gaze, and expression calculations.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("VTuber Tracking Debug Mode")
        self.resize(1020, 680)
        self.setStyleSheet("""
            QWidget {
                background-color: #121214;
                color: #e5e5e5;
                font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif;
            }
            QGroupBox {
                border: 1px solid #2d3748;
                border-radius: 6px;
                margin-top: 10px;
                font-weight: bold;
                font-size: 12px;
                color: #60a5fa;
                padding-top: 14px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 0 6px;
            }
        """)

        main_layout = QHBoxLayout(self)
        main_layout.setContentsMargins(12, 12, 12, 12)
        main_layout.setSpacing(12)

        # Left Column: Video display with landmarks
        video_frame = QFrame()
        video_frame.setStyleSheet("background-color: #000; border-radius: 8px; border: 1px solid #262626;")
        video_layout = QVBoxLayout(video_frame)
        video_layout.setContentsMargins(6, 6, 6, 6)

        self.video_label = QLabel()
        self.video_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video_label.setText("Dang cho tin hieu tu camera...")
        self.video_label.setStyleSheet("color: #737373; font-size: 14px;")
        self.video_label.setMinimumSize(480, 360)
        video_layout.addWidget(self.video_label, 1)

        # Hint under video
        hint_lbl = QLabel("Phim tat: [C] Can bang lai tu the (Recalibrate)  |  [D] An/hien Debug  |  [Esc] Thoat")
        hint_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        hint_lbl.setStyleSheet("color: #71717a; font-size: 11px; padding: 4px;")
        video_layout.addWidget(hint_lbl)

        main_layout.addWidget(video_frame, 3)

        # Right Column: Telemetry & Expressions Panel
        sidebar = QFrame()
        sidebar.setFixedWidth(360)
        sidebar.setStyleSheet("background-color: #18181b; border-radius: 8px; border: 1px solid #27272a;")
        side_layout = QVBoxLayout(sidebar)
        side_layout.setContentsMargins(12, 12, 12, 12)
        side_layout.setSpacing(10)

        # Header Badge
        self.lbl_badge = QLabel("Backend: OpenSeeFace ONNX")
        self.lbl_badge.setStyleSheet("""
            background-color: #1e3a8a;
            color: #bfdbfe;
            font-weight: bold;
            font-size: 12px;
            padding: 6px;
            border-radius: 4px;
        """)
        self.lbl_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        side_layout.addWidget(self.lbl_badge)

        # Scrollable Telemetry Cards
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("QScrollArea { border: none; background: transparent; }")
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setContentsMargins(0, 0, 4, 0)
        scroll_layout.setSpacing(8)

        # Group 1: Head Pose & Rotation
        grp_head = QGroupBox("Goc Xoay Dau (Head Pose)")
        lay_head = QVBoxLayout(grp_head)
        self.bar_pitch = TelemetryBar("Pitch (Up/Down)", -45.0, 45.0)
        self.bar_yaw = TelemetryBar("Yaw (Left/Right)", -50.0, 50.0)
        self.bar_roll = TelemetryBar("Roll (Tilt)", -35.0, 35.0)
        lay_head.addWidget(self.bar_pitch)
        lay_head.addWidget(self.bar_yaw)
        lay_head.addWidget(self.bar_roll)
        scroll_layout.addWidget(grp_head)

        # Group 2: Eyes & Gaze
        grp_eyes = QGroupBox("Mat & Huong Nhin (Eyes & Gaze)")
        lay_eyes = QVBoxLayout(grp_eyes)
        self.bar_blink_l = TelemetryBar("Blink Left", 0.0, 1.0)
        self.bar_blink_r = TelemetryBar("Blink Right", 0.0, 1.0)
        self.bar_blink_both = TelemetryBar("Blink (Both)", 0.0, 1.0)
        self.bar_gaze_x = TelemetryBar("Gaze X", -1.0, 1.0)
        self.bar_gaze_y = TelemetryBar("Gaze Y", -1.0, 1.0)
        lay_eyes.addWidget(self.bar_blink_l)
        lay_eyes.addWidget(self.bar_blink_r)
        lay_eyes.addWidget(self.bar_blink_both)
        lay_eyes.addWidget(self.bar_gaze_x)
        lay_eyes.addWidget(self.bar_gaze_y)
        scroll_layout.addWidget(grp_eyes)

        # Group 3: Mouth & Expressions (CRITICAL FOR NEUTRAL/JOY DEBUGGING)
        grp_mouth = QGroupBox("Khau Hinh & Nu Cuoi (Expressions)")
        lay_mouth = QVBoxLayout(grp_mouth)

        self.bar_smile_lift = TelemetryBar("Smile Lift (Raw)", -0.40, 0.10)
        self.bar_joy = TelemetryBar("Joy (Cuoi)", 0.0, 1.0)
        self.bar_u = TelemetryBar("Vowel U", 0.0, 1.0)
        self.bar_a = TelemetryBar("Mouth A (Mo)", 0.0, 1.0)
        self.bar_i = TelemetryBar("Vowel I", 0.0, 1.0)
        self.bar_e = TelemetryBar("Vowel E", 0.0, 1.0)
        self.bar_o = TelemetryBar("Vowel O", 0.0, 1.0)
        self.bar_brow_ratio = TelemetryBar("Brow Ratio (Raw)", 0.25, 0.75)
        self.bar_surprised = TelemetryBar("Surprised", 0.0, 1.0)
        self.bar_angry = TelemetryBar("Angry (Gian Du)", 0.0, 1.0)
        self.bar_sorrow = TelemetryBar("Sorrow", 0.0, 1.0)

        lay_mouth.addWidget(self.bar_smile_lift)
        lay_mouth.addWidget(self.bar_joy)
        lay_mouth.addWidget(self.bar_u)
        lay_mouth.addWidget(self.bar_a)
        lay_mouth.addWidget(self.bar_i)
        lay_mouth.addWidget(self.bar_e)
        lay_mouth.addWidget(self.bar_o)
        lay_mouth.addWidget(self.bar_brow_ratio)
        lay_mouth.addWidget(self.bar_surprised)
        lay_mouth.addWidget(self.bar_angry)
        lay_mouth.addWidget(self.bar_sorrow)
        scroll_layout.addWidget(grp_mouth)

        scroll_layout.addStretch(1)
        scroll.setWidget(scroll_content)
        side_layout.addWidget(scroll, 1)

        main_layout.addWidget(sidebar, 1)

    def update_debug_view(self, frame: np.ndarray, telemetry: Dict[str, Any]):
        """Render new video frame with mesh and update telemetry values."""
        if frame is not None and frame.size > 0:
            h, w = frame.shape[:2]
            # Convert BGR to RGB
            rgb = np.ascontiguousarray(frame[:, :, ::-1])
            qimg = QImage(rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)
            pixmap = QPixmap.fromImage(qimg)

            # Scale to fit label smoothly
            lbl_size = self.video_label.size()
            if lbl_size.width() > 10 and lbl_size.height() > 10:
                scaled = pixmap.scaled(
                    lbl_size,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                self.video_label.setPixmap(scaled)

        # Update Telemetry Dashboard
        backend = str(telemetry.get("backend", "OpenSeeFace ONNX"))
        fps = float(telemetry.get("fps", 0.0))
        self.lbl_badge.setText(f"{backend.upper()} | {fps:.1f} FPS" if fps > 0 else backend.upper())

        rot = telemetry.get("rotation", {})
        deg = 180.0 / np.pi
        self.bar_pitch.set_value(float(rot.get("pitch", 0.0)) * deg, active_color="#38bdf8")
        self.bar_yaw.set_value(float(rot.get("yaw", 0.0)) * deg, active_color="#38bdf8")
        self.bar_roll.set_value(float(rot.get("roll", 0.0)) * deg, active_color="#38bdf8")

        gaze = telemetry.get("gaze", {})
        self.bar_gaze_x.set_value(float(gaze.get("x", 0.0)), active_color="#a855f7")
        self.bar_gaze_y.set_value(float(gaze.get("y", 0.0)), active_color="#a855f7")

        vrm = telemetry.get("vrm_shapes", {})
        self.bar_blink_l.set_value(float(vrm.get("blink_l", 0.0)), active_color="#f59e0b", alert_threshold=0.35)
        self.bar_blink_r.set_value(float(vrm.get("blink_r", 0.0)), active_color="#f59e0b", alert_threshold=0.35)
        self.bar_blink_both.set_value(float(vrm.get("blink", 0.0)), active_color="#ef4444", alert_threshold=0.15)

        # Smile lift (deadzone is 0.015)
        smile_lift = float(telemetry.get("smile_lift", 0.0))
        self.bar_smile_lift.set_value(smile_lift, active_color="#ef4444" if smile_lift > 0.015 else "#10b981", alert_threshold=0.015)

        self.bar_joy.set_value(float(vrm.get("joy", 0.0)), active_color="#ef4444", alert_threshold=0.05)
        self.bar_a.set_value(float(vrm.get("a", 0.0)), active_color="#3b82f6", alert_threshold=0.08)
        self.bar_i.set_value(float(vrm.get("i", 0.0)), active_color="#3b82f6")
        self.bar_u.set_value(float(vrm.get("u", 0.0)), active_color="#3b82f6")
        self.bar_e.set_value(float(vrm.get("e", 0.0)), active_color="#3b82f6")
        self.bar_o.set_value(float(vrm.get("o", 0.0)), active_color="#3b82f6")
        brow_ratio = float(telemetry.get("brow_ratio", 0.0))
        brow_delta = float(telemetry.get("brow_delta", 0.0))
        self.bar_brow_ratio.set_value(brow_ratio, active_color="#f97316" if brow_delta > 0.02 else "#10b981", alert_threshold=0.55)

        self.bar_surprised.set_value(float(vrm.get("surprised", 0.0)), active_color="#f97316", alert_threshold=0.05)
        self.bar_angry.set_value(float(vrm.get("angry", 0.0)), active_color="#dc2626", alert_threshold=0.05)
        self.bar_sorrow.set_value(float(vrm.get("sorrow", 0.0)), active_color="#6366f1")

    def keyPressEvent(self, event):
        """Handle hotkeys: [C] recalibrate neutral baseline, [D] toggle hide, [Esc] close."""
        if event.key() == Qt.Key.Key_C:
            if hasattr(self, "on_recalibrate") and callable(self.on_recalibrate):
                self.on_recalibrate()
            event.accept()
        elif event.key() == Qt.Key.Key_D:
            self.hide()
            event.accept()
        elif event.key() in (Qt.Key.Key_Escape, Qt.Key.Key_Q):
            self.hide()
            event.accept()
        else:
            super().keyPressEvent(event)

    def closeEvent(self, event):
        """Hide window instead of destroying on user close."""
        self.hide()
        event.ignore()
