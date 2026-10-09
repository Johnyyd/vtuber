"""
PyQt6 Configuration Launcher Dialog: Allows users to configure tracking & display parameters
before launching the main 3D VTuber application.
"""

import sys
from PyQt6.QtWidgets import (
    QDialog,
    QVBoxLayout,
    QHBoxLayout,
    QTabWidget,
    QWidget,
    QLabel,
    QSpinBox,
    QDoubleSpinBox,
    QCheckBox,
    QComboBox,
    QPushButton,
    QGroupBox,
    QGridLayout,
    QFormLayout,
    QFrame,
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont, QIcon

from config_manager import DEFAULT_CONFIG, load_config, save_config


class ConfigLauncherDialog(QDialog):
    """Modern dark-themed parameter customization dialog."""

    def __init__(self, parent=None, config_path=None):
        super().__init__(parent)
        self.config_path = config_path
        self.current_config = load_config(self.config_path) if self.config_path else load_config()

        self.setWindowTitle("VTuber 3D Avatar - Cấu hình nhận diện & Khởi chạy")
        self.setMinimumSize(560, 620)
        self.resize(580, 650)
        self.setStyleSheet(self._get_stylesheet())

        self._init_ui()
        self._load_values_to_ui(self.current_config)

    def _get_stylesheet(self) -> str:
        return """
        QDialog {
            background-color: #18181f;
            color: #f3f4f6;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        }
        QTabWidget::pane {
            border: 1px solid #374151;
            background-color: #1f2937;
            border-radius: 8px;
            top: -1px;
        }
        QTabBar::tab {
            background-color: #111827;
            color: #9ca3af;
            padding: 10px 18px;
            margin-right: 4px;
            border-top-left-radius: 8px;
            border-top-right-radius: 8px;
            font-weight: bold;
            font-size: 13px;
        }
        QTabBar::tab:selected {
            background-color: #1f2937;
            color: #a78bfa;
            border-bottom: 2px solid #8b5cf6;
        }
        QTabBar::tab:hover:!selected {
            background-color: #1f2430;
            color: #e5e7eb;
        }
        QGroupBox {
            font-weight: bold;
            font-size: 13px;
            color: #c4b5fd;
            border: 1px solid #374151;
            border-radius: 8px;
            margin-top: 14px;
            padding-top: 18px;
            background-color: #192231;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 8px;
            background-color: #18181f;
            border-radius: 4px;
        }
        QLabel {
            color: #d1d5db;
            font-size: 12px;
        }
        QSpinBox, QDoubleSpinBox, QComboBox {
            background-color: #111827;
            border: 1px solid #4b5563;
            border-radius: 6px;
            color: #f9fafb;
            padding: 5px 10px;
            min-height: 24px;
            font-size: 13px;
            font-weight: 500;
        }
        QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {
            border: 1px solid #8b5cf6;
            background-color: #161f30;
        }
        QComboBox::drop-down {
            border: none;
            width: 20px;
        }
        QCheckBox {
            color: #e5e7eb;
            font-size: 13px;
            spacing: 8px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 4px;
            border: 1px solid #6b7280;
            background-color: #111827;
        }
        QCheckBox::indicator:checked {
            background-color: #8b5cf6;
            border: 1px solid #8b5cf6;
        }
        QPushButton {
            border-radius: 8px;
            font-size: 13px;
            font-weight: bold;
            padding: 9px 20px;
            min-height: 20px;
        }
        QPushButton#btn-start {
            background-color: #7c3aed;
            color: white;
            border: 1px solid #8b5cf6;
        }
        QPushButton#btn-start:hover {
            background-color: #6d28d9;
        }
        QPushButton#btn-default {
            background-color: #374151;
            color: #e5e7eb;
            border: 1px solid #4b5563;
        }
        QPushButton#btn-default:hover {
            background-color: #4b5563;
        }
        QPushButton#btn-cancel {
            background-color: #1f2937;
            color: #9ca3af;
            border: 1px solid #374151;
        }
        QPushButton#btn-cancel:hover {
            background-color: #374151;
            color: #ef4444;
        }
        """

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        title_label = QLabel("Tùy chỉnh thông số nhận diện VTuber")
        title_label.setStyleSheet("font-size: 18px; font-weight: bold; color: #f9fafb;")
        sub_label = QLabel("Cấu hình thiết bị và độ nhạy tracking khuôn mặt trước khi khởi động avatar.")
        sub_label.setStyleSheet("font-size: 12px; color: #9ca3af; margin-bottom: 4px;")
        main_layout.addWidget(title_label)
        main_layout.addWidget(sub_label)

        # Tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # Build individual tabs
        self.tab_general = self._create_tab_general()
        self.tab_expressions = self._create_tab_expressions()
        self.tab_camera = self._create_tab_camera()

        self.tabs.addTab(self.tab_general, "Thiết bị & Cửa sổ")
        self.tabs.addTab(self.tab_expressions, "Độ nhạy biểu cảm")
        self.tabs.addTab(self.tab_camera, "Camera & Lọc rung")

        # Bottom Buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(12)

        self.btn_reset = QPushButton("Khôi phục mặc định")
        self.btn_reset.setObjectName("btn-default")
        self.btn_reset.clicked.connect(self._reset_to_defaults)
        btn_layout.addWidget(self.btn_reset)

        btn_layout.addStretch()

        self.btn_cancel = QPushButton("Thoát")
        self.btn_cancel.setObjectName("btn-cancel")
        self.btn_cancel.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_cancel)

        self.btn_start = QPushButton("Lưu & Khởi chạy VTuber")
        self.btn_start.setObjectName("btn-start")
        self.btn_start.clicked.connect(self._on_save_and_launch)
        btn_layout.addWidget(self.btn_start)

        main_layout.addLayout(btn_layout)

    def _create_tab_general(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Camera settings
        group_cam = QGroupBox("Cấu hình Webcam")
        form_cam = QFormLayout(group_cam)
        form_cam.setSpacing(12)

        self.spin_cam_id = QSpinBox()
        self.spin_cam_id.setRange(0, 9)
        form_cam.addRow("Chỉ số Camera (0 = Webcam chính):", self.spin_cam_id)
        layout.addWidget(group_cam)

        # Window settings
        group_win = QGroupBox("Kích thước cửa sổ hiển thị")
        form_win = QFormLayout(group_win)
        form_win.setSpacing(12)

        self.combo_presets = QComboBox()
        self.combo_presets.addItems([
            "1024x768 (Mặc định)",
            "1280x720 (HD 720p)",
            "1920x1080 (FHD 1080p)",
            "800x600 (Gọn nhẹ)",
            "Tùy chỉnh...",
        ])
        self.combo_presets.currentIndexChanged.connect(self._on_preset_changed)
        form_win.addRow("Mẫu kích thước sẵn:", self.combo_presets)

        dim_layout = QHBoxLayout()
        self.spin_width = QSpinBox()
        self.spin_width.setRange(300, 3840)
        self.spin_width.setSingleStep(50)
        self.spin_height = QSpinBox()
        self.spin_height.setRange(200, 2160)
        self.spin_height.setSingleStep(50)
        dim_layout.addWidget(self.spin_width)
        dim_layout.addWidget(QLabel("x"))
        dim_layout.addWidget(self.spin_height)
        form_win.addRow("Chiều rộng x Chiều cao (px):", dim_layout)

        layout.addWidget(group_win)

        # Launcher startup toggle
        group_start = QGroupBox("Khởi chạy ứng dụng")
        v_start = QVBoxLayout(group_start)
        self.chk_show_launcher = QCheckBox("Luôn hiển thị bảng cài đặt này mỗi khi mở ứng dụng")
        v_start.addWidget(self.chk_show_launcher)
        layout.addWidget(group_start)

        layout.addStretch()
        return widget

    def _create_tab_expressions(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Joy (Smile)
        group_joy = QGroupBox("Nụ cười (Joy)")
        form_joy = QFormLayout(group_joy)
        form_joy.setSpacing(10)

        self.spin_smile_dz = QDoubleSpinBox()
        self.spin_smile_dz.setRange(0.01, 0.25)
        self.spin_smile_dz.setSingleStep(0.01)
        self.spin_smile_dz.setDecimals(2)
        form_joy.addRow("Điểm chết cười (Deadzone, nhỏ = nhạy hơn):", self.spin_smile_dz)

        self.spin_smile_gain = QDoubleSpinBox()
        self.spin_smile_gain.setRange(1.0, 5.0)
        self.spin_smile_gain.setSingleStep(0.1)
        self.spin_smile_gain.setDecimals(1)
        form_joy.addRow("Hệ số khuếch đại nụ cười (Gain):", self.spin_smile_gain)
        layout.addWidget(group_joy)

        # Speech (Mouth & U)
        group_mouth = QGroupBox("Khẩu hình miệng (Nói chuyện & Chu môi U)")
        form_mouth = QFormLayout(group_mouth)
        form_mouth.setSpacing(10)

        self.spin_mouth_dz = QDoubleSpinBox()
        self.spin_mouth_dz.setRange(0.02, 0.15)
        self.spin_mouth_dz.setSingleStep(0.01)
        self.spin_mouth_dz.setDecimals(2)
        form_mouth.addRow("Điểm chết mở miệng (Mouth Deadzone):", self.spin_mouth_dz)

        self.spin_mouth_gain = QDoubleSpinBox()
        self.spin_mouth_gain.setRange(1.0, 3.0)
        self.spin_mouth_gain.setSingleStep(0.1)
        self.spin_mouth_gain.setDecimals(1)
        form_mouth.addRow("Độ nhạy mở miệng nói (Mouth Gain):", self.spin_mouth_gain)

        self.spin_u_max = QDoubleSpinBox()
        self.spin_u_max.setRange(0.40, 0.75)
        self.spin_u_max.setSingleStep(0.02)
        self.spin_u_max.setDecimals(2)
        form_mouth.addRow("Giới hạn khẩu hình U (tránh chập mép):", self.spin_u_max)
        layout.addWidget(group_mouth)

        # Brows & Emotion
        group_emotion = QGroupBox("Chân mày & Cảm xúc khác")
        form_emotion = QFormLayout(group_emotion)
        form_emotion.setSpacing(10)

        self.spin_brow_dz = QDoubleSpinBox()
        self.spin_brow_dz.setRange(0.02, 0.20)
        self.spin_brow_dz.setSingleStep(0.01)
        self.spin_brow_dz.setDecimals(2)
        form_emotion.addRow("Điểm chết nhướng mày (Surprised):", self.spin_brow_dz)

        self.spin_brow_gain = QDoubleSpinBox()
        self.spin_brow_gain.setRange(1.0, 4.0)
        self.spin_brow_gain.setSingleStep(0.1)
        self.spin_brow_gain.setDecimals(1)
        form_emotion.addRow("Độ nhạy nhướng mày (Surprised Gain):", self.spin_brow_gain)

        self.spin_frown_dz = QDoubleSpinBox()
        self.spin_frown_dz.setRange(0.02, 0.20)
        self.spin_frown_dz.setSingleStep(0.01)
        self.spin_frown_dz.setDecimals(2)
        form_emotion.addRow("Điểm chết bĩu môi buồn (Sorrow):", self.spin_frown_dz)
        layout.addWidget(group_emotion)

        layout.addStretch()
        return widget

    def _create_tab_camera(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(14)

        # Eye Blink
        group_blink = QGroupBox("Chớp mắt & Nháy mắt")
        form_blink = QFormLayout(group_blink)
        form_blink.setSpacing(10)

        self.spin_blink_dz = QDoubleSpinBox()
        self.spin_blink_dz.setRange(0.05, 0.20)
        self.spin_blink_dz.setSingleStep(0.01)
        self.spin_blink_dz.setDecimals(2)
        form_blink.addRow("Điểm bắt đầu nhắm (Blink Deadzone):", self.spin_blink_dz)

        self.spin_blink_snap = QDoubleSpinBox()
        self.spin_blink_snap.setRange(0.15, 0.40)
        self.spin_blink_snap.setSingleStep(0.01)
        self.spin_blink_snap.setDecimals(2)
        form_blink.addRow("Điểm nhắm hoàn toàn (Snap Threshold):", self.spin_blink_snap)
        layout.addWidget(group_blink)

        # Camera Tilt
        group_tilt = QGroupBox("Góc đặt Camera")
        form_tilt = QFormLayout(group_tilt)
        form_tilt.setSpacing(10)

        self.spin_pitch_offset = QDoubleSpinBox()
        self.spin_pitch_offset.setRange(-30.0, 40.0)
        self.spin_pitch_offset.setSingleStep(1.0)
        self.spin_pitch_offset.setDecimals(1)
        form_tilt.addRow("Bù trừ góc nghiêng camera (độ):", self.spin_pitch_offset)
        layout.addWidget(group_tilt)

        # OneEuroFilter
        group_filter = QGroupBox("Bộ lọc chống rung chuyển động (OneEuroFilter)")
        form_filter = QFormLayout(group_filter)
        form_filter.setSpacing(10)

        self.spin_filter_cutoff = QDoubleSpinBox()
        self.spin_filter_cutoff.setRange(0.10, 3.00)
        self.spin_filter_cutoff.setSingleStep(0.05)
        self.spin_filter_cutoff.setDecimals(2)
        form_filter.addRow("Độ êm lúc đứng yên (Min Cutoff):", self.spin_filter_cutoff)

        self.spin_filter_beta = QDoubleSpinBox()
        self.spin_filter_beta.setRange(0.001, 0.100)
        self.spin_filter_beta.setSingleStep(0.005)
        self.spin_filter_beta.setDecimals(4)
        form_filter.addRow("Độ nhạy khi quay đầu nhanh (Beta):", self.spin_filter_beta)
        layout.addWidget(group_filter)

        layout.addStretch()
        return widget

    def _on_preset_changed(self, index: int):
        presets = {
            0: (1024, 768),
            1: (1280, 720),
            2: (1920, 1080),
            3: (800, 600),
        }
        if index in presets:
            w, h = presets[index]
            self.spin_width.setValue(w)
            self.spin_height.setValue(h)

    def _load_values_to_ui(self, cfg: dict):
        self.spin_cam_id.setValue(int(cfg.get("camera_id", 0)))
        w = int(cfg.get("window_width", 1024))
        h = int(cfg.get("window_height", 768))
        self.spin_width.setValue(w)
        self.spin_height.setValue(h)

        if (w, h) == (1024, 768): self.combo_presets.setCurrentIndex(0)
        elif (w, h) == (1280, 720): self.combo_presets.setCurrentIndex(1)
        elif (w, h) == (1920, 1080): self.combo_presets.setCurrentIndex(2)
        elif (w, h) == (800, 600): self.combo_presets.setCurrentIndex(3)
        else: self.combo_presets.setCurrentIndex(4)

        self.chk_show_launcher.setChecked(bool(cfg.get("show_launcher", True)))

        self.spin_smile_dz.setValue(float(cfg.get("smile_deadzone", 0.10)))
        self.spin_smile_gain.setValue(float(cfg.get("smile_gain", 2.8)))
        self.spin_mouth_dz.setValue(float(cfg.get("mouth_open_deadzone", 0.09)))
        self.spin_mouth_gain.setValue(float(cfg.get("mouth_open_gain", 1.6)))
        self.spin_u_max.setValue(float(cfg.get("u_max_clamp", 0.58)))

        self.spin_brow_dz.setValue(float(cfg.get("brow_raise_deadzone", 0.10)))
        self.spin_brow_gain.setValue(float(cfg.get("brow_raise_gain", 2.2)))
        self.spin_frown_dz.setValue(float(cfg.get("frown_deadzone", 0.10)))

        self.spin_blink_dz.setValue(float(cfg.get("blink_deadzone", 0.13)))
        self.spin_blink_snap.setValue(float(cfg.get("blink_snap_thresh", 0.25)))

        self.spin_pitch_offset.setValue(float(cfg.get("pitch_offset_deg", 18.0)))
        self.spin_filter_cutoff.setValue(float(cfg.get("filter_min_cutoff", 0.8)))
        self.spin_filter_beta.setValue(float(cfg.get("filter_beta", 0.02)))

    def _collect_values_from_ui(self) -> dict:
        return {
            "camera_id": self.spin_cam_id.value(),
            "window_width": self.spin_width.value(),
            "window_height": self.spin_height.value(),
            "show_launcher": self.chk_show_launcher.isChecked(),

            "smile_deadzone": self.spin_smile_dz.value(),
            "smile_gain": self.spin_smile_gain.value(),
            "mouth_open_deadzone": self.spin_mouth_dz.value(),
            "mouth_open_gain": self.spin_mouth_gain.value(),
            "u_max_clamp": self.spin_u_max.value(),
            "brow_raise_deadzone": self.spin_brow_dz.value(),
            "brow_raise_gain": self.spin_brow_gain.value(),
            "frown_deadzone": self.spin_frown_dz.value(),
            "frown_gain": self.current_config.get("frown_gain", 2.0),

            "blink_deadzone": self.spin_blink_dz.value(),
            "blink_snap_thresh": self.spin_blink_snap.value(),

            "pitch_offset_deg": self.spin_pitch_offset.value(),
            "filter_min_cutoff": self.spin_filter_cutoff.value(),
            "filter_beta": self.spin_filter_beta.value(),
        }

    def _reset_to_defaults(self):
        self._load_values_to_ui(DEFAULT_CONFIG)

    def _on_save_and_launch(self):
        self.current_config = self._collect_values_from_ui()
        if self.config_path:
            save_config(self.current_config, self.config_path)
        else:
            save_config(self.current_config)
        self.accept()

    def get_configured_settings(self) -> dict:
        return dict(self.current_config)
