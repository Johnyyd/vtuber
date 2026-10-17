import os
import sys
import unittest
import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_manager import DEFAULT_CONFIG, load_config, save_config
from main import parse_args


class MockFaceForDebug:
    def __init__(self):
        self.conf = 0.95
        self.eye_blink = [0.9, 0.9]
        self.euler = np.array([5.0, -10.0, 2.0], dtype=np.float32)
        self.current_features = {
            "eye_l": 0.0,
            "eye_r": 0.0,
            "mouth_open": 0.50,
            "mouth_wide": 0.20,
            "mouth_corner_updown_l": 0.5,
            "mouth_corner_updown_r": 0.5,
            "eyebrow_updown_l": 0.3,
            "eyebrow_updown_r": 0.3,
            "eyebrow_steepness_l": 0.1,
            "eyebrow_steepness_r": 0.1,
        }
        # 68 landmarks
        self.lms = np.zeros((68, 3), dtype=np.float32)
        for i in range(68):
            self.lms[i] = [100.0 + (i % 10) * 10, 150.0 + (i // 10) * 15, 1.0]


class TestDebugMode(unittest.TestCase):
    def test_config_debug_mode_default_and_persistence(self):
        self.assertIn("debug_mode", DEFAULT_CONFIG)
        self.assertFalse(DEFAULT_CONFIG["debug_mode"])

        import tempfile
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg_file = os.path.join(tmpdir, "config.txt")
            cfg = load_config(cfg_file)
            self.assertIn("debug_mode", cfg)
            self.assertFalse(cfg["debug_mode"])

            cfg["debug_mode"] = True
            save_config(cfg, cfg_file)
            reloaded = load_config(cfg_file)
            self.assertTrue(reloaded["debug_mode"])

    def test_main_cli_debug_flag(self):
        args = parse_args(["--debug"])
        self.assertTrue(args.debug)

        args_default = parse_args([])
        self.assertFalse(args_default.debug)

    def test_draw_openseeface_debug(self):
        from openseeface_tracker import draw_openseeface_debug

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        face = MockFaceForDebug()
        vrm_shapes = {"joy": 0.0, "a": 0.5, "blink": 0.0}
        telemetry = {
            "backend": "OpenSeeFace ONNX",
            "smile_lift": 0.005,
            "rotation": {"pitch": 5.0, "yaw": -10.0, "roll": 2.0},
            "gaze": {"x": 0.0, "y": 0.0},
        }

        debug_img = draw_openseeface_debug(frame, face, vrm_shapes, telemetry)
        self.assertIsInstance(debug_img, np.ndarray)
        self.assertEqual(debug_img.shape, (480, 640, 3))
        # Ensure original frame was not modified in-place
        self.assertTrue(np.array_equal(frame, frame_copy))
        # Ensure some visual marks/lines were drawn
        self.assertGreater(np.count_nonzero(debug_img), 0)

    def test_debug_window_instantiation_and_update(self):
        from PyQt6.QtWidgets import QApplication
        app = QApplication.instance()
        if app is None:
            app = QApplication(sys.argv)

        from debug_window import TrackingDebugWindow

        win = TrackingDebugWindow()
        self.assertIsNotNone(win)
        self.assertEqual(win.windowTitle(), "VTuber Tracking Debug Mode")

        test_frame = np.zeros((360, 480, 3), dtype=np.uint8)
        telemetry = {
            "backend": "OpenSeeFace ONNX",
            "fps": 30.0,
            "rotation": {"pitch": 1.2, "yaw": -3.4, "roll": 0.5},
            "gaze": {"x": 0.1, "y": -0.2},
            "smile_lift": 0.008,
            "brow_ratio": 0.48,
            "brow_baseline": 0.45,
            "brow_delta": 0.03,
            "vrm_shapes": {
                "joy": 0.0,
                "a": 0.45,
                "i": 0.0,
                "u": 0.0,
                "e": 0.0,
                "o": 0.1,
                "blink_l": 0.0,
                "blink_r": 0.0,
                "surprised": 0.0,
                "sorrow": 0.0,
            },
        }
        # Update view should execute cleanly without error
        win.update_debug_view(test_frame, telemetry)
        win.close()

    def test_draw_mediapipe_debug(self):
        from landmark_mapping import draw_mediapipe_debug

        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        frame_copy = frame.copy()
        landmarks = {
            "lm1": (0.5, 0.5, 0.0),
            "lm13": (0.5, 0.55, 0.0),
            "lm14": (0.5, 0.58, 0.0),
            "lm468": (0.4, 0.35, 0.0),
            "lm473": (0.6, 0.35, 0.0),
        }
        vrm_shapes = {"joy": 0.0, "a": 0.2, "blink_l": 0.0, "blink_r": 0.0}
        telemetry = {
            "backend": "Google MediaPipe",
            "fps": 30.0,
            "rotation": {"pitch": 0.0, "yaw": 0.0, "roll": 0.0},
            "gaze": {"x": 0.0, "y": 0.0},
        }

        debug_img = draw_mediapipe_debug(frame, landmarks, vrm_shapes, telemetry)
        self.assertIsInstance(debug_img, np.ndarray)
        self.assertEqual(debug_img.shape, (480, 640, 3))
        self.assertTrue(np.array_equal(frame, frame_copy))
        self.assertGreater(np.count_nonzero(debug_img), 0)


if __name__ == "__main__":
    unittest.main()
