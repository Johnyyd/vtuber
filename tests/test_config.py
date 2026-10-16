import os
import tempfile
import unittest
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
_SRC_DIR = os.path.join(_REPO_ROOT, "src")
for p in (_REPO_ROOT, _SRC_DIR):
    if p not in sys.path:
        sys.path.insert(0, p)

from config_manager import DEFAULT_CONFIG, load_config, save_config
from main import parse_args


class TestConfigManager(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.config_path = os.path.join(self.temp_dir.name, "test_config.txt")

    def tearDown(self):
        self.temp_dir.cleanup()

    def test_default_config_generated_if_missing(self):
        self.assertFalse(os.path.exists(self.config_path))
        cfg = load_config(self.config_path)
        self.assertTrue(os.path.exists(self.config_path))
        self.assertEqual(cfg["camera_id"], DEFAULT_CONFIG["camera_id"])
        self.assertEqual(cfg["smile_deadzone"], DEFAULT_CONFIG["smile_deadzone"])
        self.assertEqual(cfg["u_max_clamp"], DEFAULT_CONFIG["u_max_clamp"])
        self.assertEqual(cfg["o_max_clamp"], DEFAULT_CONFIG["o_max_clamp"])
        self.assertEqual(cfg["body_sway_gain"], DEFAULT_CONFIG["body_sway_gain"])
        self.assertEqual(cfg["idle_breath_intensity"], DEFAULT_CONFIG["idle_breath_intensity"])
        self.assertEqual(cfg["hair_gravity"], DEFAULT_CONFIG["hair_gravity"])
        self.assertEqual(cfg["hair_stiffness"], DEFAULT_CONFIG["hair_stiffness"])
        self.assertEqual(cfg["hair_drag"], DEFAULT_CONFIG["hair_drag"])

    def test_save_and_reload_modified_config(self):
        custom_cfg = dict(DEFAULT_CONFIG)
        custom_cfg["camera_id"] = 2
        custom_cfg["smile_deadzone"] = 0.12
        custom_cfg["smile_gain"] = 3.5
        custom_cfg["u_max_clamp"] = 0.55
        custom_cfg["o_max_clamp"] = 0.50
        custom_cfg["body_sway_gain"] = 1.50
        custom_cfg["idle_breath_intensity"] = 0.80
        custom_cfg["hair_gravity"] = 2.20
        custom_cfg["hair_stiffness"] = 0.30
        custom_cfg["hair_drag"] = 0.65
        custom_cfg["show_launcher"] = False

        success = save_config(custom_cfg, self.config_path)
        self.assertTrue(success)

        loaded = load_config(self.config_path)
        self.assertEqual(loaded["camera_id"], 2)
        self.assertAlmostEqual(loaded["smile_deadzone"], 0.12, places=2)
        self.assertAlmostEqual(loaded["smile_gain"], 3.5, places=2)
        self.assertAlmostEqual(loaded["u_max_clamp"], 0.55, places=2)
        self.assertAlmostEqual(loaded["o_max_clamp"], 0.50, places=2)
        self.assertAlmostEqual(loaded["body_sway_gain"], 1.50, places=2)
        self.assertAlmostEqual(loaded["idle_breath_intensity"], 0.80, places=2)
        self.assertAlmostEqual(loaded["hair_gravity"], 2.20, places=2)
        self.assertAlmostEqual(loaded["hair_stiffness"], 0.30, places=2)
        self.assertAlmostEqual(loaded["hair_drag"], 0.65, places=2)
        self.assertFalse(loaded["show_launcher"])

    def test_corrupted_config_falls_back_to_defaults(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("[GENERAL]\ncamera_id = not_a_number\n")

        loaded = load_config(self.config_path)
        self.assertEqual(loaded["camera_id"], DEFAULT_CONFIG["camera_id"])
        self.assertEqual(loaded["smile_deadzone"], DEFAULT_CONFIG["smile_deadzone"])

    def test_tracker_backend_config(self):
        from config_manager import DEFAULT_CONFIG, load_config, validate_config
        self.assertIn("tracker_backend", DEFAULT_CONFIG)
        self.assertEqual(DEFAULT_CONFIG["tracker_backend"], "openseeface")

        # Validation succeeds for valid backends
        valid_cfg = validate_config({"tracker_backend": "openseeface"})
        self.assertEqual(valid_cfg["tracker_backend"], "openseeface")

        # Validation falls back for invalid backend
        invalid_cfg = validate_config({"tracker_backend": "unknown_ai"})
        self.assertEqual(invalid_cfg["tracker_backend"], "openseeface")



class TestMainArgParsing(unittest.TestCase):
    def test_main_cli_arguments(self):
        args = parse_args(["--camera-id", "3", "--width", "1280", "--height", "720", "--no-launcher"])
        self.assertEqual(args.camera_id, 3)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertTrue(args.no_launcher)
        self.assertFalse(args.show_launcher)


class TestConfigGUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PyQt6.QtWidgets import QApplication
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication(sys.argv)

    def test_detect_available_cameras(self):
        from config_gui import detect_available_cameras
        cams = detect_available_cameras()
        self.assertIsInstance(cams, list)
        self.assertGreater(len(cams), 0)
        for idx, label in cams:
            self.assertIsInstance(idx, int)
            self.assertIsInstance(label, str)

    def test_config_launcher_dialog_initial_config(self):
        from config_gui import ConfigLauncherDialog
        with tempfile.TemporaryDirectory() as temp_dir:
            cfg_path = os.path.join(temp_dir, "cfg.txt")
            initial = {"camera_id": 1, "window_width": 1280, "window_height": 720}
            dialog = ConfigLauncherDialog(config_path=cfg_path, initial_config=initial)
            self.assertEqual(dialog.spin_cam_id.value(), 1)
            self.assertEqual(dialog.spin_width.value(), 1280)
            self.assertEqual(dialog.spin_height.value(), 720)
            configured = dialog._collect_values_from_ui()
            self.assertEqual(configured["camera_id"], 1)
            self.assertEqual(configured["window_width"], 1280)
            self.assertEqual(configured["window_height"], 720)


if __name__ == "__main__":
    unittest.main()
