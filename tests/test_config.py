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

    def test_save_and_reload_modified_config(self):
        custom_cfg = dict(DEFAULT_CONFIG)
        custom_cfg["camera_id"] = 2
        custom_cfg["smile_deadzone"] = 0.12
        custom_cfg["smile_gain"] = 3.5
        custom_cfg["u_max_clamp"] = 0.55
        custom_cfg["show_launcher"] = False

        success = save_config(custom_cfg, self.config_path)
        self.assertTrue(success)

        loaded = load_config(self.config_path)
        self.assertEqual(loaded["camera_id"], 2)
        self.assertAlmostEqual(loaded["smile_deadzone"], 0.12, places=2)
        self.assertAlmostEqual(loaded["smile_gain"], 3.5, places=2)
        self.assertAlmostEqual(loaded["u_max_clamp"], 0.55, places=2)
        self.assertFalse(loaded["show_launcher"])

    def test_corrupted_config_falls_back_to_defaults(self):
        with open(self.config_path, "w", encoding="utf-8") as f:
            f.write("[GENERAL]\ncamera_id = not_a_number\n")

        loaded = load_config(self.config_path)
        self.assertEqual(loaded["camera_id"], DEFAULT_CONFIG["camera_id"])
        self.assertEqual(loaded["smile_deadzone"], DEFAULT_CONFIG["smile_deadzone"])


class TestMainArgParsing(unittest.TestCase):
    def test_main_cli_arguments(self):
        args = parse_args(["--camera-id", "3", "--width", "1280", "--height", "720", "--no-launcher"])
        self.assertEqual(args.camera_id, 3)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)
        self.assertTrue(args.no_launcher)
        self.assertFalse(args.show_launcher)


if __name__ == "__main__":
    unittest.main()
