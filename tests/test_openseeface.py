import os
import unittest
import numpy as np
try:
    from openseeface_tracker import ensure_openseeface_models_downloaded, OpenSeeFaceTracker
except ImportError:
    from src.openseeface_tracker import ensure_openseeface_models_downloaded, OpenSeeFaceTracker


class TestOpenSeeFaceTracker(unittest.TestCase):
    def test_model_downloader_creates_files(self):
        model_dir = ensure_openseeface_models_downloaded()
        self.assertTrue(os.path.isdir(model_dir))
        for filename in ["mnv3_detection_opt.onnx", "lm_model3_opt.onnx", "mnv3_gaze32_split_opt.onnx"]:
            path = os.path.join(model_dir, filename)
            self.assertTrue(os.path.exists(path), f"Missing {filename}")
            self.assertGreater(os.path.getsize(path), 100000)

    def test_tracker_initialization_and_empty_frame(self):
        tracker = OpenSeeFaceTracker(width=640, height=480, max_threads=1)
        # Black frame with no face should return None
        black_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        result = tracker.process_frame(black_frame)
        self.assertIsNone(result)


class MockFace:
    def __init__(self):
        self.conf = 0.95
        self.eye_blink = [0.9, 0.9]  # open eyes
        self.euler = np.array([5.0, -10.0, 2.0], dtype=np.float32)
        self.current_features = {
            "eye_l": 0.0,
            "eye_r": 0.0,
            "mouth_open": 0.75,
            "mouth_wide": 0.20,
            "mouth_corner_updown_l": 0.5,
            "mouth_corner_updown_r": 0.5,
            "eyebrow_updown_l": 0.3,
            "eyebrow_updown_r": 0.3,
            "eyebrow_steepness_l": 0.1,
            "eyebrow_steepness_r": 0.1,
        }
        # 68 landmarks with dummy coordinates
        self.lms = np.zeros((68, 3), dtype=np.float32)
        # Eye landmarks for right eye (36-41) and left eye (42-47), pupils (66, 67)
        self.lms[36] = [100, 100, 1.0]  # right outer
        self.lms[39] = [120, 100, 1.0]  # right inner
        self.lms[66] = [110, 100, 1.0]  # right pupil center
        self.lms[42] = [150, 100, 1.0]  # left inner
        self.lms[45] = [170, 100, 1.0]  # left outer
        self.lms[67] = [160, 100, 1.0]  # left pupil center


class TestOpenSeeFaceMapping(unittest.TestCase):
    def test_mapping_vrm_vowels_and_expressions(self):
        try:
            from openseeface_tracker import (
                map_openseeface_to_vrm,
                map_openseeface_to_vrc,
                extract_head_pose,
                extract_iris_gaze,
            )
        except ImportError:
            from src.openseeface_tracker import (
                map_openseeface_to_vrm,
                map_openseeface_to_vrc,
                extract_head_pose,
                extract_iris_gaze,
            )

        face = MockFace()
        vrm = map_openseeface_to_vrm(face)
        vrc = map_openseeface_to_vrc(face)
        pose = extract_head_pose(face, pitch_offset_deg=18.0)
        gaze = extract_iris_gaze(face)

        # Check VRM blendshapes
        self.assertIn("a", vrm)
        self.assertIn("i", vrm)
        self.assertIn("u", vrm)
        self.assertIn("e", vrm)
        self.assertIn("o", vrm)
        self.assertIn("blink", vrm)
        self.assertIn("joy", vrm)
        self.assertGreater(vrm["a"], 0.3)  # mouth is open
        self.assertGreater(vrm["joy"], 0.2)  # smile corners up
        self.assertLess(vrm["blink"], 0.2)  # eyes are open

        # Check VRChat morphs
        self.assertIn("vrc_v_aa", vrc)
        self.assertIn("vrc_blink", vrc)

        # Check pose
        self.assertIsInstance(pose["pitch"], float)
        self.assertIsInstance(pose["yaw"], float)
        self.assertIsInstance(pose["roll"], float)

        # Check gaze
        self.assertTrue(-1.0 <= gaze["x"] <= 1.0)
        self.assertTrue(-1.0 <= gaze["y"] <= 1.0)


if __name__ == "__main__":
    unittest.main()

