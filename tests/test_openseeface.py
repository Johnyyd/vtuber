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

    def test_estimate_depth_matrix_alignment(self):
        from src.openseeface.tracker import FaceInfo
        tracker = OpenSeeFaceTracker(width=640, height=480, max_threads=1)
        face_info = FaceInfo(0, tracker.tracker)
        face_info.lms = np.random.rand(66, 3).astype(np.float32) * 200 + 200
        face_info.eye_state = np.array([
            [1.0, 300.0, 240.0, 1.0],
            [1.0, 340.0, 240.0, 1.0]
        ], dtype=np.float32)

        # 1st frame: rotation & translation are initially None
        success, quat, euler, pnp_error, pts_3d, lms = tracker.tracker.estimate_depth(face_info)
        self.assertTrue(success)
        self.assertEqual(pts_3d.shape, (70, 3))

        # 2nd frame: continuous tracking with prior rotation & translation matrices
        success2, quat2, euler2, pnp_error2, pts_3d2, lms2 = tracker.tracker.estimate_depth(face_info)
        self.assertTrue(success2)
        self.assertEqual(pts_3d2.shape, (70, 3))


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

    def test_openseeface_end_to_end_packet_compatibility(self):
        try:
            from vrm_renderer import format_motion_packet
            from openseeface_tracker import map_openseeface_to_vrm, map_openseeface_to_vrc, extract_head_pose, extract_iris_gaze
        except ImportError:
            from src.vrm_renderer import format_motion_packet
            from src.openseeface_tracker import map_openseeface_to_vrm, map_openseeface_to_vrc, extract_head_pose, extract_iris_gaze

        face = MockFace()
        vrm = map_openseeface_to_vrm(face, config={"pitch_offset_deg": 18.0})
        vrc = map_openseeface_to_vrc(face)
        rotation = extract_head_pose(face, pitch_offset_deg=18.0)
        gaze = extract_iris_gaze(face)
        body = {"sway_gain": 1.0, "breath_gain": 1.0}

        packet = format_motion_packet(vrc=vrc, rotation=rotation, vrm=vrm, gaze=gaze, body=body)

        self.assertIn("vrm", packet)
        self.assertIn("vrc", packet)
        self.assertIn("rotation", packet)
        self.assertIn("gaze", packet)
        self.assertIn("body", packet)
        self.assertEqual(packet["rotation"]["pitch"], rotation["pitch"])
        self.assertEqual(packet["gaze"], gaze)
        self.assertEqual(packet["body"], body)

    def test_openseeface_vowel_i_specificity_and_false_trigger_rejection(self):
        try:
            from openseeface_tracker import map_openseeface_to_vrm
        except ImportError:
            from src.openseeface_tracker import map_openseeface_to_vrm

        # 1. Normal talking A (moderate opening, ordinary mouth width): I must be 0.0
        face_a = MockFace()
        face_a.current_features["mouth_open"] = 0.18
        face_a.current_features["mouth_wide"] = 0.12
        face_a.current_features["mouth_corner_updown_l"] = 0.0
        face_a.current_features["mouth_corner_updown_r"] = 0.0
        vrm_a = map_openseeface_to_vrm(face_a)
        self.assertEqual(vrm_a["i"], 0.0, "Normal speaking width must produce 0.0 'I'")
        self.assertGreater(vrm_a["a"], 0.20, "Speaking A must remain responsive")

        # 2. Closed mouth smile: I must be 0.0, Joy must be active
        face_smile = MockFace()
        face_smile.current_features["mouth_open"] = 0.03
        face_smile.current_features["mouth_wide"] = 0.25
        face_smile.current_features["mouth_corner_updown_l"] = 0.40
        face_smile.current_features["mouth_corner_updown_r"] = 0.40
        vrm_smile = map_openseeface_to_vrm(face_smile)
        self.assertEqual(vrm_smile["i"], 0.0, "Smiling with closed mouth must produce 0.0 'I'")
        self.assertGreater(vrm_smile["joy"], 0.40, "Smiling must trigger Joy")

        # 3. Deliberately pronounced 'I' (small vertical opening, wide lateral stretch)
        face_i = MockFace()
        face_i.current_features["mouth_open"] = 0.10
        face_i.current_features["mouth_wide"] = 0.45
        vrm_i = map_openseeface_to_vrm(face_i)
        self.assertGreater(vrm_i["i"], 0.40, "Pronounced 'I' must trigger responsively")
        self.assertGreater(vrm_i["i"], vrm_i["a"], "When pronouncing 'I', 'I' must dominate over 'A'")


if __name__ == "__main__":
    unittest.main()



