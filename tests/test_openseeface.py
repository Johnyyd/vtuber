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
        # Eye landmarks for right eye (36-41), pupil (66) and left eye (42-47), pupil (67)
        self.lms[36] = [100.0, 100.0, 1.0]  # right outer
        self.lms[37] = [94.0, 115.0, 1.0]   # right top lid
        self.lms[38] = [94.0, 115.0, 1.0]
        self.lms[39] = [100.0, 130.0, 1.0]  # right inner
        self.lms[40] = [106.0, 115.0, 1.0]  # right bot lid
        self.lms[41] = [106.0, 115.0, 1.0]
        self.lms[66] = [100.0, 115.0, 1.0]  # right pupil center

        self.lms[42] = [100.0, 160.0, 1.0]  # left inner
        self.lms[43] = [94.0, 175.0, 1.0]   # left top lid
        self.lms[44] = [94.0, 175.0, 1.0]
        self.lms[45] = [100.0, 190.0, 1.0]  # left outer
        self.lms[46] = [106.0, 175.0, 1.0]  # left bot lid
        self.lms[47] = [106.0, 175.0, 1.0]
        self.lms[67] = [100.0, 175.0, 1.0]  # left pupil center


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

        # Check pose (converted to radians and clamped within [-1.0, 1.0])
        self.assertIsInstance(pose["pitch"], float)
        self.assertIsInstance(pose["yaw"], float)
        self.assertIsInstance(pose["roll"], float)
        self.assertTrue(-1.0 <= pose["pitch"] <= 1.0)
        self.assertTrue(-1.0 <= pose["yaw"] <= 1.0)
        self.assertTrue(-1.0 <= pose["roll"] <= 1.0)

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

    def test_openseeface_landmark_geometric_pose(self):
        try:
            from openseeface_tracker import extract_head_pose
        except ImportError:
            from src.openseeface_tracker import extract_head_pose

        class LandmarkFace:
            def __init__(self):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # In OpenSeeFace, column 0 is Y (vertical), column 1 is X (horizontal)
                # Neutral level face at center (Y=300, X=320)
                # Right eye (landmarks 36, 39) at Y=280, X=270
                self.lms[36] = [280.0, 260.0, 1.0]
                self.lms[39] = [280.0, 280.0, 1.0]
                # Left eye (landmarks 42, 45) at Y=280, X=360
                self.lms[42] = [280.0, 350.0, 1.0]
                self.lms[45] = [280.0, 370.0, 1.0]
                # Nose bridge (27) at Y=270, X=315
                self.lms[27] = [270.0, 315.0, 1.0]
                # Nose tip (30) at Y=310, X=315
                self.lms[30] = [310.0, 315.0, 1.0]
                # Chin (8) at Y=366, X=315 (top=40, bottom=56 -> ratio=1.40)
                self.lms[8] = [366.0, 315.0, 1.0]

        face = LandmarkFace()
        pose = extract_head_pose(face, pitch_offset_deg=18.0)
        self.assertAlmostEqual(pose["roll"], 0.0, places=2)
        self.assertAlmostEqual(pose["yaw"], 0.0, places=2)
        self.assertAlmostEqual(pose["pitch"], 0.0, places=2)

        # Looking UP: chin tilts towards nose tip, bottom distance shortens -> pitch > 0
        face.lms[8] = [340.0, 315.0, 1.0]
        pose_up = extract_head_pose(face, pitch_offset_deg=18.0)
        self.assertGreater(pose_up["pitch"], 0.20, "Looking up must produce positive pitch")

        # Looking DOWN: chin tilts away from nose tip, bottom distance expands -> pitch < 0
        face.lms[8] = [390.0, 315.0, 1.0]
        pose_down = extract_head_pose(face, pitch_offset_deg=18.0)
        self.assertLess(pose_down["pitch"], -0.20, "Looking down must produce negative pitch")

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

    def test_openseeface_iris_gaze_directions_and_blink_stability(self):
        try:
            from openseeface_tracker import extract_iris_gaze
        except ImportError:
            from src.openseeface_tracker import extract_iris_gaze

        face = MockFace()
        # 1. Neutral centered gaze -> (0.0, 0.0)
        gaze_center = extract_iris_gaze(face)
        self.assertAlmostEqual(gaze_center["x"], 0.0, places=2)
        self.assertAlmostEqual(gaze_center["y"], 0.0, places=2)

        # 2. Looking Left: pupils shift to user's left (+X in image coordinates)
        face.lms[66, 1] = 120.0  # +5px towards left
        face.lms[67, 1] = 180.0
        gaze_left = extract_iris_gaze(face)
        self.assertGreater(gaze_left["x"], 0.40, "Looking left must produce positive gaze X")

        # 3. Looking Right: pupils shift to user's right (-X in image coordinates)
        face.lms[66, 1] = 110.0  # -5px towards right
        face.lms[67, 1] = 170.0
        gaze_right = extract_iris_gaze(face)
        self.assertLess(gaze_right["x"], -0.40, "Looking right must produce negative gaze X")

        # Reset X
        face.lms[66, 1] = 115.0
        face.lms[67, 1] = 175.0

        # 4. Looking Up: pupils shift upwards (-Y in image coordinates)
        face.lms[66, 0] = 97.0   # -3px upwards
        face.lms[67, 0] = 97.0
        gaze_up = extract_iris_gaze(face)
        self.assertLess(gaze_up["y"], -0.40, "Looking up must produce negative gaze Y")

        # 5. Looking Down: pupils shift downwards (+Y in image coordinates)
        face.lms[66, 0] = 103.0  # +3px downwards
        face.lms[67, 0] = 103.0
        gaze_down = extract_iris_gaze(face)
        self.assertGreater(gaze_down["y"], 0.40, "Looking down must produce positive gaze Y")

        # 6. Eye Blink: gaze must be neutral (0.0, 0.0) during blink to prevent eye twitch
        face.eye_blink = [0.1, 0.1]  # eyes closed
        gaze_blink = extract_iris_gaze(face)
        self.assertEqual(gaze_blink["x"], 0.0)
        self.assertEqual(gaze_blink["y"], 0.0)

    def test_openseeface_feature_update_state_independence(self):
        try:
            from openseeface.tracker import Feature
        except ImportError:
            from src.openseeface.tracker import Feature

        feat = Feature()
        # 5 resting frames
        for _ in range(5):
            feat.update(0.20)
        # Mouth opens wide to 0.45 without prior dipping below median
        out_open = feat.update(0.45)
        self.assertGreater(out_open, 0.50, "Feature max must initialize even when min has not been triggered")
        self.assertIsNotNone(feat.max, "Feature max bound must be recorded")

        # Mouth returns to resting 0.20
        out_rest = feat.update(0.20)
        self.assertLess(out_rest, 0.20, "Feature must decay back down when returning to rest")

    def test_openseeface_neutral_face_does_not_trigger_joy(self):
        try:
            from openseeface_tracker import map_openseeface_to_vrm
        except ImportError:
            from src.openseeface_tracker import map_openseeface_to_vrm

        # Real tracking face with neutral mouth (corners level with/slightly below center)
        class RealisticFace:
            def __init__(self):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # Nose bridge at Y=270, tip at Y=310 (nose_h = 40)
                self.lms[27] = [270.0, 315.0, 1.0]
                self.lms[30] = [310.0, 315.0, 1.0]
                # Mouth center at Y=330 (outer upper 51: Y=326, outer lower 57: Y=334)
                self.lms[51] = [326.0, 315.0, 1.0]
                self.lms[57] = [334.0, 315.0, 1.0]
                # Inner lips touching
                self.lms[59] = [329.0, 310.0, 1.0]
                self.lms[60] = [329.0, 315.0, 1.0]
                self.lms[61] = [329.0, 320.0, 1.0]
                self.lms[63] = [331.0, 320.0, 1.0]
                self.lms[64] = [331.0, 315.0, 1.0]
                self.lms[65] = [331.0, 310.0, 1.0]
                # Neutral corners at Y=332 (2px below center -> smile_lift = -0.05)
                self.lms[48] = [332.0, 290.0, 1.0]
                self.lms[54] = [332.0, 340.0, 1.0]
                self.eye_blink = [1.0, 1.0]
                # Feature extractor had small noise
                self.current_features = {
                    "mouth_open": 0.0,
                    "mouth_wide": 0.0,
                    "mouth_corner_updown_l": 0.15,
                    "mouth_corner_updown_r": 0.15,
                }

        face = RealisticFace()
        vrm = map_openseeface_to_vrm(face)
        self.assertEqual(vrm["joy"], 0.0, "Neutral face with level mouth corners must have Joy = 0.0")

        # Now simulate actual smile: corners pulled UP to Y=322 (8px above center -> smile_lift = +0.20)
        face.lms[48, 0] = 322.0
        face.lms[54, 0] = 322.0
        vrm_smile = map_openseeface_to_vrm(face)
        self.assertGreater(vrm_smile["joy"], 0.40, "Smiling face with elevated corners must trigger Joy")

    def test_openseeface_vrc_none_blink_guard(self):
        try:
            from openseeface_tracker import map_openseeface_to_vrc
        except ImportError:
            from src.openseeface_tracker import map_openseeface_to_vrc

        class FaceWithNoneBlink:
            def __init__(self):
                self.eye_blink = None
                self.current_features = {}

        face = FaceWithNoneBlink()
        # Must not raise TypeError: 'NoneType' object is not subscriptable
        vrc = map_openseeface_to_vrc(face)
        self.assertEqual(vrc["vrc_blink"], 0.0)


if __name__ == "__main__":
    unittest.main()



