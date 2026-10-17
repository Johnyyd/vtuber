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

    def test_openseeface_adaptive_pitch_baseline_and_jaw_compensation(self):
        try:
            from openseeface_tracker import extract_head_pose, OpenSeeFaceTracker
        except ImportError:
            from src.openseeface_tracker import extract_head_pose, OpenSeeFaceTracker

        class RealAnatomyFace:
            def __init__(self, mouth_open: float = 0.0):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # Right eye
                self.lms[36] = [100.0, 100.0, 1.0]
                self.lms[39] = [100.0, 120.0, 1.0]
                # Left eye
                self.lms[42] = [100.0, 160.0, 1.0]
                self.lms[45] = [100.0, 180.0, 1.0]
                # Nose bridge (27)
                self.lms[27] = [95.0, 140.0, 1.0]
                # Nose tip (30)
                self.lms[30] = [118.0, 140.0, 1.0]  # top = 23px
                # Real human resting chin (8) with ratio 2.30 (bottom = 53px)
                self.lms[8] = [171.0, 140.0, 1.0]
                self.current_features = {"mouth_open": mouth_open}

        class MockTracker:
            def __init__(self):
                self.pitch_ratio_baseline = None

        tracker = MockTracker()
        face_rest = RealAnatomyFace(mouth_open=0.0)

        # 1. On resting level face with real anatomy ratio (2.30), pitch must calibrate to 0.00
        pose_rest = extract_head_pose(face_rest, pitch_offset_deg=18.0, tracker_instance=tracker)
        self.assertAlmostEqual(pose_rest["pitch"], 0.0, places=2, msg="Resting face must have pitch 0.00 regardless of ratio")
        self.assertAlmostEqual(tracker.pitch_ratio_baseline, 53.0 / 23.0, places=2)

        # 2. Speaking/Jaw Open: chin drops by 10px, mouth_open is 0.50.
        # Jaw compensation must prevent the model head from pitching down or up!
        face_talk = RealAnatomyFace(mouth_open=0.50)
        face_talk.lms[8] = [171.0 + (0.50 * 23.0 * 0.40), 140.0, 1.0]
        pose_talk = extract_head_pose(face_talk, pitch_offset_deg=18.0, tracker_instance=tracker)
        self.assertAlmostEqual(pose_talk["pitch"], 0.0, delta=0.04, msg="Speaking must not tilt head pitch")

        # 3. Deliberate tilt: chin drops by 15px with mouth closed (mouth_open=0)
        face_down = RealAnatomyFace(mouth_open=0.0)
        face_down.lms[8] = [186.0, 140.0, 1.0]
        pose_down = extract_head_pose(face_down, pitch_offset_deg=18.0, tracker_instance=tracker)
        self.assertLess(pose_down["pitch"], -0.15, msg="Head tilt down must produce negative pitch")

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

    def test_openseeface_eyebrow_raise_and_neutral_sorrow_suppression(self):
        try:
            from openseeface_tracker import map_openseeface_to_vrm
        except ImportError:
            from src.openseeface_tracker import map_openseeface_to_vrm

        class FaceForEyebrows:
            def __init__(self):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # Nose bridge at Y=270, tip at Y=310 (nose_h = 40)
                self.lms[27] = [270.0, 315.0, 1.0]
                self.lms[30] = [310.0, 315.0, 1.0]
                # Resting mouth corners with slight downward rest (smile_lift = -0.07)
                self.lms[51] = [326.0, 315.0, 1.0]
                self.lms[57] = [334.0, 315.0, 1.0]
                self.lms[48] = [332.8, 290.0, 1.0]
                self.lms[54] = [332.8, 340.0, 1.0]
                # Right & Left eye lids (Y=265)
                self.lms[37] = [265.0, 295.0, 1.0]
                self.lms[38] = [265.0, 305.0, 1.0]
                self.lms[43] = [265.0, 325.0, 1.0]
                self.lms[44] = [265.0, 335.0, 1.0]
                # Neutral Eyebrows at Y=244.2 -> dist = 20.8px / 40px = 0.52
                for i in range(18, 21):
                    self.lms[i] = [244.2, 290.0 + (i - 18) * 10, 1.0]
                for i in range(23, 26):
                    self.lms[i] = [244.2, 325.0 + (i - 23) * 10, 1.0]
                self.eye_blink = [1.0, 1.0]
                # OpenSeeFace steepness might have natural tilt
                self.current_features = {
                    "eyebrow_steepness_l": 0.85,
                    "eyebrow_steepness_r": 0.45,
                    "eyebrow_updown_l": 0.0,
                    "eyebrow_updown_r": 0.0,
                    "mouth_corner_updown_l": 0.0,
                    "mouth_corner_updown_r": 0.0,
                    "mouth_open": 0.0,
                    "mouth_wide": 0.0,
                }

        face = FaceForEyebrows()
        vrm_neutral = map_openseeface_to_vrm(face)
        # Neutral face with brow_ratio=0.52 and smile_lift=-0.07 must be 0 for joy, surprised, sorrow, angry
        self.assertEqual(vrm_neutral["joy"], 0.0)
        self.assertEqual(vrm_neutral["surprised"], 0.0)
        self.assertEqual(vrm_neutral["sorrow"], 0.0, "Natural resting mouth must not trigger Sorrow")
        self.assertEqual(vrm_neutral["angry"], 0.0, "Natural resting eyebrows must not trigger Angry")

        # Now simulate raising eyebrows: brows move UP to Y=237 -> dist = 28px / 40px = 0.70
        for i in range(18, 21):
            face.lms[i, 0] = 237.0
        for i in range(23, 26):
            face.lms[i, 0] = 237.0

        vrm_raised = map_openseeface_to_vrm(face)
        self.assertGreater(vrm_raised["surprised"], 0.50, "Raised eyebrows must trigger Surprised")
        self.assertEqual(vrm_raised["sorrow"], 0.0)
        self.assertEqual(vrm_raised["angry"], 0.0)

    def test_openseeface_eyebrow_raise_with_low_resting_ratio(self):
        """Test user with resting brow_ratio=0.45 (as in debug screenshot). Raising to 0.52 must trigger Surprised."""
        from openseeface_tracker import map_openseeface_to_vrm

        class FaceLowRatio:
            def __init__(self):
                self.conf = 0.95
                self.eye_blink = [0.9, 0.9]
                self.euler = np.array([-25.0, 10.0, -3.0], dtype=np.float32)
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # Nose length 50px: 27=200, 30=250
                self.lms[27, 0] = 200.0
                self.lms[30, 0] = 250.0
                # Eyes at Y=230
                for i in (37, 38, 43, 44):
                    self.lms[i, 0] = 230.0
                # Brows at Y=207.5 -> dist = 22.5px -> brow_ratio = 22.5 / 50 = 0.45
                for i in (18, 19, 20, 23, 24, 25):
                    self.lms[i, 0] = 207.5
                # Mouth center 310, corners 313.5 -> smile_lift = (310 - 313.5)/50 = -0.070
                self.lms[51, 0] = 310.0
                self.lms[57, 0] = 310.0
                self.lms[48, 0] = 313.5
                self.lms[54, 0] = 313.5
                self.current_features = {
                    "eyebrow_steepness_l": 0.89,
                    "eyebrow_steepness_r": 0.50,
                    "eyebrow_updown_l": 0.0,
                    "eyebrow_updown_r": 0.0,
                    "mouth_corner_updown_l": -0.07,
                    "mouth_corner_updown_r": -0.07,
                    "mouth_open": 0.0,
                    "mouth_wide": 0.0,
                }

        face = FaceLowRatio()
        vrm_neutral = map_openseeface_to_vrm(face)
        # Neutral face must have 0.0 for joy, sorrow, angry, surprised
        self.assertEqual(vrm_neutral["joy"], 0.0)
        self.assertEqual(vrm_neutral["surprised"], 0.0)
        self.assertEqual(vrm_neutral["sorrow"], 0.0, "Resting mouth (-0.07) & natural steepness must not trigger Sorrow")
        self.assertEqual(vrm_neutral["angry"], 0.0)
        self.assertAlmostEqual(getattr(face, "brow_ratio", 0.0), 0.45, places=2)

        # Now raise eyebrows: brows move up to Y=204 -> dist = 26px -> brow_ratio = 0.52 (delta = +0.07)
        for i in (18, 19, 20, 23, 24, 25):
            face.lms[i, 0] = 204.0

        vrm_raised = map_openseeface_to_vrm(face)
        self.assertGreater(vrm_raised["surprised"], 0.60, "Eyebrow raise on 0.45 baseline must trigger Surprised > 0.60")
        self.assertEqual(vrm_raised["sorrow"], 0.0, "Surprised must suppress Sorrow")
        self.assertEqual(vrm_raised["angry"], 0.0, "Surprised must suppress Angry")

    def test_openseeface_neutral_tilt_and_eyebrow_sequence_does_not_false_trigger_surprise(self):
        """Verify that natural posture shifts and brief glances do not trap the eyebrow baseline or falsely trigger Surprised."""
        from openseeface_tracker import map_openseeface_to_vrm

        class DynamicFace:
            def __init__(self):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # Outer eye corners for iod=80
                self.lms[36] = [250.0, 200.0, 1.0]
                self.lms[45] = [250.0, 280.0, 1.0]
                self.lms[27] = [220.0, 240.0, 1.0]
                self.lms[30] = [260.0, 240.0, 1.0]
                self.lms[51] = [320.0, 240.0, 1.0]
                self.lms[57] = [330.0, 240.0, 1.0]
                self.lms[48] = [328.0, 215.0, 1.0]
                self.lms[54] = [328.0, 265.0, 1.0]
                self.eye_blink = [1.0, 1.0]
                self.current_features = {}

        class DummyTracker:
            def __init__(self):
                self.brow_baseline = None

        tracker = DummyTracker()
        face = DynamicFace()

        # Phase 1: Neutral frames with brow distance ~22px
        for i in (37, 38, 43, 44):
            face.lms[i, 0] = 250.0
        for i in (18, 19, 20, 23, 24, 25):
            face.lms[i, 0] = 228.0

        for _ in range(15):
            vrm = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertEqual(vrm["surprised"], 0.0, "Neutral face must have 0.0 surprised")

        # Phase 2: Momentary glance down or blink (distance drops to 18px for 5 frames)
        for i in (18, 19, 20, 23, 24, 25):
            face.lms[i, 0] = 232.0
        for _ in range(5):
            vrm = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertEqual(vrm["surprised"], 0.0, "Glance down must have 0.0 surprised")

        # Phase 3: Return to neutral (22px). Baseline must not be trapped at low value!
        for i in (18, 19, 20, 23, 24, 25):
            face.lms[i, 0] = 228.0
        for _ in range(10):
            vrm = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertEqual(vrm["surprised"], 0.0, "Returning to neutral must NOT trigger Surprised!")

        # Phase 4: Deliberate eyebrow raise (distance increases to 28px)
        for i in (18, 19, 20, 23, 24, 25):
            face.lms[i, 0] = 222.0
        vrm_raised = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertGreater(vrm_raised["surprised"], 0.50, "Deliberate raise must trigger Surprised")

    def test_openseeface_pitch_head_tilt_invariance(self):
        """Ensure tilting head up or looking level with upward camera does not trigger Surprised at rest."""
        from openseeface_tracker import map_openseeface_to_vrm

        class DummyTracker:
            def __init__(self):
                self.brow_baseline = None

        tracker = DummyTracker()

        class TiltFace:
            def __init__(self, pitch_tilt="level"):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                self.eye_blink = [1.0, 1.0]
                self.current_features = {}
                # IOD = 50px
                self.lms[36] = [230.0, 200.0, 1.0]
                self.lms[45] = [230.0, 250.0, 1.0]
                self.lms[27, 0] = 200.0 # bridge
                self.lms[30, 0] = 220.0 # nose tip (nose_h = 20)
                self.lms[51, 0] = 240.0
                self.lms[57, 0] = 245.0
                self.lms[48, 0] = 244.0
                self.lms[54, 0] = 244.0

                for i in (37, 38, 43, 44):
                    self.lms[i, 0] = 230.0

                if pitch_tilt == "down":
                    # Chin closer to nose (foreshortened)
                    self.lms[8, 0] = 255.0 # chin (bottom_dist = 35, top_dist = 20, ratio = 1.75)
                    brow_dist = 9.1
                elif pitch_tilt == "level":
                    self.lms[8, 0] = 261.0 # bottom_dist = 41, ratio = 2.05
                    brow_dist = 10.4
                elif pitch_tilt == "up":
                    self.lms[8, 0] = 269.0 # bottom_dist = 49, ratio = 2.45
                    brow_dist = 11.4

                for i in (18, 19, 20, 23, 24, 25):
                    self.lms[i, 0] = 230.0 - brow_dist

        # Sequence 1: Looking down (neutral)
        face_down = TiltFace("down")
        vrm_down = map_openseeface_to_vrm(face_down, tracker_instance=tracker)
        self.assertEqual(vrm_down["surprised"], 0.0)

        # Sequence 2: Shift to looking level with cam (neutral)
        face_level = TiltFace("level")
        vrm_level = map_openseeface_to_vrm(face_level, tracker_instance=tracker)
        self.assertEqual(vrm_level["surprised"], 0.0, "Looking level with camera must not trigger Surprised at rest")

        # Sequence 3: Shift to looking up (neutral)
        face_up = TiltFace("up")
        vrm_up = map_openseeface_to_vrm(face_up, tracker_instance=tracker)
        self.assertEqual(vrm_up["surprised"], 0.0, "Looking up must not trigger Surprised at rest")

    def test_openseeface_closing_eyes_does_not_open_mouth_or_trigger_surprised(self):
        """Verify that when closing both eyes, mouth remains closed and Surprised is strictly 0.0."""
        from openseeface_tracker import map_openseeface_to_vrm

        class DummyTracker:
            def __init__(self):
                self.brow_baseline = None
                self.smile_baseline = None
                self.mouth_width_baseline = None

        tracker = DummyTracker()

        class BlinkFace:
            def __init__(self):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # Nose length 40px: bridge 200, tip 240
                self.lms[27, 0] = 200.0
                self.lms[30, 0] = 240.0
                # Eye corners (36, 39, 42, 45) at Y=230
                self.lms[36] = [230.0, 180.0, 1.0]
                self.lms[39] = [230.0, 210.0, 1.0]
                self.lms[42] = [230.0, 250.0, 1.0]
                self.lms[45] = [230.0, 280.0, 1.0]
                # Brows at Y=210 (dist=20px -> brow_ratio ~ 0.30)
                for i in (18, 19, 20, 23, 24, 25):
                    self.lms[i, 0] = 210.0
                # Mouth closed (inner lips touching at Y=280)
                self.lms[51, 0] = 277.0
                self.lms[57, 0] = 283.0
                self.lms[48] = [282.0, 210.0, 1.0]
                self.lms[54] = [282.0, 250.0, 1.0]
                for i in (59, 60, 61, 63, 64, 65):
                    self.lms[i, 0] = 280.0
                self.eye_blink = [1.0, 1.0]
                self.current_features = {"mouth_open": 0.0, "mouth_wide": 0.0}

        face = BlinkFace()
        # Step 1: Open eyes at rest
        vrm_open = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertEqual(vrm_open["blink"], 0.0)
        self.assertEqual(vrm_open["surprised"], 0.0)
        self.assertEqual(vrm_open["a"], 0.0)

        # Step 2: Close both eyes (eyelids lower downward)
        face.eye_blink = [0.0, 0.0]
        # Simulate upper eyelids descending towards lower eyelids
        self.lms_down = 236.0
        vrm_closed = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertEqual(vrm_closed["blink"], 1.0, "Blink must be fully engaged")
        self.assertEqual(vrm_closed["surprised"], 0.0, "Surprised MUST NOT trigger when closing eyes")
        self.assertEqual(vrm_closed["a"], 0.0, "Mouth MUST NOT open when closing eyes")

    def test_openseeface_independent_single_eye_winking(self):
        """Verify that winking one eye does not trigger synchronized two-eye blink."""
        from openseeface_tracker import map_openseeface_to_vrm

        class WinkFace:
            def __init__(self, left_open=1.0, right_open=1.0):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                # OpenSeeFace eye_blink is [r_open, l_open]
                self.eye_blink = [right_open, left_open]
                self.current_features = {}

        # 1. Left eye wink (left closed=0.0, right open=1.0)
        face_l = WinkFace(left_open=0.0, right_open=1.0)
        vrm_l = map_openseeface_to_vrm(face_l)
        self.assertEqual(vrm_l["blink"], 0.0, "Single eye wink must NOT trigger synchronized Blink")
        self.assertEqual(vrm_l["blink_l"], 1.0, "Left wink must engage blink_l")
        self.assertEqual(vrm_l["blink_r"], 0.0, "Left wink must leave right eye open")

        # 2. Right eye wink (right closed=0.0, left open=1.0)
        face_r = WinkFace(left_open=1.0, right_open=0.0)
        vrm_r = map_openseeface_to_vrm(face_r)
        self.assertEqual(vrm_r["blink"], 0.0, "Single eye wink must NOT trigger synchronized Blink")
        self.assertEqual(vrm_r["blink_r"], 1.0, "Right wink must engage blink_r")
        self.assertEqual(vrm_r["blink_l"], 0.0, "Right wink must leave left eye open")

        # 3. Synchronized two-eye blink (both closed=0.0)
        face_both = WinkFace(left_open=0.0, right_open=0.0)
        vrm_both = map_openseeface_to_vrm(face_both)
        self.assertEqual(vrm_both["blink"], 1.0, "Two-eye closure must trigger synchronized Blink")
        self.assertEqual(vrm_both["blink_l"], 0.0)
        self.assertEqual(vrm_both["blink_r"], 0.0)

    def test_openseeface_responsive_joy_sorrow_angry_vowels(self):
        """Verify that Joy, Sorrow, Angry and Vowels E/I/U respond reliably."""
        from openseeface_tracker import map_openseeface_to_vrm

        class DummyTracker:
            def __init__(self):
                self.brow_baseline = None
                self.smile_baseline = None
                self.mouth_width_baseline = None

        tracker = DummyTracker()

        class ExpressiveFace:
            def __init__(self):
                self.lms = np.zeros((68, 3), dtype=np.float32)
                self.lms[27, 0] = 200.0
                self.lms[30, 0] = 240.0
                self.lms[36] = [230.0, 180.0, 1.0]
                self.lms[39] = [230.0, 210.0, 1.0]
                self.lms[42] = [230.0, 250.0, 1.0]
                self.lms[45] = [230.0, 280.0, 1.0]
                for i in (18, 19, 20, 23, 24, 25):
                    self.lms[i, 0] = 210.0
                # Resting mouth corners at 284 (smile_lift = -0.10)
                self.lms[51, 0] = 277.0
                self.lms[57, 0] = 283.0
                self.lms[48] = [284.0, 210.0, 1.0]
                self.lms[54] = [284.0, 250.0, 1.0]
                for i in (59, 60, 61, 63, 64, 65):
                    self.lms[i, 0] = 280.0
                self.eye_blink = [1.0, 1.0]
                self.current_features = {"mouth_open": 0.0, "mouth_wide": 0.0}

        face = ExpressiveFace()
        # Initialize resting baseline
        map_openseeface_to_vrm(face, tracker_instance=tracker)

        # 1. Test Joy: corners lift upwards from Y=284 to Y=276 (+8px lift -> smile_delta > 0.05)
        face.lms[48, 0] = 276.0
        face.lms[54, 0] = 276.0
        vrm_joy = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertGreater(vrm_joy["joy"], 0.40, "Smiling must trigger Joy > 0.40")

        # 2. Test Sorrow: corners droop downwards from resting Y=284 to Y=290
        face.lms[48, 0] = 290.0
        face.lms[54, 0] = 290.0
        vrm_sorrow = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertGreater(vrm_sorrow["sorrow"], 0.25, "Frowning must trigger Sorrow > 0.25")

        # 3. Test Angry: brows furrow downwards towards eyes (Y=210 moves to Y=215)
        face.lms[48, 0] = 284.0
        face.lms[54, 0] = 284.0
        for i in (18, 19, 20, 23, 24, 25):
            face.lms[i, 0] = 215.0
        vrm_angry = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertGreater(vrm_angry["angry"], 0.40, "Brow furrowing must trigger Angry > 0.40")

        # 4. Test Vowel U: lip puckering (mouth width narrows, moderate opening)
        face.lms[48] = [284.0, 222.0, 1.0] # narrowed width
        face.lms[54] = [284.0, 238.0, 1.0]
        face.current_features["mouth_open"] = 0.20
        face.current_features["mouth_wide"] = -0.30
        for i in (59, 60, 61):
            face.lms[i, 0] = 277.0
        for i in (63, 64, 65):
            face.lms[i, 0] = 283.0
        vrm_u = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertGreater(vrm_u["u"], 0.30, "Pronouncing U must trigger U > 0.30")

        # 5. Test Vowel E: open jaw + lateral horizontal stretch
        face.current_features["mouth_open"] = 0.45
        face.current_features["mouth_wide"] = 0.40
        face.lms[48] = [284.0, 200.0, 1.0] # wide width
        face.lms[54] = [284.0, 260.0, 1.0]
        vrm_e = map_openseeface_to_vrm(face, tracker_instance=tracker)
        self.assertGreater(vrm_e["e"], 0.30, "Pronouncing E must trigger E > 0.30")


if __name__ == "__main__":
    unittest.main()



