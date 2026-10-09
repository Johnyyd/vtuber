import unittest
import numpy as np
try:
    from landmark_mapping import map_mediapipe_to_vrc, compute_head_pose, VRC_TARGETS
except ImportError:
    from src.landmark_mapping import map_mediapipe_to_vrc, compute_head_pose, VRC_TARGETS


class TestLandmarkMapping(unittest.TestCase):
    def test_vrc_targets_contains_all_16_targets(self):
        expected_targets = {
            "vrc_blink", "vrc_v_aa", "vrc_v_oh", "vrc_v_ou", "vrc_v_ee", "vrc_v_ih", "vrc_v_sil",
            "vrc_v_ch", "vrc_v_dd", "vrc_v_ff", "vrc_v_kk", "vrc_v_nn", "vrc_v_pp", "vrc_v_rr", "vrc_v_ss", "vrc_v_th"
        }
        self.assertEqual(set(VRC_TARGETS), expected_targets)

    def test_vrc_mapping_blink_and_mouth(self):
        mp_input = {
            "eyeBlinkLeft": 0.85,
            "eyeBlinkRight": 0.85,
            "jawOpen": 0.75,
            "mouthStretchLeft": 0.1,
            "mouthStretchRight": 0.1,
        }
        vrc = map_mediapipe_to_vrc(mp_input)
        self.assertEqual(len(vrc), 16)
        self.assertIn("vrc_blink", vrc)
        self.assertIn("vrc_v_aa", vrc)
        self.assertIn("vrc_v_sil", vrc)
        self.assertGreaterEqual(vrc["vrc_blink"], 0.7)
        self.assertLessEqual(vrc["vrc_blink"], 1.0)
        self.assertGreaterEqual(vrc["vrc_v_aa"], 0.6)
        self.assertLessEqual(vrc["vrc_v_aa"], 1.0)
        for name, val in vrc.items():
            self.assertTrue(0.0 <= val <= 1.0, f"{name} was {val}, out of [0, 1]")

    def test_vrc_mapping_smile_ee(self):
        mp_input = {
            "mouthStretchLeft": 0.8,
            "mouthStretchRight": 0.8,
            "mouthSmileLeft": 0.6,
            "mouthSmileRight": 0.6,
        }
        vrc = map_mediapipe_to_vrc(mp_input)
        self.assertGreaterEqual(vrc["vrc_v_ee"], 0.5)
        self.assertLess(vrc["vrc_v_aa"], 0.2)

    def test_compute_head_pose_neutral(self):
        # Canonical face with natural proportions centered in frame
        landmarks = {
            "lm1": (0.5, 0.50, 0.0),    # nose tip
            "lm152": (0.5, 0.62, 0.0),  # chin
            "lm33": (0.40, 0.40, 0.0),  # left eye outer
            "lm263": (0.60, 0.40, 0.0), # right eye outer
            "lm61": (0.42, 0.56, 0.0),  # left mouth corner
            "lm291": (0.58, 0.56, 0.0), # right mouth corner
        }
        pose = compute_head_pose(landmarks, frame_shape=(480, 640))
        self.assertTrue("pitch" in pose and "yaw" in pose and "roll" in pose)
        self.assertLess(abs(pose["pitch"]), 0.2)
        self.assertLess(abs(pose["yaw"]), 0.2)
        self.assertLess(abs(pose["roll"]), 0.2)

    def test_compute_head_pose_empty(self):
        pose = compute_head_pose({})
        self.assertEqual(pose, {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})

    def test_vrm_mapping_presets(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm, VRM_BLENDSHAPES
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm, VRM_BLENDSHAPES

        # Test mouth opening without emotion
        vrm_a = map_mediapipe_to_vrm({"jawOpen": 0.7})
        self.assertGreater(vrm_a["a"], 0.5)

        # Test eye blinking without emotion
        vrm_blink = map_mediapipe_to_vrm({"eyeBlinkLeft": 0.8, "eyeBlinkRight": 0.8})
        self.assertGreater(vrm_blink["blink"], 0.7)

        # Test joy emotion
        vrm_joy = map_mediapipe_to_vrm({"mouthSmileLeft": 0.85, "mouthSmileRight": 0.85})
        self.assertGreater(vrm_joy["joy"], 0.6)

        for name in VRM_BLENDSHAPES:
            self.assertIn(name, vrm_joy)
            self.assertTrue(0.0 <= vrm_joy[name] <= 1.0)

    def test_emotion_overrides_other_mouth_and_eye_expressions(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # 1. Angry overrides mouth and eyes to 0 (brows down triggers angry)
        mp_angry = {
            "browDownLeft": 0.8,
            "browDownRight": 0.8,
            "jawOpen": 0.6,
            "eyeBlinkLeft": 0.7,
            "eyeBlinkRight": 0.7,
        }
        vrm_angry = map_mediapipe_to_vrm(mp_angry)
        self.assertGreater(vrm_angry["angry"], 0.5, "Angry must be active")
        self.assertEqual(vrm_angry["a"], 0.0, "Angry must override mouth opening to 0")
        self.assertEqual(vrm_angry["blink"], 0.0, "Angry must override eye blink to 0")
        self.assertEqual(vrm_angry["joy"], 0.0)
        self.assertEqual(vrm_angry["surprised"], 0.0)

        # 2. Joy overrides jawOpen and blink to 0
        mp_joy_with_jaw = {
            "mouthSmileLeft": 0.9,
            "mouthSmileRight": 0.9,
            "jawOpen": 0.5,
            "eyeBlinkLeft": 0.6,
            "eyeBlinkRight": 0.6,
        }
        vrm_joy_res = map_mediapipe_to_vrm(mp_joy_with_jaw)
        self.assertGreater(vrm_joy_res["joy"], 0.6, "Joy must be active")
        self.assertEqual(vrm_joy_res["a"], 0.0, "Joy must override mouth to 0")
        self.assertEqual(vrm_joy_res["blink"], 0.0, "Joy must override blink to 0")

        # 3. Surprised overrides vowels and blink to 0
        mp_surp = {
            "browInnerUp": 0.8,
            "eyeWideLeft": 0.4,
            "eyeWideRight": 0.4,
            "mouthStretchLeft": 0.5,
        }
        vrm_surp = map_mediapipe_to_vrm(mp_surp)
        self.assertGreater(vrm_surp["surprised"], 0.6)
        self.assertEqual(vrm_surp["e"], 0.0)
        self.assertEqual(vrm_surp["blink"], 0.0)

    def test_one_euro_filter_smoothing(self):
        try:
            from landmark_mapping import OneEuroFilter
        except ImportError:
            from src.landmark_mapping import OneEuroFilter

        f = OneEuroFilter(min_cutoff=0.8, beta=0.015)
        # Small fluctuations around 0.1 should be smoothed
        v1 = f.filter(0.10, 0.0)
        v2 = f.filter(0.12, 0.033)
        self.assertEqual(v1, 0.10)
        self.assertLess(v2, 0.12)  # Damped
        self.assertGreater(v2, 0.10)

    def test_matrix_head_pose_computation(self):
        try:
            from landmark_mapping import compute_head_pose_from_matrix
        except ImportError:
            from src.landmark_mapping import compute_head_pose_from_matrix

        # Identity 4x4 matrix -> neutral rotation
        mat = np.eye(4, dtype=np.float64)
        pose = compute_head_pose_from_matrix(mat)
        self.assertAlmostEqual(pose["pitch"], 0.0, places=4)
        self.assertAlmostEqual(pose["yaw"], 0.0, places=4)
        self.assertAlmostEqual(pose["roll"], 0.0, places=4)

    def test_blink_deadzone_and_snap(self):
        try:
            from landmark_mapping import _calibrate_blink
        except ImportError:
            from src.landmark_mapping import _calibrate_blink

        self.assertEqual(_calibrate_blink(0.05), 0.0)   # Resting eye open
        self.assertEqual(_calibrate_blink(0.15), 0.0)   # Eye open with slight lighting variation
        self.assertEqual(_calibrate_blink(0.70), 1.0)   # Complete closure snap
        self.assertGreater(_calibrate_blink(0.35), 0.3)

    def test_blink_resting_eyes_completely_open(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # Natural user state: eyes open, looking at bright monitor with high squint
        mp_input = {
            "eyeBlinkLeft": 0.10,
            "eyeBlinkRight": 0.12,
            "eyeSquintLeft": 0.35,
            "eyeSquintRight": 0.35,
        }
        vrm = map_mediapipe_to_vrm(mp_input)
        self.assertEqual(vrm["blink"], 0.0, "Avatar blink must be 0.0 when user eyes are open")
        self.assertEqual(vrm["blink_l"], 0.0, "Avatar blink_l must be 0.0 when user eyes are open")
        self.assertEqual(vrm["blink_r"], 0.0, "Avatar blink_r must be 0.0 when user eyes are open")

    def test_blink_independent_winking(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # Wink left eye
        mp_wink_l = {
            "eyeBlinkLeft": 0.80,
            "eyeBlinkRight": 0.05,
        }
        vrm_l = map_mediapipe_to_vrm(mp_wink_l)
        self.assertEqual(vrm_l["blink"], 0.0)
        self.assertEqual(vrm_l["blink_l"], 1.0)
        self.assertEqual(vrm_l["blink_r"], 0.0)

        # Wink right eye
        mp_wink_r = {
            "eyeBlinkLeft": 0.05,
            "eyeBlinkRight": 0.80,
        }
        vrm_r = map_mediapipe_to_vrm(mp_wink_r)
        self.assertEqual(vrm_r["blink"], 0.0)
        self.assertEqual(vrm_r["blink_l"], 0.0)
        self.assertEqual(vrm_r["blink_r"], 1.0)

    def test_blink_uneven_camera_lighting_registers_cleanly(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # User closes both eyes, but one side of face is shadowed / camera at angle
        mp_uneven = {
            "eyeBlinkLeft": 0.72,
            "eyeBlinkRight": 0.32,
        }
        vrm = map_mediapipe_to_vrm(mp_uneven)
        self.assertGreaterEqual(vrm["blink"], 0.8, "Both eyes must close when blinking under uneven lighting")
        self.assertEqual(vrm["blink_l"], 0.0)
        self.assertEqual(vrm["blink_r"], 0.0)

    def test_compute_ear_open_and_closed(self):
        try:
            from landmark_mapping import compute_ear, ear_to_blink
        except ImportError:
            from src.landmark_mapping import compute_ear, ear_to_blink

        # Wide open eyes: vertical distance large (~0.03), horizontal ~0.10 -> EAR ~ 0.30
        open_lm = {
            "lm33": (0.35, 0.40, 0.0), "lm133": (0.45, 0.40, 0.0),
            "lm160": (0.40, 0.385, 0.0), "lm144": (0.40, 0.415, 0.0),
            "lm158": (0.42, 0.385, 0.0), "lm153": (0.42, 0.415, 0.0),
            "lm362": (0.55, 0.40, 0.0), "lm263": (0.65, 0.40, 0.0),
            "lm385": (0.60, 0.385, 0.0), "lm380": (0.60, 0.415, 0.0),
            "lm387": (0.62, 0.385, 0.0), "lm373": (0.62, 0.415, 0.0),
        }
        ear_l, ear_r = compute_ear(open_lm)
        self.assertGreater(ear_l, 0.25)
        self.assertGreater(ear_r, 0.25)
        self.assertEqual(ear_to_blink(ear_l), 0.0)
        self.assertEqual(ear_to_blink(ear_r), 0.0)

        # Closed eyes: eyelids touching (vertical distance ~0.005) -> EAR ~ 0.05
        closed_lm = {
            "lm33": (0.35, 0.40, 0.0), "lm133": (0.45, 0.40, 0.0),
            "lm160": (0.40, 0.398, 0.0), "lm144": (0.40, 0.402, 0.0),
            "lm158": (0.42, 0.398, 0.0), "lm153": (0.42, 0.402, 0.0),
            "lm362": (0.55, 0.40, 0.0), "lm263": (0.65, 0.40, 0.0),
            "lm385": (0.60, 0.398, 0.0), "lm380": (0.60, 0.402, 0.0),
            "lm387": (0.62, 0.398, 0.0), "lm373": (0.62, 0.402, 0.0),
        }
        ear_l_c, ear_r_c = compute_ear(closed_lm)
        self.assertLess(ear_l_c, 0.16)
        self.assertLess(ear_r_c, 0.16)
        self.assertEqual(ear_to_blink(ear_l_c), 1.0)
        self.assertEqual(ear_to_blink(ear_r_c), 1.0)

    def test_compute_mar_open_and_closed(self):
        try:
            from landmark_mapping import compute_mar, mar_to_mouth_open
        except ImportError:
            from src.landmark_mapping import compute_mar, mar_to_mouth_open

        # Closed mouth: lips touching (lm13 and lm14 vertical gap ~0.002)
        closed_mouth = {
            "lm13": (0.50, 0.600, 0.0), "lm14": (0.50, 0.602, 0.0),
            "lm61": (0.45, 0.600, 0.0), "lm291": (0.55, 0.600, 0.0),
        }
        mar_closed = compute_mar(closed_mouth)
        self.assertLess(mar_closed, 0.05)
        self.assertEqual(mar_to_mouth_open(mar_closed), 0.0)

        # Open mouth: wide vertical gap (lm13 and lm14 vertical gap ~0.05)
        open_mouth = {
            "lm13": (0.50, 0.580, 0.0), "lm14": (0.50, 0.630, 0.0),
            "lm61": (0.45, 0.600, 0.0), "lm291": (0.55, 0.600, 0.0),
        }
        mar_open = compute_mar(open_mouth)
        self.assertGreater(mar_open, 0.30)
        self.assertGreater(mar_to_mouth_open(mar_open), 0.8)

    def test_compute_iris_gaze_direction(self):
        try:
            from landmark_mapping import compute_iris_gaze
        except ImportError:
            from src.landmark_mapping import compute_iris_gaze

        # Center iris: iris is mid-way between corners
        center_iris = {
            "lm33": (0.30, 0.40, 0.0), "lm133": (0.40, 0.40, 0.0),
            "lm159": (0.35, 0.38, 0.0), "lm145": (0.35, 0.42, 0.0),
            "lm468": (0.35, 0.40, 0.0),
            "lm362": (0.50, 0.40, 0.0), "lm263": (0.60, 0.40, 0.0),
            "lm386": (0.55, 0.38, 0.0), "lm374": (0.55, 0.42, 0.0),
            "lm473": (0.55, 0.40, 0.0),
        }
        gaze = compute_iris_gaze(center_iris)
        self.assertAlmostEqual(gaze["x"], 0.0, delta=0.1)
        self.assertAlmostEqual(gaze["y"], 0.0, delta=0.1)

    def test_biomechanical_fusion_with_landmarks(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # Eyes physically wide open (EAR ~ 0.30) even if ARKit classification is noisy (e.g. 0.20 droop)
        lm_open = {
            "lm33": (0.35, 0.40, 0.0), "lm133": (0.45, 0.40, 0.0),
            "lm160": (0.40, 0.385, 0.0), "lm144": (0.40, 0.415, 0.0),
            "lm158": (0.42, 0.385, 0.0), "lm153": (0.42, 0.415, 0.0),
            "lm362": (0.55, 0.40, 0.0), "lm263": (0.65, 0.40, 0.0),
            "lm385": (0.60, 0.385, 0.0), "lm380": (0.60, 0.415, 0.0),
            "lm387": (0.62, 0.385, 0.0), "lm373": (0.62, 0.415, 0.0),
            "lm13": (0.50, 0.58, 0.0), "lm14": (0.50, 0.63, 0.0),
            "lm61": (0.45, 0.60, 0.0), "lm291": (0.55, 0.60, 0.0),
        }
        mp_noisy = {
            "eyeBlinkLeft": 0.20,
            "eyeBlinkRight": 0.20,
            "jawOpen": 0.0,
        }
        vrm = map_mediapipe_to_vrm(mp_noisy, landmarks=lm_open)
        # Suppressed by EAR: blink must be 0.0!
        self.assertEqual(vrm["blink"], 0.0, "Physical wide open EAR must suppress false eyelid droop")
        # Mouth open driven by MAR even when jawOpen is 0.0!
        self.assertGreater(vrm["a"], 0.8, "Physical MAR must drive mouth opening")

    def test_vowel_u_safe_capping_prevents_corner_collapse(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # Strong mouth pucker
        mp_pucker = {"mouthPucker": 0.85}
        vrm = map_mediapipe_to_vrm(mp_pucker)
        self.assertGreater(vrm["u"], 0.4, "U vowel should activate on pucker")
        self.assertLessEqual(vrm["u"], 0.60, "U vowel must be capped <= 0.60 to prevent mouth corner crossover")

        # Resting mouth (pucker < 0.12)
        mp_rest = {"mouthPucker": 0.05}
        vrm_rest = map_mediapipe_to_vrm(mp_rest)
        self.assertEqual(vrm_rest["u"], 0.0, "U vowel must be 0.0 at rest")

    def test_eyebrow_raise_triggers_surprised_not_sorrow(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # User raises eyebrows (browInnerUp & browOuterUp)
        mp_brow_up = {
            "browInnerUp": 0.50,
            "browOuterUpLeft": 0.45,
            "browOuterUpRight": 0.45,
            "eyeWideLeft": 0.30,
            "eyeWideRight": 0.30,
        }
        vrm = map_mediapipe_to_vrm(mp_brow_up)
        self.assertGreater(vrm["surprised"], 0.5, "Raising eyebrows must trigger surprised expression")
        self.assertEqual(vrm["sorrow"], 0.0, "Raising eyebrows must NEVER trigger sorrow")

        # User genuinely frowns / sad mouth
        mp_sad = {
            "mouthFrownLeft": 0.40,
            "mouthFrownRight": 0.40,
            "browInnerUp": 0.25,
        }
        vrm_sad = map_mediapipe_to_vrm(mp_sad)
        self.assertGreater(vrm_sad["sorrow"], 0.5, "Genuine mouth frown must trigger sorrow")
        self.assertEqual(vrm_sad["surprised"], 0.0, "Sad face must not trigger surprised")

    def test_thick_lips_resting_mouth_does_not_open(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # Thick lips resting baseline: jawOpen is ~0.07-0.08, smile is ~0.08
        mp_resting_thick = {
            "jawOpen": 0.08,
            "mouthSmileLeft": 0.08,
            "mouthSmileRight": 0.08,
        }
        cfg = {"mouth_open_deadzone": 0.09, "smile_deadzone": 0.10}
        vrm = map_mediapipe_to_vrm(mp_resting_thick, config=cfg)
        self.assertEqual(vrm["a"], 0.0, "Resting thick lips must not trigger mouth open fluttering")
        self.assertEqual(vrm["joy"], 0.0, "Resting lip corners must not trigger false smile")

        # When speaking / opening mouth
        mp_speaking = {"jawOpen": 0.35}
        vrm_speaking = map_mediapipe_to_vrm(mp_speaking, config=cfg)
        self.assertGreater(vrm_speaking["a"], 0.5, "Speaking mouth must open smoothly")

    def test_glasses_shadow_does_not_cause_eyelid_droop(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # Eyeglasses frame noise baseline: ~0.12 blink score when eyes are wide open
        mp_glasses_noise = {
            "eyeBlinkLeft": 0.12,
            "eyeBlinkRight": 0.12,
        }
        cfg = {"blink_deadzone": 0.13, "blink_snap_thresh": 0.25}
        vrm = map_mediapipe_to_vrm(mp_glasses_noise, config=cfg)
        self.assertEqual(vrm["blink"], 0.0, "Eyeglasses shadow noise must be filtered out")

        # Deliberate blink
        mp_blink = {
            "eyeBlinkLeft": 0.65,
            "eyeBlinkRight": 0.65,
        }
        vrm_blink = map_mediapipe_to_vrm(mp_blink, config=cfg)
        self.assertEqual(vrm_blink["blink"], 1.0, "Deliberate blink must snap cleanly")

    def test_emotion_mutual_exclusivity_prevents_simultaneous_joy_and_surprised(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # Combined expression: Strong smile + Moderate eyebrow raise
        # Joy is dominant: only Joy should be active, Surprised must be 0.0
        mp_combined_joy_dominant = {
            "mouthSmileLeft": 0.85,
            "mouthSmileRight": 0.85,
            "browInnerUp": 0.35,
            "browOuterUpLeft": 0.35,
            "browOuterUpRight": 0.35,
        }
        vrm_joy = map_mediapipe_to_vrm(mp_combined_joy_dominant)
        self.assertGreater(vrm_joy["joy"], 0.5, "Dominant joy must be active")
        self.assertEqual(vrm_joy["surprised"], 0.0, "Non-dominant surprised must be strictly 0.0")
        self.assertEqual(vrm_joy["sorrow"], 0.0)
        self.assertEqual(vrm_joy["angry"], 0.0)

        # Combined expression: Extreme eyebrow raise + Small smile
        # Surprised is dominant: only Surprised should be active, Joy must be 0.0
        mp_combined_surp_dominant = {
            "mouthSmileLeft": 0.15,
            "mouthSmileRight": 0.15,
            "browInnerUp": 0.85,
            "browOuterUpLeft": 0.85,
            "browOuterUpRight": 0.85,
            "eyeWideLeft": 0.40,
            "eyeWideRight": 0.40,
        }
        vrm_surp = map_mediapipe_to_vrm(mp_combined_surp_dominant)
        self.assertGreater(vrm_surp["surprised"], 0.5, "Dominant surprised must be active")
        self.assertEqual(vrm_surp["joy"], 0.0, "Non-dominant joy must be strictly 0.0")
        self.assertEqual(vrm_surp["sorrow"], 0.0)
        self.assertEqual(vrm_surp["angry"], 0.0)

    def test_all_vowels_activate_responsively_without_false_emotions(self):
        try:
            from landmark_mapping import map_mediapipe_to_vrm
        except ImportError:
            from src.landmark_mapping import map_mediapipe_to_vrm

        # 1. Vowel 'A' (Jaw Open): must activate 'a' cleanly without triggering false emotions
        mp_a = {"jawOpen": 0.40}
        vrm_a = map_mediapipe_to_vrm(mp_a)
        self.assertGreater(vrm_a["a"], 0.5, "Vowel 'A' must be active")
        self.assertEqual(vrm_a["joy"], 0.0)
        self.assertEqual(vrm_a["surprised"], 0.0)

        # 2. Vowel 'I' (Lips stretched horizontally with small jaw): must activate 'i'
        mp_i = {"mouthStretchLeft": 0.25, "mouthStretchRight": 0.25, "jawOpen": 0.05}
        vrm_i = map_mediapipe_to_vrm(mp_i)
        self.assertGreater(vrm_i["i"], 0.4, "Vowel 'I' must be active")
        self.assertEqual(vrm_i["joy"], 0.0, "Horizontal stretch for 'I' must not false trigger Joy")

        # 3. Vowel 'E' (Lips stretched horizontally with open jaw): must activate 'e'
        mp_e = {"mouthStretchLeft": 0.25, "mouthStretchRight": 0.25, "jawOpen": 0.35}
        vrm_e = map_mediapipe_to_vrm(mp_e)
        self.assertGreater(vrm_e["e"], 0.4, "Vowel 'E' must be active")
        self.assertEqual(vrm_e["joy"], 0.0, "Horizontal stretch for 'E' must not false trigger Joy")

        # 4. Vowel 'U' (Lips puckered forward): must activate 'u'
        mp_u = {"mouthPucker": 0.25}
        vrm_u = map_mediapipe_to_vrm(mp_u)
        self.assertGreater(vrm_u["u"], 0.3, "Vowel 'U' must be active")
        self.assertEqual(vrm_u["joy"], 0.0)

        # 5. Vowel 'O' (Mouth funnel or open rounded lips): must activate 'o'
        mp_o = {"mouthFunnel": 0.25}
        vrm_o = map_mediapipe_to_vrm(mp_o)
        self.assertGreater(vrm_o["o"], 0.4, "Vowel 'O' must be active")
        self.assertEqual(vrm_o["joy"], 0.0)

        # Open jaw + pucker also produces 'O'
        mp_o_round = {"jawOpen": 0.25, "mouthPucker": 0.15}
        vrm_o_round = map_mediapipe_to_vrm(mp_o_round)
        self.assertGreater(vrm_o_round["o"], 0.3, "Rounded lips with open jaw must activate 'O'")


if __name__ == "__main__":
    unittest.main()


