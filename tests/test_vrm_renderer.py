import json
import unittest
from landmark_mapping import map_mediapipe_to_vrc, map_mediapipe_to_vrm, compute_head_pose
try:
    from vrm_renderer import format_motion_packet
except ImportError:
    from src.vrm_renderer import format_motion_packet


class TestVRMRenderer(unittest.TestCase):
    def test_format_motion_packet_structure(self):
        vrc_input = map_mediapipe_to_vrc({"jawOpen": 0.6, "eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9})
        vrm_input = map_mediapipe_to_vrm({"jawOpen": 0.6, "eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9})
        pose_input = {"pitch": 0.12, "yaw": -0.25, "roll": 0.05}
        packet = format_motion_packet(vrc=vrc_input, rotation=pose_input, vrm=vrm_input)

        self.assertIn("vrm", packet)
        self.assertIn("vrc", packet)
        self.assertIn("rotation", packet)
        self.assertIn("gaze", packet)
        self.assertEqual(packet["vrm"]["blink"], vrm_input["blink"])
        self.assertEqual(packet["vrm"]["a"], vrm_input["a"])
        self.assertEqual(packet["vrc"]["vrc_blink"], vrc_input["vrc_blink"])
        self.assertEqual(packet["rotation"]["pitch"], 0.12)
        self.assertEqual(packet["rotation"]["yaw"], -0.25)
        self.assertEqual(packet["rotation"]["roll"], 0.05)
        self.assertEqual(packet["gaze"], {"x": 0.0, "y": 0.0})

        # Ensure valid JSON string serialization
        json_str = json.dumps(packet)
        self.assertIsInstance(json_str, str)
        self.assertIn("blink", json_str)
        self.assertIn("vrc_v_aa", json_str)
        self.assertIn("pitch", json_str)
        self.assertIn("gaze", json_str)

    def test_format_motion_packet_empty(self):
        packet = format_motion_packet(None, None)
        self.assertIn("vrm", packet)
        self.assertIn("vrc", packet)
        self.assertIn("rotation", packet)
        self.assertIn("gaze", packet)
        self.assertEqual(packet["vrm"]["neutral"], 1.0)
        self.assertEqual(packet["rotation"], {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})
        self.assertEqual(packet["gaze"], {"x": 0.0, "y": 0.0})

    def test_camera_reader_class_lifecycle(self):
        try:
            from vrm_renderer import CameraReader
        except ImportError:
            from src.vrm_renderer import CameraReader

        # Invalid camera ID to verify safe handling and cleanup without hanging
        reader = CameraReader(camera_id=999)
        self.assertFalse(reader.is_opened())
        ret, frame = reader.read_latest()
        self.assertFalse(ret)
        self.assertIsNone(frame)
        reader.release()


if __name__ == "__main__":
    unittest.main()
