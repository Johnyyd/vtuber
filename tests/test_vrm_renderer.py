import json
import unittest
from landmark_mapping import map_mediapipe_to_vrc, compute_head_pose
try:
    from vrm_renderer import format_motion_packet
except ImportError:
    from src.vrm_renderer import format_motion_packet


class TestVRMRenderer(unittest.TestCase):
    def test_format_motion_packet_structure(self):
        vrc_input = map_mediapipe_to_vrc({"jawOpen": 0.6, "eyeBlinkLeft": 0.9, "eyeBlinkRight": 0.9})
        pose_input = {"pitch": 0.12, "yaw": -0.25, "roll": 0.05}
        packet = format_motion_packet(vrc_input, pose_input)

        self.assertIn("vrc", packet)
        self.assertIn("rotation", packet)
        self.assertEqual(packet["vrc"]["vrc_blink"], vrc_input["vrc_blink"])
        self.assertEqual(packet["rotation"]["pitch"], 0.12)
        self.assertEqual(packet["rotation"]["yaw"], -0.25)
        self.assertEqual(packet["rotation"]["roll"], 0.05)

        # Ensure valid JSON string serialization
        json_str = json.dumps(packet)
        self.assertIsInstance(json_str, str)
        self.assertIn("vrc_v_aa", json_str)
        self.assertIn("pitch", json_str)

    def test_format_motion_packet_empty(self):
        packet = format_motion_packet(None, None)
        self.assertIn("vrc", packet)
        self.assertIn("rotation", packet)
        self.assertEqual(packet["rotation"], {"pitch": 0.0, "yaw": 0.0, "roll": 0.0})


if __name__ == "__main__":
    unittest.main()
