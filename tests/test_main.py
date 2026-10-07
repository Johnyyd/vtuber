import unittest
try:
    from main import parse_args
except ImportError:
    from src.main import parse_args


class TestMainCLI(unittest.TestCase):
    def test_default_args(self):
        args = parse_args([])
        self.assertEqual(args.camera_id, 0)
        self.assertEqual(args.width, 1024)
        self.assertEqual(args.height, 768)

    def test_custom_args(self):
        args = parse_args(["--camera-id", "2", "--width", "1280", "--height", "720"])
        self.assertEqual(args.camera_id, 2)
        self.assertEqual(args.width, 1280)
        self.assertEqual(args.height, 720)


if __name__ == "__main__":
    unittest.main()
