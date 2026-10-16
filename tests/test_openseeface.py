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


if __name__ == "__main__":
    unittest.main()
