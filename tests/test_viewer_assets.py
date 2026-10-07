import os
import unittest


class TestViewerAssets(unittest.TestCase):
    def test_viewer_libraries_exist(self):
        three_path = "viewer/libs/three.min.js"
        vrm_path = "viewer/libs/three-vrm.min.js"
        self.assertTrue(os.path.exists(three_path), f"{three_path} does not exist")
        self.assertTrue(os.path.exists(vrm_path), f"{vrm_path} does not exist")
        self.assertGreater(os.path.getsize(three_path), 100000)
        self.assertGreater(os.path.getsize(vrm_path), 50000)

    def test_yong_model_deployed_and_old_model_removed(self):
        yong_path = "assets/Yong.vrm"
        old_char_path = "assets/character.vrm"
        self.assertTrue(os.path.exists(yong_path), f"{yong_path} must exist as the active model")
        self.assertGreater(os.path.getsize(yong_path), 1000000, "Yong.vrm must be valid size (> 1MB)")
        self.assertFalse(os.path.exists(old_char_path), f"{old_char_path} should be deleted")


if __name__ == "__main__":
    unittest.main()
