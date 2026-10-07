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


if __name__ == "__main__":
    unittest.main()
