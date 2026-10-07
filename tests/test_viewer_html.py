import os
import unittest


class TestViewerHtml(unittest.TestCase):
    def test_html_files_exist(self):
        self.assertTrue(os.path.exists("viewer/index.html"))
        self.assertTrue(os.path.exists("viewer/style.css"))
        self.assertTrue(os.path.exists("viewer/app.js"))

    def test_html_contains_required_scripts(self):
        with open("viewer/index.html", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("libs/three.min.js", content)
        self.assertIn("libs/GLTFLoader.js", content)
        self.assertIn("libs/three-vrm.min.js", content)
        self.assertIn("app.js", content)

    def test_app_js_defines_update_motion(self):
        with open("viewer/app.js", "r", encoding="utf-8") as f:
            content = f.read()
        self.assertIn("window.updateMotion", content)
        self.assertIn("vrc_blink", content)
        self.assertIn("vrc_v_aa", content)
        self.assertIn("mixamorig:Head", content)
        self.assertIn("mixamorig:Neck", content)


if __name__ == "__main__":
    unittest.main()
