"""
Main Entry Point: Desktop VTuber 3D Avatar Application
"""

import os
import sys
import argparse

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
for _p in (_REPO_ROOT, _CURRENT_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QCoreApplication

from vrm_renderer import VTuberWindow


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VTuber 3D Avatar Desktop Application")
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Initial window width (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Initial window height (default: 768)",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    app = QApplication(sys.argv)
    app.setApplicationName("VTuber 3D Avatar")
    app.setOrganizationName("VTuber")

    window = VTuberWindow(
        camera_id=args.camera_id,
        width=args.width,
        height=args.height,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()