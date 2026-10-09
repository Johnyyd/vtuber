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

from vrm_renderer import VTuberWindow
from config_manager import load_config, DEFAULT_CONFIG_PATH
from config_gui import ConfigLauncherDialog


def parse_args(argv=None):
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="VTuber 3D Avatar Desktop Application")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to config.txt (default: config.txt)",
    )
    parser.add_argument(
        "--camera-id",
        type=int,
        default=0,
        help="Webcam device index override (default: 0)",
    )
    parser.add_argument(
        "--width",
        type=int,
        default=1024,
        help="Initial window width override (default: 1024)",
    )
    parser.add_argument(
        "--height",
        type=int,
        default=768,
        help="Initial window height override (default: 768)",
    )
    parser.add_argument(
        "--show-launcher",
        action="store_true",
        default=False,
        help="Force show parameter configuration panel before starting",
    )
    parser.add_argument(
        "--no-launcher",
        action="store_true",
        default=False,
        help="Skip configuration panel and launch avatar directly",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args(sys.argv[1:])

    app = QApplication(sys.argv)
    app.setApplicationName("VTuber 3D Avatar")
    app.setOrganizationName("VTuber")

    # 1. Load config from config.txt
    cfg = load_config(args.config)

    # Command line overrides if explicitly supplied
    raw_args = sys.argv[1:]
    if any(a == "--camera-id" or a.startswith("--camera-id=") for a in raw_args):
        cfg["camera_id"] = args.camera_id
    if any(a == "--width" or a.startswith("--width=") for a in raw_args):
        cfg["window_width"] = args.width
    if any(a == "--height" or a.startswith("--height=") for a in raw_args):
        cfg["window_height"] = args.height

    # 2. Check if configuration panel should be displayed
    should_show_launcher = (args.show_launcher or cfg.get("show_launcher", True)) and not args.no_launcher

    if should_show_launcher:
        dialog = ConfigLauncherDialog(config_path=args.config)
        # If user closes or clicks Cancel/Thoát, exit cleanly
        if not dialog.exec():
            sys.exit(0)
        # Retrieve updated/saved configuration
        cfg = dialog.get_configured_settings()

    # 3. Launch 3D VTuber Window
    window = VTuberWindow(
        camera_id=cfg["camera_id"],
        width=cfg["window_width"],
        height=cfg["window_height"],
        config=cfg,
    )
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()