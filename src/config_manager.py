"""
Configuration Manager: Handles loading and saving tracking parameters to config.txt.
"""

import os
import configparser
from typing import Dict, Any

# Root repository directory path
_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_CURRENT_DIR)
DEFAULT_CONFIG_PATH = os.path.join(_REPO_ROOT, "config.txt")

# Canonical default parameter values
DEFAULT_CONFIG: Dict[str, Any] = {
    # [GENERAL]
    "camera_id": 0,
    "window_width": 1024,
    "window_height": 768,
    "show_launcher": True,

    # [TRACKING_SENSITIVITY]
    "smile_deadzone": 0.08,
    "smile_gain": 2.5,
    "mouth_open_deadzone": 0.06,
    "mouth_open_gain": 1.5,
    "u_max_clamp": 0.60,
    "blink_deadzone": 0.10,
    "blink_snap_thresh": 0.22,
    "brow_raise_deadzone": 0.08,
    "brow_raise_gain": 2.2,
    "frown_deadzone": 0.08,
    "frown_gain": 2.0,

    # [CAMERA_AND_FILTER]
    "pitch_offset_deg": 18.0,
    "filter_min_cutoff": 0.8,
    "filter_beta": 0.02,
}


def generate_default_config_text(config: Dict[str, Any] = None) -> str:
    """Generate human-readable config.txt content with detailed comments."""
    cfg = dict(DEFAULT_CONFIG)
    if config:
        cfg.update(config)

    return f"""# ==============================================================================
# VTUBER 3D AVATAR - FILE CAU HINH THONG SO (CONFIG.TXT)
# File cau hinh nhan dien khuon mat va he thong.
# Co the chinh sua truc tiep hoac thong qua bang dieu khien truoc khi vao ung dung.
# ==============================================================================

[GENERAL]
# Chi so camera webcam (0: webcam mac dinh, 1, 2: webcam roi hoac DroidCam...)
camera_id = {cfg['camera_id']}

# Kich thuoc cua so mac dinh (pixels)
window_width = {cfg['window_width']}
window_height = {cfg['window_height']}

# Luon hien thi bang tuy chinh thong so truoc khi chay (True / False)
show_launcher = {str(cfg['show_launcher']).lower()}


[TRACKING_SENSITIVITY]
# Nụ cười (Joy):
# - smile_deadzone: Diem bat dau nhan cuoi (mac dinh: 0.08, pham vi: 0.02 - 0.20)
# - smile_gain: He so khuech dai nu cuoi (mac dinh: 2.5, pham vi: 1.0 - 5.0)
smile_deadzone = {cfg['smile_deadzone']:.2f}
smile_gain = {cfg['smile_gain']:.2f}

# Khau hinh mieng noi (Speech / A):
# - mouth_open_deadzone: Diem bat dau mo mieng (mac dinh: 0.06, pham vi: 0.02 - 0.15)
# - mouth_open_gain: Do nhay mo mieng khi noi (mac dinh: 1.5, pham vi: 1.0 - 3.0)
mouth_open_deadzone = {cfg['mouth_open_deadzone']:.2f}
mouth_open_gain = {cfg['mouth_open_gain']:.2f}

# Khau hinh chu moi U (Pucker):
# - u_max_clamp: Gioi han toi da khau hinh U tranh chập 2 mep mieng (mac dinh: 0.60, pham vi: 0.40 - 0.70)
u_max_clamp = {cfg['u_max_clamp']:.2f}

# Chop mat (Blink):
# - blink_deadzone: Diem bat dau nhan chop mat (mac dinh: 0.10, pham vi: 0.05 - 0.20)
# - blink_snap_thresh: Diem nham mat hoan toan (mac dinh: 0.22, pham vi: 0.15 - 0.40)
blink_deadzone = {cfg['blink_deadzone']:.2f}
blink_snap_thresh = {cfg['blink_snap_thresh']:.2f}

# Nhuong chan may (Surprised):
# - brow_raise_deadzone: Diem bat dau nhan nhuong may (mac dinh: 0.08, pham vi: 0.02 - 0.20)
# - brow_raise_gain: Do nhay bieu cam ngac nhien (mac dinh: 2.2, pham vi: 1.0 - 4.0)
brow_raise_deadzone = {cfg['brow_raise_deadzone']:.2f}
brow_raise_gain = {cfg['brow_raise_gain']:.2f}

# Meu / Buon (Sorrow):
# - frown_deadzone: Diem bat dau nhan khoe mieng triu xuong (mac dinh: 0.08, pham vi: 0.02 - 0.20)
# - frown_gain: Do nhay bieu cam buon (mac dinh: 2.0, pham vi: 1.0 - 4.0)
frown_deadzone = {cfg['frown_deadzone']:.2f}
frown_gain = {cfg['frown_gain']:.2f}


[CAMERA_AND_FILTER]
# Goc bu tru do nghieng camera khi dat tren/duoi man hinh (do, mac dinh: 18.0)
pitch_offset_deg = {cfg['pitch_offset_deg']:.1f}

# Bo loc chong rung OneEuroFilter:
# - filter_min_cutoff: Tan so loc rung luc nghi (mac dinh: 0.8, cang nho cang muot)
# - filter_beta: Do nhay dap ung khi cu dong nhanh (mac dinh: 0.02, cang lon cang it tre)
filter_min_cutoff = {cfg['filter_min_cutoff']:.2f}
filter_beta = {cfg['filter_beta']:.4f}
"""


def load_config(file_path: str = DEFAULT_CONFIG_PATH) -> Dict[str, Any]:
    """
    Load configuration from config.txt.
    If the file does not exist, automatically creates it with defaults.
    """
    if not os.path.exists(file_path):
        save_config(DEFAULT_CONFIG, file_path)
        return dict(DEFAULT_CONFIG)

    parser = configparser.ConfigParser()
    try:
        parser.read(file_path, encoding="utf-8")
    except Exception as e:
        print(f"[Config] Error reading {file_path}: {e}. Using defaults.")
        return dict(DEFAULT_CONFIG)

    config = dict(DEFAULT_CONFIG)

    # Helper getters with type safety
    def get_int(section, key, default):
        try:
            return parser.getint(section, key)
        except Exception:
            return default

    def get_float(section, key, default):
        try:
            return parser.getfloat(section, key)
        except Exception:
            return default

    def get_bool(section, key, default):
        try:
            return parser.getboolean(section, key)
        except Exception:
            return default

    # Parse GENERAL
    config["camera_id"] = get_int("GENERAL", "camera_id", config["camera_id"])
    config["window_width"] = get_int("GENERAL", "window_width", config["window_width"])
    config["window_height"] = get_int("GENERAL", "window_height", config["window_height"])
    config["show_launcher"] = get_bool("GENERAL", "show_launcher", config["show_launcher"])

    # Parse TRACKING_SENSITIVITY
    config["smile_deadzone"] = get_float("TRACKING_SENSITIVITY", "smile_deadzone", config["smile_deadzone"])
    config["smile_gain"] = get_float("TRACKING_SENSITIVITY", "smile_gain", config["smile_gain"])
    config["mouth_open_deadzone"] = get_float("TRACKING_SENSITIVITY", "mouth_open_deadzone", config["mouth_open_deadzone"])
    config["mouth_open_gain"] = get_float("TRACKING_SENSITIVITY", "mouth_open_gain", config["mouth_open_gain"])
    config["u_max_clamp"] = get_float("TRACKING_SENSITIVITY", "u_max_clamp", config["u_max_clamp"])
    config["blink_deadzone"] = get_float("TRACKING_SENSITIVITY", "blink_deadzone", config["blink_deadzone"])
    config["blink_snap_thresh"] = get_float("TRACKING_SENSITIVITY", "blink_snap_thresh", config["blink_snap_thresh"])
    config["brow_raise_deadzone"] = get_float("TRACKING_SENSITIVITY", "brow_raise_deadzone", config["brow_raise_deadzone"])
    config["brow_raise_gain"] = get_float("TRACKING_SENSITIVITY", "brow_raise_gain", config["brow_raise_gain"])
    config["frown_deadzone"] = get_float("TRACKING_SENSITIVITY", "frown_deadzone", config["frown_deadzone"])
    config["frown_gain"] = get_float("TRACKING_SENSITIVITY", "frown_gain", config["frown_gain"])

    # Parse CAMERA_AND_FILTER
    config["pitch_offset_deg"] = get_float("CAMERA_AND_FILTER", "pitch_offset_deg", config["pitch_offset_deg"])
    config["filter_min_cutoff"] = get_float("CAMERA_AND_FILTER", "filter_min_cutoff", config["filter_min_cutoff"])
    config["filter_beta"] = get_float("CAMERA_AND_FILTER", "filter_beta", config["filter_beta"])

    return config


def save_config(config: Dict[str, Any], file_path: str = DEFAULT_CONFIG_PATH) -> bool:
    """Save configuration dictionary to config.txt."""
    try:
        content = generate_default_config_text(config)
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(content)
        return True
    except Exception as e:
        print(f"[Config] Error saving {file_path}: {e}")
        return False
