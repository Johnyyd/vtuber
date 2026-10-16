"""
Configuration Manager: Handles loading and saving tracking parameters to config.txt.
"""

import os
import configparser
from typing import Dict, Any

import sys

# Root repository or executable directory path
if getattr(sys, "frozen", False):
    _APP_ROOT = os.path.dirname(sys.executable)
else:
    _CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
    _APP_ROOT = os.path.dirname(_CURRENT_DIR)

DEFAULT_CONFIG_PATH = os.path.join(_APP_ROOT, "config.txt")

# Canonical default parameter values (calibrated for natural expressions, fuller lips & eyeglasses)
DEFAULT_CONFIG: Dict[str, Any] = {
    # [GENERAL]
    "camera_id": 0,
    "tracker_backend": "openseeface",  # "openseeface" hoac "mediapipe"
    "window_width": 800,
    "window_height": 600,
    "show_launcher": True,

    # [TRACKING_SENSITIVITY]
    "smile_deadzone": 0.10,
    "smile_gain": 2.8,
    "mouth_open_deadzone": 0.06,
    "mouth_open_gain": 1.8,
    "u_max_clamp": 0.58,
    "o_deadzone": 0.10,
    "o_gain": 2.4,
    "o_max_clamp": 0.55,
    "blink_deadzone": 0.13,
    "blink_snap_thresh": 0.25,
    "brow_raise_deadzone": 0.08,
    "brow_raise_gain": 2.4,
    "frown_deadzone": 0.04,
    "frown_gain": 3.2,

    # [BODY_AND_IDLE_MOTION]
    "body_sway_gain": 1.0,
    "idle_breath_intensity": 1.0,

    # [HAIR_PHYSICS]
    "hair_gravity": 0.15,
    "hair_stiffness": 0.95,
    "hair_drag": 0.50,

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

# Mo hinh AI nhan dien khuon mat (mediapipe hoac openseeface)
# - mediapipe: Google MediaPipe (Mac dinh - Sieu nhe, muot tren moi may)
# - openseeface: OpenSeeFace ONNX (Chuyen dung VTuber - Tracking mat va mieng chinh xac cao)
tracker_backend = {cfg['tracker_backend']}

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
# - mouth_open_gain: Do nhay mo mieng khi noi (mac dinh: 1.8, pham vi: 1.0 - 3.0)
mouth_open_deadzone = {cfg['mouth_open_deadzone']:.2f}
mouth_open_gain = {cfg['mouth_open_gain']:.2f}

# Khau hinh chu moi U (Pucker):
# - u_max_clamp: Gioi han toi da khau hinh U tranh chập 2 mep mieng (mac dinh: 0.60, pham vi: 0.40 - 0.70)
u_max_clamp = {cfg['u_max_clamp']:.2f}

# Khau hinh tron mieng O (Funnel / Round):
# - o_deadzone: Diem bat dau nhan khau hinh O (mac dinh: 0.10, pham vi: 0.05 - 0.20)
# - o_gain: Do nhay khau hinh O (mac dinh: 2.4, pham vi: 1.0 - 4.0)
# - o_max_clamp: Gioi han toi da do mo khau hinh O tranh mo mieng qua to (mac dinh: 0.55, pham vi: 0.35 - 0.75)
o_deadzone = {cfg['o_deadzone']:.2f}
o_gain = {cfg['o_gain']:.2f}
o_max_clamp = {cfg['o_max_clamp']:.2f}

# Chop mat (Blink):
# - blink_deadzone: Diem bat dau nhan chop mat (mac dinh: 0.10, pham vi: 0.05 - 0.20)
# - blink_snap_thresh: Diem nham mat hoan toan (mac dinh: 0.22, pham vi: 0.15 - 0.40)
blink_deadzone = {cfg['blink_deadzone']:.2f}
blink_snap_thresh = {cfg['blink_snap_thresh']:.2f}

# Nhuong chan may (Surprised):
# - brow_raise_deadzone: Diem bat dau nhan nhuong may (mac dinh: 0.08, pham vi: 0.02 - 0.20)
# - brow_raise_gain: Do nhay bieu cam ngac nhien (mac dinh: 2.4, pham vi: 1.0 - 4.0)
brow_raise_deadzone = {cfg['brow_raise_deadzone']:.2f}
brow_raise_gain = {cfg['brow_raise_gain']:.2f}

# Meu / Buon (Sorrow):
# - frown_deadzone: Diem bat dau nhan khoe mieng triu xuong (mac dinh: 0.04, pham vi: 0.02 - 0.20)
# - frown_gain: Do nhay bieu cam buon (mac dinh: 3.2, pham vi: 1.0 - 5.0)
frown_deadzone = {cfg['frown_deadzone']:.2f}
frown_gain = {cfg['frown_gain']:.2f}


[BODY_AND_IDLE_MOTION]
# Dung dua than minh theo camera (Kinematic Torso Sway):
# - body_sway_gain: He so uon cong cot song/nguc khi nghieng dau (mac dinh: 1.0, pham vi: 0.0 - 2.0, 0.0 la tat)
body_sway_gain = {cfg['body_sway_gain']:.2f}

# Nhip tho va dung dua vi mo tu dong khi ngoi yen (Procedural Idle & Breathing):
# - idle_breath_intensity: Do manh nhip tho va chuyen dong tu dong (mac dinh: 1.0, pham vi: 0.0 - 2.0, 0.0 la tat)
idle_breath_intensity = {cfg['idle_breath_intensity']:.2f}


[HAIR_PHYSICS]
# Do nang va vat ly cua toc (Hair Physics & Weight):
# - hair_gravity: Trong luc keo toc triu xuong tu nhien (mac dinh: 1.60, pham vi: 0.0 - 5.0, 0.0 la bay bong khong trong luong)
# - hair_stiffness: Do cung dan hoi cua soi toc (mac dinh: 0.38, pham vi: 0.05 - 2.0, cang nho toc cang mem ru)
# - hair_drag: Luc can khong khi / do dam quan tinh cua toc (mac dinh: 0.58, pham vi: 0.05 - 0.95, cang lon toc cang dam va nang)
hair_gravity = {cfg['hair_gravity']:.2f}
hair_stiffness = {cfg['hair_stiffness']:.2f}
hair_drag = {cfg['hair_drag']:.2f}


[CAMERA_AND_FILTER]
# Goc bu tru do nghieng camera khi dat tren/duoi man hinh (do, mac dinh: 18.0)
pitch_offset_deg = {cfg['pitch_offset_deg']:.1f}

# Bo loc chong rung OneEuroFilter:
# - filter_min_cutoff: Tan so loc rung luc nghi (mac dinh: 0.8, cang nho cang muot)
# - filter_beta: Do nhay dap ung khi cu dong nhanh (mac dinh: 0.02, cang lon cang it tre)
filter_min_cutoff = {cfg['filter_min_cutoff']:.2f}
filter_beta = {cfg['filter_beta']:.4f}
"""


def validate_config(config: Dict[str, Any]) -> Dict[str, Any]:
    """Validate configuration dictionary, sanitizing and filling default fallbacks."""
    validated = dict(DEFAULT_CONFIG)
    if not isinstance(config, dict):
        return validated

    validated.update(config)

    # Validate tracker_backend
    backend = str(validated.get("tracker_backend", "openseeface")).strip().lower()
    if backend not in {"mediapipe", "openseeface"}:
        backend = "openseeface"
    validated["tracker_backend"] = backend

    # Clamp numeric values
    try:
        validated["camera_id"] = max(0, int(validated["camera_id"]))
        validated["window_width"] = max(400, int(validated["window_width"]))
        validated["window_height"] = max(300, int(validated["window_height"]))
        validated["show_launcher"] = bool(validated["show_launcher"])

        for float_key in [
            "smile_deadzone", "smile_gain", "mouth_open_deadzone", "mouth_open_gain",
            "u_max_clamp", "o_deadzone", "o_gain", "o_max_clamp",
            "blink_deadzone", "blink_snap_thresh", "brow_raise_deadzone", "brow_raise_gain",
            "frown_deadzone", "frown_gain", "body_sway_gain", "idle_breath_intensity",
            "hair_gravity", "hair_stiffness", "hair_drag",
            "pitch_offset_deg", "filter_min_cutoff", "filter_beta"
        ]:
            if float_key in validated:
                validated[float_key] = float(validated[float_key])
    except Exception:
        pass

    return validated


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

    def get_str(section, key, default):
        try:
            val = parser.get(section, key).strip()
            return val if val else default
        except Exception:
            return default

    # Parse GENERAL
    config["camera_id"] = get_int("GENERAL", "camera_id", config["camera_id"])
    config["tracker_backend"] = get_str("GENERAL", "tracker_backend", config["tracker_backend"])
    config["window_width"] = get_int("GENERAL", "window_width", config["window_width"])
    config["window_height"] = get_int("GENERAL", "window_height", config["window_height"])
    config["show_launcher"] = get_bool("GENERAL", "show_launcher", config["show_launcher"])

    # Parse TRACKING_SENSITIVITY
    config["smile_deadzone"] = get_float("TRACKING_SENSITIVITY", "smile_deadzone", config["smile_deadzone"])
    config["smile_gain"] = get_float("TRACKING_SENSITIVITY", "smile_gain", config["smile_gain"])
    config["mouth_open_deadzone"] = get_float("TRACKING_SENSITIVITY", "mouth_open_deadzone", config["mouth_open_deadzone"])
    config["mouth_open_gain"] = get_float("TRACKING_SENSITIVITY", "mouth_open_gain", config["mouth_open_gain"])
    config["u_max_clamp"] = get_float("TRACKING_SENSITIVITY", "u_max_clamp", config["u_max_clamp"])
    config["o_deadzone"] = get_float("TRACKING_SENSITIVITY", "o_deadzone", config["o_deadzone"])
    config["o_gain"] = get_float("TRACKING_SENSITIVITY", "o_gain", config["o_gain"])
    config["o_max_clamp"] = get_float("TRACKING_SENSITIVITY", "o_max_clamp", config["o_max_clamp"])
    config["blink_deadzone"] = get_float("TRACKING_SENSITIVITY", "blink_deadzone", config["blink_deadzone"])
    config["blink_snap_thresh"] = get_float("TRACKING_SENSITIVITY", "blink_snap_thresh", config["blink_snap_thresh"])
    config["brow_raise_deadzone"] = get_float("TRACKING_SENSITIVITY", "brow_raise_deadzone", config["brow_raise_deadzone"])
    config["brow_raise_gain"] = get_float("TRACKING_SENSITIVITY", "brow_raise_gain", config["brow_raise_gain"])
    config["frown_deadzone"] = get_float("TRACKING_SENSITIVITY", "frown_deadzone", config["frown_deadzone"])
    config["frown_gain"] = get_float("TRACKING_SENSITIVITY", "frown_gain", config["frown_gain"])

    # Parse BODY_AND_IDLE_MOTION
    config["body_sway_gain"] = get_float("BODY_AND_IDLE_MOTION", "body_sway_gain", config.get("body_sway_gain", 1.0))
    config["idle_breath_intensity"] = get_float("BODY_AND_IDLE_MOTION", "idle_breath_intensity", config.get("idle_breath_intensity", 1.0))

    # Parse HAIR_PHYSICS
    config["hair_gravity"] = get_float("HAIR_PHYSICS", "hair_gravity", config.get("hair_gravity", 1.60))
    config["hair_stiffness"] = get_float("HAIR_PHYSICS", "hair_stiffness", config.get("hair_stiffness", 0.38))
    config["hair_drag"] = get_float("HAIR_PHYSICS", "hair_drag", config.get("hair_drag", 0.58))

    # Parse CAMERA_AND_FILTER
    config["pitch_offset_deg"] = get_float("CAMERA_AND_FILTER", "pitch_offset_deg", config["pitch_offset_deg"])
    config["filter_min_cutoff"] = get_float("CAMERA_AND_FILTER", "filter_min_cutoff", config["filter_min_cutoff"])
    config["filter_beta"] = get_float("CAMERA_AND_FILTER", "filter_beta", config["filter_beta"])

    return validate_config(config)


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
