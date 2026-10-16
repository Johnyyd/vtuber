# OpenSeeFace Dual-Backend Tracking Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate OpenSeeFace (MobileNetV3 ONNX) as an optional, high-precision face tracking backend alongside Google MediaPipe, selectable via config and Launcher GUI.

**Architecture:** Create an `OpenSeeFaceTracker` adapter that matches the exact output format of MediaPipe tracking (`vrm_shapes`, `vrc_targets`, `rotation`, `gaze`). `TrackingWorker` routes camera frames to either `FaceLandmarker` or `OpenSeeFaceTracker` based on `tracker_backend` configuration, feeding through the existing `OneEuroFilter`, Kinematic Torso Sway, and Three.js VRM pipeline without breaking any downstream components.

**Tech Stack:** Python 3.11, ONNX Runtime (`onnxruntime`), OpenCV (`cv2`), NumPy, PyQt6, Three.js VRM.

**Spec:** Dual-Backend selector supporting `mediapipe` (default) and `openseeface`. Auto-download ONNX weights on first run. Zero-lag vowel synthesis (`a, i, u, e, o`), natural blink detection, 3D head pose, and iris gaze tracking mapped to universal VRM blendshapes.

## Global Constraints
- Must not break existing MediaPipe tracking behavior or performance.
- Both backends must output standard VRM blendshape dictionaries compatible with `format_motion_packet`.
- All ONNX model weights (`models/openseeface/`) must be downloaded automatically if not found locally, and ignored by Git.
- 100% test pass rate across `pytest tests/`.

---

### Task 1: OpenSeeFace Core Tracking Engine & Model Downloader

**Files:**
- Create: `src/openseeface/__init__.py`
- Create: `src/openseeface/remedian.py`
- Create: `src/openseeface/similaritytransform.py`
- Create: `src/openseeface/retinaface.py`
- Create: `src/openseeface/tracker.py`
- Create: `src/openseeface_tracker.py`
- Modify: `requirements.txt`
- Modify: `.gitignore`
- Test: `tests/test_openseeface.py`

**Interfaces:**
- Produces:
  - `ensure_openseeface_models_downloaded(model_dir: Optional[str] = None) -> str`
  - `class OpenSeeFaceTracker`:
    - `__init__(width: int = 1280, height: int = 720, model_dir: Optional[str] = None, max_threads: int = 2)`
    - `process_frame(frame: np.ndarray, config: Optional[dict] = None) -> Optional[Tuple[Dict[str, float], Dict[str, float], Dict[str, float], Dict[str, float]]]`

- [ ] **Step 1: Write failing test for OpenSeeFace model downloader & tracker initialization**

In `tests/test_openseeface.py`:
```python
import os
import unittest
import numpy as np
from openseeface_tracker import ensure_openseeface_models_downloaded, OpenSeeFaceTracker

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_openseeface.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'openseeface_tracker'`

- [ ] **Step 3: Add `onnxruntime` to requirements, update `.gitignore`, and implement OpenSeeFace modules**

In `requirements.txt`:
```text
mediapipe>=0.10.35
opencv-python>=4.7.0.72
numpy
pygltflib>=1.16
PyQt6>=6.5.0
PyQt6-WebEngine>=6.5.0
matplotlib
onnxruntime>=1.15.0
```

In `.gitignore`:
```text
/venv
*__pycache__
*.pyc
*.pytest_cache
/build
/dist
VTuberAvatar-v1.0.0-windows.zip
models/face_landmarker.task
models/openseeface/
```

Implement `src/openseeface/`:
- `__init__.py`: package marker
- `remedian.py`: robust running median estimator
- `similaritytransform.py`: similarity transform and Procrustes analysis
- `retinaface.py`: lightweight detector module
- `tracker.py`: OpenSeeFace Tracker pipeline modified for package-relative imports

Implement `src/openseeface_tracker.py` with `ensure_openseeface_models_downloaded()` and `OpenSeeFaceTracker`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_openseeface.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add requirements.txt .gitignore src/openseeface/ src/openseeface_tracker.py tests/test_openseeface.py
git commit -m "feat(openseeface): add core tracker engine and model downloader"
```

---

### Task 2: OpenSeeFace Feature to VRM & VRChat Blendshape Mapping

**Files:**
- Modify: `src/openseeface_tracker.py`
- Test: `tests/test_openseeface.py`

**Interfaces:**
- Consumes: OpenSeeFace face object (`face.current_features`, `face.eye_blink`, `face.euler`, `face.lms`)
- Produces:
  - `map_openseeface_to_vrm(face, config: Optional[dict] = None) -> Dict[str, float]`
  - `map_openseeface_to_vrc(face) -> Dict[str, float]`
  - `extract_head_pose(face, pitch_offset_deg: float = 18.0) -> Dict[str, float]`
  - `extract_iris_gaze(face) -> Dict[str, float]`

- [ ] **Step 1: Write failing tests for mapping functions with mock OpenSeeFace face objects**

In `tests/test_openseeface.py`:
```python
class MockFace:
    def __init__(self):
        self.conf = 0.95
        self.eye_blink = [0.9, 0.9] # open eyes
        self.euler = np.array([5.0, -10.0, 2.0], dtype=np.float32)
        self.current_features = {
            "eye_l": 0.0,
            "eye_r": 0.0,
            "mouth_open": 0.75,
            "mouth_wide": 0.20,
            "mouth_corner_updown_l": 0.5,
            "mouth_corner_updown_r": 0.5,
            "eyebrow_updown_l": 0.3,
            "eyebrow_updown_r": 0.3,
        }
        # 68 landmarks with dummy coordinates
        self.lms = np.zeros((68, 3), dtype=np.float32)
        # Eye landmarks for right eye (36-41) and left eye (42-47), pupils (66, 67)
        self.lms[36] = [100, 100, 1.0] # right outer
        self.lms[39] = [120, 100, 1.0] # right inner
        self.lms[66] = [110, 100, 1.0] # right pupil center
        self.lms[42] = [150, 100, 1.0] # left inner
        self.lms[45] = [170, 100, 1.0] # left outer
        self.lms[67] = [160, 100, 1.0] # left pupil center

def test_mapping_vrm_vowels_and_expressions():
    from openseeface_tracker import map_openseeface_to_vrm, map_openseeface_to_vrc, extract_head_pose, extract_iris_gaze
    face = MockFace()
    vrm = map_openseeface_to_vrm(face)
    vrc = map_openseeface_to_vrc(face)
    pose = extract_head_pose(face, pitch_offset_deg=18.0)
    gaze = extract_iris_gaze(face)

    # Check VRM blendshapes
    assert "a" in vrm
    assert "i" in vrm
    assert "u" in vrm
    assert "e" in vrm
    assert "o" in vrm
    assert "blink" in vrm
    assert "joy" in vrm
    assert vrm["a"] > 0.3  # mouth is open
    assert vrm["joy"] > 0.2  # smile corners up
    assert vrm["blink"] < 0.2  # eyes are open

    # Check VRChat morphs
    assert "vrc_v_aa" in vrc
    assert "vrc_blink" in vrc

    # Check pose
    assert isinstance(pose["pitch"], float)
    assert isinstance(pose["yaw"], float)
    assert isinstance(pose["roll"], float)

    # Check gaze
    assert -1.0 <= gaze["x"] <= 1.0
    assert -1.0 <= gaze["y"] <= 1.0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_openseeface.py -k test_mapping -v`
Expected: FAIL with `ImportError: cannot import name 'map_openseeface_to_vrm'`

- [ ] **Step 3: Implement mapping logic in `src/openseeface_tracker.py`**

Implement:
- `_clamp(val, min_v, max_v)`
- `map_openseeface_to_vrm(face, config=None)`:
  - Natural resting mouth stability (neutral when `mouth_open < deadzone`)
  - Vowel synthesis: `a` (tall opening), `i` (wide stretch), `u` (pucker), `e` (wide + open), `o` (round with `o_max_clamp`)
  - Emotion blendshapes: `joy` (mouth corners raised), `sorrow` (eyebrow steepness / frown), `surprised` (brows up + open mouth), `fun`
  - Blink mapping: `1.0 - eye_openness`, threshold snap for eyeglasses
- `map_openseeface_to_vrc(face)`: VRChat viseme targets
- `extract_head_pose(face, pitch_offset_deg=18.0)`: maps Euler angles `pitch = euler[0] + pitch_offset_deg`, `yaw = -euler[1]`, `roll = euler[2]`
- `extract_iris_gaze(face)`: relative displacement of pupil landmarks (66, 67) within eye bounds (36-41, 42-47)
- Wire into `OpenSeeFaceTracker.process_frame(frame, config)` to return `(vrm_shapes, vrc, raw_rotation, raw_gaze)`

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_openseeface.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/openseeface_tracker.py tests/test_openseeface.py
git commit -m "feat(openseeface): implement VRM and VRChat blendshape and head pose mapping"
```

---

### Task 3: Configuration System & Launcher GUI Backend Selector

**Files:**
- Modify: `src/config_manager.py`
- Modify: `config.txt`
- Modify: `src/config_gui.py`
- Modify: `tests/test_config.py`

**Interfaces:**
- Produces:
  - `DEFAULT_CONFIG["tracker_backend"] = "mediapipe"`
  - `validate_config()` validates `tracker_backend` in `{"mediapipe", "openseeface"}`
  - `ConfigLauncherDialog` displays backend selection dropdown in General Tab

- [ ] **Step 1: Write failing test in `tests/test_config.py` for `tracker_backend`**

```python
def test_tracker_backend_config(self):
    from config_manager import DEFAULT_CONFIG, load_config, validate_config
    self.assertIn("tracker_backend", DEFAULT_CONFIG)
    self.assertEqual(DEFAULT_CONFIG["tracker_backend"], "mediapipe")

    # Validation succeeds for valid backends
    valid_cfg = validate_config({"tracker_backend": "openseeface"})
    self.assertEqual(valid_cfg["tracker_backend"], "openseeface")

    # Validation falls back for invalid backend
    invalid_cfg = validate_config({"tracker_backend": "unknown_ai"})
    self.assertEqual(invalid_cfg["tracker_backend"], "mediapipe")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_config.py -k test_tracker_backend -v`
Expected: FAIL with `AssertionError: 'tracker_backend' not found in DEFAULT_CONFIG`

- [ ] **Step 3: Update `src/config_manager.py`, `config.txt`, and `src/config_gui.py`**

In `src/config_manager.py`:
- Add `"tracker_backend": "mediapipe"` to `DEFAULT_CONFIG` under `[GENERAL]`
- Add validation in `validate_config` ensuring it is lowercased and in `{"mediapipe", "openseeface"}`, defaulting to `"mediapipe"`
- Update `generate_default_config_text` with explanatory comments in Vietnamese

In `config.txt`:
- Add `tracker_backend = mediapipe` under `[GENERAL]` with comments explaining `mediapipe` vs `openseeface`

In `src/config_gui.py`:
- Add Backend Selector `QComboBox` in `_create_tab_general()`:
  - Item 0: `"Google MediaPipe (Mặc định - Nhẹ, mượt mọi cấu hình)"`, data: `"mediapipe"`
  - Item 1: `"OpenSeeFace ONNX (Chuyên dụng VTuber - Độ chuẩn xác cao)"`, data: `"openseeface"`
- Wire `tracker_backend` in `_load_values_to_ui()` and `get_config()`

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_config.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/config_manager.py config.txt src/config_gui.py tests/test_config.py
git commit -m "feat(config): add tracker_backend option and GUI backend selector"
```

---

### Task 4: Integrate Dual-Backend Tracking in `TrackingWorker`

**Files:**
- Modify: `src/vrm_renderer.py`
- Modify: `tests/test_vrm_renderer.py`

**Interfaces:**
- Consumes: `tracker_backend` from `config`
- Produces: `TrackingWorker.run()` conditionally initializes and processes frames with `OpenSeeFaceTracker` or `create_face_landmarker`

- [ ] **Step 1: Write test verifying backend selection in `TrackingWorker`**

In `tests/test_vrm_renderer.py`:
```python
def test_tracking_worker_backend_selection(self):
    try:
        from vrm_renderer import TrackingWorker
    except ImportError:
        from src.vrm_renderer import TrackingWorker

    worker_mp = TrackingWorker(camera_id=0, config={"tracker_backend": "mediapipe"})
    self.assertEqual(worker_mp.backend, "mediapipe")

    worker_osf = TrackingWorker(camera_id=0, config={"tracker_backend": "openseeface"})
    self.assertEqual(worker_osf.backend, "openseeface")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_vrm_renderer.py -k test_tracking_worker_backend_selection -v`
Expected: FAIL with `AttributeError: 'TrackingWorker' object has no attribute 'backend'`

- [ ] **Step 3: Update `src/vrm_renderer.py` with dual-backend support**

In `src/vrm_renderer.py`:
- Store `self.backend = str(self.config.get("tracker_backend", "mediapipe")).lower()` in `TrackingWorker.__init__`
- In `TrackingWorker.run()`:
  - If `self.backend == "openseeface"`:
    - Emit `"Initializing webcam & OpenSeeFace ONNX..."`
    - Initialize `OpenSeeFaceTracker(w, h, max_threads=2)`
    - On each frame: `tracked = tracker.process_frame(frame, self.config)`
    - If `tracked` is not None:
      - Unpack `vrm_shapes, vrc, raw_rotation, raw_gaze`
      - Pass through `OneEuroFilter` (rotation & gaze)
      - Apply gaze freeze on blink
      - Format `packet` with `format_motion_packet` and emit `self.motion_ready.emit(packet)`
    - If `tracked` is None:
      - Reset filters and emit neutral packet with decay
  - If `self.backend == "mediapipe"`:
    - Keep current MediaPipe FaceLandmarker logic intact

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_vrm_renderer.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/vrm_renderer.py tests/test_vrm_renderer.py
git commit -m "feat(renderer): integrate OpenSeeFace dual-backend into TrackingWorker"
```

---

### Task 5: End-to-End Verification & Documentation

**Files:**
- Modify: `README.md`
- Run test suite: `pytest tests/`

- [ ] **Step 1: Run complete test suite**

Run: `./venv/bin/pytest tests/ -v`
Expected: All tests PASS with 0 failures

- [ ] **Step 2: Verify launcher and tracker initialization on synthetic test images**

In `tests/test_openseeface.py`, test full cycle with a generated test frame containing synthetic face points to guarantee end-to-end packet emission.

- [ ] **Step 3: Update `README.md`**

Add section detailing:
- AI Tracking Backend: Google MediaPipe vs OpenSeeFace ONNX
- How to switch between backends in GUI and `config.txt`
- Characteristics of each backend (MediaPipe = ultra-lightweight; OpenSeeFace = maximum precision VTuber tracking)

- [ ] **Step 4: Commit**

```bash
git add README.md tests/
git commit -m "docs: update documentation for OpenSeeFace dual-backend tracking"
```
