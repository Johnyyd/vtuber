# Desktop VTuber 3D Avatar & Facial Mapping Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a desktop application that tracks facial landmarks and blendshapes from the webcam in real time, maps them to the 3D VRM model's 16 morph targets and head bones, and renders the avatar inside a native desktop window using direct in-memory IPC.

**Architecture:** A Python desktop application (`PyQt6` / `QWebEngineView`) embedding Three.js and `@pixiv/three-vrm` in a dedicated native window. Facial tracking runs in a background worker thread using MediaPipe; each frame's calculated VRC visemes, eye blink, and head Euler angles are pushed directly into the JS runtime via `runJavaScript` (zero WebSockets, zero open ports).

**Tech Stack:** Python 3.11, MediaPipe, OpenCV, PyQt6, PyQt6-WebEngine, Three.js, @pixiv/three-vrm

**Spec:** docs/superpowers/specs/2026-10-07-desktop-vrm-mapping-design.md

## Global Constraints
- Target environment: Linux / Wayland with Python 3.11 inside `./venv`
- Zero WebSockets and zero network ports (direct in-memory IPC only via `QWebEnginePage.runJavaScript`)
- Complete offline capability (all Three.js and VRM libs bundled locally under `viewer/libs/`)
- Exact mapping to `Mesh_InteriorMouth2`'s 16 VRC morph targets in `assets/character.vrm`
- Clean shutdown on window close / `Esc` / `Ctrl+C`

---

### Task 1: VRC Morph Target Mapping & Head Pose Computation in `src/landmark_mapping.py`

**Files:**
- Modify: `src/landmark_mapping.py`
- Test: `tests/test_landmark_mapping.py`

**Interfaces:**
- Consumes:
  - MediaPipe blendshapes dict: `{"eyeBlinkLeft": float, "jawOpen": float, ...}`
  - MediaPipe landmarks dict: `{"lm0": (x, y, z), ...}`
- Produces:
  - `map_mediapipe_to_vrc(mp_blendshapes: dict) -> dict` returning 16 VRC morph targets:
    `{"vrc_blink": float, "vrc_v_aa": float, "vrc_v_ee": float, "vrc_v_ih": float, "vrc_v_oh": float, "vrc_v_ou": float, "vrc_v_sil": float, ...}`
  - `compute_head_pose(landmarks: dict, frame_shape: tuple = (480, 640)) -> dict` returning:
    `{"pitch": float, "yaw": float, "roll": float}`

- [ ] **Step 1: Write the failing test for VRC morph target mapping and head pose**

Create `tests/test_landmark_mapping.py`:
```python
import pytest
from landmark_mapping import map_mediapipe_to_vrc, compute_head_pose

def test_vrc_mapping_blink_and_mouth():
    mp_input = {
        "eyeBlinkLeft": 0.8,
        "eyeBlinkRight": 0.8,
        "jawOpen": 0.7,
        "mouthStretchLeft": 0.1,
        "mouthStretchRight": 0.1,
    }
    vrc = map_mediapipe_to_vrc(mp_input)
    assert "vrc_blink" in vrc
    assert "vrc_v_aa" in vrc
    assert "vrc_v_sil" in vrc
    assert 0.7 <= vrc["vrc_blink"] <= 1.0
    assert 0.5 <= vrc["vrc_v_aa"] <= 1.0

def test_compute_head_pose_neutral():
    # Canonical face centered
    landmarks = {
        "lm1": (0.5, 0.5, 0.0),    # nose tip
        "lm152": (0.5, 0.7, 0.0),  # chin
        "lm33": (0.4, 0.45, 0.0),  # left eye outer
        "lm263": (0.6, 0.45, 0.0), # right eye outer
        "lm61": (0.42, 0.6, 0.0),  # left mouth
        "lm291": (0.58, 0.6, 0.0), # right mouth
    }
    pose = compute_head_pose(landmarks)
    assert "pitch" in pose and "yaw" in pose and "roll" in pose
    assert abs(pose["pitch"]) < 0.3
    assert abs(pose["yaw"]) < 0.3
    assert abs(pose["roll"]) < 0.3
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_landmark_mapping.py -v`  
Expected: FAIL (ImportError or AttributeError: `map_mediapipe_to_vrc` / `compute_head_pose` not defined).

- [ ] **Step 3: Implement mapping and head pose algorithms in `src/landmark_mapping.py`**

Update `src/landmark_mapping.py` with:
- `VRC_TARGETS` list matching the 16 morph targets of `character.vrm`.
- `map_mediapipe_to_vrc(mp_blendshapes: dict) -> dict` implementing the formulas defined in the design spec.
- `compute_head_pose(landmarks: dict, frame_shape: tuple = (480, 640)) -> dict` using `cv2.solvePnP` with 3D canonical face coordinates or vector geometry to compute pitch, yaw, roll (bounded within [-0.5, 0.5] rad for pitch, [-0.7, 0.7] rad for yaw, [-0.4, 0.4] rad for roll).

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_landmark_mapping.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add src/landmark_mapping.py tests/test_landmark_mapping.py
git commit -m "feat: add VRC blendshape mapping and head pose calculation"
```

---

### Task 2: Bundle Three.js and Three-VRM Offline Libraries

**Files:**
- Create: `viewer/libs/three.min.js`
- Create: `viewer/libs/three-vrm.min.js`
- Test: `tests/test_viewer_assets.py`

**Interfaces:**
- Produces: Standalone local Javascript libraries so the 3D desktop viewer works completely offline.

- [ ] **Step 1: Write test to verify library files exist and are non-empty**

Create `tests/test_viewer_assets.py`:
```python
import os

def test_viewer_libraries_exist():
    three_path = "viewer/libs/three.min.js"
    vrm_path = "viewer/libs/three-vrm.min.js"
    assert os.path.exists(three_path) and os.path.getsize(three_path) > 100000
    assert os.path.exists(vrm_path) and os.path.getsize(vrm_path) > 50000
```

- [ ] **Step 2: Run test to verify it fails**

Run: `./venv/bin/pytest tests/test_viewer_assets.py -v`  
Expected: FAIL (files do not exist).

- [ ] **Step 3: Download and bundle local Three.js and Three-VRM files**

Download stable UMD builds of `three.min.js` (r154+) and `three-vrm.min.js` (v2.x) directly to `viewer/libs/` using python `urllib.request`.

- [ ] **Step 4: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_viewer_assets.py -v`  
Expected: PASS.

- [ ] **Step 5: Commit changes**

```bash
git add viewer/libs/ tests/test_viewer_assets.py
git commit -m "feat: bundle local three.js and three-vrm libraries for offline rendering"
```

---

### Task 3: Build 3D Studio Desktop Viewport (`viewer/index.html`, `viewer/style.css`, `viewer/app.js`)

**Files:**
- Create: `viewer/index.html`
- Create: `viewer/style.css`
- Create: `viewer/app.js`

**Interfaces:**
- Consumes:
  - Local model: `assets/character.vrm`
  - In-memory call: `window.updateMotion(data)` where `data` is `{"vrc": {...}, "rotation": {"pitch": p, "yaw": y, "roll": r}}`
- Produces:
  - 3D WebGL scene with Three.js WebGLRenderer.
  - Smooth animation loop (60 FPS) with LERP interpolation for VRC morph targets and head/neck bone rotations.

- [ ] **Step 1: Create `viewer/index.html` and `viewer/style.css`**

Build clean UI with dark studio background, canvas, loading indicator, and debug status bar.
Include local script tags:
```html
<script src="libs/three.min.js"></script>
<script src="libs/three-vrm.min.js"></script>
<script src="app.js"></script>
```

- [ ] **Step 2: Create `viewer/app.js`**

Implement:
- Scene, PerspectiveCamera, WebGLRenderer, studio lighting (Key, Ambient, Rim).
- VRM loading via `GLTFLoader` with `VRMLoaderPlugin` pointing to `../assets/character.vrm`.
- Cache `Mesh_InteriorMouth2` morph target dictionary and bone references (`mixamorig:Head`, `mixamorig:Neck`).
- `window.updateMotion(data)` function receiving the latest target values from Python.
- Animation loop (`requestAnimationFrame`) with LERP smoothing:
  - `currentMorph[k] += (targetMorph[k] - currentMorph[k]) * lerpMorph`
  - Neck rotation: `pitch * 0.3`, `yaw * 0.3`, `roll * 0.3`
  - Head rotation: `pitch * 0.7`, `yaw * 0.7`, `roll * 0.7`
  - Handles face-loss timeout: if no motion received for 1 second, smoothly return to rest pose (`vrc_v_sil = 1.0`, zero rotation).

- [ ] **Step 3: Verify syntax and file structure**

Check files exist and have valid structure using python syntax/file checks.

- [ ] **Step 4: Commit changes**

```bash
git add viewer/
git commit -m "feat: create 3D studio viewport with VRM loader and LERP smoothing"
```

---

### Task 4: Implement Desktop Native GUI & In-Memory IPC Runner in `src/vrm_renderer.py`

**Files:**
- Modify: `src/vrm_renderer.py`
- Test: `tests/test_vrm_renderer.py`

**Interfaces:**
- Consumes:
  - `face_detector.py` (`create_face_landmarker`, `parse_landmarks`, `parse_blendshapes`)
  - `landmark_mapping.py` (`map_mediapipe_to_vrc`, `compute_head_pose`)
- Produces:
  - `VTuberApp` class running `QMainWindow` with `QWebEngineView`.
  - `TrackingWorker(QThread)` capturing webcam and computing motion packets.
  - Direct call: `page.runJavaScript(f"window.updateMotion({json_data});")`.

- [ ] **Step 1: Write headless test for `TrackingWorker` data formatting**

Create `tests/test_vrm_renderer.py`:
```python
import json
from landmark_mapping import map_mediapipe_to_vrc, compute_head_pose

def test_motion_packet_json_serializable():
    dummy_vrc = map_mediapipe_to_vrc({"jawOpen": 0.5})
    dummy_pose = {"pitch": 0.1, "yaw": -0.2, "roll": 0.05}
    packet = {"vrc": dummy_vrc, "rotation": dummy_pose}
    encoded = json.dumps(packet)
    assert "vrc_v_aa" in encoded
    assert "pitch" in encoded
```

- [ ] **Step 2: Run test to verify it passes**

Run: `./venv/bin/pytest tests/test_vrm_renderer.py -v`  
Expected: PASS.

- [ ] **Step 3: Implement `src/vrm_renderer.py` with PyQt6 Desktop GUI**

Implement:
- `TrackingWorker(QThread)`:
  - Runs webcam capture in background thread.
  - Detects landmarks & blendshapes with MediaPipe.
  - Maps to VRC targets and head pose.
  - Emits Qt signal `motion_ready(dict)` with serialized motion payload.
- `VTuberWindow(QMainWindow)`:
  - Embeds `QWebEngineView`.
  - Sets window title to "VTuber 3D Avatar", window size 1024x768.
  - Enables local file access flags in `QWebEngineSettings`.
  - Loads `viewer/index.html` via `QUrl.fromLocalFile(...)`.
  - Slots `motion_ready` to call `self.web_view.page().runJavaScript(f"window.updateMotion({json_str});")`.
  - Intercepts close event to gracefully terminate `TrackingWorker` and camera.

- [ ] **Step 4: Commit changes**

```bash
git add src/vrm_renderer.py tests/test_vrm_renderer.py
git commit -m "feat: implement PyQt6 native desktop window with in-memory IPC"
```

---

### Task 5: Update Entry Point `src/main.py` and Project Setup

**Files:**
- Modify: `src/main.py`
- Modify: `requirements.txt`
- Modify: `README.md`

**Interfaces:**
- Produces: Single command launcher `python src/main.py` launching the full desktop app.

- [ ] **Step 1: Update `src/main.py`**

Replace old stub loop in `src/main.py` with:
- PyQt6 `QApplication` initialization.
- Creation of `VTuberWindow`.
- Support for CLI arguments (e.g. `--camera-id`, `--width`, `--height`).
- Clean exit code handling with `sys.exit(app.exec())`.

- [ ] **Step 2: Update `requirements.txt` and `README.md`**

Ensure `requirements.txt` lists:
```text
mediapipe>=0.10.35
opencv-python>=4.7.0.72
numpy
pygltflib>=1.16
PyQt6>=6.5.0
PyQt6-WebEngine>=6.5.0
```
Update `README.md` with instructions on running the native desktop application.

- [ ] **Step 3: Commit changes**

```bash
git add src/main.py requirements.txt README.md
git commit -m "feat: wire up main entry point and update documentation"
```

---

### Task 6: Comprehensive Verification & End-to-End Validation

**Files:**
- All touched files

- [ ] **Step 1: Run full test suite**

Run: `./venv/bin/pytest tests/ -v`  
Expected: All tests PASS.

- [ ] **Step 2: Test model file and mapping completeness**

Run a Python script checking that every VRC morph target key mapped in `src/landmark_mapping.py` matches the morph target names in `assets/character.vrm`.

- [ ] **Step 3: Perform short execution dry-run**

Run `python src/main.py --test-mode` or run headless verification to ensure no syntax errors or runtime import crashes.

- [ ] **Step 4: Final commit and cleanup**

```bash
git status
git commit -am "chore: complete desktop VTuber mapping implementation"
```
