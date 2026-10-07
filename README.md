# VTuber Facial Tracking & 3D Desktop Avatar

A real-time VTuber desktop application that tracks facial expressions and head movements from your webcam, maps them to the 3D VRM model's morph targets and humanoid bones, and renders the avatar inside a native desktop window using direct in-memory IPC (no WebSockets, no open network ports).

---

## Features

- **Live Webcam Facial Tracking**: Uses MediaPipe FaceLandmarker (video mode) capturing 478 landmarks and 52 ARKit facial blendshapes.
- **Accurate Morph Target Mapping**: Maps MediaPipe blendshapes to standard VRM 0.x blendshape presets for `assets/Yong.vrm` (`a`, `i`, `u`, `e`, `o`, `blink`, `blink_l`, `blink_r`, `joy`, `angry`, `sorrow`, `fun`) as well as VRChat morph targets.
- **3D Head Pose Computation**: Computes natural 3D head rotation (Pitch, Yaw, Roll) to drive neck (`J_Bip_C_Neck`) and head (`J_Bip_C_Head`) humanoid bones with natural looking direction.
- **Physics & Secondary Motion**: Supports VRM Spring Bones for hair and clothing physics.
- **Smooth 60 FPS Viewport**: Studio 3D rendering with Three.js and `@pixiv/three-vrm` with LERP smoothing for jitter-free animation.
- **Zero-Network Desktop Architecture**: Runs inside a native desktop window (PyQt6 / QWebEngineView) exchanging tracking data directly in RAM (`runJavaScript`), 100% offline.

---

## Getting Started

### 1. Requirements

- Linux / macOS / Windows
- Python 3.10+
- Webcam

### 2. Installation

```bash
# Activate your virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Run Application

```bash
python src/main.py
```

Options:
```bash
python src/main.py --camera-id 0 --width 1280 --height 720
```

### 4. Controls

- **Esc** or **Q**: Close the window and exit safely.

---

## Architecture

```
Webcam -> MediaPipe FaceLandmarker -> VRC Morph & Head Pose Mapper -> In-Memory IPC -> Three.js 3D Viewport
```

- `src/face_detector.py`: Webcam frame acquisition & MediaPipe landmark extraction.
- `src/landmark_mapping.py`: Calculates 16 VRC morph targets and head pose Euler angles.
- `src/vrm_renderer.py`: PyQt6 native desktop window hosting the 3D viewport.
- `src/main.py`: Application entry point.
- `viewer/`: 3D studio viewport (HTML, CSS, Three.js, Three-VRM).