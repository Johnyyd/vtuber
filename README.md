# VTuber Facial Tracking & 3D Desktop Avatar

A real-time VTuber desktop application that tracks facial expressions and head movements from your webcam, maps them to a 3D VRM model's morph targets and humanoid bones, and renders the avatar inside a native desktop window using direct in-memory IPC (no WebSockets, no open network ports).

---

## 🔥 Features

- **Live Webcam Facial Tracking**: Uses MediaPipe FaceLandmarker capturing 478 landmarks and 52 ARKit facial blendshapes at 720p resolution for high tracking stability.
- **Accurate Viseme Mapping**: High-sensitivity mapping for standard VRM blendshapes (`a`, `i`, `u`, `e`, `o`, `blink`, `joy`, `angry`, `sorrow`, `fun`). Handles thick glasses and low-light environments robustly.
- **Mirror Mode Rotation**: Head rotation (Pitch, Yaw, Roll) and Eye Gaze are mirrored (left becomes left) so the avatar moves exactly like a reflection in a mirror.
- **Zero-Network Desktop Architecture**: Runs inside a native frameless desktop window (PyQt6) exchanging tracking data directly in RAM (`runJavaScript`), operating 100% offline.

---

## 🚀 Getting Started

### 1. Requirements

- Linux / macOS / Windows
- Python 3.10+
- Webcam (or Virtual Webcam via Phone for professional quality)

### 2. Installation

```bash
# Create and activate your virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 3. Usage

The easiest way to start the application is using the provided shell script:

```bash
bash run.sh
```

**Advanced Options:**
You can specify the camera ID, width, and height of the window:
```bash
bash run.sh --camera-id 0 --width 1280 --height 720
```

### 4. Controls
- **Click & Drag**: Move the frameless window around your desktop.
- **Esc** or **Q**: Close the window and exit safely.

---

## 📸 Pro-Tip: Using Your Phone as a Webcam

For the absolute best tracking quality (especially if you wear glasses), it is highly recommended to use your phone's camera via **DroidCam** or **Iriun Webcam**.

**Setup on Linux (Arch/Ubuntu):**
1. Install DroidCam on your phone and PC.
2. Ensure the Virtual Webcam driver is active:
   ```bash
   sudo modprobe v4l2loopback_dc
   ```
   *(Note: If DKMS fails to build the module, make sure you have the `linux-headers` package installed that matches your exact running kernel version, then reboot).*
3. Connect DroidCam via USB or WiFi.
4. Launch the VTuber software targeting the newly created virtual webcam (usually ID 1 or 2):
   ```bash
   bash run.sh --camera-id 1
   ```

---

## 🎨 Changing Your Avatar

To load a custom `.vrm` character:
1. Place your `MyCharacter.vrm` file inside the `assets/` folder.
2. Open `viewer/app.js` in a text editor.
3. Find the `loadModel` function around line 122:
   ```javascript
   // Change this line to point to your new model:
   loadModel("../assets/MyCharacter.vrm");
   ```
4. Restart the application.

---

## 🧠 Architecture

```
Webcam -> MediaPipe FaceLandmarker -> Python VRC/VRM Mapper -> In-Memory IPC -> Three.js 3D Viewport
```

- `src/face_detector.py`: Webcam frame acquisition & MediaPipe landmark extraction (configured for 720p).
- `src/landmark_mapping.py`: Calculates VRM blendshapes, applies non-linear deadzones, and computes 3D head rotation.
- `src/vrm_renderer.py`: PyQt6 native desktop window hosting the 3D viewport.
- `viewer/app.js`: 3D rendering pipeline using Three.js and `@pixiv/three-vrm`. Applies LERP smoothing and Mirror Mode inversions.