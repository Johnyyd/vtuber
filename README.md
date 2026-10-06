# VTuber Facial-Landmark → VRM Demo

A tiny Python project that:

1. Captures webcam input.
2. Uses **MediaPipe Face Mesh** (478 landmarks).
3. Maps a subset of landmarks to **VRM blendshapes**.
4. Renders the model with the chosen blendshapes.

> **Note** – The current `vrm_renderer.py` is a **stub**; it simply prints the blendshape values.
> Plug in your own renderer (Unity, Three.js, PyOpenGL, etc.) to see the model move.

## Getting Started

```bash
# (1) Install dependencies
pip install -r requirements.txt

# (2) Put your `.vrm` file into a folder named `resource` at the root of the repo
# (3) Run the demo
python src/main.py
```

Use **Esc** to exit the window.

## Customisation

- **Add more blendshapes** – extend `src/landmark_mapping.py`.
- **Replace the renderer** – modify `src/vrm_renderer.py` to call your engine.
- **Smooth landmark data** – implement filtering in `parse_landmarks` or the mapping function.

## Credits

- MediaPipe – https://google.github.io/mediapipe/
- pyvrm – https://github.com/nyatla/pyvrm