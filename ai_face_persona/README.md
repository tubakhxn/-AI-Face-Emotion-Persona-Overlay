# AI Face Emotion & Persona Overlay

Created with love by tubakhxn

Realtime webcam app that detects faces with MediaPipe, infers emotions (fast heuristics or optional DL), and draws a cyberpunk neon HUD with persona labels and screenshot support.

This README collects everything you need to run, tune, and publish the project on GitHub.

---

## Table of contents

- Features
- Requirements & notes
- Quick start (recommended: lightweight / ONNX)
- Full DL install (optional)
- How it works (architecture)
- Running the app (detailed)
- Keyboard controls & runtime tuning
- Adding an ONNX model (optional)
- Troubleshooting
- Packaging for GitHub (tips)
- Contributing & license

---

## Features

- Real-time webcam (OpenCV)
- Face detection + dense landmarks (MediaPipe Face Mesh)
- Fast image-based emotion heuristics (default, low-latency)
- Optional DL inference via ONNX (recommended) or DeepFace/TensorFlow (optional)
- Cyberpunk neon HUD (glow, rounded rect, animated scanline, glitch header text)
- FPS counter, smooth label/bbox interpolation, fade-on-change animation
- Screenshot saving (press `S`) with a short shutter sound on Windows

---

## Requirements & notes

- OS: Windows (tested). Linux/macOS should work but audio uses winsound on Windows only.
- Python: 3.10+ (3.11 recommended)
- Camera: any webcam supported by OpenCV

This project lives inside the `ai_face_persona/` folder. To avoid import errors run `main.py` from that folder or use the module runner (examples below).

---

## Quick start (recommended: lightweight / ONNX)

Run these PowerShell commands from the repository root.

```powershell
# Create a venv and upgrade pip
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip

# Install lightweight dependencies (includes onnxruntime for optional DL)
.venv\Scripts\python -m pip install -r .\ai_face_persona\requirements_nodl.txt

# Run the app from the package folder so local imports resolve
cd ai_face_persona
..\.venv\Scripts\python main.py
```

Notes:

- Default behavior: the app runs in a fast landmark-based heuristic mode (`mode='image'`) so it's smooth on most machines.
- To enable DL inference in-app, press `d`. The app will prefer ONNX (no TensorFlow required) if `onnxruntime` and a model file are present.

---

## Full DL install (optional)

If you want DeepFace (TensorFlow) or full Hugging Face text-classifier support, install the full set:

```powershell
.venv\Scripts\python -m pip install -r .\ai_face_persona\requirements.txt
```

Warning: DeepFace pulls TensorFlow which is large and may have native-ABI issues on Windows. If you see crashes, try `pip install tensorflow-cpu` for a CPU-only wheel compatible with your Python version, or prefer ONNX.

---

## How it works (architecture)

- `main.py` — webcam loop, runs detection, queries `EmotionModel`, and draws overlays.
- `face_detector.py` — wrapper around MediaPipe Face Mesh. Returns bounding box and landmark coordinates in pixel space.
- `emotion_model.py` — image heuristics (fast), optional Hugging Face text adapter, optional DL via ONNX/DeepFace. Implements temporal smoothing and persona mapping.
- `overlay_utils.py` — HUD drawing primitives (rounded glow rectangles, scanline animation, glitch text, status panel, FPS, screenshot save).

---

## Running the app (tips & examples)

- Run from package folder (recommended):

```powershell
cd ai_face_persona
..\.venv\Scripts\python main.py
```

- Run as a module (works from repo root):

```powershell
.venv\Scripts\python -m ai_face_persona.main
```

- If you get `ModuleNotFoundError: No module named 'face_detector'`, ensure you ran the command from the `ai_face_persona` folder or use the `-m` module invocation above.

---

## Keyboard controls & runtime tuning

- `S` — save screenshot (written to `screenshots/` with timestamp). Plays a short shutter sound on Windows.
- `+` / `=` — increase label smoothing (makes displayed label less jittery)
- `-` — decrease label smoothing (more reactive)
- `]` / `[` — increase / decrease bounding-box lerp (affects HUD follow speed)
- `d` — toggle DL inference on/off (OFF by default)
- `m` — switch DL backend between `onnx` and `deepface` (if available)
- `ESC` — exit the app

There is a small status panel in the HUD that shows smoothing and DL backend state.

---

## Adding an ONNX model (recommended for DL)

To use DL without installing TensorFlow, drop an ONNX emotion model at:

```
ai_face_persona/assets/emotion_model.onnx
```

Behavior:

- If DL is enabled and `emotion_model.onnx` exists, the app will use `onnxruntime` for inference.
- If the file is missing the app attempts to download a default FER+ ONNX model into `assets/` (network required). If the download fails it falls back to heuristics.

Model format notes:

- The included ONNX helper expects a FER+ style model (grayscale 64x64). If you provide a different model, update `ai_face_persona/emotion_model.py` preprocessing and label mapping accordingly.

---

## Troubleshooting

- Camera errors: make sure no other program is using the webcam and check Windows Privacy > Camera to allow desktop apps.
- App won't start / `ModuleNotFoundError`: run from the `ai_face_persona` folder or use `python -m ai_face_persona.main`.
- ONNX missing: install `onnxruntime` in the venv (it's included in `requirements_nodl.txt`) or provide your model file.
- DeepFace/TensorFlow crashes: try `pip install tensorflow-cpu` for a CPU-only wheel compatible with your Python version, or stick to ONNX.

---

## Packaging for GitHub

- Add an explicit `LICENSE` (MIT recommended). I can add one for you if you'd like.
- Add a screenshot or short GIF to the README to show the HUD — place images in `assets/` and reference them from the repo README.
- Keep `requirements_nodl.txt` for demo installs and `requirements.txt` for full DL installs.

---

## Contributing

PRs welcome. Suggested improvements:

- Add a pre-trained ONNX model and label mapping for better emotion accuracy.
- Add small unit tests for import and smoke tests.
- Improve UI polish and add more persona mappings.

---

## License

This project is provided as-is. If you want, I can add an `LICENSE` file (MIT) to the repository.

---

Created with love by tubakhxn

# AI Face Emotion & Persona Overlay

Created with love by tubakhxn

A realtime webcam application that detects faces (MediaPipe), infers an emotion (fast landmark heuristics or optional DL via ONNX/DeepFace), and draws a cyberpunk neon HUD with persona overlays and screenshot support.

This README contains everything you need to run, tweak, and publish the project on GitHub.

## Table of contents
- Features
- Requirements & notes
- Quick start (recommended: lightweight / ONNX)
- Full DL install (optional)
- How it works (architecture)
- Running the app (detailed)
- Keyboard controls & runtime tuning
- Adding an ONNX model (optional)
- Troubleshooting
- Packaging for GitHub (tips)
- Contributing & license

## Features
- Real-time webcam (OpenCV)
- Face detection + dense landmarks (MediaPipe Face Mesh)
- Fast image-based emotion heuristics (default, low-latency)
- Optional DL inference via ONNX (recommended) or DeepFace/TensorFlow
- Cyberpunk neon HUD (glow, rounded rect, scanline, glitch text)
- FPS counter, smooth label/bbox interpolation, fade-on-change animation
- Screenshot saving (press `S`) with a shutter sound on Windows

## Requirements & notes
- OS: Windows (tested), other OS should work but audio uses winsound on Windows only
- Python: 3.10+ (3.11 recommended)
- Camera: any webcam supported by OpenCV

Files of interest are inside the `ai_face_persona/` folder. When running, prefer executing from within that folder so relative imports resolve.

## Quick start (recommended: lightweight / ONNX)
1. Open PowerShell in the project root (where this README is).
2. Create a virtual environment and install lightweight (ONNX-capable) dependencies:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r .\ai_face_persona\requirements_nodl.txt
```

3. Run the app from the `ai_face_persona` folder:

```powershell
cd ai_face_persona
..\.venv\Scripts\python main.py
```

Created with love by tubakhxn
# AI Face Emotion & Persona Overlay

Build by Tuba khan

This project shows a real-time webcam feed with face detection (MediaPipe), an
emotion classifier (Hugging Face "joeddav/distilbert-base-uncased-go-emotions"), and
an animated cyberpunk neon HUD overlay.

Features
- Real-time webcam using OpenCV
- Face detection + landmarks using MediaPipe Face Mesh
- Emotion recognition using Hugging Face (adapter from facial heuristics)
- Cyberpunk neon HUD (glow, rounded face rectangle, moving scanline, glitch text)
- FPS display
- Press `S` to save a screenshot with overlays (plays a short sound)

Quick start (Windows)
1. Create and activate a virtual environment (PowerShell):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
python -m pip install -r requirements.txt
```

3. Run the app:

```powershell
python main.py
```

Notes
- The first time you run the app, the Hugging Face model will be downloaded (internet required).
- If the model can't be loaded the app will fallback gracefully to a neutral persona.
- If you encounter camera access errors, ensure other apps are not using the camera.

Files
- `main.py` - app entrypoint
- `face_detector.py` - MediaPipe Face Mesh wrapper
- `emotion_model.py` - Hugging Face classifier adapter + persona mapping
- `overlay_utils.py` - HUD drawing functions
- `assets/` - placeholder assets (hud image & font)

License & attribution
This project is provided as-is for demo purposes.
