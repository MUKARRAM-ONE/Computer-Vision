# Computer Vision Projects by MUKARRAM-ONE

[---]

> **Computer Vision**: Machine understanding of images & video—object detection, gesture tracking, emotion recognition, etc.  
> A portfolio of Python / OpenCV / MediaPipe projects with live demos and clean code.

---

## 🔧 Projects

| Project | Description | Live Demo | Tech / Skills |
|---------|-------------|-----------|---------------|
| **Invisible Cloak** | Use color segmentation to create Harry Potter-style cloak effect. | [Streamlit Demo](link-here) | OpenCV, HSV masks, real-time video |
| **Finger Counter** | Count number of fingers shown in camera feed. | [Streamlit Demo](your-link) | Contour detection, convexity defects |
| **Hand Tracking / Gesture Recognition** | Track complex hand movements and gestures. | [Demo](...) | MediaPipe Hands, gesture mapping |
| **Face & Emotion Detector** | Detect face, classify basic emotions (happy, sad, neutral, etc.). | [Demo](...) | Haar cascades or DNN, small emotion model |

---

## Requirements

To run the projects locally, install the following dependencies:

- **Python 3.8+**
- **Libraries:**
  - opencv-python
  - numpy
  - mediapipe (for hand tracking, optional)
  - streamlit (for web apps)
  - torch / tensorflow (for emotion recognition models)
  - scikit-learn
  - matplotlib (for visualizations)


## 📥 Installation & Running Locally

```bash
git clone https://github.com/MUKARRAM-ONE/Computer-Vision.git
cd Computer-Vision
pip install -r requirements.txt


---Install all requirements with:
pip install -r requirements.txt
```

---

## Cloud vs Local (what runs where)

This repository contains both code intended to run on a cloud-hosted Streamlit app and code that is intended to run locally (or be included in the app bundle). Here's a short mapping so it's clear what executes where:

- `invisible-cloak/app.py` — Cloud (Streamlit web app)
  - This is the Streamlit entrypoint that users interact with in the browser. When deployed to Streamlit Cloud (or any host running Streamlit), `app.py` runs on the server and serves the web UI.
  - NOTE: Server-side camera access (cv2.VideoCapture) usually only works on your local machine. On Streamlit Cloud the recommended mode is the Browser (client) which uses the browser's webcam (`st.camera_input`). `app.py` supports both modes; choose "Browser (client)" when running on hosted Streamlit.

- `src/invisible_cloak.py` — Local / library code
  - This module contains the image-processing logic (HSV masking, background capture, morphological operations) used by the app. It can be imported by `app.py` and is where most of the computer-vision code lives.
  - You can run or test this module locally (for example, import it into a Python REPL or a small script) without running the full Streamlit server.

Quick notes:
- If you run the app locally (your laptop/desktop), you can use the "Server (local)" camera mode to read from your machine's webcam using OpenCV. That mode uses `cv2.VideoCapture` and the `ThreadedCamera` helper in `app.py`.
- When deploying to Streamlit Cloud, use the default "Browser (client)" camera mode so users can grant webcam permission in their browser; `app.py` already handles converting `st.camera_input` frames into OpenCV-compatible arrays and processing them with `src/invisible_cloak.py`.
- The repository includes `requirements.txt` (prefer headless OpenCV on cloud hosts) and `packages.txt` if the host needs system dependencies (e.g. libGL). See the repo top-level files for more deployment notes.
