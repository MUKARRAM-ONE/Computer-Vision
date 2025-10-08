# Finger Counter Application

This application uses computer vision techniques to detect hands and count extended fingers in real-time. It leverages MediaPipe for hand landmark detection and OpenCV for image processing.
# Finger Counter

A real-time finger counting and gesture recognition project using MediaPipe and OpenCV. The repository contains three main implementations:

- Basic finger counter: `finger_counter.py`
- Advanced finger counter with extra gestures and visualizations: `advanced_finger_counter.py`
- Streamlit web interface: `streamlit_app.py`

## Table of Contents

- Overview
- Features
- Requirements
- Installation
- Usage
- How it works
- Controls
- Troubleshooting
- Future improvements

## Overview

This project detects hands in a webcam feed, computes hand landmarks using MediaPipe, and counts extended fingers per hand. The Streamlit version provides a configurable web UI for experimenting with thresholds and visualization options.

## Features

- Real-time hand detection and finger counting
- Multi-hand support (up to 2 hands)
- Gesture recognition (fist, peace, thumbs up, OK, etc.)
- Visual landmarks overlay and FPS counter
- Streamlit-based web interface for easy experimentation

## Requirements

Install the packages listed below (see `requirements.txt` inside this folder):

- opencv-python >= 4.5.0
- mediapipe >= 0.8.9
- numpy >= 1.19.0
- streamlit >= 1.10.0 (only required for the Streamlit app)

## Installation

1. Create and activate a virtual environment (recommended):

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

## Usage

Run the basic finger counter:

```powershell
python finger_counter.py
```

Run the advanced finger counter:

```powershell
python advanced_finger_counter.py
```

Run the Streamlit web interface:

```powershell
streamlit run streamlit_app.py
```

## How it works

1. Capture frames from the webcam using OpenCV.
2. Detect hand landmarks using MediaPipe Hands (21 landmarks per hand).
3. For each hand, determine which fingers are extended using simple geometric rules:
   - Thumb: compared against thumb MCP joint and relative hand orientation
   - Other fingers: tip position relative to PIP joint
4. Aggregate counts and (optionally) recognize gestures based on finger combinations.

## Controls

- Press `q` to quit (OpenCV windows). 
- Use the Streamlit UI to change detection confidence, enable/disable landmark overlays, and switch visualization modes.

## Troubleshooting

- Poor detection quality: improve lighting and ensure the hand is within the camera frame.
- Slow performance: lower the frame resolution or use a faster machine.
- Detection not starting: confirm camera permissions and correct index being used (0, 1, etc.).

## Future improvements

- Add custom gesture training and export functionality
- Improve robustness with temporal smoothing and tracking
- Add unit tests and CI for code quality

## License

Add a LICENSE file to the repository root to declare the project license (MIT, Apache-2.0, etc.).
