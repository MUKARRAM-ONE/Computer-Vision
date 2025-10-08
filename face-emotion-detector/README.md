# Face Emotion & Gender Detector

A small project that detects faces from a live camera feed and predicts both emotion and gender using lightweight MobileNetV2-based models.

This README explains how the project works, what files and models are included, how to set up the environment, how to run the detector, and troubleshooting tips.

## Table of Contents
- Overview
- How it works
- Models
- Requirements
- Installation
- Usage
- Expected output
- Troubleshooting
- License


## Overview
The Face Emotion & Gender Detector captures frames from your webcam, detects faces, and runs two separate classifiers on each detected face:
- Emotion classifier (MobileNetV2) — predicts emotions like Happy, Sad, Angry, Neutral, etc.
- Gender classifier (MobileNetV2) — predicts Male/Female.

Both models are optimized for real-time inference on a CPU but also support GPU if PyTorch detects CUDA.

## How it works
1. Frame capture: the app reads frames continuously from a webcam using OpenCV.
2. Face detection: detected using a light-weight detector (Haar cascade, MTCNN, or MediaPipe face detection depending on code implementation).
3. Preprocessing: detected face regions are resized, normalized, and converted to tensors suitable for the MobileNetV2 models.
4. Inference: each preprocessed face is passed to the emotion and gender models (separate networks) to get predictions.
5. Postprocessing: predictions are mapped to human-readable labels and drawn on the frame for real-time display.

## Models
- `emotion_mobilenetv2.pth` — trained MobileNetV2 weights for emotion recognition.
- `gender_mobilenetv2.pth` — trained MobileNetV2 weights for gender classification.

Place these model files in the `face-emotion-detector/models/` directory. If the models are large they are included in the repository; consider using Git LFS for large model files in production.

## Requirements
Minimum recommended packages (see `requirements.txt`):
- opencv-python >= 4.5.0
- mediapipe >= 0.8.9
- numpy >= 1.19.0
- torch >= 1.9.0
- torchvision >= 0.10.0
- pillow >= 8.0.0

## Installation
1. Create a Python virtual environment (recommended):
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```
2. Upgrade pip and install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```
3. (Optional) If you plan to use GPU acceleration, install a compatible PyTorch build from https://pytorch.org

## Usage
Start the app from the `face-emotion-detector` folder:
```powershell
cd face-emotion-detector
python app.py
```

Controls and behavior:
- The app will open a window with a live camera feed.
- Detected faces will be boxed and annotated with predicted emotion and gender.
- Press `q` to quit the application.

## Expected output
A real-time webcam window showing bounding boxes around faces with text annotations like:
- Emotion: Happy (0.92)
- Gender: Male (0.87)

Scores are confidence values from the network.

## Troubleshooting
- Camera not detected: verify camera index and permissions. Try `cv2.VideoCapture(0)` or `1`.
- Slow inference: use smaller frame size, reduce model input size, or use GPU.
- Wrong predictions: ensure models are loaded with the correct label mapping and normalization.
- Model file missing: ensure `models/` contains `emotion_mobilenetv2.pth` and `gender_mobilenetv2.pth`.

## Notes
- Consider using Git LFS for the `.pth` model files if they are large.
- The project is designed for demonstration and may need larger datasets and regularization for production accuracy.

## License
This project carries your chosen license (add a LICENSE file in the repo root). If none is present, treat the code as private.
