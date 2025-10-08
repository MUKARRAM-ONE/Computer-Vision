# Finger Counter Application

This application uses computer vision techniques to detect hands and count extended fingers in real-time. It leverages MediaPipe for hand landmark detection and OpenCV for image processing.

## Features

- Real-time hand detection and finger counting
- Multiple application versions:
  - Basic finger counter (`finger_counter.py`)
  - Advanced finger counter with gesture recognition (`advanced_finger_counter.py`)
  - Interactive Streamlit web interface (`streamlit_app.py`)
- Multi-hand detection and total finger counting (up to 2 hands simultaneously)
- Gesture recognition for common hand gestures
- Visualization of hand landmarks and finger status
- Performance metrics (FPS counter)

## Requirements

The application requires the following packages:

- opencv-python>=4.5.0
- mediapipe>=0.8.9
- numpy>=1.19.0
- streamlit>=1.10.0 (for the web interface version)

## Installation

1. Make sure you have Python installed (Python 3.8 or higher recommended)
2. Install the required packages:

```bash
pip install opencv-python mediapipe numpy streamlit
```

## Usage

### Basic Finger Counter

Run the basic finger counter application:

```bash
python finger_counter.py
```

### Advanced Finger Counter

Run the advanced finger counter with gesture recognition:

```bash
python advanced_finger_counter.py
```

### Streamlit Web Interface

Run the interactive web interface:

```bash
streamlit run streamlit_app.py
```

## How It Works

### Hand Detection

The application uses MediaPipe Hands to detect hand landmarks in the camera feed. MediaPipe provides 21 landmarks for each hand, representing different points on the hand (fingertips, joints, etc.).

### Finger Counting Logic

The finger counting logic works as follows:

1. **Thumb**: The thumb is considered extended if its tip is to the right of the thumb MCP joint (for right hand) or to the left (for left hand).
2. **Other fingers**: A finger is considered extended if its tip is above its PIP joint (the second joint from the tip).

### Gesture Recognition

The advanced version includes recognition of common gestures:

- **Fist**: All fingers closed
- **One/Index**: Only index finger extended
- **Peace/Victory**: Index and middle fingers extended
- **Three**: Index, middle, and ring fingers extended
- **Four**: All fingers except thumb extended
- **Five/Open Hand**: All fingers extended
- **Special gestures**: Thumbs up, thumbs down, OK sign, rock sign, pinch

## Controls

- Press 'q' to quit the application (for OpenCV windows)
- Use the sidebar controls in the Streamlit version to adjust settings

## Troubleshooting

- If the hand detection is not working well, try adjusting the lighting in your environment
- Make sure your hand is clearly visible in the camera frame
- Adjust the detection confidence threshold in the Streamlit version if needed
- For better performance, ensure your computer meets the minimum requirements for running MediaPipe

## Future Improvements

- Support for multiple hands with individual finger counting
- More advanced gesture recognition
- Custom gesture training capability
- Integration with other applications via API