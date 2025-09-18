# Invisible Cloak Application

This application creates an "invisible cloak" effect using OpenCV and Streamlit. It allows you to become invisible by holding a cloth of a specific color in front of the camera.

## Features

- Real-time invisible cloak effect using computer vision
- Support for a wide range of cloak colors:
  - Basic colors: Red, Blue, Green
  - Pink variations: Pink, Light Pink, Hot Pink, Pastel Pink
  - Blue variations: Sky Blue, Light Blue
  - Additional colors: Yellow, Orange, Purple
  - Custom color option with HSV sliders
- Color detection helper to identify your cloth's HSV values
- User-friendly Streamlit interface
- Background capture and reset functionality
- Optimized performance for smoother camera experience

## Requirements

The application requires the following packages:

- opencv-python==4.8.0.76 (or newer)
- numpy==1.24.3 (or newer)
- streamlit==1.26.0 (or newer)
- python-dotenv==1.0.0 (or newer)

## Installation

1. Make sure you have Python installed (Python 3.8 or higher recommended)
2. Install the required packages using the requirements.txt file:

```bash
pip install -r ../requirements.txt
```

## How to Run

1. Navigate to the invisible-cloak directory:

```bash
cd Computer-Vision/invisible-cloak
```

2. Run the Streamlit application:

```bash
streamlit run app.py
```

3. The application will open in your default web browser.

## Usage Instructions

1. Click the "Start Camera" button to activate your webcam.
2. Make sure you're not in the frame, then click "Capture Background".
3. Select a color for your cloak from the dropdown menu:
   - Choose from predefined colors (Red, Blue, Green, Pink variations, etc.)
   - Or select "Custom" to define your own color using HSV sliders
4. If your cloth color isn't working well with predefined options:
   - Enable the "Color Detection Helper" checkbox
   - Point your cloth at the center green circle
   - Note the HSV values displayed on screen
   - Select "Custom" color and adjust the sliders to match those values
5. Hold a cloth of the selected color in front of the camera.
6. Watch as the cloth becomes "invisible", showing the background instead!
7. Adjust the performance settings in the .env file if needed.

## Performance Settings

You can adjust the performance settings in the `.env` file:

```
# Performance settings
CAMERA_RESIZE_WIDTH=640  # Adjust to change the processing resolution
FRAME_DELAY=0.001        # Adjust to balance between performance and CPU usage
```

- Lower `CAMERA_RESIZE_WIDTH` for better performance on slower devices
- Increase `FRAME_DELAY` if CPU usage is too high
- Decrease `FRAME_DELAY` for smoother camera experience on powerful devices
6. You can reset the background at any time by clicking "Reset Background".

## How It Works

The application uses color detection in the HSV color space to identify the cloak. When the cloak is detected, it replaces those pixels with the corresponding pixels from the previously captured background image, creating the illusion of invisibility.

## Troubleshooting

- If the cloak effect isn't working well, try adjusting the lighting in your room.
- Make sure the cloth color matches the selected color option.
- For best results, use a cloth with a solid, vibrant color.
- If your specific cloth color isn't detected properly:
  - Use the Color Detection Helper to find the exact HSV values of your cloth
  - Use the Custom color option to fine-tune the HSV range
  - Try reducing the saturation minimum value for light/pastel colors
  - Try adjusting the hue range to accommodate slight color variations