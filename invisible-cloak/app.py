import streamlit as st
import cv2
import numpy as np
import os
import time
from src.invisible_cloak import InvisibleCloak
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get performance settings from environment variables
CAMERA_RESIZE_WIDTH = int(os.getenv('CAMERA_RESIZE_WIDTH', 640))
FRAME_DELAY = float(os.getenv('FRAME_DELAY', 0.0))

# Set page configuration
st.set_page_config(
    page_title="Invisible Cloak",
    page_icon="🧙‍♂️",
    layout="wide"
)

# Initialize session state variables
if 'cloak' not in st.session_state:
    st.session_state.cloak = InvisibleCloak()

if 'background_captured' not in st.session_state:
    st.session_state.background_captured = False

if 'camera_started' not in st.session_state:
    st.session_state.camera_started = False

if 'camera' not in st.session_state:
    st.session_state.camera = None

# Title and description
st.title("🧙‍♂️ Invisible Cloak")
st.markdown("""
    Become invisible with the magic of computer vision! 
    This application uses OpenCV to create an 'invisible cloak' effect.
    
    **Instructions:**
    1. Start the camera
    2. Capture the background (make sure you're not in the frame)
    3. Select a color for your cloak
    4. Hold a cloth of the selected color in front of the camera
    5. Watch yourself disappear!
""")

# Sidebar for controls
st.sidebar.title("Controls")

# Camera controls
camera_placeholder = st.empty()

# Start/Stop camera button
if st.sidebar.button("Start Camera" if not st.session_state.camera_started else "Stop Camera"):
    if st.session_state.camera_started:
        if st.session_state.camera is not None:
            st.session_state.camera.release()
        st.session_state.camera = None
        st.session_state.camera_started = False
    else:
        st.session_state.camera = cv2.VideoCapture(0)
        st.session_state.camera_started = True

# Background capture button
background_button = st.sidebar.button(
    "Capture Background", 
    disabled=not st.session_state.camera_started
)

# Reset background button
reset_button = st.sidebar.button(
    "Reset Background", 
    disabled=not st.session_state.background_captured
)

# Color selection
color_options = ["Red", "Blue", "Green", "Pink", "Light Pink", "Hot Pink", "Pastel Pink", "Sky Blue", "Light Blue", "Yellow", "Orange", "Purple", "Custom"]
selected_color = st.sidebar.selectbox(
    "Select Cloak Color",
    color_options,
    index=0
)

# Custom color controls
if selected_color == "Custom":
    st.sidebar.subheader("Custom Color HSV Range")
    min_hue = st.sidebar.slider("Min Hue", 0, 179, 140)
    max_hue = st.sidebar.slider("Max Hue", 0, 179, 170)
    min_sat = st.sidebar.slider("Min Saturation", 0, 255, 50)
    max_sat = st.sidebar.slider("Max Saturation", 0, 255, 255)
    min_val = st.sidebar.slider("Min Value", 0, 255, 50)
    max_val = st.sidebar.slider("Max Value", 0, 255, 255)
    
    # Apply custom HSV range
    st.session_state.cloak.lower_red = np.array([min_hue, min_sat, min_val])
    st.session_state.cloak.upper_red = np.array([max_hue, max_sat, max_val])
    st.session_state.cloak.lower_red2 = None
    st.session_state.cloak.upper_red2 = None
    
    # Show color preview
    st.sidebar.subheader("Color Preview")
    # Create a sample color based on the HSV values
    preview = np.ones((100, 100, 3), dtype=np.uint8)
    # Use the middle values of the HSV range for preview
    h = (min_hue + max_hue) // 2
    s = (min_sat + max_sat) // 2
    v = (min_val + max_val) // 2
    preview[:] = (h, s, v)
    preview_bgr = cv2.cvtColor(preview, cv2.COLOR_HSV2BGR)
    preview_rgb = cv2.cvtColor(preview_bgr, cv2.COLOR_BGR2RGB)
    st.sidebar.image(preview_rgb, caption="Selected Color", width=100)
    
    # Add HSV color picker tips
    with st.sidebar.expander("HSV Color Picking Tips"):
        st.markdown("""
        **Tips for finding your cloth color:**
        1. **Hue (H):** Represents the color type (red, green, blue, etc.)
           - Red: 0-10 or 170-180
           - Pink/Purple: 140-170
           - Blue: 100-140
           - Green: 40-80
           - Yellow: 20-40
           - Orange: 10-25
        2. **Saturation (S):** Color intensity (0=gray, 255=vivid)
           - For light/pastel colors: Use lower values (30-150)
           - For vivid colors: Use higher values (150-255)
        3. **Value (V):** Brightness (0=dark, 255=bright)
           - For dark colors: Use lower values (50-150)
           - For bright colors: Use higher values (150-255)
        
        **For light pink:** Try H:140-170, S:30-100, V:150-255
        """)


# Color helper toggle
if st.sidebar.checkbox("Enable Color Detection Helper", value=st.session_state.get('show_color_helper', False)):
    st.session_state.show_color_helper = True
else:
    st.session_state.show_color_helper = False

# Apply color selection
if selected_color:
    st.session_state.cloak.set_color_range(selected_color)

# Main display area
frame_placeholder = st.empty()

# Function to process camera feed
def process_camera_feed():
    if st.session_state.camera is not None and st.session_state.camera.isOpened():
        ret, frame = st.session_state.camera.read()
        if ret:
            # Flip the frame horizontally for a more intuitive mirror view
            frame = cv2.flip(frame, 1)
            
            # Resize frame for better performance using environment variable settings
            height, width = frame.shape[:2]
            if width > CAMERA_RESIZE_WIDTH:
                scale_factor = CAMERA_RESIZE_WIDTH / width
                frame = cv2.resize(frame, (int(width * scale_factor), int(height * scale_factor)))
            
            # Add color detection helper if enabled
            if 'show_color_helper' not in st.session_state:
                st.session_state.show_color_helper = False
                
            if st.session_state.show_color_helper and st.session_state.camera_started:
                # Convert to HSV for color detection
                hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                
                # Draw a target circle in the center for color sampling
                center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
                cv2.circle(frame, (center_x, center_y), 5, (0, 255, 0), 2)
                
                # Get the HSV color at the center point
                center_hsv = hsv_frame[center_y, center_x]
                h, s, v = center_hsv
                
                # Display the HSV values on the frame
                hsv_text = f"H: {h}, S: {s}, V: {v}"
                cv2.putText(frame, hsv_text, (center_x + 10, center_y + 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            
            # Handle background capture button press
            if background_button:
                st.session_state.cloak.capture_background(frame.copy())
                st.session_state.background_captured = True
                st.sidebar.success("Background captured successfully!")
            
            # Handle reset button press
            if reset_button:
                st.session_state.cloak.reset()
                st.session_state.background_captured = False
                st.sidebar.info("Background reset. Capture a new background.")
            
            # Process the frame if background is captured
            if st.session_state.background_captured:
                processed_frame, success = st.session_state.cloak.process_frame(frame)
                if success:
                    frame = processed_frame
            
            # Convert from BGR to RGB for display in Streamlit
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
            
            return True
    return False

# Main loop
if st.session_state.camera_started:
    status_text = st.empty()
    while process_camera_feed():
        # Use environment variable for frame delay
        time.sleep(FRAME_DELAY)
        
        # Display status
        if st.session_state.background_captured:
            status_text.success("Invisible cloak is active! Hold up your " + selected_color.lower() + " cloth.")
        else:
            status_text.warning("Please capture the background first (make sure you're not in the frame).")
        
        # Check if the session state has changed (e.g., if the user has clicked stop)
        if not st.session_state.camera_started:
            break
else:
    # Display placeholder when camera is not started
    camera_placeholder.info("Click 'Start Camera' to begin")

# Clean up resources when the app is closed
def cleanup():
    if st.session_state.camera is not None:
        st.session_state.camera.release()
        st.session_state.camera = None
        st.session_state.camera_started = False

# Register the cleanup function to be called when the script ends
import atexit
atexit.register(cleanup)