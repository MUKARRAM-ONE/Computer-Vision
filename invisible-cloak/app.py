import streamlit as st
# NOTE: This Streamlit app runs on the web (browser). We avoid server-side OpenCV
# to reduce native dependency issues on cloud hosts. Heavy OpenCV code is kept
# in `src/invisible_cloak.py` for local use.
import numpy as np
import os
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
import colorsys
from PIL import Image

# Load environment variables: check app folder first, then `src/.env` as fallback
base_dir = Path(__file__).resolve().parent
env_path = base_dir / '.env'
if not env_path.exists():
    alt = base_dir / 'src' / '.env'
    if alt.exists():
        env_path = alt

if env_path.exists():
    load_dotenv(dotenv_path=str(env_path))
else:
    # fallback to default behavior (searches current working dir and parents)
    load_dotenv()

# Get performance settings from environment variables
CAMERA_RESIZE_WIDTH = int(os.getenv('CAMERA_RESIZE_WIDTH', 640))
# Use a small non-zero default so the main loop yields to other threads
FRAME_DELAY = max(0.01, float(os.getenv('FRAME_DELAY', 0.01)))


class SimpleCloak:
    """A browser-friendly cloak implementation using PIL + numpy.

    This is intentionally simpler than the OpenCV version and designed to run
    safely on cloud hosts where native OpenCV wheels or system libs may not be
    available. It uses RGB-range thresholds for color masking.
    """
    PRESETS = {
        'red':   ((150, 0, 0), (255, 120, 120)),
        'blue':  ((0, 0, 100), (120, 120, 255)),
        'green': ((0, 100, 0), (120, 255, 120)),
        'pink':  ((200, 100, 150), (255, 200, 230)),
        'yellow':((200, 180, 0), (255, 255, 120)),
        'orange':((200, 80, 0), (255, 180, 80)),
        'purple':((120, 40, 120), (200, 140, 200)),
    }

    def __init__(self):
        self.background = None
        self.background_captured = False
        # default color
        self.lower_rgb, self.upper_rgb = self.PRESETS['red']

    def capture_background(self, pil_img):
        # store a copy resized to current frame
        self.background = pil_img.copy()
        self.background_captured = True
        return True

    def set_color_range(self, color_name):
        name = color_name.lower()
        if name in self.PRESETS:
            self.lower_rgb, self.upper_rgb = self.PRESETS[name]
        return True

    def process_frame(self, pil_img):
        """Return (pil_output, success)."""
        if not self.background_captured or self.background is None:
            return pil_img, False

        # Ensure background matches size
        if self.background.size != pil_img.size:
            try:
                self.background = self.background.resize(pil_img.size, Image.BILINEAR)
            except Exception:
                return pil_img, False

        arr = np.array(pil_img)
        low = np.array(self.lower_rgb, dtype=np.uint8)
        high = np.array(self.upper_rgb, dtype=np.uint8)

        # Build mask in RGB space
        mask = ((arr >= low) & (arr <= high)).all(axis=2)

        # Smooth mask a little using small blur on PIL-level
        mask_pil = Image.fromarray((mask * 255).astype('uint8'))
        mask_pil = mask_pil.filter(Image.Filter.GaussianBlur(radius=1)) if hasattr(Image, 'Filter') else mask_pil
        mask = (np.array(mask_pil) > 128)

        bg_arr = np.array(self.background)
        out = arr.copy()
        out[mask] = bg_arr[mask]
        return Image.fromarray(out), True

    def reset(self):
        self.background = None
        self.background_captured = False
        return True

# Set page configuration
st.set_page_config(
    page_title="Invisible Cloak",
    page_icon="🧙‍♂️",
    layout="wide"
)

# Initialize session state variables
if 'cloak' not in st.session_state:
    st.session_state.cloak = SimpleCloak()

if 'background_captured' not in st.session_state:
    st.session_state.background_captured = False

# This web app uses the browser webcam (st.camera_input). Server-side camera
# code (cv2.VideoCapture) is intentionally omitted to avoid native dependency
# issues on hosted platforms.

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

st.sidebar.info("This app runs in the browser. Use the webcam widget to take a picture or enable Live Processing.")


# No server-side camera code in the cloud app.

# Background capture button (browser-only app)
background_button = st.sidebar.button("Capture Background")

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
    
    # Show color preview (approximate by mixing min/max RGB)
    st.sidebar.subheader("Color Preview")
    # Use middle values of the selected RGB range from SimpleCloak presets if available
    preset = st.session_state.cloak.PRESETS.get('pink') if 'pink' in st.session_state.cloak.PRESETS else None
    # Build preview from selected_color preset when available
    try:
        lower, upper = st.session_state.cloak.PRESETS.get(selected_color.lower(), st.session_state.cloak.PRESETS['red'])
        lr = np.array(lower, dtype=np.uint8)
        ur = np.array(upper, dtype=np.uint8)
        mid = ((lr.astype(int) + ur.astype(int)) // 2).astype(np.uint8)
        preview = np.ones((100, 100, 3), dtype=np.uint8) * mid.reshape((1, 1, 3))
        st.sidebar.image(preview, caption="Selected Color", width=100)
    except Exception:
        pass
    
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

# Placeholders to capture and hold images (browser-only)
browser_capture_placeholder = st.empty()
browser_background_captured = False
browser_background = None

# Session flags for browser frame and live processing
if 'browser_frame' not in st.session_state:
    st.session_state.browser_frame = None

if 'processing' not in st.session_state:
    # When True the app will continuously re-run to provide a live view
    st.session_state.processing = False

# Function to process camera feed
def process_camera_feed():
    """Browser-only processing: accept a PIL image from st.camera_input, run SimpleCloak, and display."""
    cam_upload = browser_capture_placeholder.camera_input("Use your webcam — take a picture to process")
    pil_img = None
    if cam_upload is None:
        pil_img = st.session_state.get('browser_frame')
        if pil_img is None:
            return False
    else:
        pil_img = Image.open(cam_upload).convert('RGB')
        st.session_state.browser_frame = pil_img

    # Color helper (show center RGB value)
    if st.session_state.get('show_color_helper', False):
        arr = np.array(pil_img)
        cy, cx = arr.shape[0] // 2, arr.shape[1] // 2
        r, g, b = arr[cy, cx].tolist()
        st.sidebar.markdown(f"**Center RGB:** R={r} G={g} B={b}")

    # Background capture
    if background_button:
        st.session_state.cloak.capture_background(pil_img.copy())
        st.session_state.background_captured = True
        st.sidebar.success("Background captured successfully!")

    if reset_button:
        st.session_state.cloak.reset()
        st.session_state.background_captured = False
        st.sidebar.info("Background reset. Capture a new background.")

    # Process and display
    if st.session_state.background_captured:
        out_pil, ok = st.session_state.cloak.process_frame(pil_img)
        if ok:
            frame_to_show = out_pil
        else:
            frame_to_show = pil_img
    else:
        frame_to_show = pil_img

    frame_placeholder.image(frame_to_show, use_column_width=True)
    return True

# Processing controls and main-run behavior (Streamlit friendly)
status_text = st.empty()

# Manual processing controls
if st.sidebar.button("Process / Refresh Frame"):
    process_camera_feed()

# Status messaging
if st.session_state.background_captured:
    status_text.success("Invisible cloak is active! Hold up your " + selected_color.lower() + " cloth.")
else:
    status_text.warning("Please capture the background first (make sure you're not in the frame).")

camera_placeholder.info("Use the webcam widget to take a picture and press 'Process / Refresh Frame' to apply the cloak.")

# Clean up resources when the app is closed
def cleanup():
    # For browser-only app there may be no camera session objects; clear cloak state
    try:
        st.session_state.cloak.reset()
    except Exception:
        pass

# Register the cleanup function to be called when the script ends
import atexit
atexit.register(cleanup)