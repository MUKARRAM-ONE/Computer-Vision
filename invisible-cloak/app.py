import streamlit as st
import cv2
import numpy as np
import os
import time
import threading
from pathlib import Path
from dotenv import load_dotenv
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


class InvisibleCloak:
    """Lightweight invisible cloak processor optimized for low-latency use.

    Designed to work with already-resized frames (keeps processing small).
    """
    def __init__(self):
        # Default color range for red cloak (supports wrap-around with two ranges)
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])

        # Background frame (should match processed frame size)
        self.background = None
        self.background_captured = False

        # Small kernel for faster morphological ops
        self._kernel = np.ones((2, 2), np.uint8)

    def capture_background(self, frame):
        """Capture the background frame (store a copy sized to the incoming frame)."""
        # store a copy so external changes don't affect background
        self.background = frame.copy()
        self.background_captured = True
        return True

    def set_color_range(self, color_name):
        """Set some handy presets for cloak colors."""
        name = color_name.lower()
        if name == 'red':
            self.lower_red = np.array([0, 120, 70])
            self.upper_red = np.array([10, 255, 255])
            self.lower_red2 = np.array([170, 120, 70])
            self.upper_red2 = np.array([180, 255, 255])
        elif name == 'blue':
            self.lower_red = np.array([100, 150, 0])
            self.upper_red = np.array([140, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif name == 'green':
            self.lower_red = np.array([40, 100, 50])
            self.upper_red = np.array([80, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif name in ('pink', 'light pink', 'hot pink', 'pastel pink'):
            self.lower_red = np.array([140, 30, 100])
            self.upper_red = np.array([170, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif name in ('sky blue', 'light blue'):
            self.lower_red = np.array([90, 50, 100])
            self.upper_red = np.array([110, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif name == 'yellow':
            self.lower_red = np.array([20, 100, 100])
            self.upper_red = np.array([40, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif name == 'orange':
            self.lower_red = np.array([10, 100, 100])
            self.upper_red = np.array([25, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif name == 'purple':
            self.lower_red = np.array([125, 50, 50])
            self.upper_red = np.array([150, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        return True

    def process_frame(self, frame):
        """Process a frame and return (output_frame, success).

        The method is optimized for speed: expects `frame` to be reasonably small
        (e.g., width around 640 or less). It ensures the background matches
        the incoming frame size and uses minimal morphology.
        """
        if not self.background_captured or self.background is None:
            return frame, False

        # make sure background size matches incoming frame
        if self.background.shape != frame.shape:
            try:
                self.background = cv2.resize(self.background, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
            except Exception:
                # if resize fails, return original frame to avoid crashes
                return frame, False

        # Convert to HSV (fast)
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

        # Create mask for primary range
        mask = cv2.inRange(hsv, self.lower_red, self.upper_red)

        # If a wrap-around range exists (for red), include it
        if getattr(self, 'lower_red2', None) is not None and getattr(self, 'upper_red2', None) is not None:
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = cv2.bitwise_or(mask, mask2)

        # Fast morphological cleaning: small kernel and single pass
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, self._kernel)

        # Optional smoothing to reduce blockiness while keeping it fast
        mask = cv2.GaussianBlur(mask, (3, 3), 0)

        # Inverse mask for foreground
        mask_inv = cv2.bitwise_not(mask)

        # Combine frames using masks
        res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)
        res2 = cv2.bitwise_and(self.background, self.background, mask=mask)
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)

        return final_output, True

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

# Camera mode: server (uses cv2 on the server) or browser (uses user's webcam via browser)
camera_mode = st.sidebar.selectbox("Camera Mode", ["Browser (client)", "Server (local)"], index=0,
                                 help="Browser: uses your browser webcam (asks permission). Server: uses server-side camera (not available on Streamlit Cloud).")


class ThreadedCamera:
    """Background camera reader to always keep the latest frame available."""
    def __init__(self, src=0, resize_width=CAMERA_RESIZE_WIDTH):
        self.src = src
        self.resize_width = resize_width
        self.cap = cv2.VideoCapture(self.src)
        # Try to set capture resolution to reduce work per-frame
        try:
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(self.resize_width))
            # height will adjust automatically; keep default FPS
        except Exception:
            pass

        self.running = False
        self.frame = None
        self.lock = threading.Lock()

    def start(self):
        if self.running:
            return self
        self.running = True
        self.thread = threading.Thread(target=self._update, daemon=True)
        self.thread.start()
        return self

    def _update(self):
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.01)
                continue
            # keep the raw frame; resizing will be done in main thread to control CPU
            with self.lock:
                self.frame = frame

    def read(self):
        with self.lock:
            if self.frame is None:
                return None
            return self.frame.copy()

    def stop(self):
        self.running = False
        try:
            if hasattr(self, 'thread'):
                self.thread.join(timeout=0.5)
        except Exception:
            pass
        try:
            if self.cap is not None:
                self.cap.release()
        except Exception:
            pass

# Start/Stop camera button
if st.sidebar.button("Start Camera" if not st.session_state.camera_started else "Stop Camera"):
    if st.session_state.camera_started:
        if st.session_state.camera is not None:
            try:
                st.session_state.camera.stop()
            except Exception:
                try:
                    st.session_state.camera.cap.release()
                except Exception:
                    pass
        st.session_state.camera = None
        st.session_state.camera_started = False
    else:
        # Start background reader for lower latency
        cam = ThreadedCamera(src=0, resize_width=CAMERA_RESIZE_WIDTH)
        cam.start()
        st.session_state.camera = cam
        st.session_state.camera_started = True

# Background capture button
# Enabled for Browser mode (no server camera required). For Server mode, require camera started.
background_button = st.sidebar.button(
    "Capture Background",
    disabled=(camera_mode == "Server (local)" and not st.session_state.camera_started)
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

# For browser mode: placeholders to capture and hold images
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
    """Process one frame from either server camera or browser camera_input.

    Returns True if a frame was displayed, False otherwise.
    """
    frame = None

    # 1) Acquire frame depending on mode
    if camera_mode == "Server (local)":
        cam = st.session_state.get('camera')
        if cam is None:
            return False
        raw = cam.read()
        if raw is None:
            return False
        frame = raw
    else:
        # Browser: camera_input may be used as a snapshot upload
        cam_upload = browser_capture_placeholder.camera_input("Use your webcam — take a picture to process")
        if cam_upload is None:
            # If there's a previously stored browser frame, reuse it
            frame = st.session_state.browser_frame
            if frame is None:
                return False
        else:
            pil_img = Image.open(cam_upload).convert('RGB')
            frame = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
            # store last browser frame for re-runs
            st.session_state.browser_frame = frame

    # 2) Normalize frame (flip, resize)
    try:
        frame = cv2.flip(frame, 1)
    except Exception:
        pass

    h, w = frame.shape[:2]
    if w > CAMERA_RESIZE_WIDTH:
        scale = CAMERA_RESIZE_WIDTH / w
        frame = cv2.resize(frame, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # 3) Color helper
    if st.session_state.get('show_color_helper', False):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        cx, cy = frame.shape[1] // 2, frame.shape[0] // 2
        cv2.circle(frame, (cx, cy), 5, (0, 255, 0), 2)
        ch, cs, cvv = hsv_frame[cy, cx]
        hsv_text = f"H: {ch}, S: {cs}, V: {cvv}"
        cv2.putText(frame, hsv_text, (cx + 10, cy + 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 4) Handle background capture & reset
    if background_button:
        st.session_state.cloak.capture_background(frame.copy())
        st.session_state.background_captured = True
        st.sidebar.success("Background captured successfully!")

    if reset_button:
        st.session_state.cloak.reset()
        st.session_state.background_captured = False
        st.sidebar.info("Background reset. Capture a new background.")

    # 5) Process frame if background captured
    if st.session_state.background_captured:
        processed, ok = st.session_state.cloak.process_frame(frame)
        if ok:
            frame = processed

    # 6) Display
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    except Exception:
        rgb = frame
    frame_placeholder.image(rgb, channels="RGB", use_column_width=True)

    return True

# Processing controls and main-run behavior (Streamlit friendly)
status_text = st.empty()

# Live processing toggle (works for both modes)
live_toggle = st.sidebar.checkbox("Live Processing (auto-refresh)", value=st.session_state.get('processing', False))
st.session_state.processing = live_toggle

# Process one frame per run
frame_shown = process_camera_feed()

# Status messaging
if st.session_state.background_captured:
    status_text.success("Invisible cloak is active! Hold up your " + selected_color.lower() + " cloth.")
else:
    status_text.warning("Please capture the background first (make sure you're not in the frame).")

# If live processing is enabled, sleep a bit and rerun the app to emulate a live feed
if st.session_state.processing:
    time.sleep(FRAME_DELAY)
    # Re-run the script to fetch/process the next frame
    st.experimental_rerun()
else:
    # Display placeholder when camera is not started
    camera_placeholder.info("Click 'Start Camera' to begin (Server mode) or switch to Browser mode to use your webcam in the browser")

# Clean up resources when the app is closed
def cleanup():
    if st.session_state.camera is not None:
        # Stop threaded camera if it provides a stop() method
        try:
            st.session_state.camera.stop()
        except Exception:
            try:
                st.session_state.camera.release()
            except Exception:
                pass
        st.session_state.camera = None
        st.session_state.camera_started = False

# Register the cleanup function to be called when the script ends
import atexit
atexit.register(cleanup)