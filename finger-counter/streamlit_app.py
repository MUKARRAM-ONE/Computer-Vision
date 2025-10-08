import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
import time
from src.hand_detector import HandDetector

# Set page configuration
st.set_page_config(
    page_title="Finger Counter",
    page_icon="👋",
    layout="wide"
)

# Title and description
st.title("✋ Finger Counter")
st.markdown("""
    Count your fingers using computer vision! This application uses MediaPipe to detect hand landmarks 
    and count the number of extended fingers in real-time.
    
    **Instructions:**
    1. Start the camera using the button in the sidebar
    2. Show one or both hands to the camera
    3. The application will count your extended fingers from all visible hands
    4. Try different gestures or use both hands to see the total finger count
""")

# Initialize session state variables
if 'camera_started' not in st.session_state:
    st.session_state.camera_started = False

if 'detector' not in st.session_state:
    st.session_state.detector = HandDetector(min_detection_confidence=0.7, max_hands=2)

if 'finger_count_history' not in st.session_state:
    st.session_state.finger_count_history = []
    st.session_state.history_length = 5
    st.session_state.stable_count = 0

if 'gesture_history' not in st.session_state:
    st.session_state.gesture_history = []
    st.session_state.current_gesture = ""

if 'frame_count' not in st.session_state:
    st.session_state.frame_count = 0
    st.session_state.fps = 0
    st.session_state.previous_time = 0

# Sidebar controls
st.sidebar.title("Controls")

# Camera controls
if st.sidebar.button("Start Camera" if not st.session_state.camera_started else "Stop Camera"):
    st.session_state.camera_started = not st.session_state.camera_started

# Detection settings
st.sidebar.subheader("Detection Settings")
detection_confidence = st.sidebar.slider("Detection Confidence", 0.5, 1.0, 0.7, 0.05)
st.session_state.detector.min_detection_confidence = detection_confidence

# Hand detection settings
max_hands = st.sidebar.slider("Maximum Hands to Detect", 1, 2, 2, 1)
if max_hands != st.session_state.detector.max_hands:
    st.session_state.detector.max_hands = max_hands
    # Reinitialize the detector with new max_hands setting
    st.session_state.detector.hands = st.session_state.detector.mp_hands.Hands(
        static_image_mode=st.session_state.detector.mode,
        max_num_hands=max_hands,
        model_complexity=st.session_state.detector.model_complexity,
        min_detection_confidence=st.session_state.detector.min_detection_confidence,
        min_tracking_confidence=st.session_state.detector.min_tracking_confidence
    )

# Visualization settings
st.sidebar.subheader("Visualization Settings")
show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True)
show_bounding_box = st.sidebar.checkbox("Show Bounding Box", value=False)

# Define gestures
gestures = {
    0: "Fist",
    1: "One",
    2: "Peace/Victory",
    3: "Three",
    4: "Four",
    5: "Five/Open Hand"
}

# Function to stabilize finger count
def stabilize_finger_count(count):
    # Add current count to history
    st.session_state.finger_count_history.append(count)
    
    # Keep history at specified length
    if len(st.session_state.finger_count_history) > st.session_state.history_length:
        st.session_state.finger_count_history.pop(0)
    
    # Only update stable count if we have enough history
    if len(st.session_state.finger_count_history) == st.session_state.history_length:
        # Count occurrences of each finger count
        count_occurrences = {}
        for c in st.session_state.finger_count_history:
            if c in count_occurrences:
                count_occurrences[c] += 1
            else:
                count_occurrences[c] = 1
        
        # Find the most common count
        max_count = 0
        most_common = st.session_state.stable_count
        for c, occurrences in count_occurrences.items():
            if occurrences > max_count:
                max_count = occurrences
                most_common = c
        
        # Only update if the most common count appears enough times
        if max_count > st.session_state.history_length // 2:
            st.session_state.stable_count = most_common
    
    return st.session_state.stable_count

# Main display area with two columns
col1, col2 = st.columns([3, 1])

# Camera view in the first column
with col1:
    frame_placeholder = st.empty()

# Finger count and gesture display in the second column
with col2:
    st.subheader("Finger Count")
    finger_count_placeholder = st.empty()
    
    st.subheader("Detected Gesture")
    gesture_placeholder = st.empty()
    
    st.subheader("Performance")
    fps_placeholder = st.empty()

# Function to process camera feed
def process_camera_feed():
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    while st.session_state.camera_started:
        # Read frame from webcam
        success, img = cap.read()
        if not success:
            st.error("Failed to capture image from camera")
            break
        
        # Flip the image horizontally for a more intuitive mirror view
        img = cv2.flip(img, 1)
        
        # Find hands in the image
        if show_landmarks:
            img = st.session_state.detector.find_hands(img)
        else:
            img = st.session_state.detector.find_hands(img, draw=False)
        
        # Get hand landmark positions
        landmark_list, bbox = st.session_state.detector.find_position(img, draw=False)
        
        # Draw bounding box if enabled
        if show_bounding_box and bbox:
            cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Count fingers from all detected hands
        if st.session_state.detector.results and st.session_state.detector.results.multi_hand_landmarks:
            # Count total fingers from all hands
            finger_count = st.session_state.detector.count_all_fingers()
            
            # Count hands
            num_hands = len(st.session_state.detector.results.multi_hand_landmarks)
            
            # Stabilize finger count
            stable_count = stabilize_finger_count(finger_count)
            
            # Get gesture name or show total count
            if num_hands == 1:
                gesture_name = gestures.get(stable_count, "Unknown")
            else:
                gesture_name = f"Total from {num_hands} hands"
            
            # Update UI elements
            finger_count_placeholder.markdown(f"<h1 style='text-align: center; color: #FF4B4B;'>{stable_count}</h1>", unsafe_allow_html=True)
            gesture_placeholder.markdown(f"<h3 style='text-align: center;'>{gesture_name}</h3>", unsafe_allow_html=True)
            
            # Draw finger count on image
            cv2.rectangle(img, (img.shape[1]-90, 10), (img.shape[1]-10, 90), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(stable_count), (img.shape[1]-70, 75), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)
        else:
            # Reset history when no hand is detected
            st.session_state.finger_count_history = []
            finger_count_placeholder.markdown(f"<h1 style='text-align: center; color: gray;'>-</h1>", unsafe_allow_html=True)
            gesture_placeholder.markdown(f"<h3 style='text-align: center; color: gray;'>No Hand Detected</h3>", unsafe_allow_html=True)
        
        # Calculate and display FPS
        current_time = time.time()
        st.session_state.frame_count += 1
        
        if (current_time - st.session_state.previous_time) > 1:
            st.session_state.fps = st.session_state.frame_count / (current_time - st.session_state.previous_time)
            st.session_state.previous_time = current_time
            st.session_state.frame_count = 0
        
        fps_placeholder.markdown(f"<p>FPS: {int(st.session_state.fps)}</p>", unsafe_allow_html=True)
        
        # Convert from BGR to RGB for display in Streamlit
        rgb_frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(rgb_frame, channels="RGB", use_column_width=True)
    
    # Release resources when stopped
    cap.release()

# Display instructions when camera is not started
if not st.session_state.camera_started:
    frame_placeholder.info("Click 'Start Camera' in the sidebar to begin")
    finger_count_placeholder.markdown(f"<h1 style='text-align: center; color: gray;'>-</h1>", unsafe_allow_html=True)
    gesture_placeholder.markdown(f"<h3 style='text-align: center; color: gray;'>Camera not started</h3>", unsafe_allow_html=True)
    fps_placeholder.markdown(f"<p>FPS: 0</p>", unsafe_allow_html=True)
else:
    # Process camera feed when camera is started
    process_camera_feed()

# Footer
st.markdown("""---
### How it works
This application uses MediaPipe Hands to detect hand landmarks and OpenCV for image processing. 
It identifies the positions of your finger joints to determine whether each finger is extended or not.

- 👍 **Thumb**: Detected based on the horizontal position of the thumb tip relative to the thumb base
- ☝️ **Other fingers**: Detected by comparing the vertical positions of fingertips to their middle joints
- ✌️ **Multiple hands**: The app can detect up to 2 hands simultaneously and count the total number of extended fingers
""")