import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math
from hand_detector import HandDetector

# Initialize the webcam
cap = cv2.VideoCapture(0)

# Set camera resolution for better performance
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Initialize hand detector
detector = HandDetector(min_detection_confidence=0.7, max_hands=1)

# Variables for FPS calculation
previous_time = 0
current_time = 0

# Main loop
while True:
    # Read frame from webcam
    success, img = cap.read()
    if not success:
        print("Failed to capture image from camera")
        break
    
    # Flip the image horizontally for a more intuitive mirror view
    img = cv2.flip(img, 1)
    
    # Find hands in the image
    img = detector.find_hands(img)
    
    # Get hand landmark positions
    landmark_list, bbox = detector.find_position(img)
    
    # Count fingers if hand is detected
    finger_count = 0
    if landmark_list:
        # Count fingers
        finger_count = detector.count_fingers()
        
        # Display finger count
        cv2.rectangle(img, (20, 20), (170, 120), (0, 255, 0), cv2.FILLED)
        cv2.putText(img, str(finger_count), (45, 100), cv2.FONT_HERSHEY_PLAIN, 5, (255, 0, 0), 10)
    
    # Calculate and display FPS
    current_time = time.time()
    fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
    previous_time = current_time
    
    cv2.putText(img, f"FPS: {int(fps)}", (500, 30), cv2.FONT_HERSHEY_PLAIN, 1, (255, 0, 0), 2)
    
    # Display the image
    cv2.imshow("Finger Counter", img)
    
    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()