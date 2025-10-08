import cv2
import numpy as np
import mediapipe as mp
import time
import os
import math
from hand_detector import HandDetector

class AdvancedFingerCounter:
    def __init__(self):
        # Initialize webcam
        self.cap = cv2.VideoCapture(0)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        # Initialize hand detector
        self.detector = HandDetector(min_detection_confidence=0.8, max_hands=2)
        
        # Variables for FPS calculation
        self.previous_time = 0
        self.current_time = 0
        
        # Variables for finger counting history
        self.finger_count_history = []
        self.history_length = 10  # Number of frames to keep in history
        self.stable_count = 0     # Current stable finger count
        
        # Gesture recognition variables
        self.gestures = {
            0: "Fist",
            1: "One",
            2: "Peace",
            3: "Three",
            4: "Four",
            5: "Five/Open Hand"
        }
        
        # Special gesture recognition
        self.special_gestures = {
            "thumbs_up": "Thumbs Up",
            "thumbs_down": "Thumbs Down",
            "ok": "OK",
            "rock": "Rock",
            "pinch": "Pinch"
        }
        self.current_gesture = ""
        
        # UI variables
        self.overlay_alpha = 0.3
        self.bg_color = (245, 117, 16)  # Orange
        self.text_color = (255, 255, 255)  # White
        self.accent_color = (0, 255, 0)  # Green
        
    def detect_special_gestures(self, landmark_list):
        """Detect special hand gestures based on landmark positions"""
        if not landmark_list:
            return ""
            
        # Get finger states (0 = closed, 1 = open)
        fingers = self.get_finger_states(landmark_list)
        
        # Thumbs up: only thumb is up, hand is vertical
        if fingers == [1, 0, 0, 0, 0]:
            # Check if thumb is pointing up (y-coordinate of thumb tip is less than thumb MCP)
            if landmark_list[4][2] < landmark_list[2][2]:
                return "thumbs_up"
        
        # Thumbs down: only thumb is up, but pointing down
        if fingers == [1, 0, 0, 0, 0]:
            # Check if thumb is pointing down
            if landmark_list[4][2] > landmark_list[2][2]:
                return "thumbs_down"
        
        # OK sign: thumb and index finger form a circle, other fingers up
        if fingers[0] == 1 and fingers[1] == 1:
            # Calculate distance between thumb tip and index tip
            distance = math.sqrt(
                (landmark_list[4][1] - landmark_list[8][1])**2 + 
                (landmark_list[4][2] - landmark_list[8][2])**2
            )
            # If distance is small, it's likely an OK sign
            if distance < 30:
                return "ok"
        
        # Rock sign: index and pinky up, others down
        if fingers == [0, 1, 0, 0, 1]:
            return "rock"
        
        # Pinch gesture: thumb and index close together, others can be in any position
        distance = math.sqrt(
            (landmark_list[4][1] - landmark_list[8][1])**2 + 
            (landmark_list[4][2] - landmark_list[8][2])**2
        )
        if distance < 30:
            return "pinch"
            
        return ""
    
    def get_finger_states(self, landmark_list):
        """Get the state of each finger (0 = closed, 1 = open)"""
        fingers = []
        
        # Thumb: Check if thumb tip is to the right of thumb MCP for right hand
        if landmark_list[4][1] > landmark_list[3][1]:
            fingers.append(1)
        else:
            fingers.append(0)
        
        # Other fingers: Check if finger tip is above finger PIP joint
        tip_ids = [8, 12, 16, 20]  # Index, middle, ring, pinky tips
        for id in tip_ids:
            if landmark_list[id][2] < landmark_list[id-2][2]:
                fingers.append(1)
            else:
                fingers.append(0)
                
        return fingers
    
    def stabilize_finger_count(self, count):
        """Stabilize finger count using a history buffer"""
        # Add current count to history
        self.finger_count_history.append(count)
        
        # Keep history at specified length
        if len(self.finger_count_history) > self.history_length:
            self.finger_count_history.pop(0)
        
        # Only update stable count if we have enough history
        if len(self.finger_count_history) == self.history_length:
            # Count occurrences of each finger count
            count_occurrences = {}
            for c in self.finger_count_history:
                if c in count_occurrences:
                    count_occurrences[c] += 1
                else:
                    count_occurrences[c] = 1
            
            # Find the most common count
            max_count = 0
            most_common = self.stable_count
            for c, occurrences in count_occurrences.items():
                if occurrences > max_count:
                    max_count = occurrences
                    most_common = c
            
            # Only update if the most common count appears enough times
            if max_count > self.history_length // 2:
                self.stable_count = most_common
        
        return self.stable_count
    
    def create_ui_overlay(self, img, finger_count, gesture_name):
        """Create a nice UI overlay with finger count and gesture name"""
        h, w, c = img.shape
        
        # Create semi-transparent overlay for the bottom info panel
        overlay = img.copy()
        cv2.rectangle(overlay, (0, h-150), (w, h), self.bg_color, -1)
        cv2.addWeighted(overlay, self.overlay_alpha, img, 1 - self.overlay_alpha, 0, img)
        
        # Display finger count
        cv2.putText(img, f"Fingers: {finger_count}", (20, h-100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.text_color, 3)
        
        # Display gesture name
        if gesture_name:
            cv2.putText(img, f"Gesture: {gesture_name}", (20, h-40), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, self.text_color, 3)
        
        # Display FPS
        cv2.putText(img, f"FPS: {int(self.fps)}", (w-150, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, self.accent_color, 2)
        
        return img
    
    def run(self):
        """Main loop for the finger counter application"""
        while True:
            # Read frame from webcam
            success, img = self.cap.read()
            if not success:
                print("Failed to capture image from camera")
                break
            
            # Flip the image horizontally for a more intuitive mirror view
            img = cv2.flip(img, 1)
            
            # Find hands in the image
            img = self.detector.find_hands(img)
            
            # Get hand landmark positions
            landmark_list, bbox = self.detector.find_position(img, draw=False)
            
            # Process hand landmarks if detected
            if landmark_list:
                # Count fingers
                raw_finger_count = self.detector.count_fingers()
                
                # Stabilize finger count
                stable_finger_count = self.stabilize_finger_count(raw_finger_count)
                
                # Detect special gestures
                special_gesture = self.detect_special_gestures(landmark_list)
                
                # Determine gesture name
                if special_gesture:
                    gesture_name = self.special_gestures[special_gesture]
                else:
                    gesture_name = self.gestures.get(stable_finger_count, "")
                
                # Create UI overlay
                img = self.create_ui_overlay(img, stable_finger_count, gesture_name)
                
                # Draw hand landmarks with custom style
                for id, (_, cx, cy) in enumerate(landmark_list):
                    # Draw larger circles for fingertips
                    if id in [4, 8, 12, 16, 20]:  # Fingertips
                        cv2.circle(img, (cx, cy), 12, (0, 255, 0), cv2.FILLED)
                    else:
                        cv2.circle(img, (cx, cy), 7, (0, 0, 255), cv2.FILLED)
            else:
                # No hand detected
                self.finger_count_history = []  # Reset history
                img = self.create_ui_overlay(img, 0, "No Hand Detected")
            
            # Calculate FPS
            self.current_time = time.time()
            self.fps = 1 / (self.current_time - self.previous_time) if (self.current_time - self.previous_time) > 0 else 0
            self.previous_time = self.current_time
            
            # Display the image
            cv2.imshow("Advanced Finger Counter", img)
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Release resources
        self.cap.release()
        cv2.destroyAllWindows()

# Run the application
if __name__ == "__main__":
    app = AdvancedFingerCounter()
    app.run()