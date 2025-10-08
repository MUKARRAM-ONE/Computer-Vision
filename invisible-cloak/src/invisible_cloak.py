import cv2
import numpy as np
import os
from dotenv import load_dotenv

load_dotenv()

class InvisibleCloak:
    def __init__(self):
        # Default color range for red cloak
        self.lower_red = np.array([0, 120, 70])
        self.upper_red = np.array([10, 255, 255])
        self.lower_red2 = np.array([170, 120, 70])
        self.upper_red2 = np.array([180, 255, 255])
        
        # Initialize background as None
        self.background = None
        self.background_captured = False
    
    def capture_background(self, frame):
        """Capture the background frame"""
        self.background = frame
        self.background_captured = True
        return True
    
    def set_color_range(self, color_name):
        """Set the color range for the cloak"""
        if color_name.lower() == 'red':
            self.lower_red = np.array([0, 120, 70])
            self.upper_red = np.array([10, 255, 255])
            self.lower_red2 = np.array([170, 120, 70])
            self.upper_red2 = np.array([180, 255, 255])
        elif color_name.lower() == 'blue':
            self.lower_red = np.array([100, 150, 0])
            self.upper_red = np.array([140, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'green':
            self.lower_red = np.array([40, 100, 50])
            self.upper_red = np.array([80, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'pink':
            self.lower_red = np.array([140, 100, 100])
            self.upper_red = np.array([170, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'light pink':
            self.lower_red = np.array([140, 30, 100])
            self.upper_red = np.array([170, 100, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'hot pink':
            self.lower_red = np.array([150, 100, 100])
            self.upper_red = np.array([165, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'pastel pink':
            self.lower_red = np.array([145, 30, 150])
            self.upper_red = np.array([165, 120, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'sky blue':
            self.lower_red = np.array([90, 100, 100])
            self.upper_red = np.array([110, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'light blue':
            self.lower_red = np.array([90, 50, 100])
            self.upper_red = np.array([110, 150, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'yellow':
            self.lower_red = np.array([20, 100, 100])
            self.upper_red = np.array([40, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'orange':
            self.lower_red = np.array([10, 100, 100])
            self.upper_red = np.array([25, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        elif color_name.lower() == 'purple':
            self.lower_red = np.array([125, 50, 50])
            self.upper_red = np.array([150, 255, 255])
            self.lower_red2 = None
            self.upper_red2 = None
        return True
    
    def process_frame(self, frame):
        """Process a frame to create the invisible cloak effect"""
        if not self.background_captured:
            return frame, False
        
        # Convert the frame to HSV color space
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask for the cloak color
        mask1 = cv2.inRange(hsv, self.lower_red, self.upper_red)
        
        if self.lower_red2 is not None and self.upper_red2 is not None:
            mask2 = cv2.inRange(hsv, self.lower_red2, self.upper_red2)
            mask = mask1 + mask2
        else:
            mask = mask1
        
        # Pre-compute kernel for morphological operations
        kernel = np.ones((3, 3), np.uint8)
        
        # Refining the mask - use more efficient morphological operations
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, kernel, iterations=1)
        
        # Create the inverse mask
        mask_inv = cv2.bitwise_not(mask)
        
        # Use the mask to extract the foreground and background
        res1 = cv2.bitwise_and(frame, frame, mask=mask_inv)
        res2 = cv2.bitwise_and(self.background, self.background, mask=mask)
        
        # Combine the foreground and background
        final_output = cv2.addWeighted(res1, 1, res2, 1, 0)
        
        return final_output, True
    
    def reset(self):
        """Reset the background"""
        self.background = None
        self.background_captured = False
        return True