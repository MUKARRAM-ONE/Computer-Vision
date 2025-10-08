import cv2
import mediapipe as mp
import math
import numpy as np

class HandDetector:
    def __init__(self, mode=False, max_hands=2, model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5):
        """
        Initialize the hand detector with MediaPipe Hands.
        
        Args:
            mode: False for tracking, True for detection only
            max_hands: Maximum number of hands to detect
            model_complexity: Complexity of the hand landmark model (0, 1, or 2)
            min_detection_confidence: Minimum confidence for hand detection
            min_tracking_confidence: Minimum confidence for hand tracking
        """
        self.mode = mode
        self.max_hands = max_hands
        self.model_complexity = model_complexity
        self.min_detection_confidence = min_detection_confidence
        self.min_tracking_confidence = min_tracking_confidence
        
        # Initialize MediaPipe Hands
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.max_hands,
            model_complexity=self.model_complexity,
            min_detection_confidence=self.min_detection_confidence,
            min_tracking_confidence=self.min_tracking_confidence
        )
        
        # For drawing hand landmarks
        self.mp_draw = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # Store hand landmarks
        self.results = None
        self.landmark_list = []
        self.bbox = []
        
        # Finger tip IDs (thumb, index, middle, ring, pinky)
        self.tip_ids = [4, 8, 12, 16, 20]
        
    def find_hands(self, img, draw=True):
        """
        Find hands in the image.
        
        Args:
            img: Input image
            draw: Whether to draw hand landmarks on the image
            
        Returns:
            Image with or without hand landmarks drawn
        """
        # Convert BGR image to RGB
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Process the image with MediaPipe Hands
        self.results = self.hands.process(img_rgb)
        
        # Draw hand landmarks if hands are detected and draw is True
        if self.results.multi_hand_landmarks and draw:
            for hand_landmarks in self.results.multi_hand_landmarks:
                self.mp_draw.draw_landmarks(
                    img,
                    hand_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
        
        return img
    
    def find_position(self, img, hand_no=0, draw=True):
        """
        Find the position of hand landmarks.
        
        Args:
            img: Input image (can be None when just extracting landmark data)
            hand_no: Which hand to detect (0 for first hand)
            draw: Whether to draw landmark points on the image
            
        Returns:
            List of landmark positions and bounding box
        """
        self.landmark_list = []
        self.bbox = []
        
        # Check if hands are detected
        if self.results.multi_hand_landmarks:
            # Get the specified hand
            if len(self.results.multi_hand_landmarks) > hand_no:
                my_hand = self.results.multi_hand_landmarks[hand_no]
                
                # Get image dimensions if img is provided
                if img is not None:
                    h, w, c = img.shape
                    
                    # Initialize bounding box variables
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                else:
                    # Use default dimensions if img is None (for internal processing)
                    w, h = 640, 480
                    x_min, y_min = w, h
                    x_max, y_max = 0, 0
                
                # Process each landmark
                for id, lm in enumerate(my_hand.landmark):
                    # Convert normalized coordinates to pixel coordinates
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    
                    # Update bounding box
                    x_min = min(x_min, cx)
                    y_min = min(y_min, cy)
                    x_max = max(x_max, cx)
                    y_max = max(y_max, cy)
                    
                    # Add landmark to list
                    self.landmark_list.append([id, cx, cy])
                    
                    # Draw landmark point if img is provided
                    if img is not None and draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
                
                # Add padding to bounding box
                padding = 20
                self.bbox = [
                    max(0, x_min - padding),
                    max(0, y_min - padding),
                    min(w, x_max + padding),
                    min(h, y_max + padding)
                ]
                
                # Draw bounding box if img is provided
                if img is not None and draw:
                    cv2.rectangle(img, (self.bbox[0], self.bbox[1]), 
                                 (self.bbox[2], self.bbox[3]), (0, 255, 0), 2)
        
        return self.landmark_list, self.bbox
    
    def count_fingers(self, hand_no=0):
        """
        Count the number of extended fingers for a specific hand.
        
        Args:
            hand_no: Which hand to count fingers for (0 for first hand)
            
        Returns:
            Number of extended fingers (0-5) for the specified hand
        """
        fingers = []
        
        # Check if hand landmarks are detected
        if self.landmark_list:
            # Thumb: Check if thumb tip is to the right of thumb MCP for right hand
            # For left hand, we would check if thumb tip is to the left of thumb MCP
            # This is a simplified approach and may need adjustment based on hand orientation
            if self.landmark_list[self.tip_ids[0]][1] > self.landmark_list[self.tip_ids[0] - 1][1]:
                fingers.append(1)
            else:
                fingers.append(0)
            
            # Other fingers: Check if finger tip is above finger PIP joint
            for id in range(1, 5):
                if self.landmark_list[self.tip_ids[id]][2] < self.landmark_list[self.tip_ids[id] - 2][2]:
                    fingers.append(1)
                else:
                    fingers.append(0)
        
        return sum(fingers)
        
    def count_all_fingers(self):
        """
        Count the number of extended fingers across all detected hands.
        
        Returns:
            Total number of extended fingers across all hands
        """
        total_fingers = 0
        
        # Check if hands are detected
        if self.results.multi_hand_landmarks:
            # Process each hand
            for hand_no in range(len(self.results.multi_hand_landmarks)):
                # Find position for this hand
                self.find_position(None, hand_no=hand_no, draw=False)
                
                # Count fingers for this hand
                fingers_this_hand = self.count_fingers(hand_no)
                total_fingers += fingers_this_hand
        
        return total_fingers
    
    def find_distance(self, p1, p2, img=None, draw=True, r=15, t=3):
        """
        Find the distance between two landmarks.
        
        Args:
            p1: First landmark ID
            p2: Second landmark ID
            img: Image to draw on (optional)
            draw: Whether to draw the connection
            r: Radius of circles at landmark points
            t: Thickness of the connection line
            
        Returns:
            Distance between landmarks, image with drawn connection, and coordinates
        """
        x1, y1 = self.landmark_list[p1][1], self.landmark_list[p1][2]
        x2, y2 = self.landmark_list[p2][1], self.landmark_list[p2][2]
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        
        # Calculate Euclidean distance
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        
        # Draw connection if image is provided
        if img is not None and draw:
            cv2.circle(img, (x1, y1), r, (255, 0, 255), cv2.FILLED)
            cv2.circle(img, (x2, y2), r, (255, 0, 255), cv2.FILLED)
            cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), t)
            cv2.circle(img, (cx, cy), r, (0, 0, 255), cv2.FILLED)
        
        return distance, img, [x1, y1, x2, y2, cx, cy]