import cv2
import numpy as np
from dotenv import load_dotenv

load_dotenv()


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
