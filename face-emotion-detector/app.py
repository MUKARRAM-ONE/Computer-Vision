# app.py
"""
Face Emotion and Gender Detector using PyTorch MobileNetV2 models and MediaPipe

Detects face landmarks and classifies:
 - Emotions: Happy, Surprised, Neutral (using both MediaPipe landmarks and MobileNetV2)
 - Gender: Male, Female (using MobileNetV2)

Run:
    python app.py
Press 'q' to quit.
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image

# Create models directory if it doesn't exist
os.makedirs('models', exist_ok=True)

# MediaPipe Face Mesh setup
mp_face_mesh = mp.solutions.face_mesh

# Indices for lips / mouth corners in MediaPipe Face Mesh (468-point)
LEFT_MOUTH = 61
RIGHT_MOUTH = 291
UPPER_LIP = 13
LOWER_LIP = 14
LEFT_EYE_OUTER = 33
RIGHT_EYE_OUTER = 263

# Parameters (tune these if needed)
SMILE_WIDTH_RATIO_THRESHOLD = 0.45
MOUTH_OPEN_RATIO_THRESHOLD = 0.035
SMALL_FACE_AREA = 1000

# Gender and Emotion classes
GENDER_CLASSES = ['Male', 'Female']
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

# Define MobileNetV2 model for emotion and gender detection
class EmotionGenderModel(nn.Module):
    def __init__(self, num_classes):
        super(EmotionGenderModel, self).__init__()
        # Load pretrained MobileNetV2 model
        self.base = torch.hub.load('pytorch/vision:v0.10.0', 'mobilenet_v2', pretrained=True)
        # Replace the classifier
        in_features = self.base.classifier[1].in_features
        self.base.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x):
        return self.base(x)

# Load PyTorch models
def load_models():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load emotion model
    emotion_model = EmotionGenderModel(len(EMOTION_CLASSES))
    emotion_model.load_state_dict(torch.load('models/emotion_mobilenetv2.pth', map_location=device))
    emotion_model.eval()
    
    # Load gender model
    gender_model = EmotionGenderModel(len(GENDER_CLASSES))
    gender_model.load_state_dict(torch.load('models/gender_mobilenetv2.pth', map_location=device))
    gender_model.eval()
    
    return emotion_model, gender_model, device

# Image preprocessing for PyTorch models
def preprocess_image(image, device):
    # Convert to PIL Image
    image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    
    # Define transformations
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Apply transformations
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)  # Add batch dimension
    
    # Move to device
    return input_batch.to(device)

def normalized_distance(a, b):
    """Euclidean distance between two (x,y) tuples."""
    return np.linalg.norm(np.array(a) - np.array(b))

def landmarks_to_np(landmarks, frame_w, frame_h):
    """Convert normalized landmarks to 2D pixel coords list"""
    pts = []
    for lm in landmarks:
        pts.append((int(lm.x * frame_w), int(lm.y * frame_h)))
    return pts

def classify_emotion(landmarks_px):
    """
    Rule-based classifier using mouth width & mouth opening relative to face width.
    landmarks_px: list of (x,y) in pixel coords (length >= 468)
    Returns: (emotion_label, score_dict)
    """
    # Get relevant points
    lm_left_mouth = landmarks_px[LEFT_MOUTH]
    lm_right_mouth = landmarks_px[RIGHT_MOUTH]
    lm_upper = landmarks_px[UPPER_LIP]
    lm_lower = landmarks_px[LOWER_LIP]
    lm_left_face = landmarks_px[LEFT_EYE_OUTER]
    lm_right_face = landmarks_px[RIGHT_EYE_OUTER]

    # face width ~ distance between eye outer corners
    face_width = normalized_distance(lm_left_face, lm_right_face)
    if face_width < 1:
        face_width = 1.0

    # mouth width and mouth opening (height)
    mouth_width = normalized_distance(lm_left_mouth, lm_right_mouth)
    mouth_opening = normalized_distance(lm_upper, lm_lower)

    # normalized ratios
    width_ratio = mouth_width / face_width
    open_ratio = mouth_opening / face_width  # normalize by face width for stability

    # pack scores for transparency/debugging
    scores = {
        "width_ratio": width_ratio,
        "open_ratio": open_ratio,
        "mouth_width": mouth_width,
        "mouth_opening": mouth_opening,
        "face_width": face_width
    }

    # simple decision logic
    # priority: surprised (big mouth opening), then happy (wide mouth but not very open), else neutral
    if open_ratio > MOUTH_OPEN_RATIO_THRESHOLD * 3:  # very open mouth -> definitely surprised
        return "Surprised", scores
    if open_ratio > MOUTH_OPEN_RATIO_THRESHOLD:
        # mouth moderately open — treat as surprised unless it's also very wide (smile)
        if width_ratio < SMILE_WIDTH_RATIO_THRESHOLD:
            return "Surprised", scores

    if width_ratio > SMILE_WIDTH_RATIO_THRESHOLD:
        return "Happy", scores

    return "Neutral", scores

def draw_label(frame, text, x, y, bg_color=(0, 120, 255), text_color=(255,255,255)):
    """Draw a filled label on the frame at (x,y) top-left."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.8
    thickness = 2
    (w, h), _ = cv2.getTextSize(text, font, scale, thickness)
    cv2.rectangle(frame, (x-5, y-5), (x + w + 5, y + h + 5), bg_color, -1)
    cv2.putText(frame, text, (x, y + h - 2), font, scale, text_color, thickness, cv2.LINE_AA)

def main(camera_id=0):
    # Load PyTorch models
    try:
        print("Loading PyTorch models...")
        emotion_model, gender_model, device = load_models()
        use_ml_models = True
        print("PyTorch models loaded successfully!")
    except Exception as e:
        print(f"Error loading PyTorch models: {e}")
        print("Falling back to rule-based detection only.")
        use_ml_models = False
    
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        print("ERROR: Could not open webcam.")
        return

    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # MediaPipe Face Mesh
    with mp_face_mesh.FaceMesh(static_image_mode=False,
                               max_num_faces=1,
                               refine_landmarks=True,
                               min_detection_confidence=0.5,
                               min_tracking_confidence=0.5) as face_mesh:

        prev_time = 0.0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Flip for mirror view
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Process MediaPipe Face Mesh
            results = face_mesh.process(rgb)
            
            # Variables for results
            rule_emotion_label = "No face"
            ml_emotion_label = ""
            gender_label = ""
            scores = {}
            
            # Process MediaPipe face landmarks for emotion
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                landmarks_px = landmarks_to_np(face_landmarks, frame_w, frame_h)
                
                # Rule-based emotion classification
                rule_emotion_label, scores = classify_emotion(landmarks_px)
                
                # Extract face for ML-based classification
                if use_ml_models:
                    # Get face bounding box from landmarks
                    x_coords = [lm[0] for lm in landmarks_px]
                    y_coords = [lm[1] for lm in landmarks_px]
                    
                    x_min, x_max = int(min(x_coords)), int(max(x_coords))
                    y_min, y_max = int(min(y_coords)), int(max(y_coords))
                    
                    # Add padding
                    padding = 20
                    x_min = max(0, x_min - padding)
                    y_min = max(0, y_min - padding)
                    x_max = min(frame_w, x_max + padding)
                    y_max = min(frame_h, y_max + padding)
                    
                    # Extract face
                    face_img = frame[y_min:y_max, x_min:x_max]
                    
                    if face_img.size > 0:
                        # Preprocess for PyTorch models
                        try:
                            input_tensor = preprocess_image(face_img, device)
                            
                            # Emotion prediction
                            with torch.no_grad():
                                emotion_output = emotion_model(input_tensor)
                                emotion_probs = torch.nn.functional.softmax(emotion_output, dim=1)
                                emotion_idx = torch.argmax(emotion_probs, dim=1).item()
                                ml_emotion_label = EMOTION_CLASSES[emotion_idx]
                                emotion_conf = emotion_probs[0][emotion_idx].item() * 100
                                
                                # Gender prediction
                                gender_output = gender_model(input_tensor)
                                gender_probs = torch.nn.functional.softmax(gender_output, dim=1)
                                gender_idx = torch.argmax(gender_probs, dim=1).item()
                                gender_label = GENDER_CLASSES[gender_idx]
                                gender_conf = gender_probs[0][gender_idx].item() * 100
                        except Exception as e:
                            print(f"Error in ML prediction: {e}")
                
                # Draw face bounding box
                if use_ml_models and face_img.size > 0:
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
                
                # Draw important landmark markers (mouth corners and inner lips)
                for idx in [LEFT_MOUTH, RIGHT_MOUTH, UPPER_LIP, LOWER_LIP]:
                    (cx, cy) = landmarks_px[idx]
                    cv2.circle(frame, (cx, cy), 3, (0, 0, 255), -1)
                
                # Draw the face mesh (optional small dots)
                for (x, y) in landmarks_px[0:468:4]:  # draw every 4th point to reduce clutter
                    cv2.circle(frame, (x, y), 1, (200,200,200), -1)
                
                # Display the numeric scores (optional)
                if scores:
                    score_text = f"W:{scores.get('width_ratio', 0):.2f} O:{scores.get('open_ratio', 0):.3f}"
                    draw_label(frame, score_text, 10, 150, bg_color=(70,70,70))

            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if prev_time else 0.0
            prev_time = curr_time
            fps_text = f"FPS: {int(fps)}"
            
            # Display results
            draw_label(frame, f"Rule Emotion: {rule_emotion_label}", 10, 10, bg_color=(70,70,70))
            if ml_emotion_label:
                draw_label(frame, f"ML Emotion: {ml_emotion_label} ({emotion_conf:.1f}%)", 10, 45, bg_color=(70,70,70))
            if gender_label:
                draw_label(frame, f"Gender: {gender_label} ({gender_conf:.1f}%)", 10, 80, bg_color=(70,70,70))
            draw_label(frame, fps_text, 10, 115, bg_color=(70,70,70))

            cv2.imshow("Face Emotion & Gender Detector", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
