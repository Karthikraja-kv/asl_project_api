import numpy as np
import tensorflow as tf
import json
from pathlib import Path
import cv2
import mediapipe as mp

# ----------------------------
# Paths
# ----------------------------
BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "asl_lstm_model.keras"
LABEL_MAP_PATH = BASE_DIR / "processed" / "label_map.json"

# ----------------------------
# Constants
# ----------------------------
SEQUENCE_LENGTH = 80   # must match training
FEATURES = 126         # 2 hands × 63

# ----------------------------
# Load model
# ----------------------------
model = tf.keras.models.load_model(str(MODEL_PATH), compile=False)

# ----------------------------
# Load label map
# ----------------------------
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)
index_to_label = {int(v): k for k, v in label_map.items()}

# ----------------------------
# MediaPipe Hands setup
# ----------------------------
mp_hands = mp.solutions.hands

def extract_landmarks_from_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    landmarks_seq = []

    with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as hands:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            frame_landmarks = [0.0] * FEATURES
            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    handedness = results.multi_handedness[idx].classification[0].label
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])
                    if handedness == "Left":
                        frame_landmarks[0:63] = coords
                    else:
                        frame_landmarks[63:126] = coords

            landmarks_seq.append(frame_landmarks)

    cap.release()
    return np.array(landmarks_seq)

def pad_or_sample(seq):
    num_frames = len(seq)
    if num_frames < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - num_frames, FEATURES))
        seq = np.vstack([seq, padding])
    else:
        indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        seq = seq[indices]
    return seq

def predict_signs(video_path):
    landmarks_seq = extract_landmarks_from_video(video_path)
    if len(landmarks_seq) == 0:
        return "No hands detected"

    X = pad_or_sample(landmarks_seq)
    X = np.expand_dims(X, axis=0)
    probs = model.predict(X, verbose=0)[0]
    pred_idx = np.argmax(probs)
    label = index_to_label[pred_idx]

    return label
