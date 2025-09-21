# services/predict_service_multi.py

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
MOVEMENT_THRESHOLD = 0.002
MIN_FRAMES_PER_SIGN = 15
MIN_GAP = 10           # frames of "hands down" before cutting

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
    landmarks_seq = np.array(landmarks_seq)
    print(f"📹 Total frames extracted: {len(landmarks_seq)}")
    return landmarks_seq


def segment_signs(landmarks_seq):
    if len(landmarks_seq) == 0:
        return []

    segments = []
    start_idx = 0
    hands_down_count = 0

    for i in range(1, len(landmarks_seq)):
        movement = np.linalg.norm(landmarks_seq[i] - landmarks_seq[i-1])
        is_hands_down = np.all(np.abs(landmarks_seq[i]) < 1e-4)

        if is_hands_down:
            hands_down_count += 1
        else:
            hands_down_count = 0

        # Cut if hands down long enough
        if hands_down_count >= MIN_GAP:
            if i - start_idx >= MIN_FRAMES_PER_SIGN:
                segment = landmarks_seq[start_idx:i-hands_down_count]
                if np.any(np.linalg.norm(segment, axis=1) > 0.01):
                    segments.append(segment)
                    print(f"✂️ Segment [{start_idx}:{i-hands_down_count}] length={len(segment)}")
            start_idx = i

        # Alternative: cut on low movement after 60 frames
        elif movement < MOVEMENT_THRESHOLD and (i - start_idx) > 60:
            segment = landmarks_seq[start_idx:i]
            if len(segment) >= MIN_FRAMES_PER_SIGN and np.any(np.linalg.norm(segment, axis=1) > 0.01):
                segments.append(segment)
                print(f"✂️ Segment (low movement) [{start_idx}:{i}] length={len(segment)}")
            start_idx = i

    # Last segment
    last_segment = landmarks_seq[start_idx:]
    if len(last_segment) >= MIN_FRAMES_PER_SIGN and np.any(np.linalg.norm(last_segment, axis=1) > 0.01):
        segments.append(last_segment)
        print(f"✂️ Last segment [{start_idx}:end] length={len(last_segment)}")

    print(f"✅ Total segments detected: {len(segments)}")
    return segments


def pad_or_sample(seq):
    num_frames = len(seq)
    if num_frames < SEQUENCE_LENGTH:
        padding = np.zeros((SEQUENCE_LENGTH - num_frames, FEATURES))
        seq = np.vstack([seq, padding])
    else:
        indices = np.linspace(0, num_frames - 1, SEQUENCE_LENGTH, dtype=int)
        seq = seq[indices]
    return seq


def predict_multi_signs(video_path):
    landmarks_seq = extract_landmarks_from_video(video_path)
    segments = segment_signs(landmarks_seq)

    predictions = []
    for idx, seg in enumerate(segments):
        X = pad_or_sample(seg)
        X = np.expand_dims(X, axis=0)
        probs = model.predict(X, verbose=0)[0]
        pred_idx = np.argmax(probs)
        label = index_to_label[pred_idx]
        predictions.append(label)
        print(f"🔮 Segment {idx}: predicted={label} (len={len(seg)})")

    return predictions
