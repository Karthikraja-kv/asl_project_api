# services/predict_service.py
from __future__ import annotations
import json
from pathlib import Path
from typing import Optional

import numpy as np
import tensorflow as tf
import cv2
import mediapipe as mp

# ----------------------------
# Constants / Paths
# ----------------------------
SEQUENCE_LENGTH = 80      # must match your preprocessing/training
FEATURES = 126            # 2 hands × (21 landmarks × 3)

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DIR = BASE_DIR / "processed"
MODEL_CANDIDATES = [
    BASE_DIR / "asl_lstm_model.keras",
    BASE_DIR / "asl_lstm_model.h5",
]
LABEL_MAP_PATH = PROCESSED_DIR / "label_map.json"

# ----------------------------
# Lazy singletons
# ----------------------------
_model: Optional[tf.keras.Model] = None
_index_to_label: Optional[dict[int, str]] = None

def model_path_available() -> Optional[Path]:
    """Return first existing model path or None."""
    for p in MODEL_CANDIDATES:
        if p.exists():
            return p
    return None

def label_map_available() -> bool:
    return LABEL_MAP_PATH.exists()

def load_label_map_if_needed() -> dict[int, str]:
    global _index_to_label
    if _index_to_label is not None:
        return _index_to_label

    if not LABEL_MAP_PATH.exists():
        raise FileNotFoundError(
            f"label_map.json not found at {LABEL_MAP_PATH}. "
            "Run /preprocess first to generate it."
        )
    with open(LABEL_MAP_PATH, "r") as f:
        label_map = json.load(f)          # {"label_name": idx}
    # invert -> {idx:int: label_name:str}
    _index_to_label = {int(v): k for k, v in label_map.items()}
    return _index_to_label

def load_model_if_needed() -> tf.keras.Model:
    global _model
    if _model is not None:
        return _model

    mpth = model_path_available()
    if mpth is None:
        raise FileNotFoundError(
            "Model not found. Expected one of: "
            + ", ".join(str(p) for p in MODEL_CANDIDATES)
            + ". Run /train to create it."
        )
    _model = tf.keras.models.load_model(str(mpth), compile=False)
    print(f"✅ Loaded model: {mpth}")
    return _model

# ----------------------------
# Landmark extraction (2 hands)
# ----------------------------
mp_hands = mp.solutions.hands

def extract_landmarks_from_video(video_path: Path) -> np.ndarray:
    cap = cv2.VideoCapture(str(video_path))
    seq = []

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

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(rgb)

            frame_vec = [0.0] * FEATURES  # [0:63]=Left, [63:126]=Right
            if results.multi_hand_landmarks and results.multi_handedness:
                for idx, hand_lms in enumerate(results.multi_hand_landmarks):
                    handed = results.multi_handedness[idx].classification[0].label
                    coords = []
                    for lm in hand_lms.landmark:
                        coords.extend([lm.x, lm.y, lm.z])  # 63
                    if handed == "Left":
                        frame_vec[0:63] = coords
                    else:
                        frame_vec[63:126] = coords

            seq.append(frame_vec)

    cap.release()
    return np.asarray(seq, dtype=np.float32)

def pad_or_sample(seq: np.ndarray) -> np.ndarray:
    n = len(seq)
    if n == 0:
        return seq
    if n < SEQUENCE_LENGTH:
        pad = np.zeros((SEQUENCE_LENGTH - n, FEATURES), dtype=seq.dtype)
        return np.vstack([seq, pad])
    # uniform sampling
    idx = np.linspace(0, n - 1, SEQUENCE_LENGTH, dtype=int)
    return seq[idx]

# ----------------------------
# Public API used by router
# ----------------------------
def predict_signs(video_path: Path) -> dict:
    """
    Returns a dict with either:
      {"label": "<predicted_label>", "confidence": float}
    or raises a FileNotFoundError with guidance.
    """
    # Lazy load requirements
    model = load_model_if_needed()
    idx2label = load_label_map_if_needed()

    seq = extract_landmarks_from_video(video_path)
    if len(seq) == 0:
        return {"error": "No hands detected in video."}

    X = pad_or_sample(seq)
    X = np.expand_dims(X, axis=0)   # (1, 80, 126)

    probs = model.predict(X, verbose=0)[0]
    pred_idx = int(np.argmax(probs))
    label = idx2label.get(pred_idx, f"<unknown:{pred_idx}>")
    conf = float(probs[pred_idx])
    return {"label": label, "confidence": conf}

def runtime_status() -> dict:
    """For /predict/health to avoid exceptions at import time."""
    mpth = model_path_available()
    return {
        "model_present": mpth is not None,
        "model_path": str(mpth) if mpth else None,
        "label_map_present": label_map_available(),
        "label_map_path": str(LABEL_MAP_PATH),
    }
