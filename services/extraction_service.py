# services/extraction_service.py

import os
import cv2
import mediapipe as mp
import numpy as np
from typing import Tuple

FEATURES = 126  # 2 hands × 63


def extract_hand_landmarks(video_path: str, output_npy_path: str) -> int:
    """
    Extract per-frame 2-hand landmarks and save as .npy.
    Returns number of frames saved.
    """
    os.makedirs(os.path.dirname(output_npy_path), exist_ok=True)

    mp_hands = mp.solutions.hands
    landmarks_all_frames = []

    cap = cv2.VideoCapture(video_path)
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
                    handedness = results.multi_handedness[idx].classification[0].label  # "Left"/"Right"
                    coords = []
                    for lm in hand_landmarks.landmark:
                        coords.extend([lm.x, lm.y, lm.z])  # 21*3 = 63
                    if handedness == "Left":
                        frame_landmarks[0:63] = coords
                    else:
                        frame_landmarks[63:126] = coords

            landmarks_all_frames.append(frame_landmarks)

    cap.release()

    arr = np.array(landmarks_all_frames)
    np.save(output_npy_path, arr)
    return arr.shape[0]


def batch_extract_from_samples(samples_dir: str, npy_output_dir: str) -> dict:
    """
    Go through samples/<label>/*.mov and extract hand landmarks.
    Save to npy_data/<label>/*.npy.
    Returns a dict of files processed.
    """
    summary = {}

    for label in os.listdir(samples_dir):
        label_folder = os.path.join(samples_dir, label)
        if not os.path.isdir(label_folder):
            continue

        output_label_folder = os.path.join(npy_output_dir, label)
        os.makedirs(output_label_folder, exist_ok=True)

        for file_name in os.listdir(label_folder):
            if not file_name.endswith(".mov"):
                continue

            video_path = os.path.join(label_folder, file_name)
            npy_name = os.path.splitext(file_name)[0] + ".npy"
            output_npy_path = os.path.join(output_label_folder, npy_name)

            try:
                num_frames = extract_hand_landmarks(video_path, output_npy_path)
                summary[video_path] = {
                    "output": output_npy_path,
                    "frames": num_frames,
                    "status": "success"
                }
            except Exception as e:
                summary[video_path] = {
                    "output": output_npy_path,
                    "frames": 0,
                    "status": f"error: {str(e)}"
                }

    return summary
