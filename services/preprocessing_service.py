import os
import json
import numpy as np
from typing import Tuple

SEQUENCE_LENGTH = 80   # increased for longer signs
FEATURES = 126         # 2 hands × 63

def preprocess_file(file_path: str, sequence_length:int = SEQUENCE_LENGTH, features:int = FEATURES) -> np.ndarray:
    data = np.load(file_path)  # [frames, 63] or [frames,126]
    num_frames, num_features = data.shape

    # Handle single-hand input
    if num_features == 63:
        padded = np.zeros((num_frames, features))
        padded[:, :63] = data
        data = padded
    elif num_features != features:
        raise ValueError(f"Unexpected features ({num_features}) in {file_path}")

    # Pad or sample
    if num_frames < sequence_length:
        padding = np.zeros((sequence_length - num_frames, features))
        data = np.concatenate((data, padding), axis=0)
    else:
        indices = np.linspace(0, num_frames - 1, num=sequence_length, dtype=int)
        data = data[indices]

    return data

def batch_preprocess(data_dir: str, output_dir: str) -> Tuple[int, dict]:
    """
    Scans data_dir for label subfolders each containing .npy files.
    Writes processed arrays and label_map.json to output_dir.
    Returns (num_processed_files, label_map).
    """
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"{data_dir} does not exist")

    X = []
    y = []
    label_map = {}
    os.makedirs(output_dir, exist_ok=True)

    for idx, label in enumerate(sorted(os.listdir(data_dir))):
        label_path = os.path.join(data_dir, label)
        if not os.path.isdir(label_path):
            continue
        label_map[label] = idx
        for fname in sorted(os.listdir(label_path)):
            if not fname.endswith(".npy"):
                continue
            fpath = os.path.join(label_path, fname)
            processed = preprocess_file(fpath)
            X.append(processed)
            y.append(label)

    X = np.array(X)   # [samples, 80, 126]
    y = np.array(y)   # [samples,]
    np.save(os.path.join(output_dir, "X_processed.npy"), X)
    np.save(os.path.join(output_dir, "y_labels.npy"), y)
    with open(os.path.join(output_dir, "label_map.json"), "w") as f:
        json.dump(label_map, f)

    return len(X), label_map
