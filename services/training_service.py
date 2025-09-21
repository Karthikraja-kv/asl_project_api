import os
import json
import numpy as np
import random
from typing import Union
from sklearn.model_selection import train_test_split, LeaveOneOut
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from tensorflow.keras.utils import to_categorical

SEQUENCE_LENGTH = 80   # must match preprocessing
FEATURES = 126

def create_model(input_shape, num_classes):
    model = Sequential([
        Input(shape=input_shape),                     # (80, 126)
        LSTM(128, return_sequences=False),
        Dropout(0.4),
        Dense(64, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train_model(processed_dir: str, model_path: str, epochs:int=50, batch_size:int=8) -> Union[float, dict]:
    X_path = os.path.join(processed_dir, "X_processed.npy")
    y_path = os.path.join(processed_dir, "y_labels.npy")
    label_map_path = os.path.join(processed_dir, "label_map.json")

    if not (os.path.exists(X_path) and os.path.exists(y_path) and os.path.exists(label_map_path)):
        raise FileNotFoundError("Processed data missing. Run preprocess first.")

    # reproducibility
    np.random.seed(42)
    random.seed(42)
    tf.random.set_seed(42)

    X = np.load(X_path)
    y = np.load(y_path)

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)
    num_classes = y_categorical.shape[1]

    if len(X) < 10:
        loo = LeaveOneOut()
        accs = []
        for train_idx, test_idx in loo.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y_categorical[train_idx], y_categorical[test_idx]
            model = create_model((SEQUENCE_LENGTH, FEATURES), num_classes)
            model.fit(X_train, y_train, epochs=20, batch_size=1, verbose=0)
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            accs.append(float(acc))
        model.save(model_path)
        return {"loo_accuracy_mean": float(np.mean(accs)), "per_split": accs}

    # normal training
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_categorical, test_size=0.2, stratify=y_categorical, random_state=42
    )
    model = create_model((SEQUENCE_LENGTH, FEATURES), num_classes)
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    model.save(model_path)
    return float(acc)
