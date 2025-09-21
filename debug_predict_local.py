# debug_predict_local.py
from services.predict_service_multi import (
    extract_landmarks_from_video,
    segment_signs,
    pad_or_sample,
    predict_signs,
    model,
    index_to_label,
)
import numpy as np

# ----------------------------
# Config
# ----------------------------
video = "/Users/kvl04/Documents/Clear/asl_project_api/predict_sign.mov"
print("🎬 Running debug for:", video)

# ----------------------------
# Extract landmarks
# ----------------------------
landmarks = extract_landmarks_from_video(video)
print(f"📹 Total frames extracted: {len(landmarks)}")

# ----------------------------
# Segment signs
# ----------------------------
segs = segment_signs(landmarks)
print(f"✂️ Segments detected: {len(segs)} → {[len(s) for s in segs]}")

# ----------------------------
# Debug each segment
# ----------------------------
for i, s in enumerate(segs):
    x = pad_or_sample(s)
    print(f"\n🔎 Segment {i}: raw_len={len(s)}, post_shape={x.shape}")

    X = np.expand_dims(x, axis=0)
    probs = model.predict(X, verbose=0)[0]
    top_idx = np.argmax(probs)
    top_label = index_to_label[top_idx]
    confidence = probs[top_idx]

    print(f"   🔮 Prediction: {top_label} (conf={confidence:.2f})")
    print(f"   📊 Full probs: {[f'{index_to_label[j]}:{probs[j]:.2f}' for j in range(len(probs))]}")

# ----------------------------
# Run final predictions (multi)
# ----------------------------
final_preds = predict_signs(video)
print("\n✅ Final predictions:", final_preds)
