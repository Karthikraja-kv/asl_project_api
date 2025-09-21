import os
from fastapi import FastAPI

from routers import extract_router, preprocess_router, train_router, predict_router

app = FastAPI(title="ASL Project API")

# Ensure runtime folders exist
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "uploads"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "npy_data"), exist_ok=True)
os.makedirs(os.path.join(BASE_DIR, "processed"), exist_ok=True)

# Include routers
app.include_router(extract_router.router, prefix="/extract", tags=["extract"])
app.include_router(preprocess_router.router, prefix="/preprocess", tags=["preprocess"])
app.include_router(train_router.router, prefix="/train", tags=["train"])
app.include_router(predict_router.router, prefix="/predict", tags=["predict"])
