from fastapi import APIRouter, Form, HTTPException
import os
from services.training_service import train_model

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")
MODEL_PATH = os.path.join(PROJECT_ROOT, "asl_lstm_model.keras")

@router.post("/")
async def train_endpoint(epochs: int = Form(50), batch_size: int = Form(8)):
    """
    Train the LSTM on processed data (runs synchronously).
    Returns accuracy or LOO results.
    """
    try:
        result = train_model(PROCESSED_DIR, MODEL_PATH, epochs=epochs, batch_size=batch_size)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"model_path": MODEL_PATH, "train_result": result}
