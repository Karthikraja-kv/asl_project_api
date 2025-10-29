# preprocess_router.py

from fastapi import APIRouter, HTTPException
import os
from services.preprocessing_service import batch_preprocess

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
NPY_DATA_DIR = os.path.join(PROJECT_ROOT, "npy_data")
PROCESSED_DIR = os.path.join(PROJECT_ROOT, "processed")

@router.post("/")
async def preprocess_endpoint():
    """
    Preprocess all npy files in npy_data/ and write outputs to processed/
    """
    try:
        count, label_map = batch_preprocess(NPY_DATA_DIR, PROCESSED_DIR)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    return {"processed_files": count, "label_map": label_map, "processed_dir": PROCESSED_DIR}
