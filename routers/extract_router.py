# extract_router.py

from fastapi import APIRouter, UploadFile, File, Form, HTTPException
import os, shutil

from services.extraction_service import extract_hand_landmarks, batch_extract_from_samples

router = APIRouter()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(BASE_DIR, ".."))
NPY_DATA_DIR = os.path.join(PROJECT_ROOT, "npy_data")
SAMPLES_DIR = os.path.join(PROJECT_ROOT, "samples")


@router.post("/")
async def extract_endpoint(file: UploadFile = File(...), label: str = Form(...)):
    """
    Upload a video and save extracted landmarks as npy under npy_data/<label>/
    """
    if not file.filename.lower().endswith((".mp4", ".mov")):
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov supported")

    label_dir = os.path.join(NPY_DATA_DIR, label)
    os.makedirs(label_dir, exist_ok=True)

    video_path = os.path.join(label_dir, file.filename)
    npy_name = os.path.splitext(file.filename)[0] + ".npy"
    npy_path = os.path.join(label_dir, npy_name)

    with open(video_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        frames = extract_hand_landmarks(video_path, npy_path)
    except Exception as e:
        if os.path.exists(video_path):
            os.remove(video_path)
        raise HTTPException(status_code=500, detail=str(e))

    if os.path.exists(video_path):
        os.remove(video_path)

    return {"saved_npy": npy_path, "frames_extracted": frames, "label": label}


@router.post("/from-samples")
async def extract_from_samples():
    """
    Extract landmarks from all videos in samples/<label>/*.mov and save to npy_data/<label>/
    """
    if not os.path.exists(SAMPLES_DIR):
        raise HTTPException(status_code=400, detail="samples/ directory not found")

    summary = batch_extract_from_samples(SAMPLES_DIR, NPY_DATA_DIR)
    return {
        "message": "Extraction from samples complete",
        "summary": summary
    }
