# routers/predict_router.py
from fastapi import APIRouter, UploadFile, File, HTTPException
import shutil
from pathlib import Path
from services.predict_service import predict_signs

# Create router WITHOUT prefix here
router = APIRouter(tags=["Predict"])

# Ensure uploads directory exists
UPLOAD_DIR = Path(__file__).resolve().parent.parent / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@router.post("/")
async def predict(file: UploadFile = File(...)):
    """
    Predict multiple ASL signs from an uploaded video.
    """
    try:
        # Save uploaded video temporarily
        temp_path = UPLOAD_DIR / file.filename
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run prediction
        predicted_labels = predict_signs(temp_path)

        # Remove temp file
        temp_path.unlink(missing_ok=True)

        return {"predicted_labels": predicted_labels}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")
