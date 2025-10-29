# routers/predict_router.py
from pathlib import Path
import shutil
from fastapi import APIRouter, UploadFile, File, HTTPException, status

from services.predict_service import predict_signs, runtime_status

router = APIRouter(tags=["Predict"])
UPLOAD_DIR = Path(__file__).resolve().parents[1] / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

@router.get("/health")
def health():
    return runtime_status()

@router.post("/")
async def predict(file: UploadFile = File(...)):
    """
    Upload a video (.mp4/.mov), get predicted label.
    Will return 503 with guidance if model/labels are not ready yet.
    """
    if not file.filename.lower().endswith((".mp4", ".mov")):
        raise HTTPException(status_code=400, detail="Only .mp4 and .mov are supported.")

    temp_path = UPLOAD_DIR / file.filename
    try:
        with open(temp_path, "wb") as buf:
            shutil.copyfileobj(file.file, buf)

        result = predict_signs(temp_path)  # may raise FileNotFoundError (handled below)
        return {"result": result}

    except FileNotFoundError as e:
        # Model or label_map missing → clear guidance, no startup crash
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(e)
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {e}")
    finally:
        try:
            temp_path.unlink(missing_ok=True)
        except Exception:
            pass
