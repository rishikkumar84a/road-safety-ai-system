from fastapi import APIRouter, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from ..services.detector import DetectionService
from ..models.schemas import AnalysisResponse
import cv2
import numpy as np
import io

router = APIRouter()
detector = DetectionService()

@router.post("/predict/image", response_model=AnalysisResponse)
async def predict_image(file: UploadFile = File(...)):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    content = await file.read()
    result = detector.process_image(content)
    return result

@router.get("/health")
def health_check():
    return {"status": "healthy", "service": "Road Safety AI API"}
