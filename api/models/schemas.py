from pydantic import BaseModel
from typing import List, Optional

class BoundingBox(BaseModel):
    x1: float
    y1: float
    x2: float
    y2: float

class Detection(BaseModel):
    box: BoundingBox
    confidence: float
    class_id: int
    class_name: str

class LaneStatus(BaseModel):
    drift_status: str
    left_line_detected: bool
    right_line_detected: bool

class AnalysisResponse(BaseModel):
    detections: List[Detection]
    lane_status: LaneStatus
    fps: Optional[float] = None
