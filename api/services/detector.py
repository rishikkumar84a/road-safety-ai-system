from ultralytics import YOLO
import cv2
import numpy as np
import os
import sys

# Import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.lane_detection import detect_lane_lines

class DetectionService:
    def __init__(self, model_path='yolov8n.pt'):
        self.model = YOLO(model_path)
    
    def process_image(self, image_bytes):
        # Convert bytes to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Lane Detection
        left, right, drift = detect_lane_lines(frame)
        
        # Object Detection
        results = self.model(frame)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = self.model.names[cls]
                
                detections.append({
                    "box": {"x1": x1, "y1": y1, "x2": x2, "y2": y2},
                    "confidence": conf,
                    "class_id": cls,
                    "class_name": name
                })
                
        return {
            "detections": detections,
            "lane_status": {
                "drift_status": drift,
                "left_line_detected": left is not None,
                "right_line_detected": right is not None
            }
        }
