"""
Inference Service for Road Safety AI
Handles model loading and prediction logic
"""

import cv2
import numpy as np
from ultralytics import YOLO
from loguru import logger
import time
from pathlib import Path
from io import BytesIO
from typing import Dict, List
import tempfile

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from api.models.schemas import Detection, BoundingBox, DetectionResponse
from utils.drawing import draw_multiple_boxes

class InferenceService:
    """Service for handling inference operations"""
    
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize inference service
        
        Args:
            model_path: Path to YOLOv8 model
            confidence_threshold: Default confidence threshold
            iou_threshold: Default IoU threshold
        """
        self.model_path = model_path
        self.default_conf = confidence_threshold
        self.default_iou = iou_threshold
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        logger.info(f"Model loaded with {len(self.class_names)} classes")
    
    async def predict_image(
        self,
        image_bytes: bytes,
        confidence_threshold: float = None,
        iou_threshold: float = None
    ) -> DetectionResponse:
        """
        Run inference on image bytes
        
        Args:
            image_bytes: Image file bytes
            confidence_threshold: Confidence threshold (uses default if None)
            iou_threshold: IoU threshold (uses default if None)
            
        Returns:
            DetectionResponse with detection results
        """
        conf = confidence_threshold or self.default_conf
        iou = iou_threshold or self.default_iou
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        h, w = image.shape[:2]
        
        # Run inference
        start_time = time.time()
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]
        processing_time = (time.time() - start_time) * 1000
        
        # Parse detections
        detections = []
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                detection = Detection(
                    class_id=class_id,
                    class_name=self.class_names[class_id],
                    confidence=confidence,
                    bounding_box=BoundingBox(
                        x1=float(x1),
                        y1=float(y1),
                        x2=float(x2),
                        y2=float(y2)
                    )
                )
                detections.append(detection)
        
        return DetectionResponse(
            success=True,
            detections=detections,
            num_detections=len(detections),
            image_width=w,
            image_height=h,
            processing_time_ms=processing_time
        )
    
    async def predict_image_annotated(
        self,
        image_bytes: bytes,
        confidence_threshold: float = None,
        iou_threshold: float = None
    ) -> BytesIO:
        """
        Run inference and return annotated image
        
        Args:
            image_bytes: Image file bytes
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            BytesIO object containing annotated JPEG image
        """
        conf = confidence_threshold or self.default_conf
        iou = iou_threshold or self.default_iou
        
        # Decode image
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Failed to decode image")
        
        # Run inference
        results = self.model.predict(
            image,
            conf=conf,
            iou=iou,
            verbose=False
        )[0]
        
        # Parse detections for drawing
        detections_list = []
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                detections_list.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                })
        
        # Draw detections
        annotated = draw_multiple_boxes(image, detections_list, self.class_names)
        
        # Encode to JPEG
        _, encoded = cv2.imencode('.jpg', annotated)
        return BytesIO(encoded.tobytes())
    
    async def process_video(
        self,
        video_path: str,
        confidence_threshold: float = None,
        iou_threshold: float = None
    ) -> str:
        """
        Process video file and return path to annotated video
        
        Args:
            video_path: Path to input video
            confidence_threshold: Confidence threshold
            iou_threshold: IoU threshold
            
        Returns:
            Path to output video
        """
        conf = confidence_threshold or self.default_conf
        iou = iou_threshold or self.default_iou
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output path
        output_path = tempfile.NamedTemporaryFile(
            delete=False, suffix='.mp4'
        ).name
        
        # Setup video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(
            output_path, fourcc, fps, (frame_width, frame_height)
        )
        
        try:
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Run inference
                results = self.model.predict(
                    frame,
                    conf=conf,
                    iou=iou,
                    verbose=False
                )[0]
                
                # Parse and draw detections
                detections_list = []
                boxes = results.boxes
                
                if boxes is not None and len(boxes) > 0:
                    for box in boxes:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        confidence = float(box.conf[0].cpu().numpy())
                        class_id = int(box.cls[0].cpu().numpy())
                        
                        detections_list.append({
                            'box': (x1, y1, x2, y2),
                            'confidence': confidence,
                            'class_id': class_id,
                            'class_name': self.class_names[class_id]
                        })
                
                annotated = draw_multiple_boxes(frame, detections_list, self.class_names)
                writer.write(annotated)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count} frames")
        
        finally:
            cap.release()
            writer.release()
        
        logger.info(f"Video processing complete: {frame_count} frames")
        return output_path
    
    def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up inference service")