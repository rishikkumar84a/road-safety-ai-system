"""
Real-time Inference Pipeline for Road Safety AI System
Supports webcam, video file, and image input with YOLOv8
"""

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
from loguru import logger
import argparse
from typing import Optional, List, Dict, Tuple

import sys
sys.path.append(str(Path(__file__).parent.parent))

from utils.drawing import draw_multiple_boxes, draw_fps, draw_warning
from utils.lane_detection import LaneDetector
from utils.fps_counter import FPSCounter


class RoadSafetyInference:
    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.25,
        iou_threshold: float = 0.45
    ):
        """
        Initialize inference pipeline
        
        Args:
            model_path: Path to trained YOLOv8 model
            confidence_threshold: Minimum confidence for detections
            iou_threshold: IoU threshold for NMS
        """
        self.model_path = model_path
        self.conf_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        
        # Load model
        logger.info(f"Loading model from {model_path}")
        self.model = YOLO(model_path)
        self.class_names = self.model.names
        
        # Initialize utilities
        self.lane_detector = LaneDetector()
        self.fps_counter = FPSCounter()
        
        logger.info("Model loaded successfully")
        logger.info(f"Classes: {self.class_names}")
    
    def predict_image(self, image: np.ndarray) -> Tuple[np.ndarray, List[Dict]]:
        """
        Run inference on a single image
        
        Args:
            image: Input image (BGR)
            
        Returns:
            Tuple of (annotated_image, detections_list)
        """
        # Run YOLO inference
        results = self.model.predict(
            image,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            verbose=False
        )[0]
        
        # Parse detections
        detections = []
        boxes = results.boxes
        
        if boxes is not None and len(boxes) > 0:
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = float(box.conf[0].cpu().numpy())
                class_id = int(box.cls[0].cpu().numpy())
                
                detections.append({
                    'box': (x1, y1, x2, y2),
                    'confidence': confidence,
                    'class_id': class_id,
                    'class_name': self.class_names[class_id]
                })
        
        # Draw detections
        annotated = draw_multiple_boxes(image, detections, self.class_names)
        
        return annotated, detections
    
    def process_frame(
        self,
        frame: np.ndarray,
        detect_lanes: bool = True,
        show_fps: bool = True
    ) -> Tuple[np.ndarray, List[Dict], List[str]]:
        """
        Process a single video frame
        
        Args:
            frame: Input frame (BGR)
            detect_lanes: Whether to perform lane detection
            show_fps: Whether to show FPS counter
            
        Returns:
            Tuple of (annotated_frame, detections, warnings)
        """
        # Update FPS
        fps = self.fps_counter.update()
        
        # Run object detection
        annotated, detections = self.predict_image(frame)
        
        # Lane detection
        warnings = []
        if detect_lanes:
            lane_image, lane_lines = self.lane_detector.detect_lanes(frame)
            
            # Check for lane drift
            if len(lane_lines) >= 2:
                is_drifting, direction = self.lane_detector.check_lane_drift(
                    lane_lines, frame.shape
                )
                if is_drifting:
                    warnings.append((f"LANE DRIFT: {direction.upper()}", "warning"))
            
            # Overlay lanes
            annotated = cv2.addWeighted(annotated, 0.8, lane_image, 0.2, 0)
        
        # Check for hazards
        hazard_warnings = self._check_hazards(detections)
        warnings.extend(hazard_warnings)
        
        # Draw warnings
        if warnings:
            y_offset = 80
            for warning_text, warning_type in warnings:
                annotated = draw_warning(
                    annotated, warning_text, warning_type,
                    position=(10, y_offset)
                )
                y_offset += 40
        
        # Draw FPS
        if show_fps:
            annotated = draw_fps(annotated, fps)
        
        return annotated, detections, [w[0] for w in warnings]
    
    def _check_hazards(self, detections: List[Dict]) -> List[Tuple[str, str]]:
        """
        Check for safety hazards in detections
        
        Args:
            detections: List of detections
            
        Returns:
            List of (warning_text, warning_type) tuples
        """
        warnings = []
        
        for det in detections:
            class_name = det['class_name']
            confidence = det['confidence']
            
            # Fire detection
            if 'fire' in class_name.lower() and confidence > 0.5:
                warnings.append(("FIRE DETECTED!", "danger"))
            
            # Smoke detection
            elif 'smoke' in class_name.lower() and confidence > 0.5:
                warnings.append(("SMOKE DETECTED!", "warning"))
            
            # Pedestrian warning
            elif 'pedestrian' in class_name.lower() and confidence > 0.6:
                warnings.append(("PEDESTRIAN ALERT", "warning"))
        
        return warnings
    
    def run_webcam(
        self,
        camera_id: int = 0,
        display_window: str = "Road Safety AI",
        save_output: Optional[str] = None
    ):
        """
        Run inference on webcam feed
        
        Args:
            camera_id: Camera device ID
            display_window: Window name for display
            save_output: Path to save output video (optional)
        """
        logger.info(f"Starting webcam inference (camera {camera_id})")
        
        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            logger.error(f"Failed to open camera {camera_id}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                save_output, fourcc, fps, (frame_width, frame_height)
            )
            logger.info(f"Saving output to {save_output}")
        
        logger.info("Press 'q' to quit, 's' to save screenshot")
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    logger.warning("Failed to read frame")
                    break
                
                # Process frame
                annotated, detections, warnings = self.process_frame(frame)
                
                # Display
                cv2.imshow(display_window, annotated)
                
                # Save if enabled
                if writer:
                    writer.write(annotated)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{frame_count}.jpg"
                    cv2.imwrite(screenshot_path, annotated)
                    logger.info(f"Screenshot saved: {screenshot_path}")
                
                frame_count += 1
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info(f"Processed {frame_count} frames")
    
    def run_video(
        self,
        video_path: str,
        display_window: str = "Road Safety AI",
        save_output: Optional[str] = None
    ):
        """
        Run inference on video file
        
        Args:
            video_path: Path to input video
            display_window: Window name for display
            save_output: Path to save output video (optional)
        """
        logger.info(f"Processing video: {video_path}")
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            logger.error(f"Failed to open video: {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        logger.info(f"Video: {frame_width}x{frame_height} @ {fps} FPS, {total_frames} frames")
        
        # Setup video writer if saving
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(
                save_output, fourcc, fps, (frame_width, frame_height)
            )
            logger.info(f"Saving output to {save_output}")
        
        frame_count = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated, detections, warnings = self.process_frame(frame)
                
                # Display
                cv2.imshow(display_window, annotated)
                
                # Save if enabled
                if writer:
                    writer.write(annotated)
                
                # Handle key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Quit requested")
                    break
                
                frame_count += 1
                
                # Progress
                if frame_count % 100 == 0:
                    logger.info(f"Processed {frame_count}/{total_frames} frames")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            logger.info(f"Video processing complete: {frame_count} frames")
    
    def run_image(self, image_path: str, save_output: Optional[str] = None):
        """
        Run inference on single image
        
        Args:
            image_path: Path to input image
            save_output: Path to save output image (optional)
        """
        logger.info(f"Processing image: {image_path}")
        
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Failed to read image: {image_path}")
            return
        
        # Process
        annotated, detections = self.predict_image(image)
        
        # Log detections
        logger.info(f"Found {len(detections)} detections:")
        for det in detections:
            logger.info(f"  - {det['class_name']}: {det['confidence']:.2f}")
        
        # Save or display
        if save_output:
            cv2.imwrite(save_output, annotated)
            logger.info(f"Output saved to {save_output}")
        else:
            cv2.imshow("Road Safety AI - Image", annotated)
            logger.info("Press any key to close")
            cv2.waitKey(0)
            cv2.destroyAllWindows()


def main():
    """Main execution"""
    parser = argparse.ArgumentParser(
        description='Road Safety AI - Real-time Inference'
    )
    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to trained YOLOv8 model'
    )
    parser.add_argument(
        '--source', type=str, default='0',
        help='Input source: webcam ID, video path, or image path'
    )
    parser.add_argument(
        '--conf', type=float, default=0.25,
        help='Confidence threshold'
    )
    parser.add_argument(
        '--iou', type=float, default=0.45,
        help='IoU threshold for NMS'
    )
    parser.add_argument(
        '--output', type=str, default=None,
        help='Path to save output'
    )
    
    args = parser.parse_args()
    
    # Initialize inference
    inference = RoadSafetyInference(
        model_path=args.model,
        confidence_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    # Determine source type
    source = args.source
    
    if source.isdigit():
        # Webcam
        inference.run_webcam(
            camera_id=int(source),
            save_output=args.output
        )
    elif Path(source).suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp']:
        # Image
        inference.run_image(
            image_path=source,
            save_output=args.output
        )
    else:
        # Video
        inference.run_video(
            video_path=source,
            save_output=args.output
        )


if __name__ == '__main__':
    main()