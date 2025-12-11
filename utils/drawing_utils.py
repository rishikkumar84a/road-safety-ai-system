import cv2
import numpy as np

def draw_detections(frame, detections, classes):
    """
    Draws bounding boxes and labels on the frame.
    detections: List of [x1, y1, x2, y2, conf, cls]
    classes: Dictionary of class names
    """
    for det in detections:
        x1, y1, x2, y2, conf, cls = det
        label = f"{classes[int(cls)]} {conf:.2f}"
        
        # Color based on class (simple hash for consistent color)
        color_seed = int(cls) * 50
        color = (color_seed % 255, (color_seed * 2) % 255, (color_seed * 3) % 255)
        
        cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
        cv2.putText(frame, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return frame

def draw_lane_lines(frame, left_line, right_line, drift_status="Normal"):
    """
    Draws detected lane lines and drift status.
    left_line, right_line: Tuple of ((x1, y1), (x2, y2)) or None
    """
    overlay = frame.copy()
    
    if left_line:
        cv2.line(overlay, left_line[0], left_line[1], (255, 0, 0), 5) # Blue for left
    if right_line:
        cv2.line(overlay, right_line[0], right_line[1], (0, 0, 255), 5) # Red for right
        
    # Draw polygon between lines if both exist
    if left_line and right_line:
        pts = np.array([left_line[0], left_line[1], right_line[1], right_line[0]], np.int32)
        cv2.fillPoly(overlay, [pts], (0, 255, 0))
    
    # Blend with original
    cv2.addWeighted(overlay, 0.3, frame, 0.7, 0, frame)
    
    # Draw status
    color = (0, 255, 0) if drift_status == "Normal" else (0, 0, 255)
    cv2.putText(frame, f"Lane Status: {drift_status}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)
    
    return frame
