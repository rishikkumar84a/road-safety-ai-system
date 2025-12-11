"""
Drawing utilities for visualization
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Optional


# Color palette for different classes
COLOR_PALETTE = {
    'vehicle': (0, 255, 0),          # Green
    'pedestrian': (255, 0, 0),       # Blue
    'bicycle': (0, 255, 255),        # Yellow
    'motorcycle': (255, 255, 0),     # Cyan
    'fire': (0, 0, 255),             # Red
    'smoke': (128, 128, 128),        # Gray
    'traffic_sign_stop': (0, 0, 200), # Dark Red
    'traffic_sign_yield': (0, 165, 255), # Orange
    'traffic_sign_speed_limit': (255, 0, 255), # Magenta
    'lane_line': (255, 255, 255)     # White
}


def get_color(class_name: str, class_id: Optional[int] = None) -> Tuple[int, int, int]:
    """
    Get color for a specific class
    
    Args:
        class_name: Name of the class
        class_id: Class ID (used if name not in palette)
        
    Returns:
        BGR color tuple
    """
    if class_name in COLOR_PALETTE:
        return COLOR_PALETTE[class_name]
    
    # Generate color based on class_id
    if class_id is not None:
        np.random.seed(class_id)
        return tuple(np.random.randint(0, 255, 3).tolist())
    
    return (0, 255, 0)  # Default green


def draw_bounding_box(
    image: np.ndarray,
    box: Tuple[int, int, int, int],
    label: str,
    confidence: float,
    color: Optional[Tuple[int, int, int]] = None,
    thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box with label
    
    Args:
        image: Input image
        box: Bounding box (x1, y1, x2, y2)
        label: Class label
        confidence: Detection confidence
        color: Box color (auto-selected if None)
        thickness: Line thickness
        
    Returns:
        Image with drawn box
    """
    x1, y1, x2, y2 = map(int, box)
    
    if color is None:
        color = get_color(label)
    
    # Draw rectangle
    cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Prepare label text
    text = f"{label}: {confidence:.2f}"
    
    # Get text size
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_thickness = 1
    (text_width, text_height), baseline = cv2.getTextSize(
        text, font, font_scale, font_thickness
    )
    
    # Draw label background
    cv2.rectangle(
        image,
        (x1, y1 - text_height - baseline - 5),
        (x1 + text_width, y1),
        color,
        -1
    )
    
    # Draw label text
    cv2.putText(
        image,
        text,
        (x1, y1 - baseline - 2),
        font,
        font_scale,
        (255, 255, 255),
        font_thickness,
        cv2.LINE_AA
    )
    
    return image


def draw_multiple_boxes(
    image: np.ndarray,
    detections: List[Dict],
    class_names: Dict[int, str]
) -> np.ndarray:
    """
    Draw multiple bounding boxes on image
    
    Args:
        image: Input image
        detections: List of detection dictionaries with keys:
                   'box', 'class_id', 'confidence'
        class_names: Dictionary mapping class_id to class_name
        
    Returns:
        Image with all boxes drawn
    """
    result = image.copy()
    
    for det in detections:
        box = det['box']
        class_id = det['class_id']
        confidence = det['confidence']
        
        class_name = class_names.get(class_id, f"Class_{class_id}")
        color = get_color(class_name, class_id)
        
        result = draw_bounding_box(
            result, box, class_name, confidence, color
        )
    
    return result


def draw_lane_lines(
    image: np.ndarray,
    lane_points: List[np.ndarray],
    color: Tuple[int, int, int] = (0, 255, 0),
    thickness: int = 3
) -> np.ndarray:
    """
    Draw lane lines on image
    
    Args:
        image: Input image
        lane_points: List of lane line point arrays
        color: Line color
        thickness: Line thickness
        
    Returns:
        Image with lane lines drawn
    """
    result = image.copy()
    
    for points in lane_points:
        if len(points) >= 2:
            cv2.polylines(
                result,
                [points.astype(np.int32)],
                False,
                color,
                thickness,
                cv2.LINE_AA
            )
    
    return result


def draw_fps(
    image: np.ndarray,
    fps: float,
    position: Tuple[int, int] = (10, 30),
    color: Tuple[int, int, int] = (0, 255, 0)
) -> np.ndarray:
    """
    Draw FPS counter on image
    
    Args:
        image: Input image
        fps: FPS value
        position: Text position
        color: Text color
        
    Returns:
        Image with FPS text
    """
    text = f"FPS: {fps:.1f}"
    cv2.putText(
        image,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA
    )
    return image


def draw_warning(
    image: np.ndarray,
    warning_text: str,
    warning_type: str = 'danger',
    position: Optional[Tuple[int, int]] = None
) -> np.ndarray:
    """
    Draw warning message on image
    
    Args:
        image: Input image
        warning_text: Warning message
        warning_type: Type of warning ('danger', 'warning', 'info')
        position: Text position (center if None)
        
    Returns:
        Image with warning
    """
    h, w = image.shape[:2]
    
    # Select color based on warning type
    warning_colors = {
        'danger': (0, 0, 255),   # Red
        'warning': (0, 165, 255), # Orange
        'info': (255, 255, 0)    # Yellow
    }
    color = warning_colors.get(warning_type, (0, 255, 255))
    
    # Default position (center)
    if position is None:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 1.0
        thickness = 2
        (text_width, text_height), _ = cv2.getTextSize(
            warning_text, font, font_scale, thickness
        )
        position = ((w - text_width) // 2, h // 2)
    
    # Draw semi-transparent background
    overlay = image.copy()
    cv2.rectangle(
        overlay,
        (position[0] - 10, position[1] - 40),
        (position[0] + len(warning_text) * 20, position[1] + 10),
        (0, 0, 0),
        -1
    )
    image = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
    
    # Draw warning text
    cv2.putText(
        image,
        warning_text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        1.0,
        color,
        2,
        cv2.LINE_AA
    )
    
    return image


def create_detection_overlay(
    image: np.ndarray,
    detections: List[Dict],
    class_names: Dict[int, str],
    fps: Optional[float] = None,
    warnings: Optional[List[Tuple[str, str]]] = None
) -> np.ndarray:
    """
    Create complete detection overlay with boxes, FPS, and warnings
    
    Args:
        image: Input image
        detections: List of detections
        class_names: Class name mapping
        fps: FPS value (optional)
        warnings: List of (warning_text, warning_type) tuples
        
    Returns:
        Image with complete overlay
    """
    result = image.copy()
    
    # Draw detections
    result = draw_multiple_boxes(result, detections, class_names)
    
    # Draw FPS
    if fps is not None:
        result = draw_fps(result, fps)
    
    # Draw warnings
    if warnings:
        y_offset = 80
        for warning_text, warning_type in warnings:
            result = draw_warning(
                result, warning_text, warning_type,
                position=(10, y_offset)
            )
            y_offset += 40
    
    return result