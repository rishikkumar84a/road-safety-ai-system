import cv2
import numpy as np

def detect_lane_lines(frame):
    """
    Basic lane detection using Color masking + Canny + Hough Transform.
    Returns left_line, right_line coordinates and drift status.
    """
    height, width = frame.shape[:2]
    
    # Region of Interest (ROI) - typically lower half triangle
    mask = np.zeros_like(frame)
    polygons = np.array([
        [(0, height), (width // 2, height // 2), (width, height)]
    ])
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Gaussian Blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Canny Edge Detection
    canny = cv2.Canny(blur, 50, 150)
    
    # Apply ROI mask
    mask = np.zeros_like(canny)
    cv2.fillPoly(mask, polygons, 255)
    masked_image = cv2.bitwise_and(canny, mask)
    
    # Hough Transform
    lines = cv2.HoughLinesP(masked_image, 2, np.pi/180, 100, np.array([]), minLineLength=40, maxLineGap=5)
    
    left_fit = []
    right_fit = []
    
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line.reshape(4)
            parameters = np.polyfit((x1, x2), (y1, y2), 1)
            slope = parameters[0]
            intercept = parameters[1]
            
            if slope < 0:
                left_fit.append((slope, intercept))
            else:
                right_fit.append((slope, intercept))
    
    left_line = make_coordinates(frame, np.average(left_fit, axis=0)) if left_fit else None
    right_line = make_coordinates(frame, np.average(right_fit, axis=0)) if right_fit else None
    
    drift_status = calculate_drift(width, left_line, right_line)
    
    return left_line, right_line, drift_status

def make_coordinates(image, line_parameters):
    if line_parameters is None or np.isnan(line_parameters).any():
        return None
    slope, intercept = line_parameters
    y1 = image.shape[0]
    y2 = int(y1 * (3/5))
    
    try:
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return ((x1, y1), (x2, y2))
    except OverflowError:
        return None

def calculate_drift(frame_width, left_line, right_line):
    """
    Estimates drift based on the center of the lane vs center of the frame.
    """
    if left_line is None or right_line is None:
        return "Unknown"
        
    lane_center = (left_line[0][0] + right_line[0][0]) / 2
    frame_center = frame_width / 2
    
    offset = lane_center - frame_center
    
    # Threshold for drift (pixels)
    threshold = 50
    
    if offset > threshold:
        return "Drifting Right"
    elif offset < -threshold:
        return "Drifting Left"
    else:
        return "Normal"
