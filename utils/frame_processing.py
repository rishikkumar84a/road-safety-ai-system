import cv2
import numpy as np

def resize_frame(frame, width=640):
    """Resizes frame maintaining aspect ratio."""
    h, w = frame.shape[:2]
    scale = width / w
    return cv2.resize(frame, (width, int(h * scale)))

def normalize_frame(frame):
    """Normalizes frame pixel values to 0-1 range."""
    return frame.astype(np.float32) / 255.0
