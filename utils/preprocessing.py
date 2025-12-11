"""
Image and frame preprocessing utilities
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def resize_with_aspect_ratio(
    image: np.ndarray,
    target_size: int = 640,
    keep_ratio: bool = True
) -> Tuple[np.ndarray, float]:
    """
    Resize image while maintaining aspect ratio
    
    Args:
        image: Input image
        target_size: Target size for the longest side
        keep_ratio: Whether to maintain aspect ratio
        
    Returns:
        Tuple of (resized_image, scale_factor)
    """
    h, w = image.shape[:2]
    
    if keep_ratio:
        scale = target_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
    else:
        new_w = new_h = target_size
        scale = 1.0
    
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return resized, scale


def letterbox(
    image: np.ndarray,
    new_shape: Tuple[int, int] = (640, 640),
    color: Tuple[int, int, int] = (114, 114, 114),
    auto: bool = True,
    scaleFill: bool = False,
    scaleup: bool = True,
    stride: int = 32
) -> Tuple[np.ndarray, Tuple[float, float], Tuple[int, int]]:
    """
    Resize and pad image while meeting stride-multiple constraints
    
    Args:
        image: Input image
        new_shape: Target shape (height, width)
        color: Padding color
        auto: Minimum rectangle
        scaleFill: Stretch to fill
        scaleup: Allow upscaling
        stride: Stride constraint
        
    Returns:
        Tuple of (padded_image, ratio, padding)
    """
    shape = image.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        image = cv2.resize(image, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    image = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    
    return image, ratio, (dw, dh)


def normalize_image(image: np.ndarray, mean: Optional[np.ndarray] = None, 
                    std: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Normalize image to [0, 1] or using mean and std
    
    Args:
        image: Input image
        mean: Mean values for normalization
        std: Standard deviation values for normalization
        
    Returns:
        Normalized image
    """
    image = image.astype(np.float32) / 255.0
    
    if mean is not None and std is not None:
        mean = np.array(mean, dtype=np.float32)
        std = np.array(std, dtype=np.float32)
        image = (image - mean) / std
    
    return image


def enhance_image(image: np.ndarray, brightness: float = 1.0, 
                 contrast: float = 1.0) -> np.ndarray:
    """
    Enhance image brightness and contrast
    
    Args:
        image: Input image
        brightness: Brightness factor (1.0 = no change)
        contrast: Contrast factor (1.0 = no change)
        
    Returns:
        Enhanced image
    """
    enhanced = cv2.convertScaleAbs(image, alpha=contrast, beta=(brightness - 1) * 255)
    return enhanced


def denoise_image(image: np.ndarray, strength: int = 10) -> np.ndarray:
    """
    Apply denoising to image
    
    Args:
        image: Input image
        strength: Denoising strength
        
    Returns:
        Denoised image
    """
    if len(image.shape) == 3:
        denoised = cv2.fastNlMeansDenoisingColored(image, None, strength, strength, 7, 21)
    else:
        denoised = cv2.fastNlMeansDenoising(image, None, strength, 7, 21)
    
    return denoised


def sharpen_image(image: np.ndarray) -> np.ndarray:
    """
    Sharpen image using kernel
    
    Args:
        image: Input image
        
    Returns:
        Sharpened image
    """
    kernel = np.array([[-1, -1, -1],
                      [-1,  9, -1],
                      [-1, -1, -1]])
    sharpened = cv2.filter2D(image, -1, kernel)
    return sharpened


def convert_color_space(image: np.ndarray, conversion: str = 'BGR2RGB') -> np.ndarray:
    """
    Convert image color space
    
    Args:
        image: Input image
        conversion: Conversion type (BGR2RGB, BGR2GRAY, etc.)
        
    Returns:
        Converted image
    """
    conversion_map = {
        'BGR2RGB': cv2.COLOR_BGR2RGB,
        'RGB2BGR': cv2.COLOR_RGB2BGR,
        'BGR2GRAY': cv2.COLOR_BGR2GRAY,
        'GRAY2BGR': cv2.COLOR_GRAY2BGR,
        'BGR2HSV': cv2.COLOR_BGR2HSV,
        'HSV2BGR': cv2.COLOR_HSV2BGR,
    }
    
    if conversion in conversion_map:
        return cv2.cvtColor(image, conversion_map[conversion])
    return image