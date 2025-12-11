"""
File handling utilities for the Road Safety AI System
"""

import os
import shutil
from pathlib import Path
from typing import List, Optional
from loguru import logger


def ensure_dir(directory: str) -> Path:
    """
    Ensure directory exists, create if not
    
    Args:
        directory: Directory path
        
    Returns:
        Path object of the directory
    """
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_image_files(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Get all image files from a directory
    
    Args:
        directory: Directory to search
        extensions: List of valid extensions (default: ['.jpg', '.jpeg', '.png', '.bmp'])
        
    Returns:
        List of image file paths
    """
    if extensions is None:
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
    
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    image_files = []
    for ext in extensions:
        image_files.extend(path.glob(f"*{ext}"))
        image_files.extend(path.glob(f"*{ext.upper()}"))
    
    return sorted(image_files)


def get_video_files(directory: str, extensions: Optional[List[str]] = None) -> List[Path]:
    """
    Get all video files from a directory
    
    Args:
        directory: Directory to search
        extensions: List of valid extensions (default: ['.mp4', '.avi', '.mov', '.mkv'])
        
    Returns:
        List of video file paths
    """
    if extensions is None:
        extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv']
    
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return []
    
    video_files = []
    for ext in extensions:
        video_files.extend(path.glob(f"*{ext}"))
        video_files.extend(path.glob(f"*{ext.upper()}"))
    
    return sorted(video_files)


def copy_files(src_files: List[Path], dst_dir: str, create_dir: bool = True) -> int:
    """
    Copy multiple files to destination directory
    
    Args:
        src_files: List of source file paths
        dst_dir: Destination directory
        create_dir: Create destination directory if not exists
        
    Returns:
        Number of files copied
    """
    if create_dir:
        ensure_dir(dst_dir)
    
    dst_path = Path(dst_dir)
    copied = 0
    
    for src in src_files:
        try:
            shutil.copy2(src, dst_path / src.name)
            copied += 1
        except Exception as e:
            logger.error(f"Failed to copy {src}: {e}")
    
    logger.info(f"Copied {copied}/{len(src_files)} files to {dst_dir}")
    return copied


def move_files(src_files: List[Path], dst_dir: str, create_dir: bool = True) -> int:
    """
    Move multiple files to destination directory
    
    Args:
        src_files: List of source file paths
        dst_dir: Destination directory
        create_dir: Create destination directory if not exists
        
    Returns:
        Number of files moved
    """
    if create_dir:
        ensure_dir(dst_dir)
    
    dst_path = Path(dst_dir)
    moved = 0
    
    for src in src_files:
        try:
            shutil.move(str(src), str(dst_path / src.name))
            moved += 1
        except Exception as e:
            logger.error(f"Failed to move {src}: {e}")
    
    logger.info(f"Moved {moved}/{len(src_files)} files to {dst_dir}")
    return moved


def clean_directory(directory: str, keep_extensions: Optional[List[str]] = None) -> int:
    """
    Clean directory by removing files not matching keep_extensions
    
    Args:
        directory: Directory to clean
        keep_extensions: Extensions to keep (None = remove all)
        
    Returns:
        Number of files removed
    """
    path = Path(directory)
    if not path.exists():
        logger.warning(f"Directory does not exist: {directory}")
        return 0
    
    removed = 0
    for item in path.iterdir():
        if item.is_file():
            if keep_extensions is None or item.suffix.lower() not in keep_extensions:
                try:
                    item.unlink()
                    removed += 1
                except Exception as e:
                    logger.error(f"Failed to remove {item}: {e}")
    
    logger.info(f"Removed {removed} files from {directory}")
    return removed


def get_file_size_mb(file_path: str) -> float:
    """
    Get file size in MB
    
    Args:
        file_path: Path to file
        
    Returns:
        File size in megabytes
    """
    return Path(file_path).stat().st_size / (1024 * 1024)


def validate_file_exists(file_path: str, file_type: str = "File") -> bool:
    """
    Validate that a file exists
    
    Args:
        file_path: Path to file
        file_type: Type description for logging
        
    Returns:
        True if file exists, False otherwise
    """
    if not os.path.exists(file_path):
        logger.error(f"{file_type} not found: {file_path}")
        return False
    return True