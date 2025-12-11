import os
import sys

def check_file_exists(path: str) -> bool:
    """Checks if a file exists."""
    return os.path.exists(path)

def create_dir_if_not_exists(path: str):
    """Creates directory if it doesn't exist."""
    if not os.path.exists(path):
        os.makedirs(path)

def get_project_root() -> str:
    """Returns the project root directory."""
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
