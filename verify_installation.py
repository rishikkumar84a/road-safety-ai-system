"""
Installation Verification Script
Checks if all dependencies are installed correctly
"""

import sys
from importlib import import_module

# Required packages
REQUIRED_PACKAGES = {
    'ultralytics': 'YOLOv8',
    'torch': 'PyTorch',
    'cv2': 'OpenCV (opencv-python)',
    'numpy': 'NumPy',
    'fastapi': 'FastAPI',
    'uvicorn': 'Uvicorn',
    'pydantic': 'Pydantic',
    'yaml': 'PyYAML',
    'PIL': 'Pillow',
    'loguru': 'Loguru',
    'matplotlib': 'Matplotlib',
    'seaborn': 'Seaborn',
    'sklearn': 'scikit-learn',
    'pandas': 'Pandas'
}

def check_package(package_name, display_name):
    """Check if a package is installed"""
    try:
        module = import_module(package_name)
        version = getattr(module, '__version__', 'unknown')
        print(f"✓ {display_name:30s} - Version: {version}")
        return True
    except ImportError:
        print(f"✗ {display_name:30s} - NOT INSTALLED")
        return False

def check_cuda():
    """Check CUDA availability"""
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            device_name = torch.cuda.get_device_name(0)
            print(f"✓ CUDA Available               - {device_count} GPU(s): {device_name}")
            return True
        else:
            print(f"⚠ CUDA Not Available           - Will use CPU (slower)")
            return False
    except:
        print(f"✗ CUDA Check Failed            - Could not check CUDA status")
        return False

def check_directory_structure():
    """Check if project structure is correct"""
    import os
    
    required_dirs = [
        'dataset',
        'models',
        'training',
        'inference',
        'api',
        'utils',
        'docs',
        'annotations'
    ]
    
    print("\nChecking directory structure:")
    all_exist = True
    for dir_name in required_dirs:
        exists = os.path.isdir(dir_name)
        symbol = "✓" if exists else "✗"
        print(f"{symbol} {dir_name}/")
        all_exist = all_exist and exists
    
    return all_exist

def main():
    """Main verification"""
    print("="*60)
    print("Road Safety AI System - Installation Verification")
    print("="*60)
    print("\nChecking Python version:")
    print(f"✓ Python {sys.version.split()[0]}")
    
    print("\nChecking required packages:")
    all_installed = True
    for package, display in REQUIRED_PACKAGES.items():
        installed = check_package(package, display)
        all_installed = all_installed and installed
    
    print("\nChecking CUDA/GPU:")
    check_cuda()
    
    print()
    structure_ok = check_directory_structure()
    
    print("\n" + "="*60)
    if all_installed and structure_ok:
        print("✓ Installation verification PASSED!")
        print("\nYou're ready to use the Road Safety AI System.")
        print("\nNext steps:")
        print("1. Prepare your dataset (see dataset/README.md)")
        print("2. Train a model: python training/train.py --config training/config.yaml")
        print("3. Run inference: python inference/realtime.py --model models/best.pt --source 0")
    else:
        print("✗ Installation verification FAILED!")
        print("\nPlease install missing packages:")
        print("  pip install -r requirements.txt")
        
        if not structure_ok:
            print("\nSome directories are missing. Please ensure you're in the")
            print("project root directory.")
    
    print("="*60)

if __name__ == '__main__':
    main()