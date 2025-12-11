"""
Example Usage Script
Demonstrates basic usage of the Road Safety AI System
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

def example_1_train_model():
    """Example: Train a model"""
    print("\n" + "="*60)
    print("EXAMPLE 1: Training a Model")
    print("="*60)
    
    print("\nCommand to train a model:")
    print("  python training/train.py --config training/config.yaml")
    
    print("\nWhat happens:")
    print("  1. Loads dataset from dataset/data.yaml")
    print("  2. Initializes YOLOv8 with pretrained weights")
    print("  3. Trains for configured epochs (default: 100)")
    print("  4. Saves best model to runs/train/road_safety_model/weights/best.pt")
    print("  5. Generates training metrics and plots")
    
    print("\nTips:")
    print("  - Adjust batch_size in config.yaml if GPU memory is low")
    print("  - Use yolov8n for faster training, yolov8m for better accuracy")
    print("  - Training typically takes 2-6 hours depending on dataset size")


def example_2_webcam_inference():
    """Example: Real-time webcam inference"""
    print("\n" + "="*60)
    print("EXAMPLE 2: Real-time Webcam Inference")
    print("="*60)
    
    print("\nCommand to run webcam inference:")
    print("  python inference/realtime.py --model models/best.pt --source 0")
    
    print("\nWhat happens:")
    print("  1. Loads trained model")
    print("  2. Opens webcam (camera ID 0)")
    print("  3. Processes frames in real-time")
    print("  4. Draws bounding boxes and labels")
    print("  5. Detects lane lines and drift")
    print("  6. Shows hazard warnings (fire, smoke)")
    
    print("\nKeyboard controls:")
    print("  q - Quit")
    print("  s - Save screenshot")


def example_3_api_server():
    """Example: Start API server"""
    print("\n" + "="*60)
    print("EXAMPLE 3: REST API Server")
    print("="*60)
    
    print("\nCommand to start API server:")
    print("  python api/main.py")
    
    print("\nServer endpoints:")
    print("  GET  /health - Health check")
    print("  POST /predict/image - Detect objects (JSON)")
    print("  POST /predict/image/annotated - Get annotated image")
    print("  POST /predict/video - Process video")
    print("  GET  /model/info - Model information")
    print("  GET  /docs - Interactive API documentation")
    
    print("\nExample API call:")
    print('  curl -X POST "http://localhost:8000/predict/image" \\')
    print('    -F "file=@test.jpg" \\')
    print('    -F "confidence=0.25"')


def example_4_python_client():
    """Example: Python API client"""
    print("\n" + "="*60)
    print("EXAMPLE 4: Python API Client")
    print("="*60)
    
    code = '''
import requests

def detect_objects(image_path):
    """Detect objects in image using API"""
    url = "http://localhost:8000/predict/image"
    
    with open(image_path, 'rb') as f:
        response = requests.post(
            url,
            files={'file': f},
            data={'confidence': 0.25}
        )
    
    result = response.json()
    
    print(f"Found {result['num_detections']} objects:")
    for det in result['detections']:
        print(f"  - {det['class_name']}: {det['confidence']:.2f}")
    
    return result

# Usage
result = detect_objects('road_image.jpg')
'''
    
    print("\nPython code example:")
    print(code)


def example_5_video_processing():
    """Example: Process video file"""
    print("\n" + "="*60)
    print("EXAMPLE 5: Video File Processing")
    print("="*60)
    
    print("\nCommand to process video:")
    print("  python inference/realtime.py \\")
    print("    --model models/best.pt \\")
    print("    --source input_video.mp4 \\")
    print("    --output output_video.mp4 \\")
    print("    --conf 0.25")
    
    print("\nWhat happens:")
    print("  1. Loads video file")
    print("  2. Processes each frame")
    print("  3. Draws detections on frames")
    print("  4. Saves annotated video to output file")
    print("  5. Shows progress during processing")


def example_6_batch_images():
    """Example: Batch image processing"""
    print("\n" + "="*60)
    print("EXAMPLE 6: Batch Image Processing")
    print("="*60)
    
    code = '''
from pathlib import Path
from ultralytics import YOLO

# Load model
model = YOLO('models/best.pt')

# Process all images in directory
image_dir = Path('test_images')
results = []

for img_path in image_dir.glob('*.jpg'):
    result = model.predict(
        str(img_path),
        conf=0.25,
        save=True,
        project='runs/detect',
        name='batch_results'
    )
    results.append(result)
    print(f"Processed: {img_path.name}")

print(f"Processed {len(results)} images")
print(f"Results saved to: runs/detect/batch_results/")
'''
    
    print("\nBatch processing code:")
    print(code)


def example_7_custom_detection():
    """Example: Custom detection with utilities"""
    print("\n" + "="*60)
    print("EXAMPLE 7: Custom Detection Pipeline")
    print("="*60)
    
    code = '''
import cv2
from ultralytics import YOLO
from utils.drawing import draw_multiple_boxes
from utils.preprocessing import resize_with_aspect_ratio
from utils.fps_counter import FPSCounter

# Initialize
model = YOLO('models/best.pt')
fps_counter = FPSCounter()

# Load and preprocess image
image = cv2.imread('test_image.jpg')
resized, scale = resize_with_aspect_ratio(image, target_size=640)

# Run detection
results = model.predict(resized, conf=0.25, verbose=False)[0]

# Parse detections
detections = []
for box in results.boxes:
    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
    detections.append({
        'box': (x1, y1, x2, y2),
        'confidence': float(box.conf[0]),
        'class_id': int(box.cls[0]),
        'class_name': model.names[int(box.cls[0])]
    })

# Draw results
annotated = draw_multiple_boxes(resized, detections, model.names)

# Update FPS
fps = fps_counter.update()
print(f"Detection FPS: {fps:.1f}")

# Save result
cv2.imwrite('output.jpg', annotated)
'''
    
    print("\nCustom pipeline code:")
    print(code)


def main():
    """Run all examples"""
    print("\n" + "#"*60)
    print("# Road Safety AI System - Usage Examples")
    print("#"*60)
    
    example_1_train_model()
    example_2_webcam_inference()
    example_3_api_server()
    example_4_python_client()
    example_5_video_processing()
    example_6_batch_images()
    example_7_custom_detection()
    
    print("\n" + "="*60)
    print("For more information, see:")
    print("  - README.md - Complete documentation")
    print("  - docs/QUICKSTART.md - Quick start guide")
    print("  - docs/API_USAGE.md - API examples")
    print("="*60 + "\n")


if __name__ == '__main__':
    main()