import cv2
import argparse
from ultralytics import YOLO
import sys
import os

# Add project root to path to import utils
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.lane_detection import detect_lane_lines
from utils.drawing_utils import draw_detections, draw_lane_lines
from utils.fps_counter import FPSCounter
from utils.frame_processing import resize_frame

def run_inference(source, model_path):
    # Load Model
    print(f"Loading model from {model_path}...")
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"Error loading model: {e}")
        print("Falling back to yolov8n.pt for demonstration")
        model = YOLO('yolov8n.pt')

    # Video Capture
    if source.isdigit():
        source = int(source)
    cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        print("Error: Could not open video source.")
        return

    fps_counter = FPSCounter()
    
    # Windows
    cv2.namedWindow("Road Safety AI System", cv2.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # 1. Preprocess
        # Optional: Resize for speed if needed
        # frame = resize_frame(frame, width=1280)
        
        # 2. Lane Detection (Algorithm)
        left_line, right_line, drift_status = detect_lane_lines(frame)
        
        # 3. Object Detection (YOLOv8)
        results = model(frame, verbose=False, stream=True)
        
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0]
                conf = box.conf[0]
                cls = box.cls[0]
                detections.append([x1, y1, x2, y2, conf, cls])
        
        # 4. Visualization
        # Draw Lane Lines
        frame = draw_lane_lines(frame, left_line, right_line, drift_status)
        
        # Draw Objects
        frame = draw_detections(frame, detections, model.names)
        
        # Draw FPS
        fps = fps_counter.update()
        cv2.putText(frame, f"FPS: {fps}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 5. Display
        cv2.imshow("Road Safety AI System", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='0', help='Video source: 0 for webcam or path to video file')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Path to trained .pt model')
    
    args = parser.parse_args()
    
    run_inference(args.source, args.model)
