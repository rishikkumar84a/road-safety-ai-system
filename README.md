# Road Safety AI System

A production-ready computer vision system for road safety monitoring. Detecting vehicles, pedestrians, traffic signs, fire/smoke hazards, and lane drift in real-time.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![YOLOv8](https://img.shields.io/badge/YOLO-v8-green)

## 📋 Features

- **Object Detection**: Vehicles, Pedestrians, Traffic Signs, Fire/Smoke.
- **Lane Drift Detection**: Real-time analysis of lane positioning with visual warnings.
- **Real-time Inference**: Optimized OpenCV pipeline for webcam or video feeds.
- **REST API**: Production-ready FastAPI backend for image analysis.
- **Training Pipeline**: Complete scripts to train custom models on your own datasets.

## 📂 Project Structure

```
road_safety_ai_system/
├── api/                  # FastAPI Backend
├── dataset/              # Dataset configuration
├── docs/                 # Documentation
├── inference/            # Real-time inference scripts
├── training/             # Model training scripts
├── utils/                # shared utilities
└── requirements.txt      # Dependencies
```

## 🚀 Getting Started

### 1. Installation

```bash
git clone https://github.com/your-username/road-safety-ai-system.git
cd road-safety-ai-system
pip install -r requirements.txt
```

### 2. Dataset Preparation

See [docs/DATASET_PREP.md](docs/DATASET_PREP.md) for detailed instructions on annotating and formatting your data for YOLOv8.

### 3. Model Training

To train the model on your dataset:

```bash
cd training
python train.py --epochs 50 --model yolov8n.pt
```

The best model will be saved to `training/runs/road_safety_exp/weights/best.pt`.

### 4. Real-time Inference

Run the detection system on a video file or webcam:

```bash
cd inference
# For webcam
python realtime_detection.py --source 0

# For video file
python realtime_detection.py --source path/to/video.mp4 --model ../training/runs/road_safety_exp/weights/best.pt
```

### 5. API Backend

Start the API server:

```bash
uvicorn api.main:app --reload
```

- **Swagger UI**: Visit `http://localhost:8000/docs` to test endpoints.
- **Endpoint**: POST `/api/v1/predict/image`

## 🛠 Tech Stack

- **Core**: Python 3.9+
- **Vision**: OpenCV, Ultralytics YOLOv8, PyTorch
- **API**: FastAPI, Uvicorn, Pydantic
- **Development**: VS Code, Windows/Linux

## 🔮 Future Improvements

- [ ] Integration with GPS for location-based hazard logging.
- [ ] Connect to Twilio API for SMS alerts on accident detection.
- [ ] TensorRT optimization for edge devices (Jetson Nano).
- [ ] Dashboard for historical analytics.

## 📄 License

This project is licensed under the MIT License.
