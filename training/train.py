from ultralytics import YOLO
import os
import argparse

def train_model(data_config, epochs=50, img_size=640, model_variant='yolov8n.pt'):
    """
    Trains a YOLOv8 model.
    """
    # Load a model
    model = YOLO(model_variant)  # load a pretrained model (recommended for training)

    # Train the model
    results = model.train(
        data=data_config,
        epochs=epochs,
        imgsz=img_size,
        device='0' if os.environ.get('CUDA_VISIBLE_DEVICES') else 'cpu', # Use GPU if available
        project='../training/runs',
        name='road_safety_exp',
        exist_ok=True
    )
    
    # Validate
    metrics = model.val()
    print(f"mAP@50-95: {metrics.box.map}")
    
    # Export the model
    success = model.export(format='onnx')
    print(f"Model exported: {success}")
    
    return model

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train YOLOv8 for Road Safety')
    parser.add_argument('--config', type=str, default='../dataset/road_safety.yaml', help='Path to data.yaml')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--model', type=str, default='yolov8n.pt', help='Model variant (n, s, m, l, x)')
    
    args = parser.parse_args()
    
    # Ensure config path is absolute or correct relative
    config_path = os.path.abspath(args.config)
    
    print(f"Starting training with config: {config_path}")
    train_model(config_path, args.epochs, model_variant=args.model)
