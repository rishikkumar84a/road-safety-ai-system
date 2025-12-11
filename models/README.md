# Models Directory

This directory stores trained YOLOv8 model weights.

## Contents

- `best.pt` - Best model from training (lowest validation loss)
- `last.pt` - Last checkpoint from training
- Custom exported models (ONNX, TensorRT, etc.)

## Usage

### After Training

Models are automatically saved here by the training script:
```bash
python training/train.py --config training/config.yaml
```

Output location:
```
runs/train/road_safety_model/weights/best.pt
```

Copy to this directory:
```bash
cp runs/train/road_safety_model/weights/best.pt models/best.pt
```

### For Inference

```bash
python inference/realtime.py --model models/best.pt --source 0
```

### For API

Set in `.env` or `api/config.py`:
```
MODEL_PATH=models/best.pt
```

## Model Files

Model files (.pt) are excluded from git due to size.
Add your trained models here locally.