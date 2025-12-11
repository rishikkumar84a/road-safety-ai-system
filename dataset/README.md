# Dataset Setup Guide

## Directory Structure

Organize your dataset with the following structure:

```
dataset/
├── data.yaml              # Dataset configuration
├── images/
│   ├── train/            # Training images
│   ├── val/              # Validation images
│   └── test/             # Test images (optional)
└── labels/
    ├── train/            # Training annotations
    ├── val/              # Validation annotations
    └── test/             # Test annotations (optional)
```

## Quick Start

### 1. Collect Images

Gather images from:
- Dashcam footage
- Traffic cameras
- Public datasets (COCO, BDD100K, etc.)
- Custom recordings

**Recommended:**
- Training: 1000+ images
- Validation: 200+ images
- Test: 100+ images

### 2. Annotate Images

Use annotation tools to label objects:

**CVAT (Recommended)**
```bash
# Web-based: https://cvat.org
# Or self-hosted:
docker run -d -p 8080:8080 cvat/server
```

**LabelImg**
```bash
pip install labelImg
labelImg
```

### 3. Organize Files

```bash
# Example structure
dataset/
├── images/
│   ├── train/
│   │   ├── img001.jpg
│   │   ├── img002.jpg
│   │   └── ...
│   └── val/
│       ├── img101.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img001.txt
    │   ├── img002.txt
    │   └── ...
    └── val/
        ├── img101.txt
        └── ...
```

### 4. Validate Annotations

```bash
python utils/validate_annotations.py --data dataset/data.yaml
```

## Annotation Format

Each `.txt` file contains one line per object:

```
<class_id> <x_center> <y_center> <width> <height>
```

Example (`img001.txt`):
```
0 0.516 0.384 0.156 0.298
1 0.234 0.512 0.089 0.234
4 0.789 0.123 0.045 0.067
```

All values are normalized (0-1).

## Class Mapping

| ID | Class Name              | Description                    |
|----|-------------------------|--------------------------------|
| 0  | vehicle                 | Cars, trucks, buses            |
| 1  | pedestrian              | People walking                 |
| 2  | bicycle                 | Bicycles                       |
| 3  | motorcycle              | Motorcycles, scooters          |
| 4  | fire                    | Fire hazards                   |
| 5  | smoke                   | Smoke detection                |
| 6  | traffic_sign_stop       | Stop signs                     |
| 7  | traffic_sign_yield      | Yield signs                    |
| 8  | traffic_sign_speed_limit| Speed limit signs              |
| 9  | lane_line               | Lane markings                  |

## Public Datasets

You can use these public datasets as starting points:

1. **COCO** - General object detection
   - http://cocodataset.org/

2. **BDD100K** - Driving dataset
   - https://www.bdd100k.com/

3. **Cityscapes** - Urban scenes
   - https://www.cityscapes-dataset.com/

4. **KITTI** - Autonomous driving
   - http://www.cvlibs.net/datasets/kitti/

## Data Augmentation

The training pipeline automatically applies:
- Random horizontal flip
- Color jittering (HSV)
- Random scaling
- Mosaic augmentation
- MixUp (optional)

Configure in `training/config.yaml`.

## Tips for Good Dataset

1. **Diversity**: Include various:
   - Weather conditions (sunny, rainy, foggy)
   - Time of day (day, night, dusk)
   - Camera angles
   - Lighting conditions

2. **Balance**: Similar number of images per class

3. **Quality**: 
   - Clear, high-resolution images
   - Accurate bounding boxes
   - Consistent annotation style

4. **Edge Cases**:
   - Partial occlusions
   - Small objects
   - Crowded scenes

## Troubleshooting

**Issue: Missing labels warning**
- Ensure every image has a corresponding `.txt` file
- Empty `.txt` files are OK for images with no objects

**Issue: Invalid class ID**
- Check class IDs are 0-9 (matching data.yaml)

**Issue: Coordinates out of range**
- All values must be between 0 and 1
- Double-check annotation tool export settings