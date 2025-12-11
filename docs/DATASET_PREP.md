# Dataset Preparation Guide

## 1. Data Collection
- Collect images or videos of road scenarios.
- Ensure diversity: day/night, different weather, urban/highway.
- Include specific scenarios for **Fire/Smoke** hazards.

## 2. Annotation (YOLO Format)
We recommend using **CVAT**, **LabelImg**, or **Label Studio**.

### Classes:
- `0`: vehicle
- `1`: pedestrian
- `2`: traffic_sign
- `3`: fire_smoke

### Annotation Files:
For each image `image_01.jpg`, create a corresponding text file `image_01.txt`.
Format:
```
<class_id> <x_center> <y_center> <width> <height>
```
Values are normalized between 0 and 1.

Example `image_01.txt`:
```
0 0.53 0.45 0.12 0.20
1 0.12 0.55 0.05 0.15
3 0.88 0.22 0.10 0.10
```

## 3. Directory Structure
Organize your `dataset/` folder as follows:

```
dataset/
├── images/
│   ├── train/
│   │   ├── img1.jpg
│   │   └── ...
│   └── val/
│       ├── img2.jpg
│       └── ...
└── labels/
    ├── train/
    │   ├── img1.txt
    │   └── ...
    └── val/
        ├── img2.txt
        └── ...
```

## 4. Updates to Config
Modify `dataset/road_safety.yaml` if your paths differ.
