# Annotation Guide

## Directory Structure
```
annotations/
├── train/
│   ├── image1.txt
│   ├── image2.txt
│   └── ...
├── val/
│   ├── image1.txt
│   └── ...
└── test/
│   ├── image1.txt
│   └── ...
```

## Annotation Format

### YOLO Format (Default)
Each `.txt` file corresponds to an image with the same name.

**Format per line:**
```
<class_id> <x_center> <y_center> <width> <height>
```

All values are normalized (0-1):
- `class_id`: Integer class index (0-9 based on data.yaml)
- `x_center`, `y_center`: Center coordinates (relative to image dimensions)
- `width`, `height`: Box dimensions (relative to image dimensions)

**Example (image1.txt):**
```
0 0.516 0.384 0.156 0.298
1 0.234 0.512 0.089 0.234
9 0.456 0.789 0.678 0.012
```

### Lane Line Annotation (Polylines)
For lane detection, use polyline annotations:
```
9 x1 y1 x2 y2 x3 y3 x4 y4 ...
```

Convert to bounding box for YOLO training or use segmentation format.

### COCO Format (Alternative)
For polygon/segmentation tasks, use COCO JSON format:

```json
{
  "images": [
    {
      "id": 1,
      "file_name": "image1.jpg",
      "width": 1920,
      "height": 1080
    }
  ],
  "annotations": [
    {
      "id": 1,
      "image_id": 1,
      "category_id": 0,
      "bbox": [100, 200, 150, 200],
      "area": 30000,
      "segmentation": [[100, 200, 250, 200, 250, 400, 100, 400]],
      "iscrowd": 0
    }
  ],
  "categories": [
    {"id": 0, "name": "vehicle"},
    {"id": 1, "name": "pedestrian"}
  ]
}
```

## Annotation Tools

### 1. CVAT (Recommended)
- **Best for:** Video annotation, team collaboration
- **Export format:** YOLO 1.1, COCO
- **URL:** https://cvat.org

**Steps:**
1. Create new task
2. Upload images/video
3. Add labels (vehicle, pedestrian, fire, smoke, etc.)
4. Annotate using rectangles, polygons, or polylines
5. Export in YOLO format
6. Extract to `dataset/` folder

### 2. LabelImg
- **Best for:** Quick image annotation
- **Export format:** YOLO, Pascal VOC
- **Installation:** `pip install labelImg`

**Steps:**
1. Run `labelImg`
2. Open directory with images
3. Set "YOLO" as save format
4. Create rectangles and assign classes
5. Save annotations (auto-creates .txt files)

### 3. Label Studio
- **Best for:** Complex projects, ML-assisted labeling
- **Export format:** YOLO, COCO, Custom
- **Installation:** `pip install label-studio`

**Steps:**
1. Start server: `label-studio`
2. Create project with Object Detection template
3. Import images
4. Annotate with bounding boxes
5. Export as YOLO format

## Annotation Best Practices

1. **Consistency:** Use same annotation style across dataset
2. **Quality over quantity:** Accurate annotations > more data
3. **Tight bounding boxes:** Minimize background inclusion
4. **Occlusion handling:** Annotate visible parts only
5. **Edge cases:** Include various lighting, weather, angles
6. **Lane lines:** Use polylines for curved roads
7. **Fire/Smoke:** Annotate entire visible region
8. **Traffic signs:** Include full sign boundary

## Converting Formats

### COCO to YOLO
```python
import json

def coco_to_yolo(coco_json_path, output_dir):
    with open(coco_json_path) as f:
        data = json.load(f)
    
    for ann in data['annotations']:
        img = next(i for i in data['images'] if i['id'] == ann['image_id'])
        img_w, img_h = img['width'], img['height']
        
        x, y, w, h = ann['bbox']
        x_center = (x + w/2) / img_w
        y_center = (y + h/2) / img_h
        width = w / img_w
        height = h / img_h
        
        class_id = ann['category_id']
        
        txt_path = f"{output_dir}/{img['file_name'].replace('.jpg', '.txt')}"
        with open(txt_path, 'a') as f:
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")
```

## Quality Check
Run validation script before training:
```bash
python utils/validate_annotations.py --data dataset/data.yaml
```