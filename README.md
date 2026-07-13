# AI Valve Detection — YOLOv8s

Production-grade object detection model for underground gas valve well inspection.

**Model Performance**: mAP50 = 92.95% | mAP50-95 = 85.20% | 4 valve classes

**Product Page**: [ValveAI Landing](https://zhuj92405-design.github.io/valve-detection-yolov8s2/)

## Quick Start

```bash
pip install ultralytics pillow
```

```python
from ultralytics import YOLO

# Load model
model = YOLO("best.pt")

# Run inference on a single image
results = model.predict(source="inspection_photo.jpg", conf=0.4)

# Access detections
for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Class: {model.names[cls]}, Confidence: {conf:.2f}, BBox: {[x1,y1,x2,y2]}")

# Save annotated result
results[0].save("output.jpg")
```

## Detection Classes

| Class ID | Name | Chinese | Typical Share |
|----------|------|---------|---------------|
| 0 | Gate Valve | 闸阀 | ~45% |
| 1 | Globe Valve | 截止阀 | ~17% |
| 2 | Ball Valve | 球阀 | ~34% |
| 3 | Other Valve | 其他 | ~4% |

## Batch Inference

Process an entire directory of inspection photos:

```bash
python inference.py --source ./inspection_photos/ --conf 0.4 --output ./results/
```

This generates:
- Annotated images with bounding boxes
- CSV report with all detections (class, confidence, coordinates)
- Per-class summary statistics

## API Server

Run a FastAPI detection server:

```bash
pip install fastapi uvicorn python-multipart
uvicorn server:app --host 0.0.0.0 --port 8000
```

Endpoints:
- `POST /detect` — Upload image, get JSON detection results
- `GET /health` — Health check

## Export to Production Formats

```python
model = YOLO("best.pt")

# ONNX (recommended for production deployment)
model.export(format="onnx")

# TensorRT (NVIDIA GPU inference)
model.export(format="engine")

# CoreML (Apple devices)
model.export(format="coreml")

# OpenVINO (Intel devices)
model.export(format="openvino")
```

## Training Pipeline

This model was trained using an **iterative pseudo-labeling** pipeline:

1. Start with 30 hand-labeled images
2. Train initial model -> predict on unlabeled data
3. Filter high-confidence predictions (>=0.5)
4. Retrain on expanded dataset
5. Repeat for 10 rounds

| Round | Images | Model | mAP50 | mAP50-95 |
|-------|--------|-------|-------|----------|
| R1 | 30 | YOLOv8n | 28.1% | -- |
| R4 | 1,626 | YOLOv8n | 65.5% | 49.6% |
| R7 | 3,937 | YOLOv8s | 80.5% | 63.6% |
| R8 | 8,506 | YOLOv8s | 83.7% | 68.9% |
| R9 | 18,608 | YOLOv8s | 81.3% | 65.0% |
| **R10** | **9,038** | **YOLOv8s** | **92.95%** | **85.20%** |

Key finding: Removing 50% of noisy pseudo-labels (filtering from 18K to 9K images) improved mAP50 by 11.2 percentage points over R9.

## Anomaly Detection

In addition to valve localization, this repository now includes an **anomaly detection model** that identifies structural and environmental anomalies inside gas valve wells.

**Model Performance (V6)**: mAP50 = 55.71% | mAP50-95 = 37.56% | 6 anomaly classes | +21.2% over V2

### Anomaly Classes

| Class ID | Name | Chinese | Severity | Prevalence (861K images) |
|----------|------|---------|----------|--------------------------|
| 0 | Water Accumulation | 积水 | Medium | 44.4% |
| 1 | Water Seepage | 渗水 | High | 46.1% |
| 2 | Corrosion / Rust | 腐蚀生锈 | High | 77.3% |
| 3 | Coating Damage | 涂层损坏 | Medium | 67.7% |
| 4 | Wall Crack | 墙体裂缝 | Critical | 59.6% |
| 5 | Fog / Condensation | 雾气结露 | Low | 11.0% |

### Quick Start — Anomaly Detection

```bash
pip install ultralytics opencv-python pillow
```

```python
from ultralytics import YOLO

model = YOLO("anomaly_detection_best_v6.pt")
results = model.predict(source="well_inspection.jpg", conf=0.3)

for r in results:
    for box in r.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        print(f"Anomaly: {model.names[cls]}, Confidence: {conf:.2f}, BBox: [{x1:.0f},{y1:.0f},{x2:.0f},{y2:.0f}]")
```

### Combined Pipeline — Valve + Anomaly Detection

Run both models in sequence for a complete inspection report:

```bash
python combined_pipeline.py --source ./inspection_photos/ --output ./pipeline_results/
```

This first detects valves, then scans the same images for anomalies, producing a unified report.

### Training Details

| Metric | V1 | V2 | V6 | Change (V6 vs V2) |
|--------|----|----|----|-------------------|
| Training images | 213 | 512 | 5,954 | +1064% |
| Total bounding boxes | 505 | 1,181 | 11,543 | +877% |
| Best epoch | 33 | 73 | 48 | — |
| Early stopped at | 64 | 110 | 78 | — |
| mAP50 | 31.15% | 45.92% | **55.71%** | **+21.2%** |
| mAP50-95 | 19.38% | 29.00% | **37.56%** | **+29.5%** |
| Precision | — | 54.09% | 60.12% | +11.1% |
| Recall | — | 42.27% | 52.84% | +25.0% |
| Optimizer | AdamW | AdamW | AdamW | — |
| Patience | 30 | 40 | 50 | — |

**Key V6 improvements**: Expanded dataset with V4 pseudo-labels (512→5,954 images, +1064%), integrated 1,301 VLM-annotated images for label quality, increased patience (40→50), larger training set with pseudo-labeling pipeline for robust anomaly detection.

### EfficientNet-B0 Anomaly Classifier

A complementary **image-level classifier** (EfficientNet-B0) is also available:

| Metric | Value |
|--------|-------|
| Anomaly type accuracy | 74.0% |
| Severity classification accuracy | 78.8% |
| Input | Full image (no cropping needed) |
| Output | 6 anomaly types + 4 severity levels |
| Training data | 510 VLM-annotated images |
| Full inference | 861,367 images (100% complete) |

### Dataset Statistics (861,367 images)

| Stat | Value |
|------|-------|
| Images with at least one anomaly | 94.5% |
| Corrosion / Rust prevalence | 77.3% |
| Coating Damage prevalence | 67.7% |
| Wall Crack prevalence | 59.6% |
| Water Seepage prevalence | 46.1% |
| Water Accumulation prevalence | 44.4% |
| Fog Condensation prevalence | 11.0% |
| Mild severity | 68.3% |
| Moderate severity | 26.2% |
| Severe severity | 1.1% |

## Resources

| Resource | Link |
|----------|------|
| **Valve Detection Model** | [HuggingFace](https://huggingface.co/lg227210/valve-detection-yolov8s) |
| **Anomaly Detection Model** | [HuggingFace](https://huggingface.co/lg227210/anomaly-detection-yolov8s) |
| **Live Demo** | [HuggingFace Spaces](https://huggingface.co/spaces/lg227210/valve-detection-demo) |
| **Dataset** | [HuggingFace Datasets](https://huggingface.co/datasets/lg227210/valve-detection-dataset) |
| **Technical Whitepaper** | [PDF](https://huggingface.co/lg227210/valve-detection-yolov8s/blob/main/ValveAI_Technical_Whitepaper.pdf) |
| **Technical Blog** | [Dev.to](https://dev.to/jins_zhu_10096cef92aba38b/how-we-trained-an-837-map50-valve-detection-model-with-iterative-pseudo-labeling-4lhi) |
| **Product Page** | [ValveAI Landing](https://zhuj92405-design.github.io/valve-detection-yolov8s2/) |

## License

- **Non-commercial use**: Free for research, education, and personal projects
- **Commercial use**: Requires a paid license. Contact for pricing.

## Citation

```bibtex
@software{valve-detection-yolov8,
  title = {AI Valve Detection: YOLOv8s for Gas Infrastructure Inspection},
  author = {ValveAI},
  year = {2026},
  note = {Trained on proprietary underground gas valve well inspection data, mAP50=92.5\%}
}
```
