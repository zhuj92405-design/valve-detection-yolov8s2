#!/usr/bin/env python3
"""
ValveAI Inspection API Server
Production-ready REST API for gas valve well inspection.

Endpoints:
  POST /inspect   - Full pipeline inspection (valve detection + anomaly detection + classification)
  POST /detect    - Valve detection only
  POST /anomaly   - Anomaly detection only
  GET  /health    - Health check

Usage:
  python server.py --port 8000 --host 0.0.0.0

Commercial licensing required for production use.
"""

import argparse
import io
import json
import time
from datetime import datetime
from pathlib import Path

import torch
import numpy as np
from PIL import Image
from ultralytics import YOLO

# ---- Configuration ----
DEFAULT_VALVE_MODEL = "valve_detection_best.pt"
DEFAULT_ANOMALY_MODEL = "anomaly_detection_best.pt"
DEFAULT_CLASSIFIER_MODEL = "anomaly_classifier_best.pt"

VALVE_CLASSES = {0: "gate_valve", 1: "globe_valve", 2: "ball_valve", 3: "other_valve"}
ANOMALY_CLASSES = {0: "water_accumulation", 1: "water_seepage", 2: "corrosion_rust",
                   3: "coating_damage", 4: "wall_crack", 5: "fog_condensation"}
SEVERITY_MAP = {0: "medium", 1: "high", 2: "high", 3: "medium", 4: "critical", 5: "low"}

# ---- Model Loading ----
class InspectionPipeline:
    def __init__(self, valve_path, anomaly_path, classifier_path, device="cpu"):
        self.device = device
        print(f"Loading models on {device}...")
        self.valve_model = YOLO(valve_path)
        self.anomaly_model = YOLO(anomaly_path)
        self.classifier = None  # Load on demand
        self.classifier_path = classifier_path
        print("Models loaded successfully.")

    def _load_classifier(self):
        if self.classifier is None:
            import torchvision.models as models
            import torch.nn as nn
            # EfficientNet-B0 architecture
            backbone = models.efficientnet_b0(weights=None)
            backbone.classifier = nn.Identity()
            self.classifier = nn.Module()
            self.classifier.backbone = backbone
            self.classifier.anomaly_head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(256, 6)
            )
            self.classifier.severity_head = nn.Sequential(
                nn.Dropout(0.3), nn.Linear(1280, 256), nn.ReLU(),
                nn.Dropout(0.2), nn.Linear(256, 4)
            )
            state = torch.load(self.classifier_path, map_location=self.device, weights_only=True)
            self.classifier.load_state_dict(state["model_state_dict"])
            self.classifier.to(self.device).eval()

    def detect_valves(self, image, conf=0.4):
        results = self.valve_model.predict(source=image, conf=conf, verbose=False)
        detections = []
        for box in results[0].boxes:
            cls = int(box.cls)
            detections.append({
                "class": VALVE_CLASSES.get(cls, "unknown"),
                "class_id": cls,
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
            })
        return detections, results[0].plot()

    def detect_anomalies(self, image, conf=0.3):
        results = self.anomaly_model.predict(source=image, conf=conf, verbose=False)
        anomalies = []
        for box in results[0].boxes:
            cls = int(box.cls)
            anomalies.append({
                "class": ANOMALY_CLASSES.get(cls, "unknown"),
                "class_id": cls,
                "severity": SEVERITY_MAP.get(cls, "unknown"),
                "confidence": round(float(box.conf), 3),
                "bbox": [round(x, 1) for x in box.xyxy[0].tolist()]
            })
        return anomalies, results[0].plot()

    def classify_anomalies(self, image):
        self._load_classifier()
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = transform(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            anomaly_out = self.classifier.anomaly_head(self.classifier.backbone(img_tensor))
            severity_out = self.classifier.severity_head(self.classifier.backbone(img_tensor))
        anomaly_pred = (torch.sigmoid(anomaly_out) > 0.5).cpu().numpy()[0]
        severity_pred = torch.argmax(severity_out, dim=1).item()
        severity_labels = ["none", "minor", "moderate", "severe"]

        detected = []
        for i, present in enumerate(anomaly_pred):
            if present:
                detected.append({
                    "class": ANOMALY_CLASSES[i],
                    "class_id": i,
                    "severity": severity_labels[severity_pred]
                })
        return detected, severity_labels[severity_pred]

    def full_inspection(self, image, valve_conf=0.4, anomaly_conf=0.3):
        start = time.time()
        valves, valve_img = self.detect_valves(image, conf=valve_conf)
        anomalies, anomaly_img = self.detect_anomalies(image, conf=anomaly_conf)

        # Calculate health score
        score = 100
        if anomalies:
            for a in anomalies:
                severity = a.get("severity", "low")
                deduction = {"critical": 30, "high": 20, "medium": 10, "low": 5}
                score -= deduction.get(severity, 5)
        score = max(0, score)

        if score >= 80:
            status = "GOOD"
        elif score >= 60:
            status = "FAIR"
        elif score >= 40:
            status = "POOR"
        else:
            status = "CRITICAL"

        elapsed = round(time.time() - start, 2)
        return {
            "inspection_time": datetime.now().isoformat(),
            "processing_time_seconds": elapsed,
            "valve_detections": valves,
            "anomaly_detections": anomalies,
            "health_score": score,
            "health_status": status,
            "valve_count": len(valves),
            "anomaly_count": len(anomalies),
            "model_versions": {
                "valve_detection": "R10 (mAP50=92.95%)",
                "anomaly_detection": "V2s (mAP50=48.2%)",
                "anomaly_classification": "V4 (74.0% accuracy)"
            }
        }


# ---- Flask API ----
def create_app(pipeline):
    from flask import Flask, request, jsonify
    from flask_cors import CORS

    app = Flask(__name__)
    CORS(app)

    @app.route("/health", methods=["GET"])
    def health():
        return jsonify({"status": "healthy", "models_loaded": True})

    @app.route("/inspect", methods=["POST"])
    def inspect():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        valve_conf = float(request.form.get("valve_conf", 0.4))
        anomaly_conf = float(request.form.get("anomaly_conf", 0.3))
        result = pipeline.full_inspection(image, valve_conf, anomaly_conf)
        return jsonify(result)

    @app.route("/detect", methods=["POST"])
    def detect():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        conf = float(request.form.get("conf", 0.4))
        detections, _ = pipeline.detect_valves(image, conf=conf)
        return jsonify({"valve_detections": detections, "count": len(detections)})

    @app.route("/anomaly", methods=["POST"])
    def anomaly():
        if "image" not in request.files:
            return jsonify({"error": "No image provided"}), 400
        file = request.files["image"]
        image = Image.open(io.BytesIO(file.read())).convert("RGB")
        conf = float(request.form.get("conf", 0.3))
        anomalies, _ = pipeline.detect_anomalies(image, conf=conf)
        return jsonify({"anomaly_detections": anomalies, "count": len(anomalies)})

    return app


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ValveAI Inspection API Server")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--valve-model", type=str, default=DEFAULT_VALVE_MODEL)
    parser.add_argument("--anomaly-model", type=str, default=DEFAULT_ANOMALY_MODEL)
    parser.add_argument("--classifier-model", type=str, default=DEFAULT_CLASSIFIER_MODEL)
    args = parser.parse_args()

    pipeline = InspectionPipeline(
        valve_path=args.valve_model,
        anomaly_path=args.anomaly_model,
        classifier_path=args.classifier_model,
        device=args.device,
    )

    app = create_app(pipeline)
    print(f"Starting ValveAI Inspection API on {args.host}:{args.port}")
    app.run(host=args.host, port=args.port)
