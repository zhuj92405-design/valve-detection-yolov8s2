#!/usr/bin/env python3
"""
Gas Valve Well Inspection Pipeline - End-to-End
Combines 3 AI models for complete inspection analysis:
1. Valve Detection (YOLOv8s, mAP50=92.95%) - Detect valve types
2. Anomaly Detection (YOLOv8s, mAP50=48.2%) - Localize anomaly regions  
3. Anomaly Classification (EfficientNet-B0, 74.0%) - Classify anomalies + severity

Output: Comprehensive inspection report with overall health score
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from ultralytics import YOLO


# ===== Configuration =====

VALVE_MODEL_PATH = "valve_detection_best.pt"
ANOMALY_DET_MODEL_PATH = "anomaly_detection_best.pt"
ANOMALY_CLS_MODEL_PATH = "best_anomaly_classifier.pt"

VALVE_CLASSES = ['gate_valve', 'globe_valve', 'ball_valve', 'other_valve']
ANOMALY_DET_CLASSES = ['water_accumulation', 'water_seepage', 'corrosion_rust',
                        'coating_damage', 'wall_crack', 'fog_condensation']
ANOMALY_CLS_CLASSES = ['water_accumulation', 'water_seepage', 'corrosion_rust',
                       'coating_damage', 'wall_crack', 'fog_condensation']
SEVERITY_LEVELS = ['none', 'minor', 'moderate', 'severe']


class AnomalyClassifierModel(nn.Module):
    """EfficientNet-B0 based multi-label anomaly classifier with severity."""
    def __init__(self, num_anomalies=6, num_severity=4):
        super().__init__()
        from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
        self.backbone = efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
        num_features = self.backbone.classifier[1].in_features  # 1280
        self.backbone.classifier = nn.Identity()
        self.anomaly_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_anomalies),
        )
        self.severity_head = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_severity),  # shared 4-class severity (none/minor/moderate/severe)
        )
        self.num_anomalies = num_anomalies
        self.num_severity = num_severity

    def forward(self, x):
        features = self.backbone(x)
        anomaly_out = self.anomaly_head(features)
        severity_out = self.severity_head(features)
        return anomaly_out, severity_out

# Health score weights (lower = more concerning)
ANOMALY_WEIGHTS = {
    'water_accumulation': 0.7,
    'water_seepage': 0.8,
    'corrosion_rust': 0.6,
    'coating_damage': 0.5,
    'wall_crack': 0.4,
    'fog_condensation': 0.9,
}

SEVERITY_MULTIPLIERS = {
    'none': 0.0,
    'minor': 0.3,
    'moderate': 0.6,
    'severe': 1.0,
}


# ===== Models =====

class ValveDetector:
    """Detect valve types in inspection images."""
    
    def __init__(self, model_path=VALVE_MODEL_PATH):
        self.model = YOLO(model_path)
        self.classes = VALVE_CLASSES
    
    def detect(self, image_path, conf=0.25):
        results = self.model(image_path, conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                detections.append({
                    'class': self.classes[cls],
                    'confidence': round(float(box.conf), 3),
                    'bbox': [round(x) for x in box.xyxy[0].tolist()],
                })
        return detections


class AnomalyDetector:
    """Detect and localize anomaly regions."""
    
    def __init__(self, model_path=ANOMALY_DET_MODEL_PATH):
        self.model = YOLO(model_path)
        self.classes = ANOMALY_DET_CLASSES
    
    def detect(self, image_path, conf=0.2):
        results = self.model(image_path, conf=conf, verbose=False)
        detections = []
        for r in results:
            for box in r.boxes:
                cls = int(box.cls)
                xyxy = box.xyxy[0].tolist()
                detections.append({
                    'class': self.classes[cls],
                    'confidence': round(float(box.conf), 3),
                    'bbox': [round(x) for x in xyxy],
                    'area_pct': round((xyxy[2]-xyxy[0]) * (xyxy[3]-xyxy[1]) / (2592*1944) * 100, 2),
                })
        return detections


class AnomalyClassifier:
    """Classify image-level anomalies and severity."""
    
    def __init__(self, model_path=ANOMALY_CLS_MODEL_PATH):
        self.device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
        ckpt = torch.load(model_path, map_location=self.device, weights_only=False)
        self.model = AnomalyClassifierModel()
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            self.model.load_state_dict(ckpt['model_state_dict'])
        else:
            self.model = ckpt
        self.model.to(self.device)
        self.model.eval()
        self.anomaly_classes = ANOMALY_CLS_CLASSES
        self.severity_levels = SEVERITY_LEVELS
    
    def classify(self, image_path):
        img = Image.open(image_path).convert('RGB').resize((224, 224))
        arr = np.array(img).astype(np.float32) / 255.0
        arr = (arr - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = torch.tensor(arr.transpose(2, 0, 1), dtype=torch.float32).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            anomaly_out, severity_out = self.model(tensor)
        
        anomaly_probs = torch.sigmoid(anomaly_out).cpu().numpy()[0]
        # Shared severity classifier: 4-class softmax (none/minor/moderate/severe)
        severity_probs = torch.softmax(severity_out, dim=1).cpu().numpy()[0]  # (4,)
        sev_idx = np.argmax(severity_probs)
        
        results = []
        for i, cls_name in enumerate(self.anomaly_classes):
            if anomaly_probs[i] > 0.3:
                results.append({
                    'class': cls_name,
                    'probability': round(float(anomaly_probs[i]), 3),
                    'severity': self.severity_levels[sev_idx],
                    'severity_confidence': round(float(severity_probs[sev_idx]), 3),
                })
        
        return sorted(results, key=lambda x: x['probability'], reverse=True)


# ===== Inspection Pipeline =====

class InspectionPipeline:
    """End-to-end pipeline combining all models."""
    
    def __init__(self, valve_model=VALVE_MODEL_PATH, 
                 anomaly_det_model=ANOMALY_DET_MODEL_PATH,
                 anomaly_cls_model=ANOMALY_CLS_MODEL_PATH):
        print("Loading models...")
        self.valve_det = ValveDetector(valve_model)
        self.anomaly_det = AnomalyDetector(anomaly_det_model)
        self.anomaly_cls = AnomalyClassifier(anomaly_cls_model)
        print("All models loaded.")
    
    def inspect(self, image_path, conf_valve=0.25, conf_anomaly=0.2):
        """Run full inspection on a single image."""
        report = {
            'image': str(image_path),
            'timestamp': datetime.now().isoformat(),
            'valve_detection': self.valve_det.detect(image_path, conf=conf_valve),
            'anomaly_detection': self.anomaly_det.detect(image_path, conf=conf_anomaly),
            'anomaly_classification': self.anomaly_cls.classify(image_path),
        }
        
        # Compute health score
        report['health_score'] = self._compute_health_score(report)
        report['overall_status'] = self._status_from_score(report['health_score'])
        
        return report
    
    def _compute_health_score(self, report):
        """Compute 0-100 health score. 100 = perfect condition."""
        penalty = 0
        for det in report.get('anomaly_detection', []):
            weight = ANOMALY_WEIGHTS.get(det['class'], 0.5)
            area_factor = min(det.get('area_pct', 1) / 10, 1.0)
            penalty += weight * area_factor * det['confidence'] * 30
        
        for cls in report.get('anomaly_classification', []):
            sev_mult = SEVERITY_MULTIPLIERS.get(cls['severity'], 0.3)
            penalty += sev_mult * cls['probability'] * 15
        
        score = max(0, min(100, 100 - penalty))
        return round(score, 1)
    
    def _status_from_score(self, score):
        if score >= 80:
            return 'GOOD'
        elif score >= 60:
            return 'FAIR'
        elif score >= 40:
            return 'POOR'
        else:
            return 'CRITICAL'
    
    def format_report(self, report):
        """Format report as readable text."""
        lines = [
            "=" * 60,
            "GAS VALVE WELL INSPECTION REPORT",
            "=" * 60,
            f"Image: {report['image']}",
            f"Time: {report['timestamp']}",
            f"Overall Status: {report['overall_status']}",
            f"Health Score: {report['health_score']}/100",
            "-" * 60,
        ]
        
        # Valve Detection
        lines.append("VALVE DETECTION:")
        valves = report['valve_detection']
        if valves:
            for v in valves:
                lines.append(f"  - {v['class']}: {v['confidence']:.1%} confidence")
        else:
            lines.append("  No valves detected")
        
        # Anomaly Detection
        lines.append("\nANOMALY DETECTION (bbox-level):")
        anomalies = report['anomaly_detection']
        if anomalies:
            for a in anomalies:
                lines.append(f"  - {a['class']}: {a['confidence']:.1%} ({a['area_pct']:.1f}% area)")
        else:
            lines.append("  No anomalies detected")
        
        # Anomaly Classification
        lines.append("\nANOMALY CLASSIFICATION (image-level):")
        classifications = report['anomaly_classification']
        if classifications:
            for c in classifications:
                lines.append(f"  - {c['class']}: {c['probability']:.1%} probability, {c['severity']} severity")
        else:
            lines.append("  No anomalies classified")
        
        lines.append("=" * 60)
        return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description='Gas Valve Well Inspection Pipeline')
    parser.add_argument('images', nargs='+', help='Image path(s)')
    parser.add_argument('--valve-model', default=VALVE_MODEL_PATH)
    parser.add_argument('--anomaly-det-model', default=ANOMALY_DET_MODEL_PATH)
    parser.add_argument('--anomaly-cls-model', default=ANOMALY_CLS_MODEL_PATH)
    parser.add_argument('--output', '-o', help='Output JSON file')
    parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    args = parser.parse_args()
    
    pipeline = InspectionPipeline(
        valve_model=args.valve_model,
        anomaly_det_model=args.anomaly_det_model,
        anomaly_cls_model=args.anomaly_cls_model,
    )
    
    reports = []
    for img_path in args.images:
        report = pipeline.inspect(img_path, conf_valve=args.conf, conf_anomaly=args.conf)
        reports.append(report)
        print(pipeline.format_report(report))
    
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(reports, f, indent=2, ensure_ascii=False)
        print(f"\nReports saved to {args.output}")


if __name__ == '__main__':
    main()
