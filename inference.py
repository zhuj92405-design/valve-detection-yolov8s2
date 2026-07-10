#!/usr/bin/env python3
"""Valve Detection & Anomaly Detection Pipeline - YOLOv8s
Valve Detection: mAP50=92.95% (4 classes)
Anomaly Detection: mAP50=48.2% (6 classes)
"""
from ultralytics import YOLO
import argparse

VALVE_CLASSES = ['gate_valve', 'globe_valve', 'ball_valve', 'other_valve']
ANOMALY_CLASSES = ['water_accumulation', 'water_seepage', 'corrosion_rust', 
                   'coating_damage', 'wall_crack', 'fog_condensation']

def detect_valves(image_path, model_path='valve_detection_best.pt', conf=0.25):
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf_val = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'class': VALVE_CLASSES[cls],
                'confidence': round(conf_val, 3),
                'bbox': [round(x) for x in xyxy]
            })
    return detections

def detect_anomalies(image_path, model_path='anomaly_detection_best.pt', conf=0.25):
    model = YOLO(model_path)
    results = model(image_path, conf=conf)
    detections = []
    for r in results:
        for box in r.boxes:
            cls = int(box.cls)
            conf_val = float(box.conf)
            xyxy = box.xyxy[0].tolist()
            detections.append({
                'class': ANOMALY_CLASSES[cls],
                'confidence': round(conf_val, 3),
                'bbox': [round(x) for x in xyxy]
            })
    return detections

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gas Valve Well Detection Pipeline')
    parser.add_argument('--image', required=True)
    parser.add_argument('--task', choices=['valve', 'anomaly', 'both'], default='both')
    parser.add_argument('--conf', type=float, default=0.25)
    args = parser.parse_args()
    
    if args.task in ['valve', 'both']:
        valves = detect_valves(args.image, conf=args.conf)
        print(f'Valve detections: {len(valves)}')
        for v in valves:
            print(f"  {v['class']}: {v['confidence']:.2f} at {v['bbox']}")
    
    if args.task in ['anomaly', 'both']:
        anomalies = detect_anomalies(args.image, conf=args.conf)
        print(f'Anomaly detections: {len(anomalies)}')
        for a in anomalies:
            print(f"  {a['class']}: {a['confidence']:.2f} at {a['bbox']}")
