#!/usr/bin/env python3
"""
AI Anomaly Detection — Inference Script

Detect anomalies in underground gas valve well inspection photos
using the YOLOv8s anomaly detection model.

Supports both PyTorch (.pt) and ONNX (.onnx) inference.

Usage:
    python anomaly_detect.py --source inspection_photo.jpg --conf 0.3
    python anomaly_detect.py --source ./photos/ --model best.onnx --output ./anomaly_results/
"""

import argparse
import csv
import os
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

ANOMALY_CLASSES = {
    0: "water_accumulation",
    1: "water_seepage",
    2: "corrosion_rust",
    3: "coating_damage",
    4: "wall_crack",
    5: "fog_condensation",
}

SEVERITY_MAP = {
    "water_accumulation": "medium",
    "water_seepage": "high",
    "corrosion_rust": "high",
    "coating_damage": "medium",
    "wall_crack": "critical",
    "fog_condensation": "low",
}

COLORS = {
    "water_accumulation": (65, 105, 225),
    "water_seepage": (255, 69, 0),
    "corrosion_rust": (0, 165, 255),
    "coating_damage": (0, 255, 127),
    "wall_crack": (0, 0, 255),
    "fog_condensation": (203, 192, 255),
}


def load_model(model_path: str) -> YOLO:
    return YOLO(model_path)


def run_inference(model, source, conf=0.3, iou=0.5):
    results = model.predict(source=source, conf=conf, iou=iou, save=False, verbose=False)
    return results


def extract_detections(results):
    all_detections = []
    class_counts = {}

    for r in results:
        img_path = r.path
        img_name = Path(img_path).name

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = ANOMALY_CLASSES.get(cls_id, f"class_{cls_id}")
            severity = SEVERITY_MAP.get(cls_name, "unknown")

            all_detections.append({
                "image": img_name,
                "class": cls_name,
                "severity": severity,
                "confidence": round(conf, 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1),
            })

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    return all_detections, class_counts


def save_annotated(results, output_dir):
    img_dir = Path(output_dir) / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    for r in results:
        img_path = r.path
        img_name = Path(img_path).stem
        img_bgr = r.orig_img.copy()

        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            cls_name = ANOMALY_CLASSES.get(cls_id, f"class_{cls_id}")
            color = COLORS.get(cls_name, (255, 255, 255))

            cv2.rectangle(img_bgr, (x1, y1), (x2, y2), color, 2)
            label = f"{cls_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
            cv2.rectangle(img_bgr, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img_bgr, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

        save_path = img_dir / f"{img_name}_anomaly.jpg"
        cv2.imwrite(str(save_path), img_bgr)


def save_csv(detections, output_dir):
    csv_path = Path(output_dir) / "anomaly_detections.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "class", "severity", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        writer.writerows(detections)
    return csv_path


def print_summary(detections, class_counts, n_images):
    severity_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "unknown": 4}
    severity_counts = {}
    for d in detections:
        s = d["severity"]
        severity_counts[s] = severity_counts.get(s, 0) + 1

    print(f"\n{'='*55}")
    print(f"  Anomaly Detection Summary")
    print(f"{'='*55}")
    print(f"  Images processed:    {n_images}")
    print(f"  Total anomalies:     {len(detections)}")
    print()
    print(f"  Per-class breakdown:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        sev = SEVERITY_MAP.get(cls_name, "?")
        print(f"    {cls_name:<22} {count:>3}  (severity: {sev})")
    print()
    print(f"  Severity breakdown:")
    for sev in sorted(severity_counts, key=lambda s: severity_order.get(s, 99)):
        print(f"    {sev:<10} {severity_counts[sev]:>3}")
    print(f"{'='*55}")


def main():
    parser = argparse.ArgumentParser(description="AI Anomaly Detection — Inference")
    parser.add_argument("--source", required=True, help="Path to image file or directory")
    parser.add_argument("--model", default="anomaly_best.pt", help="Path to model weights (.pt or .onnx)")
    parser.add_argument("--conf", type=float, default=0.3, help="Confidence threshold (default: 0.3)")
    parser.add_argument("--iou", type=float, default=0.5, help="IoU threshold for NMS (default: 0.5)")
    parser.add_argument("--output", default="./anomaly_results/", help="Output directory")
    parser.add_argument("--no-save", action="store_true", help="Skip saving annotated images and CSV")
    args = parser.parse_args()

    is_onnx = args.model.endswith(".onnx")
    print(f"Loading anomaly model from {args.model} ({'ONNX' if is_onnx else 'PyTorch'})...")
    model = load_model(args.model)

    print(f"Running anomaly detection on {args.source} (conf={args.conf}, iou={args.iou})...")
    results = run_inference(model, args.source, conf=args.conf, iou=args.iou)

    detections, class_counts = extract_detections(results)

    if not args.no_save:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        save_annotated(results, args.output)
        if detections:
            csv_path = save_csv(detections, args.output)
            print(f"CSV report saved: {csv_path}")
        print(f"Annotated images saved to: {output_dir / 'images'}")

    print_summary(detections, class_counts, len(results))

    return detections


if __name__ == "__main__":
    main()
