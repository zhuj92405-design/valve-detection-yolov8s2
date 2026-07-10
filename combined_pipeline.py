#!/usr/bin/env python3
"""
Combined Pipeline — Valve Detection + Anomaly Detection

Runs both models sequentially on inspection photos:
  1. YOLOv8s valve detection (4 classes, mAP50=92.95%)
  2. YOLOv8s anomaly detection (6 classes, mAP50=31.2%)

Outputs a unified inspection report.

Usage:
    python combined_pipeline.py --source inspection_photo.jpg
    python combined_pipeline.py --source ./inspection_photos/ --output ./pipeline_results/
"""

import argparse
import csv
import json
from pathlib import Path
from datetime import datetime

import cv2
from ultralytics import YOLO

VALVE_CLASSES = {0: "gate_valve", 1: "globe_valve", 2: "ball_valve", 3: "other_valve"}
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

VALVE_COLORS = {
    0: (255, 200, 0),
    1: (0, 200, 255),
    2: (0, 255, 0),
    3: (200, 200, 200),
}
ANOMALY_COLORS = {
    "water_accumulation": (65, 105, 225),
    "water_seepage": (255, 69, 0),
    "corrosion_rust": (0, 165, 255),
    "coating_damage": (0, 255, 127),
    "wall_crack": (0, 0, 255),
    "fog_condensation": (203, 192, 255),
}


def draw_detections(img, boxes, classes_map, color_map, prefix=""):
    for box in boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls_name = classes_map.get(cls_id, f"class_{cls_id}")
        color = color_map.get(cls_id, color_map.get(cls_name, (255, 255, 255)))

        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        label = f"{prefix}{cls_name} {conf:.2f}" if prefix else f"{cls_name} {conf:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.rectangle(img, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, label, (x1 + 2, y1 - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
    return img


def process_image(valve_model, anomaly_model, source, valve_conf, anomaly_conf, anomaly_iou):
    valve_results = valve_model.predict(source=source, conf=valve_conf, save=False, verbose=False)
    anomaly_results = anomaly_model.predict(source=source, conf=anomaly_conf, iou=anomaly_iou, save=False, verbose=False)

    image_report = {"image": str(source), "valves": [], "anomalies": []}

    r_v = valve_results[0]
    for box in r_v.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = VALVE_CLASSES.get(cls_id, f"class_{cls_id}")
        image_report["valves"].append({
            "class": cls_name, "confidence": round(conf, 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    r_a = anomaly_results[0]
    for box in r_a.boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        cls_name = ANOMALY_CLASSES.get(cls_id, f"class_{cls_id}")
        severity = SEVERITY_MAP.get(cls_name, "unknown")
        image_report["anomalies"].append({
            "class": cls_name, "severity": severity, "confidence": round(conf, 4),
            "bbox": [round(x1, 1), round(y1, 1), round(x2, 1), round(y2, 1)],
        })

    img_bgr = r_v.orig_img.copy()
    draw_detections(img_bgr, r_v.boxes, VALVE_CLASSES, VALVE_COLORS, prefix="V:")
    draw_detections(img_bgr, r_a.boxes, ANOMALY_CLASSES, ANOMALY_COLORS, prefix="A:")

    return image_report, img_bgr


def main():
    parser = argparse.ArgumentParser(description="Combined Pipeline — Valve + Anomaly Detection")
    parser.add_argument("--source", required=True, help="Path to image file or directory")
    parser.add_argument("--valve-model", default="best.pt", help="Valve detection model path")
    parser.add_argument("--anomaly-model", default="anomaly_best.pt", help="Anomaly detection model path (.pt or .onnx)")
    parser.add_argument("--valve-conf", type=float, default=0.4, help="Valve detection confidence threshold")
    parser.add_argument("--anomaly-conf", type=float, default=0.3, help="Anomaly detection confidence threshold")
    parser.add_argument("--anomaly-iou", type=float, default=0.5, help="Anomaly NMS IoU threshold")
    parser.add_argument("--output", default="./pipeline_results/", help="Output directory")
    args = parser.parse_args()

    source = Path(args.source)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    print(f"Loading valve detection model: {args.valve_model}")
    valve_model = YOLO(args.valve_model)
    print(f"Loading anomaly detection model: {args.anomaly_model}")
    anomaly_model = YOLO(args.anomaly_model)

    if source.is_file():
        image_paths = [source]
    else:
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        image_paths = sorted(p for p in source.rglob("*") if p.suffix.lower() in exts)

    if not image_paths:
        print(f"No images found at {args.source}")
        return

    print(f"Processing {len(image_paths)} image(s)...")

    all_reports = []
    valve_counts = {}
    anomaly_counts = {}
    severity_counts = {}

    for img_path in image_paths:
        report, annotated_img = process_image(
            valve_model, anomaly_model, str(img_path),
            args.valve_conf, args.anomaly_conf, args.anomaly_iou,
        )
        all_reports.append(report)

        save_name = f"{img_path.stem}_pipeline.jpg"
        cv2.imwrite(str(img_dir / save_name), annotated_img)

        for v in report["valves"]:
            valve_counts[v["class"]] = valve_counts.get(v["class"], 0) + 1
        for a in report["anomalies"]:
            anomaly_counts[a["class"]] = anomaly_counts.get(a["class"], 0) + 1
            severity_counts[a["severity"]] = severity_counts.get(a["severity"], 0) + 1

    json_path = output_dir / "pipeline_report.json"
    with open(json_path, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "valve_model": args.valve_model,
            "anomaly_model": args.anomaly_model,
            "total_images": len(image_paths),
            "images": all_reports,
        }, f, indent=2, ensure_ascii=False)
    print(f"JSON report saved: {json_path}")

    csv_path = output_dir / "pipeline_report.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["image", "type", "class", "severity", "confidence", "x1", "y1", "x2", "y2"])
        writer.writeheader()
        for report in all_reports:
            img_name = Path(report["image"]).name
            for v in report["valves"]:
                writer.writerow({"image": img_name, "type": "valve", "class": v["class"],
                                 "severity": "", "confidence": v["confidence"],
                                 "x1": v["bbox"][0], "y1": v["bbox"][1], "x2": v["bbox"][2], "y2": v["bbox"][3]})
            for a in report["anomalies"]:
                writer.writerow({"image": img_name, "type": "anomaly", "class": a["class"],
                                 "severity": a["severity"], "confidence": a["confidence"],
                                 "x1": a["bbox"][0], "y1": a["bbox"][1], "x2": a["bbox"][2], "y2": a["bbox"][3]})
    print(f"CSV report saved: {csv_path}")

    total_valves = sum(valve_counts.values())
    total_anomalies = sum(anomaly_counts.values())
    sev_order = ["critical", "high", "medium", "low", "unknown"]

    print(f"\n{'='*60}")
    print(f"  Combined Inspection Report")
    print(f"{'='*60}")
    print(f"  Images processed:     {len(image_paths)}")
    print()
    print(f"  VALVE DETECTIONS      ({total_valves} total)")
    if valve_counts:
        for cls, cnt in sorted(valve_counts.items(), key=lambda x: -x[1]):
            print(f"    {cls:<20} {cnt:>3}")
    else:
        print(f"    (none)")

    print()
    print(f"  ANOMALY DETECTIONS    ({total_anomalies} total)")
    if anomaly_counts:
        for cls, cnt in sorted(anomaly_counts.items(), key=lambda x: -x[1]):
            sev = SEVERITY_MAP.get(cls, "?")
            print(f"    {cls:<20} {cnt:>3}  (severity: {sev})")
    else:
        print(f"    (none — all clear)")

    if severity_counts:
        print()
        print(f"  SEVERITY BREAKDOWN")
        for sev in sorted(severity_counts, key=lambda s: sev_order.index(s) if s in sev_order else 99):
            marker = ">>>" if sev in ("critical", "high") else "   "
            print(f"    {marker} {sev:<10} {severity_counts[sev]:>3}")

    print()
    if severity_counts.get("critical", 0) > 0:
        print(f"  ** CRITICAL anomalies detected — immediate inspection required **")
    elif severity_counts.get("high", 0) > 0:
        print(f"  * HIGH severity anomalies detected — prioritize review *")
    elif total_anomalies == 0:
        print(f"  No anomalies detected — well appears in good condition")
    else:
        print(f"  Minor anomalies detected — schedule routine follow-up")
    print(f"{'='*60}")
    print(f"  Annotated images: {img_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
