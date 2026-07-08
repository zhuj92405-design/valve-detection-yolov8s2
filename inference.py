#!/usr/bin/env python3
"""
AI Valve Detection — Batch Inference Script

Process a directory of inspection photos with the YOLOv8s valve detection model.

Usage:
    python inference.py --source ./inspection_photos/ --conf 0.4 --output ./results/
    python inference.py --source single_image.jpg --conf 0.5
"""

import argparse
import os
import csv
from pathlib import Path
from ultralytics import YOLO

CLASS_NAMES = {
    0: "gate_valve",
    1: "globe_valve",
    2: "ball_valve",
    3: "other_valve"
}


def main():
    parser = argparse.ArgumentParser(description="AI Valve Detection — Batch Inference")
    parser.add_argument("--source", required=True, help="Path to image file or directory")
    parser.add_argument("--model", default="best.pt", help="Path to model weights")
    parser.add_argument("--conf", type=float, default=0.4, help="Confidence threshold")
    parser.add_argument("--output", default="./results/", help="Output directory")
    parser.add_argument("--save-csv", action="store_true", default=True, help="Save CSV report")
    args = parser.parse_args()

    # Load model
    print(f"Loading model from {args.model}...")
    model = YOLO(args.model)

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    img_dir = output_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Run inference
    print(f"Running inference on {args.source} (conf={args.conf})...")
    results = model.predict(source=args.source, conf=args.conf, save=False, verbose=True)

    # Collect all detections
    all_detections = []
    class_counts = {}

    for r in results:
        img_path = r.path
        img_name = Path(img_path).name

        # Save annotated image
        annotated = r.plot()
        from PIL import Image
        Image.fromarray(annotated).save(str(img_dir / img_name))

        # Extract detections
        for box in r.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            cls_name = CLASS_NAMES.get(cls_id, f"class_{cls_id}")

            all_detections.append({
                "image": img_name,
                "class": cls_name,
                "confidence": round(conf, 4),
                "x1": round(x1, 1),
                "y1": round(y1, 1),
                "x2": round(x2, 1),
                "y2": round(y2, 1)
            })

            class_counts[cls_name] = class_counts.get(cls_name, 0) + 1

    # Save CSV report
    if args.save_csv and all_detections:
        csv_path = output_dir / "detections.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["image", "class", "confidence", "x1", "y1", "x2", "y2"])
            writer.writeheader()
            writer.writerows(all_detections)
        print(f"CSV report saved: {csv_path}")

    # Print summary
    print(f"\n{'='*50}")
    print(f"Detection Summary")
    print(f"{'='*50}")
    print(f"Total images processed: {len(results)}")
    print(f"Total detections: {len(all_detections)}")
    print(f"\nPer-class breakdown:")
    for cls_name, count in sorted(class_counts.items(), key=lambda x: -x[1]):
        print(f"  {cls_name}: {count}")
    print(f"\nAnnotated images saved to: {img_dir}")

    return all_detections


if __name__ == "__main__":
    main()
