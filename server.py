"""FastAPI server for valve detection model."""
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, FileResponse
from ultralytics import YOLO
from PIL import Image
import io, os, tempfile, time

app = FastAPI(title="ValveAI Detection API", version="1.0")

# Load model once at startup
model = YOLO("/app/best.pt")
CLASS_NAMES = {0: "gate_valve", 1: "globe_valve", 2: "ball_valve", 3: "other_valve"}

@app.get("/health")
async def health():
    return {"status": "healthy", "model": "valve-detection-yolov8s", "mAP50": "92.45%"}

@app.post("/detect")
async def detect(file: UploadFile = File(...), conf: float = 0.4):
    """Detect valves in an uploaded image."""
    start = time.time()
    contents = await file.read()
    img = Image.open(io.BytesIO(contents))
    
    results = model.predict(source=img, conf=conf, imgsz=640, verbose=False)
    
    detections = []
    for box in results[0].boxes:
        detections.append({
            "class": CLASS_NAMES.get(int(box.cls), "unknown"),
            "class_id": int(box.cls),
            "confidence": round(float(box.conf), 3),
            "bbox_xyxy": [round(x, 1) for x in box.xyxy[0].tolist()],
        })
    
    elapsed = round((time.time() - start) * 1000)
    return JSONResponse({
        "detections": detections,
        "count": len(detections),
        "inference_ms": elapsed,
        "model": "valve-detection-yolov8s",
        "mAP50": "92.45%"
    })

@app.get("/")
async def root():
    return {
        "service": "ValveAI Detection API",
        "version": "1.0",
        "endpoints": {
            "/detect": "POST - Upload image for valve detection",
            "/health": "GET - Health check",
        }
    }
