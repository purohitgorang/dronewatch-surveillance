from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import datetime
import os
import sys

# Ensure local detector import resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_detector import LocalDroneDetector

app = FastAPI(
    title="DroneWatch AI - Local Edge Detection API",
    description="Offline, high-accuracy UAV/Drone detection powered by local YOLOv8 neural network on PyTorch CUDA.",
    version="2.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize local offline drone detector
detector = LocalDroneDetector()

# In-memory storage for detection history logs
logs: List[Dict[str, Any]] = []

class DetectionLog(BaseModel):
    id: int
    timestamp: str
    drone_name: Optional[str]
    confidence: float
    box: Optional[List[float]] = None
    box_norm: Optional[List[float]] = None
    device: Optional[str] = None
    inference_ms: Optional[float] = None

class ConfigRequest(BaseModel):
    api_key: Optional[str] = None
    confidence_threshold: Optional[float] = 0.25

@app.get("/")
async def root():
    return {
        "system": "DroneWatch AI",
        "mode": "Offline Edge Detection",
        "model": "YOLOv8n-Drone (Fine-Tuned)",
        "device": detector.device_name,
        "classes": list(LocalDroneDetector.FRIENDLY_NAMES.values())[:2],
        "status": "active"
    }

@app.get("/status")
async def get_status():
    return {
        "status": "ready",
        "engine": "Local YOLOv8 Offline Edge AI",
        "device": detector.device_name,
        "cuda_accelerated": "cuda" in detector.device,
        "logs_count": len(logs)
    }

@app.post("/config")
async def configure(request: Optional[ConfigRequest] = None):
    # Backward compatibility endpoint - no external API key needed
    return {
        "status": "configured",
        "message": "Local offline model ready. No cloud API key required.",
        "device": detector.device_name
    }

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.25, ge=0.05, le=0.95, description="Confidence threshold")
):
    contents = await file.read()
    result = detector.detect(contents, conf_threshold=conf)
    
    if result.get("is_drone"):
        log_entry = {
            "id": len(logs) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "drone_name": result.get("drone_name", "Quadcopter (UAV)"),
            "confidence": result.get("confidence", 0.0),
            "box": result.get("box"),
            "box_norm": result.get("box_norm"),
            "device": result.get("device"),
            "inference_ms": result.get("inference_ms"),
        }
        logs.insert(0, log_entry)  # Prepend newest detection
        return {
            "detected": True,
            "data": log_entry,
            "all_detections": result.get("detections", []),
            "inference_ms": result.get("inference_ms"),
            "device": result.get("device")
        }
    
    return {
        "detected": False,
        "inference_ms": result.get("inference_ms"),
        "device": result.get("device")
    }

@app.get("/logs", response_model=List[DetectionLog])
async def get_logs():
    return logs

@app.delete("/logs")
async def clear_logs():
    global logs
    logs = []
    return {"status": "cleared"}
