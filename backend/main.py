from fastapi import FastAPI, UploadFile, File, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import datetime
import os
import sys

# Ensure local detector import resolves
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from local_detector import LocalDroneDetector

app = FastAPI(
    title="Drone Hawk X - Local Edge Detection API",
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

# Mount test samples statically
samples_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_samples")
if os.path.exists(samples_dir):
    app.mount("/samples", StaticFiles(directory=samples_dir), name="samples")

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
    confidence_threshold: Optional[float] = 0.55

@app.get("/")
async def root():
    return {
        "system": "Drone Hawk X",
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

@app.get("/api/sample-images")
async def get_sample_images():
    """Return available benchmark test samples with category tags."""
    return [
        {
            "id": "drone_phantom",
            "filename": "drone_phantom.jpg",
            "name": "DJI Phantom Quadcopter",
            "category": "drone",
            "url": "http://localhost:8000/samples/drone_phantom.jpg"
        },
        {
            "id": "drone_hover",
            "filename": "drone_hover.jpg",
            "name": "Commercial Drone in Hover",
            "category": "drone",
            "url": "http://localhost:8000/samples/drone_hover.jpg"
        },
        {
            "id": "drone_fpv",
            "filename": "drone_fpv.jpg",
            "name": "FPV Multi-Rotor Racer",
            "category": "drone",
            "url": "http://localhost:8000/samples/drone_fpv.jpg"
        },
        {
            "id": "drone_aerial",
            "filename": "drone_aerial.jpg",
            "name": "Aerial Cinematography Drone",
            "category": "drone",
            "url": "http://localhost:8000/samples/drone_aerial.jpg"
        },
        {
            "id": "bird_sky",
            "filename": "bird_sky.jpg",
            "name": "Bird in Flight (Negative)",
            "category": "distractor",
            "url": "http://localhost:8000/samples/bird_sky.jpg"
        },
        {
            "id": "airplane_sky",
            "filename": "airplane_sky.jpg",
            "name": "Commercial Aircraft (Negative)",
            "category": "distractor",
            "url": "http://localhost:8000/samples/airplane_sky.jpg"
        },
        {
            "id": "office_interior",
            "filename": "office_interior.jpg",
            "name": "Indoor Office (Negative)",
            "category": "distractor",
            "url": "http://localhost:8000/samples/office_interior.jpg"
        },
        {
            "id": "street_cars",
            "filename": "street_cars.jpg",
            "name": "City Traffic (Negative)",
            "category": "distractor",
            "url": "http://localhost:8000/samples/street_cars.jpg"
        }
    ]

@app.post("/api/test-sample/{filename}")
async def test_sample(
    filename: str,
    conf: float = Query(0.55, ge=0.10, le=0.95)
):
    filepath = os.path.join(samples_dir, filename)
    if not os.path.exists(filepath):
        raise HTTPException(status_code=404, detail="Sample image not found")
    
    with open(filepath, "rb") as f:
        contents = f.read()
        
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
        logs.insert(0, log_entry)
        return {
            "detected": True,
            "data": log_entry,
            "all_detections": result.get("detections", []),
            "inference_ms": result.get("inference_ms"),
            "device": result.get("device"),
            "sample_url": f"http://localhost:8000/samples/{filename}"
        }
    
    return {
        "detected": False,
        "inference_ms": result.get("inference_ms"),
        "device": result.get("device"),
        "sample_url": f"http://localhost:8000/samples/{filename}"
    }

@app.post("/config")
async def configure(request: Optional[ConfigRequest] = None):
    return {
        "status": "configured",
        "message": "Local offline model ready. No cloud API key required.",
        "device": detector.device_name
    }

@app.post("/detect")
async def detect(
    file: UploadFile = File(...),
    conf: float = Query(0.55, ge=0.10, le=0.95, description="Confidence threshold")
):
    contents = await file.read()
    if len(contents) < 500:
        return {"detected": False, "message": "Frame buffer empty"}
        
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
        logs.insert(0, log_entry)
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
