from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List, Optional
import datetime
from gemini_service import GeminiService
import os

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for logs
logs = []

class DetectionLog(BaseModel):
    id: int
    timestamp: str
    drone_name: Optional[str]
    confidence: float

class ConfigRequest(BaseModel):
    api_key: str

gemini_service = None

# Auto-configure if GEMINI_API_KEY environment variable or .env file exists
_env_key = os.getenv("GEMINI_API_KEY")
_env_file = os.path.join(os.path.dirname(__file__), ".env")
if not _env_key and os.path.exists(_env_file):
    try:
        with open(_env_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip().startswith("GEMINI_API_KEY="):
                    _env_key = line.strip().split("=", 1)[1].strip().strip('"').strip("'")
    except Exception:
        pass

if _env_key:
    try:
        gemini_service = GeminiService(_env_key)
        print("[+] Gemini Service auto-initialized from environment.")
    except Exception as e:
        print(f"[-] Could not auto-initialize Gemini Service: {e}")

@app.post("/config")
async def configure(request: ConfigRequest):
    global gemini_service
    gemini_service = GeminiService(request.api_key)
    return {"status": "configured"}

@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    global gemini_service
    if not gemini_service:
        raise HTTPException(status_code=400, detail="API Key not configured. Please supply GEMINI_API_KEY via UI config or .env")
    
    contents = await file.read()
    result = gemini_service.analyze_image(contents)
    
    if result.get("is_drone"):
        log_entry = {
            "id": len(logs) + 1,
            "timestamp": datetime.datetime.now().isoformat(),
            "drone_name": result.get("drone_name", "Unknown Drone"),
            "confidence": result.get("confidence", 0.0)
        }
        logs.insert(0, log_entry) # Add to top
        return {"detected": True, "data": log_entry}
    
    return {"detected": False}

@app.get("/logs", response_model=List[DetectionLog])
async def get_logs():
    return logs
