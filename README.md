# 🦅 Drone Hawk X: Real-Time Offline Edge Airspace Surveillance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Edge%20Vision-orange.svg)](https://github.com/ultralytics/ultralytics)
[![PyTorch CUDA](https://img.shields.io/badge/PyTorch-CUDA%20Accelerated-76b900.svg)](https://pytorch.org/)

**Drone Hawk X** is an enterprise-grade, offline edge AI airspace perimeter monitoring platform. Designed specifically for zero-trust, air-gapped, or field tactical environments, Drone Hawk X operates **100% locally on-device with zero external API keys, zero cloud egress, and zero internet dependencies**.

---

## 🌟 Key Features

* **100% Local Offline Edge AI:** Powered by an edge-optimized YOLOv8 neural network fine-tuned specifically on UAV and aerial drone datasets.
* **Hardware Accelerated Inference:** Automatically leverages local NVIDIA GPUs via PyTorch CUDA (e.g. NVIDIA GeForce GTX 1650) achieving sub-50ms inference latency; includes automatic CPU fallback.
* **Dual Airspace Inspection Modes:**
  * **Live Optical Feed:** Continuous real-time webcam surveillance with military HUD bounding-box target overlays.
  * **File / Image Inspection:** Rapid drag-and-drop analysis of surveillance imagery and snapshot feeds.
* **Tactical Target Tracking:** Detects multi-rotor quadcopters and fixed-wing reconnaissance drones, calculates normalized and absolute bounding coordinates, confidence metrics, and incident timestamps.
* **Audit Trail & Telemetry:** Instant incident logging with timestamped logs, device telemetry, and one-click export/clearing.

---

## 🏗️ Architecture

```
┌────────────────────────────────────────────────────────┐
│                  Drone Hawk X Client                   │
│      React 19 + Vite + Lucide HUD Interface (Port 5173)│
└───────────────────────────┬────────────────────────────┘
                            │ HTTP POST (Multipart Image)
                            ▼
┌────────────────────────────────────────────────────────┐
│              FastAPI Edge Server (Port 8000)           │
│                  main.py & /detect                     │
└───────────────────────────┬────────────────────────────┘
                            │ In-Memory Image Buffer
                            ▼
┌────────────────────────────────────────────────────────┐
│             Local YOLOv8 Drone Engine                  │
│       local_detector.py + drone_yolov8n.pt             │
│   (Hardware-Accelerated on Local NVIDIA GPU / CUDA)   │
└────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Requirements
* Python 3.10+
* Node.js 18+
* Optional: NVIDIA GPU with CUDA 12.x for maximum hardware acceleration (CPU supported out of the box)

### 2. Turnkey Launcher (Windows)
Double-click [`run_dronewatch.bat`](file:///C:/Users/Animesh/Projects/04-dronewatch/run_dronewatch.bat):
```bash
run_dronewatch.bat
```
* **Frontend Tactical HUD:** `http://localhost:5173`
* **Interactive OpenAPI Docs:** `http://localhost:8000/docs`
* **System Health Endpoint:** `http://localhost:8000/status`

### 3. Manual Installation
```bash
# Setup backend
cd backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt

# Setup frontend
cd ../frontend
npm install
npm run dev
```

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/status` | Telemetry on loaded model, GPU hardware, and audit counts |
| `POST` | `/detect?conf=0.25` | Multipart image inspection returning drone target bounding boxes |
| `GET` | `/logs` | Full history of airspace intrusions and timestamps |
| `DELETE` | `/logs` | Flush in-memory incident log |

---

## 📄 License
MIT License
