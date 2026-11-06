# 🛸 DroneWatch: Real-Time Airspace Threat Surveillance

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19.0-61dafb.svg)](https://react.dev/)
[![Gemini](https://img.shields.io/badge/Google%20Gemini-1.5%20Flash-4285f4.svg)](https://ai.google.dev/)

DroneWatch is a full-stack, AI-powered airspace perimeter monitoring platform. It captures camera feeds, detects unauthorized unmanned aerial vehicles (UAVs), classifies specific drone models using Google's Gemini Vision API, and records audit trails.

---

## 🌟 Key Features

* **Multimodal Threat Detection:** Harnesses Google Gemini 1.5 Flash Vision for zero-shot drone detection, payload analysis, and model classification.
* **Modern Reactive Dashboard:** Built with React 19, Vite, and Lucide icons for high-framerate visual feeds and incident history.
* **Turnkey Dual Execution:** Launches asynchronous FastAPI backend and Vite development server in one command.
* **Interactive OpenAPI Docs:** Auto-generated interactive Swagger UI at `/docs`.

---

## 🏗️ Architecture

```
[React 19 Frontend (Port 5173)] ──HTTP POST /detect──> [FastAPI Backend (Port 8000)] ──> [Google Gemini API]
```

---

## 🚀 Quick Start

### 1. Environment Setup
Create a `.env` file in `backend/`:
```env
GEMINI_API_KEY="your-google-gemini-api-key-here"
```

### 2. Install Dependencies
```bash
# Backend
cd backend
pip install -r requirements.txt

# Frontend
cd ../frontend
npm install
```

### 3. Run the System
```bash
# Windows turnkey launcher (starts frontend + backend)
run_dronewatch.bat
```
* **Frontend Dashboard:** `http://localhost:5173`
* **Backend API Docs:** `http://localhost:8000/docs`

---

## 📄 License
MIT License
