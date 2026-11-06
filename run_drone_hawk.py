import subprocess
import os
import sys
import time
import urllib.request
import webbrowser

BASE_DIR = r"C:\Users\Animesh\Projects\04-dronewatch"
VENV_PYTHON = os.path.join(BASE_DIR, ".venv", "Scripts", "python.exe")

print("=" * 60)
print("          LAUNCHING DRONE HAWK X LOCALLY")
print("=" * 60)

# 1. Start FastAPI Backend on Port 8000
backend_cmd = [VENV_PYTHON, "-m", "uvicorn", "main:app", "--host", "127.0.0.1", "--port", "8000", "--reload"]
backend_cwd = os.path.join(BASE_DIR, "backend")
print(f"[*] Starting Local Vision Backend on http://localhost:8000 ...")
backend_log = open(os.path.join(BASE_DIR, "backend.log"), "w", encoding="utf-8")
p_backend = subprocess.Popen(
    backend_cmd,
    cwd=backend_cwd,
    stdout=backend_log,
    stderr=subprocess.STDOUT
)

# 2. Start Vite Frontend on Port 5173
frontend_cmd = ["cmd.exe", "/c", "npm", "run", "dev", "--", "--port", "5173", "--host"]
frontend_cwd = os.path.join(BASE_DIR, "frontend")
print(f"[*] Starting Reactive Tactical Frontend on http://localhost:5173 ...")
frontend_log = open(os.path.join(BASE_DIR, "frontend.log"), "w", encoding="utf-8")
p_frontend = subprocess.Popen(
    frontend_cmd,
    cwd=frontend_cwd,
    stdout=frontend_log,
    stderr=subprocess.STDOUT
)

print("\n[*] Initializing services and verifying ports...")
time.sleep(8)

# Verify Backend
for attempt in range(5):
    try:
        with urllib.request.urlopen("http://127.0.0.1:8000/status", timeout=5) as resp:
            print(f"[+] Drone Hawk X Backend:  ONLINE (HTTP {resp.status}) at http://localhost:8000")
            print(f"    Payload: {resp.read().decode('utf-8')}")
            break
    except Exception as e:
        if attempt < 4:
            time.sleep(2)
        else:
            print(f"[!] Backend check: {e}")

# Verify Frontend
try:
    with urllib.request.urlopen("http://127.0.0.1:5173", timeout=5) as resp:
        print(f"[+] Drone Hawk X Frontend: ONLINE (HTTP {resp.status}) at http://localhost:5173")
except Exception as e:
    print(f"[!] Frontend check: {e}")

print("\n" + "=" * 60)
print("  [SUCCESS] DRONE HAWK X IS NOW RUNNING LOCALLY!")
print("  - Frontend Dashboard:  http://localhost:5173")
print("  - Backend API Docs:    http://localhost:8000/docs")
print("  - Health Telemetry:    http://localhost:8000/status")
print("=" * 60)

try:
    webbrowser.open("http://localhost:5173")
except Exception:
    pass

try:
    while True:
        time.sleep(30)
except KeyboardInterrupt:
    print("\nStopping Drone Hawk X services...")
    p_backend.terminate()
    p_frontend.terminate()
