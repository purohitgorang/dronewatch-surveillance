import React, { useState, useEffect, useRef } from 'react';
import { Camera, Shield, AlertTriangle, Settings, Activity } from 'lucide-react';

const API_URL = 'http://localhost:8000';

function App() {
  const [apiKey, setApiKey] = useState(localStorage.getItem('gemini_api_key') || '');
  const [isConfigured, setIsConfigured] = useState(false);
  const [logs, setLogs] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const intervalRef = useRef(null);

  useEffect(() => {
    if (apiKey) {
      configureBackend(apiKey);
    }
    startCamera();
    return () => stopCamera();
  }, []);

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: true });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Error accessing webcam:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
    stopDetection();
  };

  const configureBackend = async (key) => {
    try {
      const response = await fetch(`${API_URL}/config`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ api_key: key }),
      });
      if (response.ok) {
        setIsConfigured(true);
        localStorage.setItem('gemini_api_key', key);
      }
    } catch (error) {
      console.error("Config error:", error);
    }
  };

  const startDetection = () => {
    if (!isConfigured) return;
    setIsDetecting(true);
    intervalRef.current = setInterval(captureAndDetect, 3000); // Every 3 seconds
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
    }
  };

  const captureAndDetect = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const context = canvasRef.current.getContext('2d');
    context.drawImage(videoRef.current, 0, 0, canvasRef.current.width, canvasRef.current.height);

    canvasRef.current.toBlob(async (blob) => {
      if (!blob) return;

      const formData = new FormData();
      formData.append('file', blob, 'capture.jpg');

      try {
        const response = await fetch(`${API_URL}/detect`, {
          method: 'POST',
          body: formData,
        });
        const data = await response.json();

        if (data.detected) {
          setLogs(prev => [data.data, ...prev]);
        }
      } catch (error) {
        console.error("Detection error:", error);
      }
    }, 'image/jpeg');
  };

  const handleApiKeySubmit = (e) => {
    e.preventDefault();
    const key = e.target.elements.apiKey.value;
    setApiKey(key);
    configureBackend(key);
  };

  return (
    <div className="container">
      {!isConfigured && (
        <div className="api-key-modal">
          <div className="modal-content">
            <h2><Settings size={24} /> Configuration</h2>
            <p>Please enter your Google Gemini API Key to start.</p>
            <form onSubmit={handleApiKeySubmit}>
              <input
                name="apiKey"
                type="password"
                className="input-field"
                placeholder="Enter API Key"
                required
              />
              <button type="submit" className="btn">Save & Continue</button>
            </form>
          </div>
        </div>
      )}

      <header className="header">
        <div className="title">
          <Shield style={{ marginRight: '10px', verticalAlign: 'middle' }} />
          DroneWatch AI
        </div>
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div className={`status-dot`} style={{ backgroundColor: isDetecting ? '#22c55e' : '#ef4444' }}></div>
          <span>{isDetecting ? 'System Active' : 'System Standby'}</span>
        </div>
      </header>

      <div className="main-grid">
        <div className="glass-panel">
          <div className="webcam-container">
            <video ref={videoRef} autoPlay playsInline muted />
            <canvas ref={canvasRef} width="640" height="480" style={{ display: 'none' }} />
            <div className="overlay">
              <Camera size={16} />
              <span>Live Feed Analysis</span>
            </div>
          </div>

          <div className="controls">
            {!isDetecting ? (
              <button className="btn" onClick={startDetection} disabled={!isConfigured}>
                Start Detection
              </button>
            ) : (
              <button className="btn btn-secondary" onClick={stopDetection}>
                Stop Detection
              </button>
            )}
          </div>
        </div>

        <div className="glass-panel logs-container">
          <div className="logs-header">
            <Activity size={20} />
            Detection Logs
          </div>
          <div className="logs-list">
            {logs.length === 0 ? (
              <div style={{ textAlign: 'center', color: '#64748b', marginTop: '2rem' }}>
                No drones detected yet.
              </div>
            ) : (
              logs.map((log) => (
                <div key={log.id} className="log-item">
                  <div className="log-time">{new Date(log.timestamp).toLocaleTimeString()}</div>
                  <div className="log-name">
                    <AlertTriangle size={14} style={{ marginRight: '5px', color: '#eab308' }} />
                    {log.drone_name}
                  </div>
                  <div className="log-confidence">
                    Confidence: {(log.confidence * 100).toFixed(1)}%
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
