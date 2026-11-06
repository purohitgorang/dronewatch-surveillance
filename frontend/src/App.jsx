import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, Shield, AlertTriangle, Activity, 
  Cpu, Crosshair, Upload, Play, Square, Trash2, CheckCircle2 
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

function App() {
  const [systemStatus, setSystemStatus] = useState({
    status: 'connecting',
    engine: 'Local YOLOv8 Drone AI',
    device: 'Detecting GPU/CPU...',
    cuda_accelerated: false
  });
  const [logs, setLogs] = useState([]);
  const [isDetecting, setIsDetecting] = useState(false);
  const [mode, setMode] = useState('webcam'); // 'webcam' | 'upload'
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.25);
  const [lastLatency, setLastLatency] = useState(null);
  const [latestDetection, setLatestDetection] = useState(null);
  const [uploadedImageSrc, setUploadedImageSrc] = useState(null);
  const [isProcessingUpload, setIsProcessingUpload] = useState(false);

  const videoRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const uploadOverlayCanvasRef = useRef(null);
  const uploadImgRef = useRef(null);
  const intervalRef = useRef(null);

  // Check backend offline AI health on load
  useEffect(() => {
    fetchSystemStatus();
    fetchLogs();
    startCamera();
    return () => {
      stopCamera();
      stopDetection();
    };
  }, []);

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/status`);
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
    } catch (err) {
      console.warn("Backend status poll error:", err);
      setSystemStatus(prev => ({ ...prev, status: 'offline' }));
    }
  };

  const fetchLogs = async () => {
    try {
      const res = await fetch(`${API_URL}/logs`);
      if (res.ok) {
        const data = await res.json();
        setLogs(data);
      }
    } catch (err) {
      console.warn("Could not fetch logs:", err);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
      }
    } catch (err) {
      console.error("Webcam access error:", err);
    }
  };

  const stopCamera = () => {
    if (videoRef.current && videoRef.current.srcObject) {
      videoRef.current.srcObject.getTracks().forEach(track => track.stop());
    }
    stopDetection();
  };

  const startDetection = () => {
    setIsDetecting(true);
    // Poll detection every 1.5 seconds for responsive edge detection
    intervalRef.current = setInterval(captureAndDetect, 1500);
  };

  const stopDetection = () => {
    setIsDetecting(false);
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }
    clearOverlay(overlayCanvasRef);
  };

  const clearOverlay = (canvasRef) => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const drawBoundingBoxes = (canvas, boxes, containerWidth, containerHeight) => {
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    canvas.width = containerWidth;
    canvas.height = containerHeight;
    ctx.clearRect(0, 0, containerWidth, containerHeight);

    if (!boxes || boxes.length === 0) return;

    boxes.forEach(det => {
      // box_norm format: [ymin, xmin, ymax, xmax] (0.0 to 1.0)
      if (det.box_norm) {
        const [ymin, xmin, ymax, xmax] = det.box_norm;
        const x = xmin * containerWidth;
        const y = ymin * containerHeight;
        const w = (xmax - xmin) * containerWidth;
        const h = (ymax - ymin) * containerHeight;

        // Draw tactical radar box
        ctx.strokeStyle = '#ef4444'; // Red alert
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Corner brackets for military HUD style
        const bracketLen = 12;
        ctx.strokeStyle = '#22c55e'; // Green accents
        ctx.lineWidth = 4;
        
        // Top-left
        ctx.beginPath();
        ctx.moveTo(x, y + bracketLen); ctx.lineTo(x, y); ctx.lineTo(x + bracketLen, y);
        ctx.stroke();
        // Top-right
        ctx.beginPath();
        ctx.moveTo(x + w - bracketLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + bracketLen);
        ctx.stroke();
        // Bottom-left
        ctx.beginPath();
        ctx.moveTo(x, y + h - bracketLen); ctx.lineTo(x, y + h); ctx.lineTo(x + bracketLen, y + h);
        ctx.stroke();
        // Bottom-right
        ctx.beginPath();
        ctx.moveTo(x + w - bracketLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - bracketLen);
        ctx.stroke();

        // Label pill
        const label = `${det.drone_name} (${(det.confidence * 100).toFixed(1)}%)`;
        ctx.font = 'bold 12px Inter, sans-serif';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = 'rgba(239, 68, 68, 0.9)';
        ctx.fillRect(x, Math.max(0, y - 24), textWidth + 14, 22);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x + 7, Math.max(14, y - 8));
      }
    });
  };

  const captureAndDetect = async () => {
    if (!videoRef.current || !captureCanvasRef.current) return;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    const canvas = captureCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('file', blob, 'feed_frame.jpg');

      try {
        const res = await fetch(`${API_URL}/detect?conf=${confidenceThreshold}`, {
          method: 'POST',
          body: formData,
        });
        const data = await res.json();

        if (data.inference_ms) setLastLatency(data.inference_ms);

        if (data.detected) {
          setLogs(prev => [data.data, ...prev]);
          setLatestDetection(data.data);
          drawBoundingBoxes(
            overlayCanvasRef.current,
            data.all_detections || [data.data],
            video.clientWidth,
            video.clientHeight
          );
        } else {
          clearOverlay(overlayCanvasRef);
          setLatestDetection(null);
        }
      } catch (err) {
        console.error("Frame detection error:", err);
      }
    }, 'image/jpeg', 0.85);
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = (event) => {
      setUploadedImageSrc(event.target.result);
      runUploadInference(file);
    };
    reader.readAsDataURL(file);
  };

  const runUploadInference = async (file) => {
    setIsProcessingUpload(true);
    clearOverlay(uploadOverlayCanvasRef);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const res = await fetch(`${API_URL}/detect?conf=${confidenceThreshold}`, {
        method: 'POST',
        body: formData,
      });
      const data = await res.json();
      if (data.inference_ms) setLastLatency(data.inference_ms);

      if (data.detected) {
        setLogs(prev => [data.data, ...prev]);
        setLatestDetection(data.data);
        
        // Wait for image render then draw overlay
        setTimeout(() => {
          if (uploadImgRef.current) {
            drawBoundingBoxes(
              uploadOverlayCanvasRef.current,
              data.all_detections || [data.data],
              uploadImgRef.current.clientWidth,
              uploadImgRef.current.clientHeight
            );
          }
        }, 150);
      } else {
        setLatestDetection(null);
        clearOverlay(uploadOverlayCanvasRef);
      }
    } catch (err) {
      console.error("Upload inference failed:", err);
    } finally {
      setIsProcessingUpload(false);
    }
  };

  const handleClearLogs = async () => {
    try {
      await fetch(`${API_URL}/logs`, { method: 'DELETE' });
      setLogs([]);
    } catch (err) {
      setLogs([]);
    }
  };

  return (
    <div className="container">
      {/* Top Navigation / Brand */}
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Shield size={32} color="#60a5fa" />
          <div>
            <div className="title">Drone Hawk X</div>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Offline Edge Airspace Surveillance Engine • Local PyTorch CUDA
            </div>
          </div>
        </div>

        {/* System Health Badge */}
        <div style={{ display: 'flex', gap: '1rem', alignItems: 'center' }}>
          <div className="edge-badge">
            <Cpu size={14} color="#22c55e" />
            <span>{systemStatus.device || 'NVIDIA GTX 1650'}</span>
          </div>

          <div className="status-pill">
            <div 
              className="status-dot" 
              style={{ backgroundColor: isDetecting || mode === 'upload' ? '#22c55e' : '#eab308' }} 
            />
            <span>{isDetecting ? 'Live Radar Active' : (mode === 'upload' ? 'Upload Ready' : 'Standby')}</span>
          </div>
        </div>
      </header>

      {/* Mode Selector Tabs */}
      <div className="mode-tabs">
        <button 
          className={`tab-btn ${mode === 'webcam' ? 'active' : ''}`}
          onClick={() => { setMode('webcam'); startCamera(); }}
        >
          <Camera size={16} /> Live Webcam Surveillance
        </button>
        <button 
          className={`tab-btn ${mode === 'upload' ? 'active' : ''}`}
          onClick={() => { setMode('upload'); stopDetection(); }}
        >
          <Upload size={16} /> Test Drone Image / File
        </button>
      </div>

      <div className="main-grid">
        {/* Visual Stream Panel */}
        <div className="glass-panel">
          {mode === 'webcam' ? (
            <div className="webcam-container">
              <video ref={videoRef} autoPlay playsInline muted />
              <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
              <canvas ref={overlayCanvasRef} className="hud-overlay-canvas" />

              <div className="overlay">
                <Crosshair size={16} color="#ef4444" />
                <span>Real-Time Edge Airspace Feed</span>
              </div>

              {latestDetection && (
                <div className="alert-hud">
                  <AlertTriangle size={18} color="#ef4444" />
                  <span>ALERT: {latestDetection.drone_name} Detected!</span>
                </div>
              )}
            </div>
          ) : (
            <div className="upload-container">
              {uploadedImageSrc ? (
                <div style={{ position: 'relative', width: '100%', display: 'flex', justifyContent: 'center' }}>
                  <img 
                    ref={uploadImgRef} 
                    src={uploadedImageSrc} 
                    alt="Drone Test" 
                    className="uploaded-img-preview" 
                  />
                  <canvas ref={uploadOverlayCanvasRef} className="hud-overlay-canvas" />
                </div>
              ) : (
                <div className="upload-dropzone">
                  <Upload size={48} color="#60a5fa" style={{ marginBottom: '1rem' }} />
                  <h3>Upload Airspace Image to Inspect</h3>
                  <p style={{ color: '#94a3b8', fontSize: '0.9rem' }}>
                    Drag and drop or select a drone snapshot for instant local YOLOv8 evaluation.
                  </p>
                </div>
              )}

              <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
                <label className="btn btn-secondary file-input-label">
                  <Upload size={16} /> Select Drone Image
                  <input type="file" accept="image/*" onChange={handleFileUpload} style={{ display: 'none' }} />
                </label>
              </div>

              {isProcessingUpload && (
                <div style={{ textAlign: 'center', marginTop: '0.5rem', color: '#60a5fa' }}>
                  Analyzing with local CUDA model...
                </div>
              )}
            </div>
          )}

          {/* Tactical Controls & Tuning */}
          <div className="controls-row">
            {mode === 'webcam' && (
              <div>
                {!isDetecting ? (
                  <button className="btn btn-detect" onClick={startDetection}>
                    <Play size={16} /> Start Edge Detection
                  </button>
                ) : (
                  <button className="btn btn-stop" onClick={stopDetection}>
                    <Square size={16} /> Halt Detection
                  </button>
                )}
              </div>
            )}

            {/* Sensitivity Slider */}
            <div className="slider-group">
              <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                Confidence Threshold: <strong>{(confidenceThreshold * 100).toFixed(0)}%</strong>
              </span>
              <input 
                type="range" 
                min="0.10" 
                max="0.90" 
                step="0.05"
                value={confidenceThreshold}
                onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                className="range-slider"
              />
            </div>

            {/* Live Metrics */}
            <div className="metrics-badge">
              <Activity size={14} color="#60a5fa" />
              <span>Inference: {lastLatency ? `${lastLatency}ms` : 'Ready'}</span>
            </div>
          </div>
        </div>

        {/* Tactical Log Stream */}
        <div className="glass-panel logs-container">
          <div className="logs-header">
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
              <Activity size={20} color="#60a5fa" />
              <span>Surveillance Log ({logs.length})</span>
            </div>
            {logs.length > 0 && (
              <button className="btn-icon" onClick={handleClearLogs} title="Clear history">
                <Trash2 size={16} />
              </button>
            )}
          </div>

          <div className="logs-list">
            {logs.length === 0 ? (
              <div className="empty-logs">
                <CheckCircle2 size={36} color="#64748b" style={{ marginBottom: '0.5rem' }} />
                <div>Airspace Clear</div>
                <div style={{ fontSize: '0.8rem', color: '#64748b', marginTop: '0.25rem' }}>
                  No unauthorized drones detected yet.
                </div>
              </div>
            ) : (
              logs.map((log) => (
                <div key={log.id} className="log-item">
                  <div className="log-meta">
                    <span className="log-time">{new Date(log.timestamp).toLocaleTimeString()}</span>
                    <span className="log-chip">ID #{log.id}</span>
                  </div>
                  <div className="log-name">
                    <AlertTriangle size={15} color="#ef4444" style={{ marginRight: '6px' }} />
                    {log.drone_name}
                  </div>
                  <div className="log-footer">
                    <span className="log-confidence">
                      Confidence: {(log.confidence * 100).toFixed(1)}%
                    </span>
                    {log.inference_ms && (
                      <span className="log-latency">{log.inference_ms}ms</span>
                    )}
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
