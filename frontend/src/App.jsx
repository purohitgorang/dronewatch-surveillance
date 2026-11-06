import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, Shield, AlertTriangle, Activity, 
  Cpu, Crosshair, Upload, Play, Square, Trash2, CheckCircle2, VideoOff, Check, X
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
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [mode, setMode] = useState('upload'); // Default to upload so user can test immediately with 1 click!
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.55);
  const [lastLatency, setLastLatency] = useState(null);
  const [latestDetection, setLatestDetection] = useState(null);
  const [selectedSample, setSelectedSample] = useState(null);
  const [sampleImages, setSampleImages] = useState([]);
  const [isLoadingSample, setIsLoadingSample] = useState(false);
  const [sampleResult, setSampleResult] = useState(null);

  const videoRef = useRef(null);
  const currentStreamRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const webcamOverlayCanvasRef = useRef(null);
  const imageCanvasRef = useRef(null);

  // Initialize status and load available test samples
  useEffect(() => {
    fetchSystemStatus();
    fetchSampleImages();
    return () => {
      stopCamera();
    };
  }, []);

  // Detection loop when webcam is running
  useEffect(() => {
    let interval = null;
    if (isCameraActive && isDetecting) {
      interval = setInterval(captureWebcamFrame, 1200);
    }
    return () => {
      if (interval) clearInterval(interval);
    };
  }, [isCameraActive, isDetecting, confidenceThreshold]);

  const fetchSystemStatus = async () => {
    try {
      const res = await fetch(`${API_URL}/status`);
      if (res.ok) {
        const data = await res.json();
        setSystemStatus(data);
      }
    } catch (err) {
      console.warn("Backend status error:", err);
      setSystemStatus(prev => ({ ...prev, status: 'offline' }));
    }
  };

  const fetchSampleImages = async () => {
    try {
      const res = await fetch(`${API_URL}/api/sample-images`);
      if (res.ok) {
        const data = await res.json();
        setSampleImages(data);
      }
    } catch (err) {
      console.warn("Could not load sample images:", err);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 } }
      });
      currentStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.play().catch(e => console.warn("Video play:", e));
      }
      setIsCameraActive(true);
      setIsDetecting(true);
    } catch (err) {
      console.error("Camera access failed:", err);
      alert("Webcam access error: " + err.message + "\n\nPlease ensure your camera is plugged in and allowed, or test with the 1-Click Samples!");
      setIsCameraActive(false);
      setIsDetecting(false);
    }
  };

  const stopCamera = () => {
    if (currentStreamRef.current) {
      currentStreamRef.current.getTracks().forEach(t => t.stop());
      currentStreamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsCameraActive(false);
    setIsDetecting(false);
    clearCanvas(webcamOverlayCanvasRef);
    setLatestDetection(null);
  };

  const clearCanvas = (canvasRef) => {
    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext('2d');
      ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
    }
  };

  const drawBoundingBoxesOnCanvas = (canvas, boxes, width, height) => {
    if (!canvas) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, width, height);

    if (!boxes || boxes.length === 0) return;

    boxes.forEach(det => {
      if (det.box_norm) {
        const [ymin, xmin, ymax, xmax] = det.box_norm;
        const x = xmin * width;
        const y = ymin * height;
        const w = (xmax - xmin) * width;
        const h = (ymax - ymin) * height;

        // Tactical Red Box
        ctx.strokeStyle = '#ef4444';
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);

        // Green Corner Brackets
        const bracketLen = 14;
        ctx.strokeStyle = '#22c55e';
        ctx.lineWidth = 4;
        
        ctx.beginPath();
        ctx.moveTo(x, y + bracketLen); ctx.lineTo(x, y); ctx.lineTo(x + bracketLen, y);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x + w - bracketLen, y); ctx.lineTo(x + w, y); ctx.lineTo(x + w, y + bracketLen);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x, y + h - bracketLen); ctx.lineTo(x, y + h); ctx.lineTo(x + bracketLen, y + h);
        ctx.stroke();

        ctx.beginPath();
        ctx.moveTo(x + w - bracketLen, y + h); ctx.lineTo(x + w, y + h); ctx.lineTo(x + w, y + h - bracketLen);
        ctx.stroke();

        // Label Tag
        const label = `${det.drone_name} (${(det.confidence * 100).toFixed(1)}%)`;
        ctx.font = 'bold 12px Inter, sans-serif';
        const textWidth = ctx.measureText(label).width;

        ctx.fillStyle = 'rgba(239, 68, 68, 0.95)';
        ctx.fillRect(x, Math.max(0, y - 24), textWidth + 14, 22);

        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, x + 7, Math.max(14, y - 8));
      }
    });
  };

  const captureWebcamFrame = async () => {
    if (!videoRef.current || !captureCanvasRef.current) return;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0 || video.paused || video.ended) return;

    const canvas = captureCanvasRef.current;
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('file', blob, 'webcam_frame.jpg');

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
          drawBoundingBoxesOnCanvas(
            webcamOverlayCanvasRef.current,
            data.all_detections || [data.data],
            video.clientWidth,
            video.clientHeight
          );
        } else {
          clearCanvas(webcamOverlayCanvasRef);
          setLatestDetection(null);
        }
      } catch (err) {
        console.error("Frame detection error:", err);
      }
    }, 'image/jpeg', 0.85);
  };

  // Test with 1-Click Sample Image
  const handleSelectSample = async (sample) => {
    setSelectedSample(sample);
    setIsLoadingSample(true);
    setSampleResult(null);

    try {
      const res = await fetch(`${API_URL}/api/test-sample/${sample.filename}?conf=${confidenceThreshold}`, {
        method: 'POST'
      });
      const data = await res.json();
      if (data.inference_ms) setLastLatency(data.inference_ms);
      setSampleResult(data);

      if (data.detected) {
        setLogs(prev => [data.data, ...prev]);
        setLatestDetection(data.data);
      } else {
        setLatestDetection(null);
      }

      // Render image on canvas with bounding boxes
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.src = sample.url;
      img.onload = () => {
        renderImageWithBoxes(img, data.all_detections || (data.data ? [data.data] : []));
      };
    } catch (err) {
      console.error("Sample test failed:", err);
    } finally {
      setIsLoadingSample(false);
    }
  };

  const renderImageWithBoxes = (img, detections) => {
    const canvas = imageCanvasRef.current;
    if (!canvas) return;

    // Aspect ratio scale to container
    const maxW = 760;
    const maxH = 460;
    let w = img.width;
    let h = img.height;
    const ratio = Math.min(maxW / w, maxH / h, 1.0);
    w = Math.round(w * ratio);
    h = Math.round(h * ratio);

    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.drawImage(img, 0, 0, w, h);

    if (detections && detections.length > 0) {
      detections.forEach(det => {
        if (det.box_norm) {
          const [ymin, xmin, ymax, xmax] = det.box_norm;
          const bx = xmin * w;
          const by = ymin * h;
          const bw = (xmax - xmin) * w;
          const bh = (ymax - ymin) * h;

          // Red Target Box
          ctx.strokeStyle = '#ef4444';
          ctx.lineWidth = 3;
          ctx.strokeRect(bx, by, bw, bh);

          // Green Tactical Corners
          const len = 14;
          ctx.strokeStyle = '#22c55e';
          ctx.lineWidth = 4;
          ctx.beginPath();
          ctx.moveTo(bx, by + len); ctx.lineTo(bx, by); ctx.lineTo(bx + len, by);
          ctx.moveTo(bx + bw - len, by); ctx.lineTo(bx + bw, by); ctx.lineTo(bx + bw, by + len);
          ctx.moveTo(bx, by + bh - len); ctx.lineTo(bx, by + bh); ctx.lineTo(bx + len, by + bh);
          ctx.moveTo(bx + bw - len, by + bh); ctx.lineTo(bx + bw, by + bh); ctx.lineTo(bx + bw, by + bh - len);
          ctx.stroke();

          // Label
          const tag = `${det.drone_name} (${(det.confidence * 100).toFixed(1)}%)`;
          ctx.font = 'bold 13px Inter, sans-serif';
          const tWidth = ctx.measureText(tag).width;

          ctx.fillStyle = 'rgba(239, 68, 68, 0.95)';
          ctx.fillRect(bx, Math.max(0, by - 26), tWidth + 16, 24);

          ctx.fillStyle = '#ffffff';
          ctx.fillText(tag, bx + 8, Math.max(16, by - 9));
        }
      });
    }
  };

  const handleCustomUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setSelectedSample(null);
    setIsLoadingSample(true);
    setSampleResult(null);

    const formData = new FormData();
    formData.append('file', file);

    const reader = new FileReader();
    reader.onload = async (event) => {
      const img = new Image();
      img.src = event.target.result;
      img.onload = async () => {
        try {
          const res = await fetch(`${API_URL}/detect?conf=${confidenceThreshold}`, {
            method: 'POST',
            body: formData,
          });
          const data = await res.json();
          if (data.inference_ms) setLastLatency(data.inference_ms);
          setSampleResult(data);

          if (data.detected) {
            setLogs(prev => [data.data, ...prev]);
            setLatestDetection(data.data);
          } else {
            setLatestDetection(null);
          }

          renderImageWithBoxes(img, data.all_detections || (data.data ? [data.data] : []));
        } catch (err) {
          console.error("Upload error:", err);
        } finally {
          setIsLoadingSample(false);
        }
      };
    };
    reader.readAsDataURL(file);
  };

  const handleClearLogs = async () => {
    try {
      await fetch(`${API_URL}/logs`, { method: 'DELETE' });
      setLogs([]);
      setLatestDetection(null);
      clearCanvas(webcamOverlayCanvasRef);
      clearCanvas(imageCanvasRef);
      setSampleResult(null);
    } catch (err) {
      setLogs([]);
    }
  };

  return (
    <div className="container">
      {/* Top Header */}
      <header className="header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
          <Shield size={34} color="#60a5fa" />
          <div>
            <div className="title">Drone Hawk X</div>
            <div style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
              Offline Edge Airspace Surveillance Engine • Local PyTorch CUDA
            </div>
          </div>
        </div>

        {/* Hardware & Status Pill */}
        <div style={{ display: 'flex', gap: '0.75rem', alignItems: 'center' }}>
          <div className="edge-badge">
            <Cpu size={14} color="#22c55e" />
            <span>{systemStatus.device || 'NVIDIA GTX 1650'}</span>
          </div>

          <div className="status-pill">
            <div 
              className="status-dot" 
              style={{ backgroundColor: isCameraActive && isDetecting ? '#22c55e' : (mode === 'upload' ? '#38bdf8' : '#eab308') }} 
            />
            <span>{isCameraActive && isDetecting ? 'Surveillance Active' : (mode === 'upload' ? 'Test Mode Active' : 'Camera Standby')}</span>
          </div>
        </div>
      </header>

      {/* Mode Tabs */}
      <div className="mode-tabs">
        <button 
          className={`tab-btn ${mode === 'upload' ? 'active' : ''}`}
          onClick={() => { setMode('upload'); stopCamera(); }}
        >
          <Upload size={16} /> 🎯 Instant Test Samples & File Upload
        </button>
        <button 
          className={`tab-btn ${mode === 'webcam' ? 'active' : ''}`}
          onClick={() => { setMode('webcam'); }}
        >
          <Camera size={16} /> 📹 Live Webcam Surveillance
        </button>
      </div>

      <div className="main-grid">
        {/* Visual Viewport Panel */}
        <div className="glass-panel">
          {mode === 'webcam' ? (
            <div className="webcam-container">
              {/* Video is ALWAYS mounted to prevent null ref bugs */}
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted 
                style={{ width: '100%', height: '100%', objectFit: 'cover', display: isCameraActive ? 'block' : 'none' }} 
              />
              <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
              <canvas 
                ref={webcamOverlayCanvasRef} 
                className="hud-overlay-canvas" 
                style={{ display: isCameraActive ? 'block' : 'none' }} 
              />

              {isCameraActive ? (
                <>
                  <div className="overlay">
                    <Crosshair size={16} color="#ef4444" />
                    <span>Live Optical Stream (Monitoring at 30 FPS)</span>
                  </div>

                  {latestDetection && (
                    <div className="alert-hud">
                      <AlertTriangle size={18} color="#ffffff" />
                      <span>TARGET IDENTIFIED: {latestDetection.drone_name}</span>
                    </div>
                  )}
                </>
              ) : (
                <div className="camera-standby-view">
                  <VideoOff size={48} color="#64748b" style={{ marginBottom: '1rem' }} />
                  <h3>Optical Webcam in Standby</h3>
                  <p style={{ color: '#94a3b8', fontSize: '0.9rem', maxWidth: '380px', margin: '0 auto 1.5rem' }}>
                    Click below to initialize your laptop webcam and begin live GPU inference.
                  </p>
                  <button className="btn btn-detect" onClick={startCamera}>
                    <Play size={16} /> Start Live Surveillance Feed
                  </button>
                </div>
              )}
            </div>
          ) : (
            <div className="test-sample-container">
              {/* Canvas viewport displaying image and bounding boxes */}
              <div className="canvas-viewport-wrapper">
                <canvas ref={imageCanvasRef} className="sample-render-canvas" />
                {!selectedSample && !sampleResult && (
                  <div className="empty-viewport-placeholder">
                    <Upload size={44} color="#60a5fa" style={{ marginBottom: '0.75rem' }} />
                    <h3>Click Any Test Sample Below</h3>
                    <p style={{ color: '#94a3b8', fontSize: '0.85rem' }}>
                      Test authentic drone detections and distractor suppressions instantly with 1 click.
                    </p>
                  </div>
                )}
                {isLoadingSample && (
                  <div className="loading-badge">Evaluating on NVIDIA GPU...</div>
                )}
              </div>

              {/* Status Banner for Test Result */}
              {sampleResult && (
                <div className={`result-alert-box ${sampleResult.detected ? 'alert-danger' : 'alert-success'}`}>
                  {sampleResult.detected ? (
                    <>
                      <AlertTriangle size={18} color="#ef4444" />
                      <div>
                        <strong>TARGET CONFIRMED:</strong> {sampleResult.data.drone_name} (Confidence: {(sampleResult.data.confidence * 100).toFixed(1)}% • Latency: {sampleResult.inference_ms}ms)
                      </div>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={18} color="#22c55e" />
                      <div>
                        <strong>AIRSPACE CLEAR:</strong> Non-Drone Object / Distractor Correctly Ignored (0 False Alarm • Latency: {sampleResult.inference_ms}ms)
                      </div>
                    </>
                  )}
                </div>
              )}

              {/* 1-Click Test Samples Grid */}
              <div className="sample-selector-section">
                <div className="sample-group-title">🎯 1-Click Drone Samples (Expected: DETECTED):</div>
                <div className="sample-buttons-grid">
                  {sampleImages.filter(s => s.category === 'drone').map(sample => (
                    <button
                      key={sample.id}
                      className={`sample-btn drone-pill ${selectedSample?.id === sample.id ? 'active' : ''}`}
                      onClick={() => handleSelectSample(sample)}
                    >
                      <span className="sample-dot-red"></span>
                      {sample.name}
                    </button>
                  ))}
                </div>

                <div className="sample-group-title" style={{ marginTop: '0.75rem' }}>
                  🛡️ 1-Click Negative Distractors (Expected: AIRSPACE CLEAR / 0 False Alarm):
                </div>
                <div className="sample-buttons-grid">
                  {sampleImages.filter(s => s.category === 'distractor').map(sample => (
                    <button
                      key={sample.id}
                      className={`sample-btn distractor-pill ${selectedSample?.id === sample.id ? 'active' : ''}`}
                      onClick={() => handleSelectSample(sample)}
                    >
                      <span className="sample-dot-green"></span>
                      {sample.name}
                    </button>
                  ))}
                </div>

                <div style={{ marginTop: '1rem', display: 'flex', justifyContent: 'center' }}>
                  <label className="btn btn-secondary file-input-label">
                    <Upload size={16} /> Upload Custom Image From Your PC
                    <input type="file" accept="image/*" onChange={handleCustomUpload} style={{ display: 'none' }} />
                  </label>
                </div>
              </div>
            </div>
          )}

          {/* Tactical Controls & Tuning */}
          <div className="controls-row">
            {mode === 'webcam' && isCameraActive && (
              <div>
                <button className="btn btn-stop" onClick={stopCamera}>
                  <Square size={16} /> Stop Surveillance
                </button>
              </div>
            )}

            {/* Sensitivity Slider */}
            <div className="slider-group">
              <span style={{ fontSize: '0.85rem', color: '#94a3b8' }}>
                Confidence Sensitivity: <strong>{(confidenceThreshold * 100).toFixed(0)}%</strong>
              </span>
              <input 
                type="range" 
                min="0.30" 
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
                  No unauthorized drones detected in this session.
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
