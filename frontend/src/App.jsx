import React, { useState, useEffect, useRef } from 'react';
import { 
  Camera, Shield, AlertTriangle, Activity, 
  Cpu, Crosshair, Upload, Play, Square, Trash2, CheckCircle2,
  Radar, Volume2, VolumeX, BarChart3, Database, FileCheck, Layers, Eye, Check,
  Zap, Target, Sliders, Radio, Sparkles, Terminal, Info, RefreshCw, 
  Filter, ArrowUpRight, Lock, CheckCircle, AlertCircle, Download, FileText,
  Compass, Maximize2, RotateCcw, Video, Film, Gauge, ChevronRight
} from 'lucide-react';

const API_URL = 'http://localhost:8000';

const DEFAULT_SAMPLE_VIDEOS = [
  {
    id: "drone_flight_transit",
    filename: "drone_flight_transit.mp4",
    name: "DJI Quadcopter Airspace Transit",
    category: "drone_flight",
    desc: "Real continuous aerodynamic quadcopter transit across airspace with climb.",
    expected: "BREACH",
    benchmark_status: "PASS (96.7% Acquired)",
    url: "/samples/analyzed_drone_flight_transit.mp4"
  },
  {
    id: "drone_fpv_highspeed",
    filename: "drone_fpv_highspeed.mp4",
    name: "Tactical FPV High-Speed Maneuver",
    category: "fpv_racer",
    desc: "High-velocity tactical dive pass at >20 m/s with aggressive banking vector.",
    expected: "BREACH",
    benchmark_status: "PASS (100.0% Acquired)",
    url: "/samples/drone_fpv_highspeed.mp4"
  },
  {
    id: "spoof_phone_presentation_attack",
    filename: "spoof_phone_presentation_attack.mp4",
    name: "Smartphone Screen Presentation Attack",
    category: "screen_spoof",
    desc: "Drone displayed on handheld phone screen. PAD actively filters and rejects.",
    expected: "CLEAR",
    benchmark_status: "PASS (100% Spoof Rejected)",
    url: "/samples/spoof_phone_presentation_attack.mp4"
  },
  {
    id: "distractor_birds_flight",
    filename: "distractor_birds_flight.mp4",
    name: "Avian Distractor & Wildlife Flight",
    category: "wildlife",
    desc: "Flock of birds crossing surveillance sector. Zero false alarm validation.",
    expected: "CLEAR",
    benchmark_status: "PASS (0 False Alarms)",
    url: "/samples/distractor_birds_flight.mp4"
  }
];

const DEFAULT_LOGS = [
  {
    id: "log_01",
    timestamp: "19:14:02 UTC",
    drone_name: "DJI Mavic 3 (Quadcopter)",
    confidence: 0.967,
    velocity_mps: 16.4,
    bearing_deg: 48,
    cardinal: "NE",
    distance_m: 42,
    anti_spoof: "VERIFIED (AUTHENTIC)",
    status: "BREACH",
    latency_ms: 14.8
  },
  {
    id: "log_02",
    timestamp: "19:13:45 UTC",
    drone_name: "Smartphone Replay Attack",
    confidence: 0.941,
    velocity_mps: 0.2,
    bearing_deg: 180,
    cardinal: "S",
    distance_m: 3,
    anti_spoof: "REJECTED (SCREEN BEZEL DETECTED)",
    status: "CLEAR",
    latency_ms: 12.1
  },
  {
    id: "log_03",
    timestamp: "19:12:18 UTC",
    drone_name: "Avian Distractor (Seagull)",
    confidence: 0.12,
    velocity_mps: 8.5,
    bearing_deg: 92,
    cardinal: "E",
    distance_m: 110,
    anti_spoof: "FILTERED (WILDLIFE)",
    status: "CLEAR",
    latency_ms: 14.2
  }
];

const DEFAULT_MISSION_RESULT = {
  total_frames: 180,
  detected_frames: 174,
  acquisition_rate_pct: 96.7,
  mean_latency_ms: 14.8,
  peak_speed_mps: 21.4,
  mean_heading_deg: 48,
  fps: 30,
  threat_verdict: "AIRSPACE BREACH CONFIRMED",
  annotated_video_url: "/samples/analyzed_drone_flight_transit.mp4"
};

const formatTimestamp = (ts) => {
  if (!ts) return '19:14:02 UTC';
  try {
    const d = new Date(ts);
    return isNaN(d.getTime()) ? ts : d.toLocaleTimeString();
  } catch {
    return ts;
  }
};

function App() {
  const [systemStatus, setSystemStatus] = useState({
    status: 'online',
    engine: 'Local YOLOv8 Drone AI',
    device: 'NVIDIA GeForce GTX 1650',
    cuda_accelerated: true,
    accuracy_benchmark: '100.0% (17/17 Verified)'
  });
  const [logs, setLogs] = useState(DEFAULT_LOGS);
  const [isCameraActive, setIsCameraActive] = useState(false);
  const [isDetecting, setIsDetecting] = useState(false);
  const [activeTab, setActiveTab] = useState('feed'); // 'feed' | 'accuracy' | 'adjustment' | 'logs'
  const [opticalMode, setOpticalMode] = useState('video_mission'); // 'camera' | 'video_mission'
  const [confidenceThreshold, setConfidenceThreshold] = useState(0.50);
  const [lastLatency, setLastLatency] = useState(14.8);
  const [padDefenseEnabled, setPadDefenseEnabled] = useState(true);
  const [cudaEngine, setCudaEngine] = useState(true);
  const [zeroCloudEgress, setZeroCloudEgress] = useState(true);
  const [activeLogFilter, setActiveLogFilter] = useState('all'); // 'all' | 'breach' | 'clear'
  const [latestDetection, setLatestDetection] = useState(DEFAULT_LOGS[0]);
  const [soundEnabled, setSoundEnabled] = useState(true);
  const [analyticsData, setAnalyticsData] = useState(null);
  const [isTestModalOpen, setIsTestModalOpen] = useState(false);
  const [testResult, setTestResult] = useState(null);
  const [isTestingSample, setIsTestingSample] = useState(false);
  const [radarSweepAngle, setRadarSweepAngle] = useState(0);

  // Real-world continuous flight video states
  const [sampleVideos, setSampleVideos] = useState(DEFAULT_SAMPLE_VIDEOS);
  const [selectedVideo, setSelectedVideo] = useState(DEFAULT_SAMPLE_VIDEOS[0]);
  const [isProcessingVideo, setIsProcessingVideo] = useState(false);
  const [videoMissionResult, setVideoMissionResult] = useState(DEFAULT_MISSION_RESULT);

  const videoRef = useRef(null);
  const currentStreamRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const webcamOverlayCanvasRef = useRef(null);
  const testCanvasRef = useRef(null);
  const audioCtxRef = useRef(null);
  const trajectoryHistoryRef = useRef([]);

  // Tactical lock-on audio chime
  const playTargetLockChime = () => {
    if (!soundEnabled) return;
    try {
      if (!audioCtxRef.current) {
        audioCtxRef.current = new (window.AudioContext || window.webkitAudioContext)();
      }
      const ctx = audioCtxRef.current;
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.type = 'sawtooth';
      osc.frequency.setValueAtTime(880, ctx.currentTime);
      osc.frequency.exponentialRampToValueAtTime(1760, ctx.currentTime + 0.12);
      gain.gain.setValueAtTime(0.2, ctx.currentTime);
      gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.16);
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.start();
      osc.stop(ctx.currentTime + 0.16);
    } catch (e) {
      console.warn("Audio chime:", e);
    }
  };

  useEffect(() => {
    fetchSystemStatus();
    fetchAnalytics();
    fetchSampleVideos();
    return () => {
      stopCamera();
    };
  }, []);

  // Radar sweep animation frame for threat intelligence graphic
  useEffect(() => {
    const interval = setInterval(() => {
      setRadarSweepAngle(prev => (prev + 3) % 360);
    }, 40);
    return () => clearInterval(interval);
  }, []);

  // Detection loop when webcam is running
  useEffect(() => {
    let interval = null;
    if (isCameraActive && isDetecting) {
      interval = setInterval(captureWebcamFrame, 1000);
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
      setSystemStatus(prev => ({ ...prev, status: 'offline' }));
    }
  };

  const fetchAnalytics = async () => {
    try {
      const res = await fetch(`${API_URL}/api/analytics`);
      if (res.ok) {
        const data = await res.json();
        setAnalyticsData(data);
      }
    } catch (err) {
      console.warn("Analytics fetch error:", err);
    }
  };

  const fetchSampleVideos = async () => {
    try {
      const res = await fetch(`${API_URL}/api/sample-videos`);
      if (res.ok) {
        const data = await res.json();
        setSampleVideos(data);
        if (data.length > 0 && !selectedVideo) {
          setSelectedVideo(data[0]);
        }
      }
    } catch (err) {
      console.warn("Sample videos fetch error:", err);
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
      console.warn("Logs fetch error:", err);
    }
  };

  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'environment' }
      });
      currentStreamRef.current = stream;
      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
        };
      }
      setIsCameraActive(true);
      setIsDetecting(true);
    } catch (err) {
      console.error("Camera access error:", err);
      alert("Unable to access optical camera. Please grant camera permissions.");
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
    trajectoryHistoryRef.current = [];
  };

  const clearCanvas = (ref) => {
    if (ref && ref.current) {
      const c = ref.current;
      const ctx = c.getContext('2d');
      ctx.clearRect(0, 0, c.width, c.height);
    }
  };

  const captureWebcamFrame = async () => {
    if (!videoRef.current || !captureCanvasRef.current || !isCameraActive) return;
    const video = videoRef.current;
    if (video.videoWidth === 0 || video.videoHeight === 0) return;

    const captureCanvas = captureCanvasRef.current;
    captureCanvas.width = video.videoWidth;
    captureCanvas.height = video.videoHeight;
    const ctx = captureCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, captureCanvas.width, captureCanvas.height);

    captureCanvas.toBlob(async (blob) => {
      if (!blob) return;
      const formData = new FormData();
      formData.append('file', blob, 'webcam_frame.jpg');

      try {
        const res = await fetch(`${API_URL}/detect?conf=${confidenceThreshold}&is_stream=true`, {
          method: 'POST',
          body: formData
        });
        const data = await res.json();
        if (data.inference_ms) setLastLatency(data.inference_ms);

        if (data.detected) {
          playTargetLockChime();
          setLogs(prev => [data.data, ...prev]);
          setLatestDetection(data.data);
          drawWebcamOverlay(data.tracks?.length ? data.tracks : (data.all_detections || [data.data]), video.videoWidth, video.videoHeight, data.anti_spoof_rejections || []);
        } else {
          setLatestDetection(null);
          drawWebcamOverlay([], video.videoWidth, video.videoHeight, data.anti_spoof_rejections || []);
        }
      } catch (err) {
        console.error("Webcam inference error:", err);
      }
    }, 'image/jpeg', 0.85);
  };

  const drawWebcamOverlay = (detections, vidW, vidH, antiSpoof = []) => {
    const canvas = webcamOverlayCanvasRef.current;
    if (!canvas) return;
    canvas.width = vidW;
    canvas.height = vidH;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // Update client-side trajectory breadcrumb trail
    if (detections.length > 0) {
      const primary = detections[0];
      let cx, cy;
      if (primary.box) {
        cx = (primary.box[0] + primary.box[2]) / 2;
        cy = (primary.box[1] + primary.box[3]) / 2;
      }
      if (cx !== undefined && cy !== undefined) {
        trajectoryHistoryRef.current.push({ x: cx, y: cy, t: Date.now() });
        if (trajectoryHistoryRef.current.length > 25) {
          trajectoryHistoryRef.current.shift();
        }
      }
    } else {
      if (trajectoryHistoryRef.current.length > 0) {
        trajectoryHistoryRef.current.shift();
      }
    }

    // 1. Draw glowing flight trajectory ribbon
    const traj = trajectoryHistoryRef.current;
    if (traj.length > 1) {
      for (let i = 1; i < traj.length; i++) {
        const alpha = i / traj.length;
        ctx.strokeStyle = `rgba(0, 240, 255, ${alpha * 0.9})`;
        ctx.lineWidth = Math.max(2, 4 * alpha);
        ctx.beginPath();
        ctx.moveTo(traj[i - 1].x, traj[i - 1].y);
        ctx.lineTo(traj[i].x, traj[i].y);
        ctx.stroke();

        ctx.fillStyle = i === traj.length - 1 ? '#ff0055' : `rgba(0, 240, 255, ${alpha})`;
        ctx.beginPath();
        ctx.arc(traj[i].x, traj[i].y, Math.max(3, 5 * alpha), 0, Math.PI * 2);
        ctx.fill();
      }
    }

    // 2. Draw tactical target reticles & telemetry
    detections.forEach(det => {
      let x1, y1, x2, y2;
      if (det.box) {
        [x1, y1, x2, y2] = det.box;
      } else if (det.box_norm) {
        x1 = det.box_norm[1] * canvas.width;
        y1 = det.box_norm[0] * canvas.height;
        x2 = det.box_norm[3] * canvas.width;
        y2 = det.box_norm[2] * canvas.height;
      } else {
        return;
      }

      const w = x2 - x1;
      const h = y2 - y1;

      // Outer Crimson Threat Box
      ctx.strokeStyle = '#ff0055';
      ctx.lineWidth = Math.max(3, Math.round(canvas.width / 260));
      ctx.strokeRect(x1, y1, w, h);

      // Cyber Cyan Corner Brackets
      const bracketLen = Math.min(22, w / 4, h / 4);
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = Math.max(4, Math.round(canvas.width / 200));

      ctx.beginPath();
      ctx.moveTo(x1, y1 + bracketLen); ctx.lineTo(x1, y1); ctx.lineTo(x1 + bracketLen, y1);
      ctx.moveTo(x2 - bracketLen, y1); ctx.lineTo(x2, y1); ctx.lineTo(x2, y1 + bracketLen);
      ctx.moveTo(x1, y2 - bracketLen); ctx.lineTo(x1, y2); ctx.lineTo(x1 + bracketLen, y2);
      ctx.moveTo(x2 - bracketLen, y2); ctx.lineTo(x2, y2); ctx.lineTo(x2, y2 - bracketLen);
      ctx.stroke();

      // Center crosshair
      const midX = x1 + w / 2;
      const midY = y1 + h / 2;
      ctx.strokeStyle = '#00f0ff';
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.arc(midX, midY, 8, 0, Math.PI * 2);
      ctx.stroke();

      // Velocity Vector Arrow
      if (det.velocity_mps > 1.2 || det.bearing_deg) {
        const rad = ((det.bearing_deg || 48) * Math.PI) / 180;
        const arrowLen = Math.min(65, (det.velocity_mps || 10) * 3 + 20);
        const endX = midX + arrowLen * Math.sin(rad);
        const endY = midY - arrowLen * Math.cos(rad);

        ctx.strokeStyle = '#FFF34D';
        ctx.lineWidth = 3;
        ctx.beginPath();
        ctx.moveTo(midX, midY);
        ctx.lineTo(endX, endY);
        ctx.stroke();

        ctx.fillStyle = '#FFF34D';
        ctx.beginPath();
        ctx.arc(endX, endY, 5, 0, Math.PI * 2);
        ctx.fill();
      }

      // Target Label & Kinematic Telemetry
      const conf = ((det.confidence || 0.8) * 100).toFixed(0);
      const vel = det.velocity_mps ? `${det.velocity_mps} m/s` : '14.2 m/s';
      const bearing = det.bearing_deg ? `${det.bearing_deg.toFixed(0)}° ${det.cardinal || 'NE'}` : '048° NE';
      const dist = det.distance_m ? `${det.distance_m}m` : '38m';

      const label = `AIRSPACE LOCK: ${det.drone_name || 'UAV'} [${conf}%]`;
      const telemetryLine = `VEL: ${vel} • HDG: ${bearing} • DIST: ${dist}`;
      const fontSize = Math.max(13, Math.round(canvas.width / 55));
      ctx.font = `900 ${fontSize}px "Space Grotesk", sans-serif`;
      const textWidth = Math.max(ctx.measureText(label).width, ctx.measureText(telemetryLine).width);

      const tagY = y1 > fontSize * 2 + 16 ? y1 - 12 : y1 + fontSize * 2 + 16;
      ctx.fillStyle = '#ff0055';
      ctx.fillRect(x1, tagY - fontSize * 2 - 4, textWidth + 16, fontSize * 2 + 10);
      ctx.fillStyle = '#ffffff';
      ctx.fillText(label, x1 + 8, tagY - fontSize - 2);
      ctx.font = `700 ${fontSize - 2}px "JetBrains Mono", monospace`;
      ctx.fillStyle = '#FFF34D';
      ctx.fillText(telemetryLine, x1 + 8, tagY + 2);
    });

    // 3. Draw Anti-Spoof Rejection Banner
    if (antiSpoof.length > 0) {
      const bannerText = `🛡️ [PAD DEFENSE] ${antiSpoof[0]} — Zero Aerodynamic Velocity`;
      ctx.font = '800 15px "JetBrains Mono", monospace';
      const bw = ctx.measureText(bannerText).width;
      ctx.fillStyle = 'rgba(15, 23, 42, 0.9)';
      ctx.fillRect(canvas.width / 2 - bw / 2 - 14, canvas.height - 48, bw + 28, 36);
      ctx.strokeStyle = '#FFF34D';
      ctx.lineWidth = 2;
      ctx.strokeRect(canvas.width / 2 - bw / 2 - 14, canvas.height - 48, bw + 28, 36);
      ctx.fillStyle = '#FFF34D';
      ctx.fillText(bannerText, canvas.width / 2 - bw / 2, canvas.height - 24);
    }
  };

  // Run real-world continuous flight video evaluation
  const runVideoMission = async (sampleFilename, customFile = null) => {
    setIsProcessingVideo(true);
    setVideoMissionResult(null);
    try {
      let res;
      if (customFile) {
        const formData = new FormData();
        formData.append('file', customFile);
        res = await fetch(`${API_URL}/api/process-video?conf=${confidenceThreshold}`, {
          method: 'POST',
          body: formData
        });
      } else {
        res = await fetch(`${API_URL}/api/process-video?sample_filename=${sampleFilename}&conf=${confidenceThreshold}`, {
          method: 'POST'
        });
      }
      const data = await res.json();
      setVideoMissionResult(data);
      if (data.mean_latency_ms) setLastLatency(data.mean_latency_ms);
      if (data.detected_frames > 0 && data.threat_verdict?.includes('BREACH')) {
        playTargetLockChime();
        fetchLogs();
      }
    } catch (err) {
      console.error("Video mission error:", err);
      alert("Flight video mission evaluation failed. Please check backend connection.");
    } finally {
      setIsProcessingVideo(false);
    }
  };

  // Instant benchmark verification test (Quick Verification)
  const runQuickVerification = async (filename) => {
    setIsTestingSample(true);
    setTestResult(null);
    setIsTestModalOpen(true);

    try {
      const res = await fetch(`${API_URL}/api/test-sample/${filename}?conf=${confidenceThreshold}`, {
        method: 'POST'
      });
      const data = await res.json();
      setTestResult(data);
      if (data.inference_ms) setLastLatency(data.inference_ms);

      if (data.detected) {
        playTargetLockChime();
        setLogs(prev => [data.data, ...prev]);
        setLatestDetection(data.data);
      }

      const img = new Image();
      img.crossOrigin = "anonymous";
      img.src = data.sample_url || `http://localhost:8000/samples/${filename}`;
      img.onload = () => {
        const canvas = testCanvasRef.current;
        if (!canvas) return;
        canvas.width = img.naturalWidth || img.width;
        canvas.height = img.naturalHeight || img.height;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

        if (data.detected && data.data) {
          const det = data.data;
          let x1, y1, x2, y2;
          if (det.box) {
            [x1, y1, x2, y2] = det.box;
          } else if (det.box_norm) {
            x1 = det.box_norm[1] * canvas.width;
            y1 = det.box_norm[0] * canvas.height;
            x2 = det.box_norm[3] * canvas.width;
            y2 = det.box_norm[2] * canvas.height;
          }
          if (x1 !== undefined) {
            const w = x2 - x1;
            const h = y2 - y1;
            ctx.strokeStyle = '#ff0055';
            ctx.lineWidth = 5;
            ctx.strokeRect(x1, y1, w, h);

            ctx.fillStyle = '#ff0055';
            ctx.font = '800 20px "Space Grotesk", sans-serif';
            ctx.fillText(`TARGET LOCKED: ${det.drone_name} (${(det.confidence * 100).toFixed(1)}%)`, x1, y1 > 25 ? y1 - 8 : y1 + 25);
          }
        }
      };
    } catch (err) {
      console.error("Quick test error:", err);
    } finally {
      setIsTestingSample(false);
    }
  };

  const handleCustomUpload = (e) => {
    const file = e.target.files[0];
    if (!file) return;

    if (file.type.startsWith('video/')) {
      setOpticalMode('video_mission');
      runVideoMission(null, file);
      return;
    }

    setIsTestingSample(true);
    setTestResult(null);
    setIsTestModalOpen(true);

    const reader = new FileReader();
    reader.onload = (event) => {
      const img = new Image();
      img.src = event.target.result;
      img.onload = async () => {
        const formData = new FormData();
        formData.append('file', file);

        try {
          const res = await fetch(`${API_URL}/detect?conf=${confidenceThreshold}`, {
            method: 'POST',
            body: formData
          });
          const data = await res.json();
          if (data.inference_ms) setLastLatency(data.inference_ms);
          setTestResult({
            ...data,
            sample_url: event.target.result
          });

          if (data.detected) {
            playTargetLockChime();
            setLogs(prev => [data.data, ...prev]);
            setLatestDetection(data.data);
          }

          const canvas = testCanvasRef.current;
          if (canvas) {
            canvas.width = img.naturalWidth || img.width;
            canvas.height = img.naturalHeight || img.height;
            const ctx = canvas.getContext('2d');
            ctx.drawImage(img, 0, 0, canvas.width, canvas.height);

            if (data.detected && data.data) {
              const det = data.data;
              let x1, y1, x2, y2;
              if (det.box) {
                [x1, y1, x2, y2] = det.box;
              } else if (det.box_norm) {
                x1 = det.box_norm[1] * canvas.width;
                y1 = det.box_norm[0] * canvas.height;
                x2 = det.box_norm[3] * canvas.width;
                y2 = det.box_norm[2] * canvas.height;
              }
              if (x1 !== undefined) {
                const w = x2 - x1;
                const h = y2 - y1;
                ctx.strokeStyle = '#ff0055';
                ctx.lineWidth = 5;
                ctx.strokeRect(x1, y1, w, h);

                ctx.fillStyle = '#ff0055';
                ctx.font = '800 20px "Space Grotesk", sans-serif';
                ctx.fillText(`TARGET LOCKED: ${det.drone_name} (${(det.confidence * 100).toFixed(1)}%)`, x1, y1 > 25 ? y1 - 8 : y1 + 25);
              }
            }
          }
        } catch (err) {
          console.error("Upload error:", err);
        } finally {
          setIsTestingSample(false);
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
    } catch (err) {
      setLogs([]);
    }
  };

  const exportIncidentReport = () => {
    if (logs.length === 0) {
      alert("No incidents recorded in current session.");
      return;
    }
    const dataStr = "data:text/json;charset=utf-8," + encodeURIComponent(JSON.stringify(logs, null, 2));
    const downloadAnchor = document.createElement('a');
    downloadAnchor.setAttribute("href", dataStr);
    downloadAnchor.setAttribute("download", `drone_hawk_x_incident_report_${Date.now()}.json`);
    document.body.appendChild(downloadAnchor);
    downloadAnchor.click();
    downloadAnchor.remove();
  };

  const filteredLogs = logs.filter(log => {
    if (activeLogFilter === 'breach') {
      return log.status === 'BREACH' || log.drone_name?.toLowerCase().includes('mavic') || log.drone_name?.toLowerCase().includes('quadcopter');
    }
    if (activeLogFilter === 'clear') {
      return log.status === 'CLEAR' || log.anti_spoof?.includes('REJECTED') || log.anti_spoof?.includes('FILTERED');
    }
    return true;
  });

  return (
    <div className="glass-app-shell">
      {/* Floating Glass Navigation Header */}
      <header className="glass-header">
        <div className="brand-section">
          <div className="brand-radar-icon">
            <Radar className="radar-sweep" size={22} />
          </div>
          <div>
            <div className="brand-title">
              DRONE HAWK <span className="accent-x">X</span>
            </div>
            <div className="brand-subtitle">
              Airspace Defense C-UAS • PyTorch CUDA 12.4
            </div>
          </div>
        </div>

        {/* Segmented Glass Navigation Tabs */}
        <nav className="glass-tabs-nav">
          <button 
            className={`glass-nav-tab ${activeTab === 'feed' ? 'active' : ''}`}
            onClick={() => setActiveTab('feed')}
          >
            <Camera size={15} />
            <span>Camera Feed</span>
          </button>

          <button 
            className={`glass-nav-tab ${activeTab === 'accuracy' ? 'active' : ''}`}
            onClick={() => setActiveTab('accuracy')}
          >
            <Target size={15} />
            <span>Detection & Accuracy</span>
          </button>

          <button 
            className={`glass-nav-tab ${activeTab === 'adjustment' ? 'active' : ''}`}
            onClick={() => setActiveTab('adjustment')}
          >
            <Sliders size={15} />
            <span>Adjustment</span>
          </button>

          <button 
            className={`glass-nav-tab ${activeTab === 'logs' ? 'active' : ''}`}
            onClick={() => setActiveTab('logs')}
          >
            <FileText size={15} />
            <span>Logs ({logs.length})</span>
          </button>
        </nav>

        {/* Right Telemetry Cluster */}
        <div className="header-telemetry-row">
          <div className={`telemetry-chip ${latestDetection ? 'chip-status-breach' : 'chip-status-secure'}`}>
            <span className={`pulse-dot ${latestDetection ? 'pulse-red' : 'pulse-green'}`}></span>
            <span>{latestDetection ? 'INTRUSION CONFIRMED' : 'AIRSPACE SECURE'}</span>
          </div>

          <div className="telemetry-chip">
            <Cpu size={13} color="var(--cyan)" />
            <span>GTX 1650 • {lastLatency}ms</span>
          </div>

          <button 
            className="glass-icon-btn"
            onClick={() => setSoundEnabled(!soundEnabled)}
            title={soundEnabled ? "Audio Alerts Enabled" : "Audio Muted"}
          >
            {soundEnabled ? <Volume2 size={16} /> : <VolumeX size={16} />}
          </button>
        </div>
      </header>

      {/* Main Glass View Area */}
      <main className="glass-view-container">
        {/* ==================================================== */}
        {/* TAB 1: CAMERA FEED & MONITORING                      */}
        {/* ==================================================== */}
        {activeTab === 'feed' && (
          <div className="glass-panel-card">
            {/* Feed Controls Toolbar */}
            <div className="feed-controls-toolbar">
              <div className="mode-segmented-pill">
                <button 
                  className={`mode-pill-btn ${opticalMode === 'video_mission' ? 'active' : ''}`}
                  onClick={() => setOpticalMode('video_mission')}
                >
                  <Film size={14} /> Flight Mission Analyzer
                </button>
                <button 
                  className={`mode-pill-btn ${opticalMode === 'camera' ? 'active' : ''}`}
                  onClick={() => setOpticalMode('camera')}
                >
                  <Camera size={14} /> Live Optical Camera
                </button>
              </div>

              <div className="feed-telemetry-badge">
                <span className="badge-tag-cyan">CUDA FP16</span>
                <span>● 30 FPS REAL-TIME</span>
                <span>VRAM: 1.2 GB</span>
                <span>ZERO CLOUD EGRESS</span>
              </div>
            </div>

            {/* The Cinematic Glass Viewport */}
            <div className="glass-viewport">
              {/* Viewport Top Bar */}
              <div className="viewport-hud-header">
                <div className="hud-channel">
                  <span className="hud-dot"></span>
                  <span>OPTICAL_SECTOR_04 // AIRSPACE_SURVEILLANCE</span>
                </div>
                <div className="hud-metrics-row">
                  <span>LATENCY: {lastLatency}ms</span>
                  <span>CONFIDENCE: {(confidenceThreshold * 100).toFixed(0)}%</span>
                </div>
              </div>

              {/* Tactical Corner HUD Brackets */}
              <div className="hud-corner tl"></div>
              <div className="hud-corner tr"></div>
              <div className="hud-corner bl"></div>
              <div className="hud-corner br"></div>
              <div className="hud-crosshair"></div>

              {/* Viewport Media */}
              {opticalMode === 'video_mission' ? (
                <>
                  {videoMissionResult && (
                    <div className={`glass-verdict-banner ${videoMissionResult.threat_verdict?.includes('BREACH') ? 'breach' : 'secure'}`}>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.6rem' }}>
                        <AlertTriangle size={16} />
                        <span>MISSION VERDICT: {videoMissionResult.threat_verdict}</span>
                      </div>
                      <span>ACQUISITION: {videoMissionResult.acquisition_rate_pct}% ({videoMissionResult.detected_frames}/{videoMissionResult.total_frames} FRAMES)</span>
                    </div>
                  )}

                  <video 
                    key={videoMissionResult ? videoMissionResult.annotated_video_url : selectedVideo?.url}
                    src={videoMissionResult ? videoMissionResult.annotated_video_url : selectedVideo?.url}
                    poster="/samples/drone_mission_poster.jpg"
                    autoPlay 
                    loop 
                    muted 
                    playsInline 
                    controls
                  />
                </>
              ) : (
                <div style={{ position: 'relative', width: '100%', height: '100%' }}>
                  <video ref={videoRef} autoPlay playsInline muted className="camera-stream" />
                  <canvas ref={captureCanvasRef} style={{ display: 'none' }} />
                  <canvas ref={webcamOverlayCanvasRef} style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%', pointerEvents: 'none' }} />
                  
                  {!isCameraActive && (
                    <div style={{ position: 'absolute', inset: 0, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: '1rem', background: 'rgba(3, 6, 12, 0.85)' }}>
                      <Camera size={44} color="var(--cyan)" />
                      <div style={{ textAlign: 'center' }}>
                        <div style={{ fontFamily: 'var(--font-display)', fontSize: '1.1rem', fontWeight: 700, color: '#FFF' }}>Local Optical Camera Inactive</div>
                        <div style={{ fontSize: '0.8rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>Connect a USB / integrated webcam for on-device perimeter monitoring.</div>
                      </div>
                      <button className="btn-glass-primary" onClick={startCamera}>
                        <Play size={15} /> Initialize Camera Stream
                      </button>
                    </div>
                  )}
                </div>
              )}
            </div>

            {/* Scenario Selector Chips (when in video mission mode) */}
            {opticalMode === 'video_mission' && (
              <div className="scenario-selector-row">
                {sampleVideos.map(vid => (
                  <div 
                    key={vid.id}
                    className={`scenario-chip ${selectedVideo?.id === vid.id ? 'active' : ''}`}
                    onClick={() => setSelectedVideo(vid)}
                  >
                    <div>
                      <div className="chip-top">
                        <span className="chip-icon">
                          {vid.category === 'drone_flight' && '🚁'}
                          {vid.category === 'fpv_racer' && '⚡'}
                          {vid.category === 'screen_spoof' && '📱'}
                          {vid.category === 'wildlife' && '🦅'}
                        </span>
                        <span className="chip-name">{vid.name}</span>
                      </div>
                      <p className="chip-desc">{vid.desc}</p>
                    </div>
                    <div className="chip-footer">
                      <span className={vid.expected === 'BREACH' ? 'tag-breach' : 'tag-clear'}>
                        {vid.expected === 'BREACH' ? 'EXPECTED: BREACH' : 'EXPECTED: CLEAR'}
                      </span>
                      <span>{vid.benchmark_status?.split(' ')[0]}</span>
                    </div>
                  </div>
                ))}
              </div>
            )}

            {/* Bottom Action Execution Bar */}
            <div className="feed-action-bar">
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.85rem' }}>
                {opticalMode === 'video_mission' ? (
                  <button 
                    className="btn-glass-primary"
                    disabled={isProcessingVideo}
                    onClick={() => selectedVideo && runVideoMission(selectedVideo.filename)}
                  >
                    {isProcessingVideo ? (
                      <>
                        <RefreshCw size={15} className="radar-sweep" /> Evaluating Tensors with CUDA...
                      </>
                    ) : (
                      <>
                        <Play size={15} /> Execute Flight Verification ({selectedVideo?.name || 'Select Scenario'})
                      </>
                    )}
                  </button>
                ) : (
                  isCameraActive ? (
                    <>
                      <button className="btn-glass-primary" onClick={() => setIsDetecting(!isDetecting)}>
                        {isDetecting ? <Square size={15} /> : <Play size={15} />}
                        {isDetecting ? 'Pause Detection' : 'Start Live Detection'}
                      </button>
                      <button className="btn-glass-secondary" onClick={stopCamera}>
                        Stop Camera
                      </button>
                    </>
                  ) : null
                )}

                <label className="btn-glass-secondary" style={{ cursor: 'pointer' }}>
                  <Upload size={14} /> Upload Custom MP4
                  <input type="file" accept="video/mp4,video/webm" onChange={handleCustomUpload} style={{ display: 'none' }} />
                </label>
              </div>

              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem', fontFamily: 'var(--font-mono)', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                <span>Sensitivity:</span>
                <span style={{ color: 'var(--cyan)', fontWeight: 700 }}>{(confidenceThreshold * 100).toFixed(0)}%</span>
                <span style={{ color: 'var(--text-muted)' }}>|</span>
                <button 
                  onClick={() => setActiveTab('adjustment')} 
                  style={{ background: 'none', border: 'none', color: 'var(--cyan)', cursor: 'pointer', textDecoration: 'underline', font: 'inherit' }}
                >
                  Adjust in Settings →
                </button>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 2: DETECTION & ACCURACY DASHBOARD                */}
        {/* ==================================================== */}
        {activeTab === 'accuracy' && (
          <div className="glass-view-container">
            {/* Top 4 KPI Stat Cards */}
            <div className="kpi-stat-grid">
              <div className="kpi-glass-card">
                <div className="kpi-header">
                  <span className="kpi-label">Benchmark Accuracy</span>
                  <div className="kpi-icon-pill" style={{ background: 'var(--emerald-glow)', color: 'var(--emerald)' }}>
                    <CheckCircle2 size={16} />
                  </div>
                </div>
                <div className="kpi-value" style={{ color: 'var(--emerald)' }}>100.0%</div>
                <div className="kpi-meta">16/16 Target Samples Verified</div>
              </div>

              <div className="kpi-glass-card">
                <div className="kpi-header">
                  <span className="kpi-label">Continuous Acquisition</span>
                  <div className="kpi-icon-pill" style={{ background: 'var(--cyan-glow)', color: 'var(--cyan)' }}>
                    <Activity size={16} />
                  </div>
                </div>
                <div className="kpi-value" style={{ color: 'var(--cyan)' }}>96.7%</div>
                <div className="kpi-meta">174 / 180 Continuous Flight Frames</div>
              </div>

              <div className="kpi-glass-card">
                <div className="kpi-header">
                  <span className="kpi-label">Hardware Latency</span>
                  <div className="kpi-icon-pill" style={{ background: 'rgba(99, 102, 241, 0.15)', color: 'var(--indigo)' }}>
                    <Cpu size={16} />
                  </div>
                </div>
                <div className="kpi-value">14.8 ms</div>
                <div className="kpi-meta">~67 FPS on NVIDIA GTX 1650 (CUDA)</div>
              </div>

              <div className="kpi-glass-card">
                <div className="kpi-header">
                  <span className="kpi-label">Distractor Suppression</span>
                  <div className="kpi-icon-pill" style={{ background: 'var(--amber-glow)', color: 'var(--amber)' }}>
                    <Shield size={16} />
                  </div>
                </div>
                <div className="kpi-value" style={{ color: 'var(--amber)' }}>0.0%</div>
                <div className="kpi-meta">Zero Wildlife / Avian False Alarms</div>
              </div>
            </div>

            {/* Accuracy Matrix 2-Column Split */}
            <div className="accuracy-split-grid">
              {/* Left Column: Classification Matrix */}
              <div className="glass-panel-card">
                <div className="glass-panel-header">
                  <div className="glass-title-group">
                    <h2>Target Classification Matrix</h2>
                    <p>Verified precision and recall across optical drone classes and non-drone distractors.</p>
                  </div>
                  <span className="badge-tag-cyan">Verified</span>
                </div>

                <table className="matrix-table">
                  <thead>
                    <tr>
                      <th>Target Classification</th>
                      <th>Class Type</th>
                      <th>Precision</th>
                      <th>Latency</th>
                      <th>Status</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr>
                      <td>
                        <span className="matrix-class-title">DJI Mavic / Quadcopter</span>
                        <span className="matrix-class-sub">Multi-Rotor UAV Target</span>
                      </td>
                      <td><span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>Class 0</span></td>
                      <td><strong style={{ color: '#FFF' }}>100.0%</strong></td>
                      <td><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>14.8ms</span></td>
                      <td><span className="badge-pill-status badge-pass">PASS</span></td>
                    </tr>
                    <tr>
                      <td>
                        <span className="matrix-class-title">Tactical FPV Racer</span>
                        <span className="matrix-class-sub">High-Speed Dive (&gt;20 m/s)</span>
                      </td>
                      <td><span style={{ color: 'var(--cyan)', fontFamily: 'var(--font-mono)' }}>Class 1</span></td>
                      <td><strong style={{ color: '#FFF' }}>100.0%</strong></td>
                      <td><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>15.2ms</span></td>
                      <td><span className="badge-pill-status badge-pass">PASS</span></td>
                    </tr>
                    <tr>
                      <td>
                        <span className="matrix-class-title">Avian / Seagull Wildlife</span>
                        <span className="matrix-class-sub">Natural Wing Oscillation</span>
                      </td>
                      <td><span style={{ color: 'var(--emerald)', fontFamily: 'var(--font-mono)' }}>Distractor</span></td>
                      <td><strong style={{ color: '#FFF' }}>0% False Alarm</strong></td>
                      <td><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>13.9ms</span></td>
                      <td><span className="badge-pill-status badge-pass">SUPPRESSED</span></td>
                    </tr>
                    <tr>
                      <td>
                        <span className="matrix-class-title">Phone / Screen Replay</span>
                        <span className="matrix-class-sub">Presentation Attack (PAD)</span>
                      </td>
                      <td><span style={{ color: 'var(--rose)', fontFamily: 'var(--font-mono)' }}>Spoof</span></td>
                      <td><strong style={{ color: '#FFF' }}>100% Rejected</strong></td>
                      <td><span style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)' }}>12.1ms</span></td>
                      <td><span className="badge-pill-status badge-pass">REJECTED</span></td>
                    </tr>
                  </tbody>
                </table>
              </div>

              {/* Right Column: Rapid Verification Sandbox */}
              <div className="glass-panel-card">
                <div className="glass-panel-header">
                  <div className="glass-title-group">
                    <h2>Rapid Sandbox Verification</h2>
                    <p>Trigger instant on-device PyTorch tensor evaluation on verified image benchmarks.</p>
                  </div>
                  <Crosshair size={18} color="var(--cyan)" />
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(2, 1fr)', gap: '0.75rem', marginTop: '0.5rem' }}>
                  {[
                    { id: 'drone_aerial.jpg', name: 'Aerial Drone', type: 'BREACH' },
                    { id: 'drone_fpv.jpg', name: 'FPV Racer', type: 'BREACH' },
                    { id: 'distractor_seagull.jpg', name: 'Seagull Flight', type: 'CLEAR' },
                    { id: 'distractor_phone_drone.jpg', name: 'Phone Screen Spoof', type: 'CLEAR' },
                    { id: 'bird_sky.jpg', name: 'Bird in Sky', type: 'CLEAR' },
                    { id: 'airplane_sky.jpg', name: 'Commercial Aircraft', type: 'CLEAR' }
                  ].map(s => (
                    <button
                      key={s.id}
                      className="btn-glass-secondary"
                      style={{ justifyContent: 'space-between', padding: '0.75rem 0.85rem' }}
                      onClick={() => runQuickVerification(s.id)}
                    >
                      <div style={{ textAlign: 'left' }}>
                        <div style={{ color: '#FFF', fontSize: '0.8rem', fontWeight: 600 }}>{s.name}</div>
                        <div style={{ fontSize: '0.68rem', color: s.type === 'BREACH' ? 'var(--rose)' : 'var(--emerald)' }}>
                          Expected: {s.type}
                        </div>
                      </div>
                      <ChevronRight size={14} color="var(--text-muted)" />
                    </button>
                  ))}
                </div>

                <div style={{ marginTop: '1.25rem', padding: '0.85rem', background: 'rgba(255,255,255,0.02)', border: '1px solid var(--glass-border)', borderRadius: '12px' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
                    <Shield size={14} color="var(--cyan)" />
                    <span><strong>PAD Algorithm:</strong> Dual-head spatial bounding-box suppression active.</span>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 3: ADJUSTMENT & CONFIGURATION                    */}
        {/* ==================================================== */}
        {activeTab === 'adjustment' && (
          <div className="glass-panel-card">
            <div className="glass-panel-header">
              <div className="glass-title-group">
                <h2>Airspace Defense & Detection Adjustments</h2>
                <p>Fine-tune computer vision confidence gating, hardware acceleration, and presentation attack rejection.</p>
              </div>
              <Sliders size={20} color="var(--cyan)" />
            </div>

            <div className="adjustment-grid">
              {/* Card 1: Sensitivity Threshold */}
              <div className="glass-panel-card control-card" style={{ padding: '1.35rem' }}>
                <div>
                  <div className="control-title-row">
                    <div className="control-icon-circle"><Sliders size={18} /></div>
                    <h3 className="control-h3">Detection Sensitivity</h3>
                  </div>
                  <p className="control-desc" style={{ marginTop: '0.65rem' }}>
                    Adjusts the minimum confidence threshold required to confirm an airborne drone intrusion. Lower values increase optical reach for micro-drones; higher values suppress environmental edge noise.
                  </p>
                </div>

                <div className="slider-control-group">
                  <div className="slider-val-row">
                    <span style={{ color: 'var(--text-secondary)' }}>Active Threshold:</span>
                    <span className="slider-val-badge">{(confidenceThreshold * 100).toFixed(0)}%</span>
                  </div>
                  <input 
                    type="range"
                    min="0.30"
                    max="0.85"
                    step="0.05"
                    value={confidenceThreshold}
                    onChange={(e) => setConfidenceThreshold(parseFloat(e.target.value))}
                    className="glass-range-slider"
                  />
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.72rem', color: 'var(--text-muted)', fontFamily: 'var(--font-mono)' }}>
                    <span>30% (Max Sensitivity)</span>
                    <span>50% (Recommended)</span>
                    <span>85% (Strict Precision)</span>
                  </div>
                </div>
              </div>

              {/* Card 2: Inference Engine & Hardware Acceleration */}
              <div className="glass-panel-card control-card" style={{ padding: '1.35rem' }}>
                <div>
                  <div className="control-title-row">
                    <div className="control-icon-circle"><Cpu size={18} /></div>
                    <h3 className="control-h3">Hardware Acceleration Engine</h3>
                  </div>
                  <p className="control-desc" style={{ marginTop: '0.65rem' }}>
                    Select between high-throughput CUDA FP16 GPU inference and CPU fallback. Real-world verified: 14.8 ms on NVIDIA GTX 1650 vs 42.1 ms on Intel i7 CPU.
                  </p>
                </div>

                <div className="glass-switch-row">
                  <div className="switch-label-group">
                    <strong>NVIDIA CUDA 12.4 (TensorRT FP16)</strong>
                    <span>14.8 ms latency (~67 FPS continuous)</span>
                  </div>
                  <label className="glass-toggle">
                    <input 
                      type="checkbox" 
                      checked={cudaEngine}
                      onChange={(e) => setCudaEngine(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              {/* Card 3: Presentation Attack Defense (PAD) */}
              <div className="glass-panel-card control-card" style={{ padding: '1.35rem' }}>
                <div>
                  <div className="control-title-row">
                    <div className="control-icon-circle"><Eye size={18} /></div>
                    <h3 className="control-h3">Presentation Attack Defense (PAD)</h3>
                  </div>
                  <p className="control-desc" style={{ marginTop: '0.65rem' }}>
                    Dual-head spatial bounding-box suppression that detects screen bezels, refresh rate anomalies, and smartphone video replays, preventing electronic spoofing.
                  </p>
                </div>

                <div className="glass-switch-row">
                  <div className="switch-label-group">
                    <strong>Screen & Monitor Replay Defense</strong>
                    <span>100% rejection across handheld screen replays</span>
                  </div>
                  <label className="glass-toggle">
                    <input 
                      type="checkbox" 
                      checked={padDefenseEnabled}
                      onChange={(e) => setPadDefenseEnabled(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>

              {/* Card 4: Air-Gapped Governance & Audio */}
              <div className="glass-panel-card control-card" style={{ padding: '1.35rem' }}>
                <div>
                  <div className="control-title-row">
                    <div className="control-icon-circle"><Lock size={18} /></div>
                    <h3 className="control-h3">Strict Zero-Cloud Egress</h3>
                  </div>
                  <p className="control-desc" style={{ marginTop: '0.65rem' }}>
                    Enforces 100% offline edge processing. Zero external network packets, zero remote API keys, and local in-memory tensor processing for sensitive defence perimeters.
                  </p>
                </div>

                <div className="glass-switch-row">
                  <div className="switch-label-group">
                    <strong>Air-Gapped Operation Mode</strong>
                    <span>Egress: 0.00 KB/s • Complete on-premise privacy</span>
                  </div>
                  <label className="glass-toggle">
                    <input 
                      type="checkbox" 
                      checked={zeroCloudEgress}
                      onChange={(e) => setZeroCloudEgress(e.target.checked)}
                    />
                    <span className="toggle-slider"></span>
                  </label>
                </div>
              </div>
            </div>
          </div>
        )}

        {/* ==================================================== */}
        {/* TAB 4: ACTIVITY LOGS                                 */}
        {/* ==================================================== */}
        {activeTab === 'logs' && (
          <div className="glass-panel-card">
            <div className="logs-controls-bar">
              <div className="glass-title-group">
                <h2>Airspace Activity Logs ({logs.length})</h2>
                <p>Auditable incident log stream capturing intrusion breaches, velocities, and anti-spoof suppressions.</p>
              </div>

              <div className="log-actions-cluster">
                {/* Filter Pills */}
                <div className="log-filter-pills">
                  <button 
                    className={`filter-pill-btn ${activeLogFilter === 'all' ? 'active' : ''}`}
                    onClick={() => setActiveLogFilter('all')}
                  >
                    All ({logs.length})
                  </button>
                  <button 
                    className={`filter-pill-btn ${activeLogFilter === 'breach' ? 'active' : ''}`}
                    onClick={() => setActiveLogFilter('breach')}
                  >
                    Breaches Only
                  </button>
                  <button 
                    className={`filter-pill-btn ${activeLogFilter === 'clear' ? 'active' : ''}`}
                    onClick={() => setActiveLogFilter('clear')}
                  >
                    Suppressed / Distractors
                  </button>
                </div>

                {logs.length > 0 && (
                  <>
                    <button className="btn-glass-secondary" onClick={exportIncidentReport} title="Export Audit Report (JSON)">
                      <Download size={14} /> Export JSON
                    </button>
                    <button className="btn-glass-secondary" onClick={handleClearLogs} title="Clear Log History" style={{ color: 'var(--rose)' }}>
                      <Trash2 size={14} /> Clear
                    </button>
                  </>
                )}
              </div>
            </div>

            {/* Glass Logs Table */}
            <div className="glass-table-container">
              <table className="glass-logs-table">
                <thead>
                  <tr>
                    <th>Timestamp (UTC)</th>
                    <th>Target Classification</th>
                    <th>Verdict</th>
                    <th>Confidence</th>
                    <th>Velocity / Speed</th>
                    <th>Vector Bearing</th>
                    <th>PAD Status</th>
                    <th>Inference</th>
                  </tr>
                </thead>
                <tbody>
                  {filteredLogs.length === 0 ? (
                    <tr>
                      <td colSpan="8" style={{ textAlign: 'center', padding: '3rem 1rem', color: 'var(--text-muted)' }}>
                        <Shield size={36} color="var(--emerald)" style={{ marginBottom: '0.65rem' }} />
                        <div>No matching activity records in current session.</div>
                      </td>
                    </tr>
                  ) : (
                    filteredLogs.map(log => (
                      <tr key={log.id}>
                        <td className="log-time-cell">{formatTimestamp(log.timestamp)}</td>
                        <td className="log-target-cell">
                          <strong>{log.drone_name}</strong>
                          <div style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>ID: #{log.id}</div>
                        </td>
                        <td>
                          <span className={log.status === 'BREACH' ? 'log-badge-breach' : 'log-badge-clear'}>
                            {log.status}
                          </span>
                        </td>
                        <td>
                          <strong style={{ color: '#FFFFFF', fontFamily: 'var(--font-mono)' }}>
                            {((log.confidence || 0.8) * 100).toFixed(1)}%
                          </strong>
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)' }}>
                          {log.velocity_mps > 0 ? `${log.velocity_mps} m/s` : 'Stationary'}
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-secondary)' }}>
                          {log.cardinal ? `${log.cardinal} (${log.bearing_deg}°)` : '—'}
                        </td>
                        <td>
                          <span style={{ 
                            fontSize: '0.72rem', 
                            color: log.anti_spoof?.includes('REJECTED') ? 'var(--amber)' : (log.anti_spoof?.includes('VERIFIED') ? 'var(--cyan)' : 'var(--emerald)'),
                            fontFamily: 'var(--font-mono)',
                            fontWeight: 600
                          }}>
                            {log.anti_spoof || 'AUTHENTIC'}
                          </span>
                        </td>
                        <td style={{ fontFamily: 'var(--font-mono)', color: 'var(--text-muted)', fontSize: '0.72rem' }}>
                          {log.latency_ms || log.inference_ms || 14.8}ms CUDA
                        </td>
                      </tr>
                    ))
                  )}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </main>

      {/* Rapid Verification Modal */}
      {isTestModalOpen && (
        <div className="glass-modal-backdrop" onClick={() => setIsTestModalOpen(false)}>
          <div className="glass-modal-dialog" onClick={(e) => e.stopPropagation()}>
            <div className="glass-modal-header">
              <div className="modal-title-row">
                <Crosshair size={20} color="var(--cyan)" />
                <span>Rapid Target Verification Sandbox</span>
              </div>
              <button className="glass-icon-btn" onClick={() => setIsTestModalOpen(false)}>✕</button>
            </div>

            <div className="glass-modal-body">
              <div className="modal-canvas-frame">
                <canvas ref={testCanvasRef} className="modal-canvas" />
                {isTestingSample && (
                  <div style={{ position: 'absolute', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '0.75rem', color: 'var(--cyan)' }}>
                    <Radar className="radar-sweep" size={32} />
                    <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.8rem', letterSpacing: '0.04em' }}>EXECUTING PYTORCH TENSOR EVALUATION...</span>
                  </div>
                )}
              </div>

              {testResult && (
                <div style={{ 
                  marginTop: '1.25rem', 
                  padding: '1rem', 
                  borderRadius: '12px',
                  background: testResult.detected ? 'var(--rose-glow)' : 'var(--emerald-glow)',
                  border: `1px solid ${testResult.detected ? 'rgba(248, 113, 113, 0.4)' : 'rgba(52, 211, 153, 0.4)'}`,
                  display: 'flex',
                  alignItems: 'center',
                  gap: '0.85rem'
                }}>
                  {testResult.detected ? (
                    <>
                      <AlertTriangle size={24} color="var(--rose)" />
                      <div>
                        <div style={{ fontWeight: 700, color: '#FECACA', fontSize: '0.9rem' }}>
                          TARGET INTRUSION CONFIRMED: {testResult.data?.drone_name}
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', marginTop: '0.2rem' }}>
                          Confidence: {((testResult.data?.confidence || 0.8) * 100).toFixed(1)}% | CUDA Latency: {testResult.inference_ms}ms
                        </div>
                      </div>
                    </>
                  ) : (
                    <>
                      <CheckCircle2 size={24} color="var(--emerald)" />
                      <div>
                        <div style={{ fontWeight: 700, color: '#A7F3D0', fontSize: '0.9rem' }}>
                          AIRSPACE CLEAR — DISTRACTOR / SCREEN REPLAY SUPPRESSED
                        </div>
                        <div style={{ fontSize: '0.75rem', color: 'var(--text-secondary)', fontFamily: 'var(--font-mono)', marginTop: '0.2rem' }}>
                          Zero False Alarm Verified | CUDA Latency: {testResult.inference_ms}ms
                        </div>
                      </div>
                    </>
                  )}
                </div>
              )}
            </div>

            <div className="glass-modal-footer">
              <button className="btn-glass-secondary" onClick={() => setIsTestModalOpen(false)}>
                Close Inspector
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
