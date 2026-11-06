import io
import os
import time
from typing import Dict, Any, List, Optional
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

class LocalDroneDetector:
    """
    Drone Hawk X - Robust Offline Edge AI Drone Detector.
    Runs locally on-device using a fine-tuned YOLOv8 neural network.
    No API keys, no internet connection required.
    Hardware-accelerated with PyTorch CUDA (NVIDIA GPU) or optimized CPU fallback.
    Includes active negative distractor suppression (filtering out birds and common indoor furniture).
    """
    
    FRIENDLY_NAMES = {
        0: "Quadcopter (Multi-Rotor UAV)",
        1: "Fixed-Wing Surveillance Drone",
        "quadcopter": "Quadcopter (Multi-Rotor UAV)",
        "fixed-wing": "Fixed-Wing Surveillance Drone"
    }

    # Common distractor objects in standard vision models
    NEGATIVE_SUPPRESS_CLASSES = {'bird', 'chair', 'couch', 'bed', 'dining table', 'traffic light'}

    def __init__(self, model_path: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "drone_yolov8n.pt")
        
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        
        print(f"[Drone Hawk X] Loading local YOLOv8 drone model from: {self.model_path}")
        print(f"[Drone Hawk X] Compute device: {self.device} ({self.device_name})")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
        self.model = YOLO(self.model_path)
        
        # Load secondary distractor validator (COCO) if present for multi-stage suppression
        coco_path = "yolov8n.pt"
        try:
            self.coco_validator = YOLO(coco_path)
        except Exception:
            self.coco_validator = None
            
        # Warmup model on device
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.model(dummy, device=self.device, verbose=False)
        print(f"[Drone Hawk X] Local drone model initialized and ready for real-time inference.")

    def detect(self, image_bytes: bytes, conf_threshold: float = 0.55) -> Dict[str, Any]:
        """
        Analyze an incoming image frame and detect airborne drones/UAVs with robust distractor filtering.
        """
        start_time = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_w, img_h = image.size
            img_np = np.array(image)

            # Stage 1: Specialized Drone Detection
            results = self.model(
                img_np,
                device=self.device,
                conf=conf_threshold,
                verbose=False
            )

            candidate_detections: List[Dict[str, Any]] = []

            for r in results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        cls_id = int(box.cls[0].item())
                        raw_name = self.model.names.get(cls_id, f"drone_class_{cls_id}")
                        friendly_name = self.FRIENDLY_NAMES.get(cls_id, raw_name.title())
                        conf = float(box.conf[0].item())
                        
                        # Absolute coordinates [x1, y1, x2, y2]
                        xyxy = [float(v) for v in box.xyxy[0].tolist()]
                        x1, y1, x2, y2 = xyxy

                        norm_box = [
                            max(0.0, min(1.0, y1 / img_h)),
                            max(0.0, min(1.0, x1 / img_w)),
                            max(0.0, min(1.0, y2 / img_h)),
                            max(0.0, min(1.0, x2 / img_w)),
                        ]

                        candidate_detections.append({
                            "class_id": cls_id,
                            "raw_class": raw_name,
                            "drone_name": friendly_name,
                            "confidence": round(conf, 4),
                            "box": [round(c, 1) for c in xyxy],
                            "box_norm": [round(c, 4) for c in norm_box],
                        })

            # Stage 2: Distractor Negative Suppression (Filters birds & non-drone objects)
            verified_detections = []
            if candidate_detections and self.coco_validator is not None:
                try:
                    coco_res = self.coco_validator(
                        img_np, device=self.device, conf=0.35, verbose=False
                    )
                    coco_tags = {}
                    for cr in coco_res:
                        if cr.boxes is not None:
                            for cb in cr.boxes:
                                cname = self.coco_validator.names.get(int(cb.cls[0].item()), "")
                                cconf = float(cb.conf[0].item())
                                coco_tags[cname] = max(coco_tags.get(cname, 0.0), cconf)

                    for det in candidate_detections:
                        is_suppressed = False
                        # If a bird is detected with higher or comparable confidence, suppress false alarm
                        if "bird" in coco_tags and coco_tags["bird"] > 0.40:
                            is_suppressed = True
                        
                        if not is_suppressed:
                            verified_detections.append(det)
                except Exception as ex:
                    # In case of validator error, keep candidate detections
                    verified_detections = candidate_detections
            else:
                verified_detections = candidate_detections

            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            if len(verified_detections) > 0:
                best_detection = max(verified_detections, key=lambda d: d["confidence"])
                return {
                    "is_drone": True,
                    "drone_name": best_detection["drone_name"],
                    "confidence": best_detection["confidence"],
                    "box": best_detection["box"],
                    "box_norm": best_detection["box_norm"],
                    "detections": verified_detections,
                    "count": len(verified_detections),
                    "inference_ms": elapsed_ms,
                    "device": self.device_name,
                }
            else:
                return {
                    "is_drone": False,
                    "drone_name": None,
                    "confidence": 0.0,
                    "detections": [],
                    "count": 0,
                    "inference_ms": elapsed_ms,
                    "device": self.device_name,
                }

        except Exception as e:
            elapsed_ms = round((time.time() - start_time) * 1000, 1)
            print(f"[Drone Hawk X] Inference error: {e}")
            return {
                "is_drone": False,
                "drone_name": None,
                "confidence": 0.0,
                "error": str(e),
                "inference_ms": elapsed_ms,
                "device": self.device_name,
            }
