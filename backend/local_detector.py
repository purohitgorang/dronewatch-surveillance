import io
import os
import time
from typing import Dict, Any, List, Optional, Tuple
import numpy as np
from PIL import Image
import torch
from ultralytics import YOLO

class LocalDroneDetector:
    """
    Drone Hawk X - Robust Enterprise Edge AI Drone Detector.
    Runs locally on NVIDIA GeForce GTX 1650 (PyTorch CUDA 12.4) or optimized CPU fallback.
    Zero cloud API keys, zero internet egress required.
    
    Robustness Architecture:
    1. Dual-Head Ensemble: YOLOv8 Drone Neural Engine + COCO Spatial Co-Validator.
    2. Phone / Screen Presentation Attack Detection (PAD): Detects and rejects drones displayed on smartphones, tablets, or monitors.
    3. Precision Spatial Overlap Suppressor: Overlap-based suppression of birds, aircraft, and indoor clutter.
    4. Geometric Feasibility Filter: Rejects linear artifacts (curtains, shelves, door frames).
    5. Temporal Track Confirmation: Multi-frame confirmation to eliminate 1-frame webcam transients.
    """

    FRIENDLY_NAMES = {
        0: "Quadcopter (Multi-Rotor UAV)",
        1: "Fixed-Wing Surveillance Drone",
        "quadcopter": "Quadcopter (Multi-Rotor UAV)",
        "fixed-wing": "Fixed-Wing Surveillance Drone",
        "drone": "Airborne UAV Target"
    }

    SCREEN_SPOOF_CLASSES = {'cell phone', 'tv', 'laptop', 'monitor'}
    WILDLIFE_CLASSES = {'bird', 'kite', 'frisbee', 'balloon'}
    INDOOR_DISTRACTORS = {
        'chair', 'couch', 'bed', 'dining table', 'toilet', 'microwave',
        'oven', 'refrigerator', 'potted plant', 'clock', 'vase', 'book',
        'bottle', 'cup', 'bowl'
    }

    def __init__(self, model_path: str = None):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        if model_path is None:
            model_path = os.path.join(base_dir, "models", "drone_yolov8n.pt")
        
        self.model_path = model_path
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device_name = (
            torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        )
        
        print(f"[Drone Hawk X] Loading primary YOLOv8 drone model from: {self.model_path}")
        print(f"[Drone Hawk X] Accelerated device: {self.device} ({self.device_name})")
        
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file not found at: {self.model_path}")
            
        self.drone_model = YOLO(self.model_path)
        
        # Secondary validator for spatial distractor & phone screen suppression
        coco_path = "yolov8n.pt"
        try:
            self.coco_validator = YOLO(coco_path)
            print("[Drone Hawk X] COCO Spatial & Anti-Spoof Validator initialized successfully.")
        except Exception as e:
            print(f"[Drone Hawk X] Warning: COCO validator not loaded ({e}).")
            self.coco_validator = None
            
        # Temporal tracking buffer for live video stream stability
        self.temporal_history = []
        
        # Warmup models on CUDA
        dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        self.drone_model(dummy, device=self.device, verbose=False)
        if self.coco_validator:
            self.coco_validator(dummy, device=self.device, verbose=False)
        print("[Drone Hawk X] Neural inference pipeline warmed up and operational.")

    @staticmethod
    def calculate_overlap_ratio(inner_box: List[float], outer_box: List[float]) -> float:
        """Calculate fraction of inner_box area covered by outer_box."""
        xA = max(inner_box[0], outer_box[0])
        yA = max(inner_box[1], outer_box[1])
        xB = min(inner_box[2], outer_box[2])
        yB = min(inner_box[3], outer_box[3])
        
        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        inner_area = (inner_box[2] - inner_box[0]) * (inner_box[3] - inner_box[1])
        return inter_area / inner_area if inner_area > 0 else 0.0

    @staticmethod
    def calculate_iou(box1: List[float], box2: List[float]) -> float:
        """Calculate standard Intersection over Union (IoU)."""
        xA = max(box1[0], box2[0])
        yA = max(box1[1], box2[1])
        xB = min(box1[2], box2[2])
        yB = min(box1[3], box2[3])
        
        inter_area = max(0.0, xB - xA) * max(0.0, yB - yA)
        b1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        b2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union_area = b1_area + b2_area - inter_area
        return inter_area / union_area if union_area > 0 else 0.0

    def detect(self, image_bytes: bytes, conf_threshold: float = 0.50, is_video_stream: bool = False) -> Dict[str, Any]:
        """
        Robustly evaluate an image or video frame for airborne drone threats.
        Includes dual-stage spatial distractor suppression and anti-spoof display filtering.
        """
        start_time = time.time()
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            img_w, img_h = image.size
            img_np = np.array(image)

            # Stage 1: Specialized UAV Tensor Extraction
            internal_conf = max(0.35, conf_threshold - 0.10)
            drone_results = self.drone_model(
                img_np,
                device=self.device,
                conf=internal_conf,
                verbose=False
            )

            candidate_boxes = []
            for r in drone_results:
                if r.boxes is not None and len(r.boxes) > 0:
                    for b in r.boxes:
                        cls_id = int(b.cls[0].item())
                        raw_name = self.drone_model.names.get(cls_id, "drone")
                        conf = float(b.conf[0].item())
                        xyxy = [float(v) for v in b.xyxy[0].tolist()]
                        
                        # Geometric sanity check
                        bw = xyxy[2] - xyxy[0]
                        bh = xyxy[3] - xyxy[1]
                        aspect_ratio = bw / bh if bh > 0 else 0
                        
                        # Reject extreme linear artifacts (curtains, door frames)
                        if aspect_ratio < 0.28 or aspect_ratio > 3.8:
                            continue
                            
                        # Class-specific confidence calibration:
                        # Fixed-wing is noisy in indoor/cluttered rooms
                        if raw_name == 'fixed-wing' and conf < 0.65:
                            continue
                            
                        # Quadcopter calibration:
                        if raw_name != 'fixed-wing' and conf < conf_threshold:
                            continue

                        candidate_boxes.append({
                            "class_id": cls_id,
                            "raw_name": raw_name,
                            "confidence": conf,
                            "box": xyxy,
                            "bw": bw,
                            "bh": bh,
                        })

            # Stage 2: Spatial Distractor & Anti-Spoof Suppression
            suppressed_count = 0
            anti_spoof_rejections = []
            verified_detections = []

            if candidate_boxes and self.coco_validator is not None:
                try:
                    coco_results = self.coco_validator(
                        img_np,
                        device=self.device,
                        conf=0.25,
                        verbose=False
                    )
                    coco_boxes = []
                    for cr in coco_results:
                        if cr.boxes is not None:
                            for cb in cr.boxes:
                                ccls = self.coco_validator.names.get(int(cb.cls[0].item()), "")
                                cconf = float(cb.conf[0].item())
                                cbox = [float(v) for v in cb.xyxy[0].tolist()]
                                coco_boxes.append((ccls, cconf, cbox))

                    for cand in candidate_boxes:
                        cand_box = cand["box"]
                        cand_conf = cand["confidence"]
                        d_area = (cand_box[2] - cand_box[0]) * (cand_box[3] - cand_box[1])
                        is_suppressed = False

                        # Check COCO spatial overlap
                        for ccls, cconf, cbox in coco_boxes:
                            coverage = self.calculate_overlap_ratio(cand_box, cbox)
                            c_area = (cbox[2] - cbox[0]) * (cbox[3] - cbox[1])

                            # Anti-Spoofing: Reject drones displayed on cell phones, TVs, laptops, monitors
                            if ccls in self.SCREEN_SPOOF_CLASSES and coverage > 0.40 and c_area >= d_area * 0.65:
                                is_suppressed = True
                                suppressed_count += 1
                                anti_spoof_rejections.append(f"Drone on {ccls} Screen ({coverage:.1%})")
                                break

                            # Wildlife Filter: If COCO sees a bird at the same box with high confidence
                            if ccls in self.WILDLIFE_CLASSES and coverage > 0.25 and cconf >= (cand_conf - 0.12):
                                is_suppressed = True
                                suppressed_count += 1
                                break

                            # Indoor Furniture & Object Filter:
                            if ccls in self.INDOOR_DISTRACTORS and coverage > 0.35:
                                is_suppressed = True
                                suppressed_count += 1
                                break

                        if not is_suppressed:
                            x1, y1, x2, y2 = cand_box
                            norm_box = [
                                max(0.0, min(1.0, y1 / img_h)),
                                max(0.0, min(1.0, x1 / img_w)),
                                max(0.0, min(1.0, y2 / img_h)),
                                max(0.0, min(1.0, x2 / img_w)),
                            ]
                            friendly_name = self.FRIENDLY_NAMES.get(cand["class_id"], cand["raw_name"].title())
                            verified_detections.append({
                                "class_id": cand["class_id"],
                                "raw_class": cand["raw_name"],
                                "drone_name": friendly_name,
                                "confidence": round(cand["confidence"], 4),
                                "box": [round(c, 1) for c in cand_box],
                                "box_norm": [round(c, 4) for c in norm_box],
                            })
                except Exception as ex:
                    print(f"[Drone Hawk X] Validator exception: {ex}")
                    for cand in candidate_boxes:
                        x1, y1, x2, y2 = cand["box"]
                        norm_box = [y1 / img_h, x1 / img_w, y2 / img_h, x2 / img_w]
                        verified_detections.append({
                            "class_id": cand["class_id"],
                            "raw_class": cand["raw_name"],
                            "drone_name": self.FRIENDLY_NAMES.get(cand["class_id"], "UAV Target"),
                            "confidence": round(cand["confidence"], 4),
                            "box": [round(c, 1) for c in cand["box"]],
                            "box_norm": [round(c, 4) for c in norm_box],
                        })
            else:
                for cand in candidate_boxes:
                    x1, y1, x2, y2 = cand["box"]
                    norm_box = [y1 / img_h, x1 / img_w, y2 / img_h, x2 / img_w]
                    verified_detections.append({
                        "class_id": cand["class_id"],
                        "raw_class": cand["raw_name"],
                        "drone_name": self.FRIENDLY_NAMES.get(cand["class_id"], "UAV Target"),
                        "confidence": round(cand["confidence"], 4),
                        "box": [round(c, 1) for c in cand["box"]],
                        "box_norm": [round(c, 4) for c in norm_box],
                    })

            # Stage 3: Temporal Stability Tracking (Anti-Flicker for Webcam Streams)
            is_confirmed = False
            if len(verified_detections) > 0:
                best_det = max(verified_detections, key=lambda d: d["confidence"])
                if is_video_stream:
                    if best_det["confidence"] >= 0.68:
                        is_confirmed = True
                        self.temporal_history.append(1)
                    else:
                        self.temporal_history.append(1)
                        if sum(self.temporal_history[-2:]) >= 2:
                            is_confirmed = True
                    if len(self.temporal_history) > 10:
                        self.temporal_history = self.temporal_history[-5:]
                else:
                    is_confirmed = True
            else:
                self.temporal_history.append(0)
                if len(self.temporal_history) > 10:
                    self.temporal_history = self.temporal_history[-5:]

            elapsed_ms = round((time.time() - start_time) * 1000, 1)

            if is_confirmed and len(verified_detections) > 0:
                best_detection = max(verified_detections, key=lambda d: d["confidence"])
                return {
                    "is_drone": True,
                    "drone_name": best_detection["drone_name"],
                    "confidence": best_detection["confidence"],
                    "box": best_detection["box"],
                    "box_norm": best_detection["box_norm"],
                    "detections": verified_detections,
                    "count": len(verified_detections),
                    "suppressed_distractors": suppressed_count,
                    "anti_spoof_rejections": anti_spoof_rejections,
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
                    "suppressed_distractors": suppressed_count,
                    "anti_spoof_rejections": anti_spoof_rejections,
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
