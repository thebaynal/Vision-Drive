"""Laptop object detection plus class-independent metric-depth obstacle cues."""

from dataclasses import replace
import math
import os
from pathlib import Path
import numpy as np
from .decision import Detection


class LaptopDetector:
    def __init__(self, weights, config, depth_model=None, device="cpu", cpu_threads=2,
                 depth_size=392, depth_scale=1.0):
        weights = Path(weights)
        if not weights.is_file():
            raise ValueError(f"Checkpoint not found: {weights}. Run python scripts/setup_laptop_models.py")
        if cpu_threads < 1 or depth_size < 224 or depth_size > 700 or depth_size % 14 or not math.isfinite(depth_scale) or depth_scale <= 0:
            raise ValueError("Use positive CPU threads/scale and a depth size divisible by 14, from 224 to 700")
        cache = Path(__file__).resolve().parents[1] / ".cache"
        os.environ.setdefault("YOLO_CONFIG_DIR", str(cache / "ultralytics"))
        os.environ.setdefault("YOLO_AUTOINSTALL", "false")
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("USE_FLAX", "0")
        import torch
        from ultralytics import YOLO
        if device == "cpu":
            torch.set_num_threads(cpu_threads)
        self.model = YOLO(str(weights), task="detect")
        self.config, self.device = config, device
        self.names = self.model.names
        self.depth = None
        self.floor_found = False
        if depth_model is not None:
            from .depth import MetricDepth
            torch_device = "cuda:" + device if device.isdecimal() else device
            self.depth = MetricDepth(depth_model, torch_device, depth_size, depth_scale)
        self.label = "objects + depth" if self.depth else "COCO objects"
        self.name = "Laptop YOLO11n 640" + (" + metric depth" if self.depth else " / image proximity")

    def warmup(self):
        self.predict(np.zeros((360, 640, 3), dtype=np.uint8))

    def predict(self, image):
        result = self.model.predict(source=image, imgsz=640, rect=False,
            conf=self.config.confidence_threshold, iou=self.config.nms_iou_threshold,
            device=self.device, verbose=False, save=False)[0]
        depth = self.depth.predict(image) if self.depth else None
        detections = []
        if result.boxes is not None:
            for box, confidence, category in zip(result.boxes.xyxyn.cpu().tolist(),
                    result.boxes.conf.cpu().tolist(), result.boxes.cls.cpu().tolist()):
                category = int(category)
                det = Detection(*(max(0., min(1., float(v))) for v in box),
                                float(confidence), category, self.names[category])
                if depth is not None and det.valid:
                    h, w = depth.shape
                    # Interior crop reduces background pixels around the object.
                    x1, x2 = round((det.x1*.8+det.x2*.2)*w), round((det.x1*.2+det.x2*.8)*w)
                    y1, y2 = round((det.y1*.8+det.y2*.2)*h), round((det.y1*.2+det.y2*.8)*h)
                    patch = depth[y1:max(y1+1,y2), x1:max(x1+1,x2)]
                    valid = patch[np.isfinite(patch) & (patch > .02)]
                    if valid.size:
                        det = replace(det, distance_m=float(np.quantile(valid, .2)))
                detections.append(det)
        if depth is not None:
            import cv2
            from .depth import surface_detections
            small = cv2.resize(depth, (160, 120), interpolation=cv2.INTER_AREA)
            surfaces, self.floor_found = surface_detections(small, self.config)
            detections.extend(surfaces)
        return tuple(detections)
