"""Metric depth -> class-independent nearby surfaces, with flat-floor rejection.

The plane fit is a heuristic for a fixed camera on a flat indoor course. Depth
values are learned estimates, not measurements. Invalid maps fail inference.
"""

from dataclasses import dataclass
import numpy as np
from .decision import Detection


def ground_plane(depth):
    """Robustly fit inverse-depth = ax + by + c to the visible lower floor.

Perspective planes are affine in inverse depth. A positive vertical gradient,
bottom-band support, and vertical extent reject front-facing walls as floors.
Returns None if the view cannot support the assumptions.
"""
    h, w = depth.shape
    y, x = np.mgrid[:h, :w].astype(np.float64)
    x, y = x / w, y / h
    good = np.isfinite(depth) & (depth > .02) & (y >= .60)
    indices = np.flatnonzero(good)
    if len(indices) < 60:
        return None
    rng = np.random.default_rng(42)
    indices = rng.choice(indices, min(len(indices), 600), replace=False)
    matrix = np.stack((x.ravel()[indices], y.ravel()[indices], np.ones(len(indices))), axis=1)
    target = 1. / depth.ravel()[indices]
    best, best_score = None, 0
    for _ in range(100):
        sample = rng.choice(len(indices), 3, replace=False)
        if abs(np.linalg.det(matrix[sample])) < .001:
            continue
        plane = np.linalg.solve(matrix[sample], target[sample])
        if plane[1] < .15 or abs(plane[0]) > plane[1] * .8:
            continue
        predicted = matrix @ plane
        inliers = abs(target-predicted) <= np.maximum(.035, abs(predicted) * .08)
        bottom = matrix[:, 1] >= .82
        if not bottom.any() or inliers[bottom].mean() < .30:
            continue
        if inliers.sum() < 40 or np.ptp(matrix[inliers, 1]) < .20:
            continue
        score = inliers.sum()
        if score > best_score:
            best, best_score = inliers, score
    if best is None or best_score < len(indices) * .30:
        return None
    plane = np.linalg.lstsq(matrix[best], target[best], rcond=None)[0]
    if plane[1] < .15 or abs(plane[0]) > plane[1] * .8:
        return None
    return plane[0]*x + plane[1]*y + plane[2]


def surface_detections(depth, config):
    """Detect nearby non-floor components, without a list of object names."""
    import cv2
    depth = np.asarray(depth, dtype=np.float32)
    if depth.ndim != 2 or min(depth.shape) < 16:
        raise ValueError("Depth map must be a 2D image at least 16x16")
    valid = np.isfinite(depth) & (depth > .02)
    if valid.mean() < .98:
        raise ValueError("Depth map unavailable or invalid; movement stopped")
    h, w = depth.shape
    plane = ground_plane(depth)
    mask = valid & (depth <= config.avoid_distance_m)
    mask[:round(config.decision_top*h)] = False
    if plane is not None:
        inverse = np.divide(1., depth, out=np.zeros_like(depth), where=valid)
        # Above-ground objects are closer than the ground at the same pixel.
        mask &= inverse > plane + np.maximum(.06, abs(plane)*.15)
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    count, components, stats, _ = cv2.connectedComponentsWithStats(mask)
    detections = []
    # Split components by region so a rectangle enclosing a curved/wide shape
    # does not claim the empty space between its disconnected parts.
    edges = (0, round(config.left_boundary*w), round(config.right_boundary*w), w)
    for component in range(1, count):
        if stats[component, cv2.CC_STAT_AREA] < max(10, h*w*.002):
            continue
        for left, right in zip(edges, edges[1:]):
            ys, xs = np.where(components[:, left:right] == component)
            if len(xs) < max(6, h*w*.001):
                continue
            xs += left
            # Quantile rather than the single noisiest/closest depth pixel.
            distance = float(np.quantile(depth[ys, xs], .15))
            detections.append(Detection(float(xs.min()/w), float(ys.min()/h),
                float((xs.max()+1)/w), float((ys.max()+1)/h), 1., 0,
                "depth obstacle" if plane is not None else "near surface", distance))
    return tuple(detections), plane is not None


class MetricDepth:
    def __init__(self, directory, device="cpu", size=392, scale=1.0):
        from pathlib import Path
        directory = Path(directory)
        if not all((directory/name).is_file() for name in ("model.safetensors", "config.json", "preprocessor_config.json")):
            raise ValueError("Laptop depth model not found. Run python scripts/setup_laptop_models.py")
        from transformers import AutoImageProcessor, AutoModelForDepthEstimation
        import torch
        self.torch, self.device, self.scale = torch, device, scale
        self.processor = AutoImageProcessor.from_pretrained(str(directory), local_files_only=True, use_fast=False)
        self.processor.size = {"height": size, "width": size}
        self.model = AutoModelForDepthEstimation.from_pretrained(str(directory), local_files_only=True,
                                                               use_safetensors=True).to(device).eval()
        if self.model.config.depth_estimation_type != "metric":
            raise ValueError("Metric indoor checkpoint required; relative depth cannot give meter thresholds")

    def predict(self, bgr):
        # Never min-max normalize the model output: it would erase near/far.
        inputs = self.processor(images=bgr[..., ::-1], return_tensors="pt").to(self.device)
        with self.torch.inference_mode():
            output = self.model(**inputs).predicted_depth
            depth = self.torch.nn.functional.interpolate(output.unsqueeze(1), size=bgr.shape[:2],
                         mode="bicubic", align_corners=False)[0, 0]
        return depth.cpu().numpy() * self.scale
