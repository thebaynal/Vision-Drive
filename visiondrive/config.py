"""Strict configuration without importing the vision or ML runtime."""

from dataclasses import asdict, dataclass, fields
import json
import math
from pathlib import Path


@dataclass(frozen=True)
class Config:
    confidence_threshold: float = 0.50
    nms_iou_threshold: float = 0.45
    left_boundary: float = 1 / 3
    right_boundary: float = 2 / 3
    decision_top: float = 0.40
    approach_top: float = 0.60
    near_top: float = 0.92
    stop_min_area: float = 0.04
    avoid_distance_m: float = 1.5
    stop_distance_m: float = 0.45
    overlap_threshold: float = 0.10
    stable_frames: int = 3
    stale_after_ms: int = 1000

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
                raise ValueError(f"{item.name} must be a finite number")
        for name in ("confidence_threshold", "nms_iou_threshold", "overlap_threshold", "stop_min_area"):
            if not 0 < getattr(self, name) <= 1:
                raise ValueError(f"{name} must be in (0, 1]")
        if not 0 < self.left_boundary < self.right_boundary < 1:
            raise ValueError("Region boundaries must satisfy 0 < left < right < 1")
        if not 0 <= self.decision_top <= self.approach_top < self.near_top < 1:
            raise ValueError("Vertical bounds must satisfy 0 <= decision_top <= approach_top < near_top < 1")
        if not 0 < self.stop_distance_m < self.avoid_distance_m:
            raise ValueError("Distances must satisfy 0 < stop_distance_m < avoid_distance_m")
        for name in ("stable_frames", "stale_after_ms"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")

    @classmethod
    def load(cls, path=None):
        if path is None:
            return cls()
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict):
            raise ValueError("Configuration must be a JSON object")
        unknown = data.keys() - {item.name for item in fields(cls)}
        if unknown:
            raise ValueError(f"Unknown configuration keys: {', '.join(sorted(unknown))}")
        return cls(**data)

    def to_dict(self):
        return asdict(self)
