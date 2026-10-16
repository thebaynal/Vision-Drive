"""Pure geometry and stateful decisions, portable to the future firmware."""

from dataclasses import dataclass
from enum import Enum
import math
from .config import Config


class Action(str, Enum):
    STOP = "STOP"
    FORWARD = "FORWARD"
    LEFT = "LEFT"
    RIGHT = "RIGHT"


@dataclass(frozen=True)
class Detection:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float = 1.0
    class_id: int = 0
    label: str = ""

    @property
    def valid(self):
        values = (self.x1, self.y1, self.x2, self.y2, self.confidence)
        return (all(math.isfinite(x) for x in values)
                and 0 <= self.x1 < self.x2 <= 1
                and 0 <= self.y1 < self.y2 <= 1
                and 0 <= self.confidence <= 1 and type(self.class_id) is int and 0 <= self.class_id <= 2)


@dataclass(frozen=True)
class Observation:
    blocked: tuple[bool, bool, bool]
    near_center: bool
    accepted: tuple[Detection, ...]
    ignored: int = 0


@dataclass(frozen=True)
class Decision:
    action: Action
    reason: str
    observation: Observation
    frame_id: int = -1
    captured_at: float = 0.0


EMPTY = Observation((False, False, False), False, ())


def intersection(a, b):
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0.0, min(a[3], b[3]) - max(a[1], b[1]))


def observe(detections, config):
    edges = (0.0, config.left_boundary, config.right_boundary, 1.0)
    regions = [(edges[i], config.decision_top, edges[i + 1], 1.0) for i in range(3)]
    near = (edges[1], config.near_top, edges[2], 1.0)
    blocked, accepted, ignored, hazard = [False] * 3, [], 0, False
    for det in detections:
        if not det.valid or det.confidence < config.confidence_threshold:
            ignored += 1
            continue
        box = (det.x1, det.y1, det.x2, det.y2)
        area = (det.x2 - det.x1) * (det.y2 - det.y1)
        hits = [intersection(box, region) / area >= config.overlap_threshold for region in regions]
        blocked = [a or b for a, b in zip(blocked, hits)]
        hazard |= hits[1] and intersection(box, near) > 0
        accepted.append(det)
    return Observation(tuple(blocked), hazard, tuple(accepted), ignored)


def candidate(obs):
    left, middle, right = obs.blocked
    if obs.near_center:
        return Action.STOP, "Near-center obstacle"
    if not middle:
        return Action.FORWARD, "No center obstacle detected"
    if not left:
        return Action.LEFT, "Center blocked; left has no detected obstacle" + (" (tie-break)" if not right else "")
    if not right:
        return Action.RIGHT, "Center and left blocked; right has no detected obstacle"
    return Action.STOP, "All three regions blocked"


class Controller:
    def __init__(self, config=None):
        self.config = config or Config()
        self.reset()

    def reset(self, reason="Waiting for fresh observations"):
        self.last_frame = -1
        self.streak = 0
        self.pending = None
        self.current = Decision(Action.STOP, reason, EMPTY)

    def stop(self, reason):
        self.streak = 0
        self.pending = None
        self.current = Decision(Action.STOP, reason, self.current.observation,
                                self.current.frame_id, self.current.captured_at)
        return self.current

    def tick(self, now, status="running"):
        if status != "running":
            return self.stop(status)
        if self.current.frame_id < 0 or now - self.current.captured_at >= self.config.stale_after_ms / 1000:
            return self.stop("Input unavailable or stale")
        return self.current

    def update(self, detections, frame_id, captured_at, now):
        if not all(math.isfinite(x) for x in (captured_at, now)) or captured_at > now:
            return self.stop("Invalid frame timestamp")
        if now - captured_at >= self.config.stale_after_ms / 1000:
            return self.stop("Input unavailable or stale")
        if frame_id <= self.last_frame:
            return self.tick(now)
        # A gap cannot count as uninterrupted evidence for movement.
        if self.current.frame_id >= 0 and captured_at - self.current.captured_at >= self.config.stale_after_ms / 1000:
            self.stop("Observation gap; stabilizing")
        self.last_frame = frame_id
        obs = observe(detections, self.config)
        action, reason = candidate(obs)
        if action == Action.STOP:
            self.stop(reason)
            self.current = Decision(action, reason, obs, frame_id, captured_at)
            return self.current
        self.streak = self.streak + 1 if self.pending == action else 1
        self.pending = action
        if self.streak < self.config.stable_frames:
            action = self.current.action
            reason = f"Stabilizing {self.pending.value}: {self.streak}/{self.config.stable_frames}"
        self.current = Decision(action, reason, obs, frame_id, captured_at)
        return self.current
