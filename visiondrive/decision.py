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
    distance_m: float | None = None

    @property
    def valid(self):
        values = (self.x1, self.y1, self.x2, self.y2, self.confidence)
        return (all(math.isfinite(x) for x in values)
                and 0 <= self.x1 < self.x2 <= 1
                and 0 <= self.y1 < self.y2 <= 1
                and 0 <= self.confidence <= 1 and type(self.class_id) is int and self.class_id >= 0
                and isinstance(self.label, str)
                and (self.distance_m is None or (math.isfinite(self.distance_m) and self.distance_m > 0)))


@dataclass(frozen=True)
class Observation:
    blocked: tuple[bool, bool, bool]
    near_center: bool
    accepted: tuple[Detection, ...]
    ignored: int = 0
    occupancy: tuple[float, float, float] = (0., 0., 0.)
    distant: int = 0


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
    blocked, accepted, ignored, hazard = [False] * 3, [], 0, False
    occupied, distant = [[], [], []], 0
    for det in detections:
        if not det.valid or det.confidence < config.confidence_threshold:
            ignored += 1
            continue
        box = (det.x1, det.y1, det.x2, det.y2)
        area = (det.x2 - det.x1) * (det.y2 - det.y1)
        accepted.append(det)
        # Metric depth takes priority when supplied. ESPDet alone uses a fixed
        # camera's ground-contact position as a calibrated proximity proxy.
        far = (det.distance_m > config.avoid_distance_m if det.distance_m is not None
               else det.y2 < config.approach_top)
        if far:
            distant += 1
            continue
        hits = [intersection(box, region) / area >= config.overlap_threshold for region in regions]
        blocked = [a or b for a, b in zip(blocked, hits)]
        emergency = (det.distance_m <= config.stop_distance_m if det.distance_m is not None
                     else det.y2 > config.near_top and area >= config.stop_min_area)
        hazard |= hits[1] and emergency
        for i, region in enumerate(regions):
            clipped = (max(box[0], region[0]), max(box[1], region[1]),
                       min(box[2], region[2]), min(box[3], region[3]))
            if clipped[0] < clipped[2] and clipped[1] < clipped[3]:
                occupied[i].append(clipped)
    occupancy = tuple(union_area(boxes) / ((region[2]-region[0]) * (region[3]-region[1]))
                      for boxes, region in zip(occupied, regions))
    return Observation(tuple(blocked), hazard, tuple(accepted), ignored, occupancy, distant)


def union_area(boxes):
    """Small rectangle sweep; overlapping boxes must not double-count space."""
    edges = sorted({x for box in boxes for x in (box[0], box[2])})
    total = 0.
    for left, right in zip(edges, edges[1:]):
        intervals = sorted((b[1], b[3]) for b in boxes if b[0] < right and b[2] > left)
        end, height = -1., 0.
        for low, high in intervals:
            height += max(0., high - max(low, end))
            end = max(end, high)
        total += (right-left) * height
    return total


def candidate(obs):
    left, middle, right = obs.blocked
    if obs.near_center:
        return Action.STOP, "Emergency center clearance"
    if not middle:
        return Action.FORWARD, "No nearby center obstacle detected"
    if not left and not right:
        if obs.occupancy[2] + 1e-6 < obs.occupancy[0]:
            return Action.RIGHT, "Center blocked; more free image space on right"
        return Action.LEFT, ("Center blocked; more free image space on left" if obs.occupancy[0] + 1e-6 < obs.occupancy[2]
                             else "Center blocked; equal side clearance (left tie-break)")
    if not left:
        return Action.LEFT, "Center blocked; left has no nearby obstacle"
    if not right:
        return Action.RIGHT, "Center and left blocked; right has no nearby obstacle"
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
            # Do not keep driving forward into a newly blocked center while
            # waiting for enough evidence to authorize a turn.
            action = self.current.action if self.current.action == action else Action.STOP
            reason = f"Stabilizing {self.pending.value}: {self.streak}/{self.config.stable_frames}"
        self.current = Decision(action, reason, obs, frame_id, captured_at)
        return self.current
