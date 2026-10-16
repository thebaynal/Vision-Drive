"""Separate a readable model preview from expiring physical motor commands."""

from dataclasses import replace
from .decision import Controller


class Instructions:
    def __init__(self, config, preview_timeout_ms=5000):
        if type(preview_timeout_ms) is not int or not 1000 <= preview_timeout_ms <= 10000:
            raise ValueError("--preview-timeout-ms must be 1000-10000")
        # A model preview can retain the last interpreted frame while another
        # inference runs. This longer display window MUST NOT authorize motors.
        self.preview_config = replace(config, stale_after_ms=preview_timeout_ms)
        self.preview = Controller(self.preview_config)
        self.control = Controller(config)

    def update(self, detections, frame_id, captured_at, now):
        detections = tuple(detections)
        self.preview.update(detections, frame_id, captured_at, now)
        self.control.update(detections, frame_id, captured_at, now)

    def tick(self, now, status="running"):
        return self.preview.tick(now, status), self.control.tick(now, status)

    def reset(self, reason):
        self.preview.reset(reason)
        self.control.reset(reason)
