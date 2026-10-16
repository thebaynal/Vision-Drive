"""Bounded latest-frame capture and inference workers; the UI never waits on ML."""

from dataclasses import dataclass
import threading
import time


@dataclass(frozen=True)
class Packet:
    frame_id: int
    captured_at: float
    image: object
    generation: int
    source_time: float | None = None


@dataclass(frozen=True)
class Result:
    packet: Packet
    detections: tuple
    completed_at: float
    inference_ms: float


class Pipeline:
    def __init__(self, capture, detector, is_camera, fps=30):
        self.capture, self.detector = capture, detector
        self.is_camera, self.fps = is_camera, fps
        self.cv = threading.Condition()
        self.closed = False
        self.paused = False
        self.status = "running"
        self.generation = 0
        self.latest = None
        self.result = None
        self.captured = 0
        self.processed = 0
        self.threads = []

    def start(self):
        for target in (self._capture, self._infer):
            thread = threading.Thread(target=target, daemon=True)
            self.threads.append(thread)
            thread.start()

    def _capture(self):
        index, deadline = 0, time.monotonic()
        try:
            while True:
                with self.cv:
                    if self.closed:
                        return
                    if self.paused and not self.is_camera:
                        self.cv.wait(.02)
                        deadline = time.monotonic()
                        continue
                    if not self.is_camera and time.monotonic() < deadline:
                        self.cv.wait(min(.02, deadline - time.monotonic()))
                        continue
                    generation = self.generation
                ok, image = self.capture.read()
                captured = time.monotonic()
                with self.cv:
                    if self.closed:
                        return
                    if generation != self.generation or self.paused:
                        continue
                    if not ok:
                        self.status = "Camera unavailable" if self.is_camera else "VIDEO ENDED"
                        self.cv.notify_all()
                        return
                    # Video is decoded at its own cadence. If the decoder falls
                    # behind, still advance through frames; the bounded slot drops
                    # frames for inference instead of retaining a stale backlog.
                    source_time = None if self.is_camera else index / self.fps
                    self.latest = Packet(index, captured, image, generation, source_time)
                    self.captured += 1
                    index += 1
                    deadline += 1 / self.fps
                    self.cv.notify_all()
        except Exception as exc:
            with self.cv:
                self.status = f"Capture error: {exc}"
                self.cv.notify_all()
        finally:
            self.capture.release()

    def _infer(self):
        last_key = None
        while True:
            with self.cv:
                self.cv.wait_for(lambda: self.closed or (
                    not self.paused and self.status == "running" and self.latest is not None
                    and (self.latest.generation, self.latest.frame_id) != last_key), timeout=.1)
                if self.closed:
                    return
                packet = self.latest
                if self.paused or self.status != "running" or packet is None:
                    continue
                key = (packet.generation, packet.frame_id)
                if key == last_key:
                    continue
                last_key = key
            start = time.monotonic()
            try:
                detections = tuple(self.detector.predict(packet.image))
                end = time.monotonic()
                with self.cv:
                    if packet.generation == self.generation and not self.paused and self.status == "running":
                        self.result = Result(packet, detections, end, (end - start) * 1000)
                        self.processed += 1
            except Exception as exc:
                with self.cv:
                    self.status = f"Inference error: {exc}"

    def snapshot(self):
        with self.cv:
            return self.status, self.result, self.latest, self.processed

    def set_paused(self, paused):
        with self.cv:
            if self.status != "running":
                return
            self.paused = paused
            self.generation += 1
            self.result = None
            self.latest = None
            self.cv.notify_all()

    def close(self):
        with self.cv:
            self.closed = True
            self.cv.notify_all()
        # Capture owns its handle, avoiding release concurrent with a native
        # read. Workers are daemons: a stuck device/model cannot trap UI exit.
        for thread in self.threads:
            thread.join(timeout=1)
