import threading
import time
import numpy as np
from visiondrive.pipeline import Pipeline
from visiondrive.config import Config
from visiondrive.decision import Action, Controller


class Capture:
    def __init__(self, count=1000):
        self.count = count
        self.released = False

    def read(self):
        if not self.count or self.released:
            return False, None
        self.count -= 1
        time.sleep(.005)
        return True, np.zeros((12, 16, 3), dtype=np.uint8)

    def release(self):
        self.released = True


def wait_for(predicate, timeout=2):
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if predicate():
            return
        time.sleep(.01)
    raise AssertionError("Worker did not reach expected state")


def test_latest_frame_slot_does_not_build_backlog():
    class Detector:
        def predict(self, _):
            time.sleep(.05)
            return ()
    pipeline = Pipeline(Capture(), Detector(), True)
    pipeline.start()
    try:
        wait_for(lambda: pipeline.processed >= 2)
        status, result, latest, _ = pipeline.snapshot()
        assert status == "running"
        assert latest.frame_id > result.packet.frame_id
        assert pipeline.captured > pipeline.processed
    finally:
        pipeline.close()


def test_pause_discards_inflight_result():
    entered, release = threading.Event(), threading.Event()
    class Detector:
        def predict(self, _):
            entered.set()
            release.wait(2)
            return ()
    pipeline = Pipeline(Capture(), Detector(), True)
    pipeline.start()
    try:
        assert entered.wait(1)
        pipeline.set_paused(True)
        release.set()
        time.sleep(.03)
        assert pipeline.snapshot()[1] is None
    finally:
        release.set()
        pipeline.close()


def test_stalled_model_cannot_prevent_independent_stop():
    class Detector:
        def predict(self, _):
            time.sleep(.2)
            return ()
    pipeline = Pipeline(Capture(), Detector(), True)
    ctl = Controller(Config(stable_frames=1, stale_after_ms=30))
    pipeline.start()
    try:
        wait_for(lambda: pipeline.processed >= 1)
        result = pipeline.snapshot()[1]
        decision = ctl.update(result.detections, result.packet.frame_id, result.packet.captured_at, time.monotonic())
        assert decision.action == Action.STOP
        assert "stale" in decision.reason
    finally:
        pipeline.close()


def test_capture_failure_and_video_eof():
    class Detector:
        def predict(self, _):
            return ()
    for camera, expected in [(True, "Camera unavailable"), (False, "VIDEO ENDED")]:
        pipeline = Pipeline(Capture(count=0), Detector(), camera)
        pipeline.start()
        try:
            wait_for(lambda: pipeline.snapshot()[0] != "running")
            assert pipeline.snapshot()[0] == expected
        finally:
            pipeline.close()
