from types import SimpleNamespace
import numpy as np
import pytest
from visiondrive.config import Config
from visiondrive.decision import Action, candidate, observe
from visiondrive.laptop_detector import LaptopDetector


class Tensor:
    def __init__(self, data): self.data = data
    def cpu(self): return self
    def tolist(self): return self.data


def adapter(distance):
    detector = LaptopDetector.__new__(LaptopDetector)
    boxes = SimpleNamespace(xyxyn=Tensor([[.4,.4,.6,.98]]), conf=Tensor([.9]), cls=Tensor([56]))
    detector.model = SimpleNamespace(predict=lambda **kw:[SimpleNamespace(boxes=boxes)])
    detector.names = {56:"chair"}
    detector.config, detector.device = Config(), "cpu"
    detector.depth = SimpleNamespace(predict=lambda im:np.full(im.shape[:2],distance,dtype=np.float32))
    return detector


def test_broad_category_and_far_distance_are_preserved_without_image_proxy_stop():
    detector = adapter(5.)
    detections = detector.predict(np.zeros((120,160,3),dtype=np.uint8))
    assert len(detections) == 1 and detections[0].label == "chair"
    assert detections[0].class_id == 56 and detections[0].distance_m == 5.
    assert detections[0].y2 == .98
    assert candidate(observe(detections,Config()))[0] == Action.FORWARD


def test_failed_depth_cannot_silently_present_clear_path():
    with pytest.raises(ValueError,match="invalid"):
        adapter(float("nan")).predict(np.zeros((120,160,3),dtype=np.uint8))
