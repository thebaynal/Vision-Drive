from types import SimpleNamespace
import pytest
from visiondrive.config import Config
from visiondrive.detector import ESPDetector


class Tensor:
    def __init__(self, values):
        self.values = values

    def cpu(self):
        return self

    def tolist(self):
        return self.values


def model_factory(names):
    layer = type("ESPBlock", (), {"__module__": "nn.modules.esp_block"})()
    boxes = SimpleNamespace(xyxyn=Tensor([[.1, .2, .4, .7]]), conf=Tensor([.9]), cls=Tensor([0]))
    class Model:
        def __init__(self, *args, **kwargs):
            self.names = names
            self.model = SimpleNamespace(modules=lambda: [layer])
            self.kwargs = None

        def predict(self, **kwargs):
            self.kwargs = kwargs
            return [SimpleNamespace(boxes=boxes)]
    return Model


def test_adapter_uses_original_normalized_coordinates(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"trusted test fixture")
    monkeypatch.setattr("visiondrive.detector.register_esp", lambda *a, **kw: model_factory({0: "bottle", 1: "cup", 2: "box"}))
    detector = ESPDetector(weights, tmp_path, Config())
    result = detector.predict(object())
    assert (result[0].x1, result[0].y1, result[0].x2, result[0].y2) == (.1, .2, .4, .7)
    assert detector.model.kwargs["rect"] is False
    assert detector.model.kwargs["imgsz"] == 224


def test_cat_checkpoint_is_disabled_even_when_explicit(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"trusted test fixture")
    monkeypatch.setattr("visiondrive.detector.register_esp", lambda *a, **kw: model_factory({0: "cat"}))
    with pytest.raises(ValueError, match="disabled"):
        ESPDetector(weights, tmp_path, Config())
    with pytest.raises(ValueError, match="disabled"):
        ESPDetector(weights, tmp_path, Config(), expected_label="cat")


def test_three_named_object_classes_are_preserved(monkeypatch, tmp_path):
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"trusted test fixture")
    monkeypatch.setattr("visiondrive.detector.register_esp", lambda *a, **kw: model_factory({0: "bottle", 1: "cup", 2: "box"}))
    detector = ESPDetector(weights, tmp_path, Config())
    assert detector.names == ["bottle", "cup", "box"]
    assert detector.predict(object())[0].label == "bottle"
