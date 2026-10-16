import json
import pytest
from visiondrive.config import Config


@pytest.mark.parametrize("kwargs", [{"confidence_threshold": 0}, {"confidence_threshold": True},
    {"left_boundary": .8, "right_boundary": .2}, {"near_top": 1}, {"decision_top": .9},
    {"stable_frames": 1.5}, {"stable_frames": 0}, {"stale_after_ms": -1},
    {"overlap_threshold": float("nan")}, {"nms_iou_threshold": float("inf")}])
def test_invalid_settings(kwargs):
    with pytest.raises(ValueError):
        Config(**kwargs)


def test_unknown_keys_and_non_object_rejected(tmp_path):
    path = tmp_path / "config.json"
    for content in ({"typo": 1}, [1, 2]):
        path.write_text(json.dumps(content))
        with pytest.raises(ValueError):
            Config.load(path)


def test_roundtrip(tmp_path):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(Config().to_dict()))
    assert Config.load(path) == Config()
