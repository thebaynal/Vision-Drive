import hashlib
import json

import pytest

from scripts.evaluate_yolo import fixed_metrics, prepare_native_dataset, verify_records


def test_supported_scores_ignore_box_targets_and_penalize_duplicate_detections():
    prediction = dict(x1=.1, y1=.1, x2=.6, y2=.8, confidence=.9, class_id=0)
    rows = [{"objects": [{"label": "bottle", "xyxy": [.1, .1, .6, .8]},
                         {"label": "box", "xyxy": [.1, .1, .6, .8]}],
             "predictions": [prediction, prediction]}]
    result = fixed_metrics(rows, .35)
    assert set(result["per_class"]) == {"bottle", "cup"}
    assert result["micro"] == {"tp": 1, "fp": 1, "fn": 0, "precision": .5,
                               "recall": 1., "f1": 2/3}


def test_native_labels_use_coco_ids_and_keep_images_with_only_box_annotations(tmp_path):
    dataset = tmp_path / "source"
    dataset.mkdir()
    (dataset / "a.jpg").write_bytes(b"a")
    (dataset / "b.jpg").write_bytes(b"b")
    records = [
        {"image": "a.jpg", "objects": [{"label": "cup", "xyxy": [.1, .2, .5, .6]}]},
        {"image": "b.jpg", "objects": [{"label": "box", "xyxy": [.1, .2, .5, .6]}]},
    ]
    output = tmp_path / "mapped"
    yaml, ids = prepare_native_dataset(dataset, records, output, {39: "bottle", 41: "cup"})
    assert ids == [39, 41]
    assert (output / "labels/test/a.txt").read_text().split()[0] == "41"
    assert not (output / "labels/test/b.txt").read_text().strip()
    assert (output / "images/test/b.jpg").read_bytes() == b"b"
    assert json.loads(yaml.read_text())["names"]["41"] == "cup"


def test_verify_rejects_duplicate_hashes_across_splits(tmp_path):
    rows = [{"split": split, "sha256": "same"} for split in ("train", "test")]
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    with pytest.raises(ValueError, match="Duplicate"):
        verify_records(tmp_path)


def test_verify_rejects_changed_test_image_bytes(tmp_path):
    (tmp_path / "test.jpg").write_bytes(b"changed")
    row = {"split": "test", "sha256": hashlib.sha256(b"original").hexdigest(), "image": "test.jpg"}
    (tmp_path / "manifest.jsonl").write_text(json.dumps(row))
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_records(tmp_path)
