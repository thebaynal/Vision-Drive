import hashlib
import json

import pytest

from scripts.evaluate_finetuned_yolo import aggregate
from scripts.prepare_yolo_finetune_dataset import image_counts
from scripts.train_yolo import audit_dataset


def test_minimum_counts_images_instead_of_multiple_boxes():
    rows = [{"split": "train", "objects": [{"label": "box"}] * 60}]
    assert image_counts(rows, "train")["box"] == 1


def dataset(tmp_path):
    rows = []
    for index, split in enumerate(("train", "val", "test")):
        image = tmp_path / f"{split}.jpg"
        image.write_bytes(bytes([index]))
        rows.append({"image": image.name, "split": split,
                     "sha256": hashlib.sha256(image.read_bytes()).hexdigest(),
                     "objects": [{"label": name} for name in ("bottle", "cup", "box")]})
    (tmp_path / "manifest.jsonl").write_text("\n".join(json.dumps(r) for r in rows))
    (tmp_path / "dataset.yaml").write_text(json.dumps({"names": ["bottle", "cup", "box"]}))
    return rows


def test_training_audit_rejects_too_few_unique_training_images(tmp_path):
    dataset(tmp_path)
    with pytest.raises(ValueError, match="at least 60"):
        audit_dataset(tmp_path, 60)


def test_training_audit_rejects_changed_content(tmp_path):
    dataset(tmp_path)
    (tmp_path / "train.jpg").write_bytes(b"changed")
    with pytest.raises(ValueError, match="Changed or duplicate"):
        audit_dataset(tmp_path, 1)


def test_micro_metrics_include_a_missed_box():
    box = [.1, .1, .6, .8]
    rows = [{"objects": [{"label": "bottle", "xyxy": box}, {"label": "box", "xyxy": box}],
             "predictions": [dict(zip(("x1", "y1", "x2", "y2"), box), class_id=0, confidence=.9)]}]
    result = aggregate(rows, .35)
    assert result["micro"] == {"tp": 1, "fp": 0, "fn": 1, "precision": 1., "recall": .5, "f1": 2/3}
    assert result["per_class"]["box"]["fn"] == 1
