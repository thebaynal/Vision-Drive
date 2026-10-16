import json
from pathlib import Path
import pytest
from scripts.prepare_dataset import prepare


def manifest(tmp_path, duplicate=False):
    import cv2
    import numpy as np
    records = []
    for i, split in enumerate(("train", "val", "test")):
        path = tmp_path / f"{split}.png"
        image = np.full((8, 8, 3), (20 if duplicate else i * 50, 0, 0), dtype=np.uint8)
        assert cv2.imwrite(str(path), image)
        records.append({"image": path.name, "split": split,
                        "objects": [{"label": "bottle", "xyxy": [.1, .2, .5, .8]}],
                        "source": {"url": "https://example.org/fixture", "license": "test fixture"}})
    source = tmp_path / "manifest.jsonl"
    source.write_text("".join(json.dumps(row) + "\n" for row in records))
    return source


def test_conversion_and_provenance(tmp_path):
    source = manifest(tmp_path)
    output = tmp_path / "dataset"
    assert prepare([source], output) == {"train": 1, "val": 1, "test": 1}
    assert json.loads((output / "dataset.yaml").read_text())["names"] == ["obstacle"]
    label = next((output / "labels/train").glob("*.txt")).read_text().strip().split()
    assert list(map(float, label[1:])) == pytest.approx([.3, .5, .4, .6])
    assert "sha256" in (output / "manifest.jsonl").read_text()


def test_duplicate_split_leakage_rejected_before_output(tmp_path):
    source = manifest(tmp_path, duplicate=True)
    with pytest.raises(ValueError, match="Duplicate"):
        prepare([source], tmp_path / "dataset")
    assert not (tmp_path / "dataset").exists()


def test_unverified_empty_labels_rejected(tmp_path):
    source = manifest(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    rows[0]["objects"] = []
    source.write_text("\n".join(json.dumps(row) for row in rows))
    with pytest.raises(ValueError, match="verified_background"):
        prepare([source], tmp_path / "dataset")


def test_three_class_dataset_keeps_source_class_ids(tmp_path):
    source = manifest(tmp_path)
    rows = [json.loads(line) for line in source.read_text().splitlines()]
    for row, label in zip(rows, ("bottle", "cup", "box")):
        row["objects"][0]["label"] = label
    source.write_text("\n".join(json.dumps(row) for row in rows))
    output = tmp_path / "dataset"
    prepare([source], output, multiclass=True)
    assert json.loads((output / "dataset.yaml").read_text())["names"] == ["bottle", "cup", "box"]
    for split, class_id in zip(("train", "val", "test"), ("0", "1", "2")):
        assert next((output / "labels" / split).glob("*.txt")).read_text().split()[0] == class_id
