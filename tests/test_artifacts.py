import ast
import hashlib
import json
from pathlib import Path
from collections import Counter
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["train_bottle_cup_box.ipynb", "laptop_esp32_demo.ipynb", "train_yolo11.ipynb"])
def test_notebook_is_valid_structure_and_all_code_cells_compile(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            assert cell["execution_count"] is None and cell["outputs"] == []


def test_published_checkpoint_matches_manifest_and_runtime_threshold():
    directory = ROOT / "weights/obstacle"
    manifest = json.loads((directory / "model.json").read_text())
    assert manifest["classes"] == ["bottle", "cup", "box"]
    assert hashlib.sha256((directory / "best.pt").read_bytes()).hexdigest() == manifest["sha256"]
    config = json.loads((ROOT / "config/default.json").read_text())
    assert config["confidence_threshold"] == manifest["confidence_threshold"]
    assert (directory / "best.pt").stat().st_size < 5*1024*1024


def test_public_midterm_manifest_has_disjoint_hashes_splits_and_attribution():
    rows = [json.loads(line) for line in (ROOT / "data/public/bottle-cup-box-manifest.jsonl").read_text().splitlines()]
    assert Counter(row["split"] for row in rows) == {"train":280,"val":60,"test":62}
    assert len({row["sha256"] for row in rows}) == len(rows)
    assert Counter(obj["label"] for row in rows for obj in row["objects"]) == {"bottle":385,"cup":189,"box":114}
    assert all(row["source"]["license"].startswith("https://") and row["source"]["url"].startswith("https://") for row in rows)


def test_yolo_dataset_meets_training_minimum_and_preserves_held_out_assignments():
    public = ROOT / "data/public"
    baseline = [json.loads(line) for line in (public / "bottle-cup-box-manifest.jsonl").read_text().splitlines()]
    rows = [json.loads(line) for line in (public / "yolo-bottle-cup-box-v2-manifest.jsonl").read_text().splitlines()]
    assert len(rows) == len({r["sha256"] for r in rows}) == 435
    counts = Counter(label for r in rows if r["split"] == "train" for label in {o["label"] for o in r["objects"]})
    assert counts == {"bottle": 125, "cup": 108, "box": 80}
    assert all(counts[name] >= 60 for name in ("bottle", "cup", "box"))
    before = {(r["sha256"], r["split"]) for r in baseline if r["split"] != "train"}
    after = {(r["sha256"], r["split"]) for r in rows if r["split"] != "train"}
    assert before == after
    assert all(r["source"]["url"] and r["source"]["license"] for r in rows)


def test_finetuned_yolo_checkpoint_metrics_and_runtime_config_match():
    directory = ROOT / "weights/laptop"
    metadata = json.loads((directory / "yolo11n-bottle-cup-box.json").read_text())
    evaluation = json.loads((ROOT / "docs/yolo11-finetuned-evaluation.json").read_text())
    config = json.loads((ROOT / "config/yolo-finetuned.json").read_text())
    digest = hashlib.sha256((directory / "yolo11n-bottle-cup-box.pt").read_bytes()).hexdigest()
    assert digest == metadata["sha256"] == evaluation["weights_sha256"]
    assert metadata["classes"] == evaluation["classes"] == ["bottle", "cup", "box"]
    assert config["confidence_threshold"] == metadata["confidence_threshold"] == evaluation["configuration"]["confidence_threshold"]
    manifest_hash = hashlib.sha256((ROOT / "data/public/yolo-bottle-cup-box-v2-manifest.jsonl").read_bytes()).hexdigest()
    assert manifest_hash == metadata["dataset_manifest_sha256"] == evaluation["dataset_manifest_sha256"]
    grid = evaluation["validation_grid"]
    selected = max(grid, key=lambda value: (grid[value]["macro_f1"], float(value)))
    assert float(selected) == metadata["confidence_threshold"]
    for name, counts in evaluation["test"]["per_class"].items():
        assert counts["tp"] + counts["fn"] == evaluation["test_instances"][name]
