import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_notebook_is_valid_structure_and_all_code_cells_compile():
    notebook = json.loads((ROOT / "notebooks/train_bottle_cup_box.ipynb").read_text(encoding="utf-8"))
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
