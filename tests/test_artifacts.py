import ast
import hashlib
import json
from pathlib import Path
from collections import Counter
import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("name", ["train_bottle_cup_box.ipynb", "laptop_esp32_demo.ipynb", "yolo11n_pretrained_evaluation.ipynb"])
def test_notebook_is_valid_structure_and_all_code_cells_compile(name):
    notebook = json.loads((ROOT / "notebooks" / name).read_text(encoding="utf-8"))
    assert notebook["nbformat"] == 4
    ids = [cell["id"] for cell in notebook["cells"]]
    assert len(ids) == len(set(ids))
    for cell in notebook["cells"]:
        if cell["cell_type"] == "code":
            ast.parse("".join(cell["source"]))
            if name != 'yolo11n_pretrained_evaluation.ipynb':
                assert cell["execution_count"] is None and cell["outputs"] == []
            assert not any(o.get('output_type') == 'error' for o in cell['outputs'])


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


def test_indoor_evaluation_keeps_original_coco_ids_and_source_splits():
    from visiondrive.evaluation.data import read_jsonl
    rows = read_jsonl(ROOT/'data/public/indoor-evaluation-manifest.jsonl')
    assert Counter(r['split'] for r in rows) == {'dev':80,'test':320}
    assert len({r['sha256'] for r in rows}) == len(rows)
    baseline = read_jsonl(ROOT/'data/public/bottle-cup-box-manifest.jsonl')
    assert not {r['image_id'] for r in rows} & {r['image_id'] for r in baseline}
    assert not {r['group_id'] for r in rows if r['split']=='dev'} & {r['group_id'] for r in rows if r['split']=='test'}
    prior = read_jsonl(ROOT/'data/public/exploratory-indoor-manifest.jsonl')
    assert not {r['image_id'] for r in rows} & {r['image_id'] for r in prior}
    assert {o['class_id'] for r in rows for o in r['objects']} == {0}
    assert len({r['group_id'] for r in rows})==len(rows)
    assert all(r['source']['original_split'] == ('validation' if r['split']=='dev' else 'test') for r in rows)
