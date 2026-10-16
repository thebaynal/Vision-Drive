"""Calibrate confidence on validation data, then evaluate an untouched test split."""

import argparse
from collections import Counter
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visiondrive.config import Config
from visiondrive.decision import intersection
from visiondrive.detector import ESPDetector

CLASSES = ["bottle", "cup", "box"]


def score(rows, threshold):
    totals = {label: Counter(tp=0, fp=0, fn=0) for label in CLASSES}
    for row in rows:
        predictions = sorted((d for d in row["predictions"] if d["confidence"] >= threshold),
                             key=lambda d: -d["confidence"])
        remaining = list(row["objects"])
        for prediction in predictions:
            box = [prediction[key] for key in ("x1", "y1", "x2", "y2")]
            label = CLASSES[prediction["class_id"]]
            best, best_iou = None, 0.
            for index, target in enumerate(remaining):
                if target["label"] != label:
                    continue
                truth = target["xyxy"]
                overlap = intersection(box, truth)
                union = (box[2]-box[0])*(box[3]-box[1]) + (truth[2]-truth[0])*(truth[3]-truth[1]) - overlap
                iou = overlap / union if union > 0 else 0
                if iou > best_iou:
                    best, best_iou = index, iou
            if best is not None and best_iou >= .5:
                totals[label]["tp"] += 1
                remaining.pop(best)
            else:
                totals[label]["fp"] += 1
        for target in remaining:
            totals[target["label"]]["fn"] += 1
    report = {}
    for label, counts in totals.items():
        tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
        report[label] = {**counts, "precision": tp / (tp+fp) if tp+fp else 0,
                         "recall": tp / (tp+fn) if tp+fn else 0,
                         "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0}
    return {"per_class": report, "macro_f1": statistics.mean(row["f1"] for row in report.values())}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--weights", required=True, type=Path)
    cli.add_argument("--dataset", type=Path, default=ROOT / "datasets/bottle-cup-box")
    cli.add_argument("--output", type=Path, default=ROOT / "runs/object-evaluation.json")
    cli.add_argument("--calibrate-config", type=Path, help="Write validation-selected confidence into this runtime config")
    args = cli.parse_args()
    import cv2
    config = Config(confidence_threshold=.05)
    model = ESPDetector(args.weights, ROOT / "external/esp-detection", config)
    model.warmup()
    records = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    cached, timings = {"val": [], "test": []}, []
    for row in records:
        if row["split"] not in cached:
            continue
        image = cv2.imread(str(args.dataset / row["image"]))
        if image is None:
            raise ValueError(f"Cannot read held-out image {row['image']}")
        start = time.monotonic()
        predictions = model.predict(image)
        timings.append((time.monotonic()-start)*1000)
        cached[row["split"]].append({**row, "predictions": [vars(d) for d in predictions]})
    grid = [.15, .20, .25, .30, .35, .40, .50]
    validation = {str(t): score(cached["val"], t) for t in grid}
    threshold = max(grid, key=lambda t: (validation[str(t)]["macro_f1"], t))
    report = {"weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
              "classes": CLASSES, "confidence_threshold": threshold, "iou_threshold": .5,
              "selection": "macro F1 on validation images only; test unused for model/threshold selection",
              "validation_grid": validation, "test": score(cached["test"], threshold),
              "image_counts": {split: len(rows) for split, rows in cached.items()},
              "median_inference_ms": statistics.median(timings),
              "limitations": "Small public subset; original annotations may be incomplete for other present target classes; no real course validation"}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    args.output.with_suffix(".predictions.json").write_text(json.dumps(cached), encoding="utf-8")
    if args.calibrate_config:
        updated = replace(Config.load(args.calibrate_config), confidence_threshold=threshold)
        args.calibrate_config.write_text(json.dumps(updated.to_dict(), indent=2)+"\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "validation_grid"}, indent=2))


if __name__ == "__main__":
    main()
