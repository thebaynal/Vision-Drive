"""Choose confidence on validation images, then evaluate the three-class YOLO model."""

import argparse
from collections import Counter
from dataclasses import replace
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.evaluate_objects import CLASSES, score
from scripts.train_yolo import audit_dataset
from visiondrive.config import Config
from visiondrive.laptop_detector import LaptopDetector


def aggregate(rows, threshold):
    result = score(rows, threshold)
    counts = {key: sum(v[key] for v in result["per_class"].values()) for key in ("tp", "fp", "fn")}
    tp, fp, fn = (counts[k] for k in ("tp", "fp", "fn"))
    result["micro"] = {**counts, "precision": tp/(tp+fp) if tp+fp else 0.,
                       "recall": tp/(tp+fn) if tp+fn else 0.,
                       "f1": 2*tp/(2*tp+fp+fn) if 2*tp+fp+fn else 0.}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/yolo-bottle-cup-box-v3")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/yolo11n-finetuned-evaluation/results.json")
    parser.add_argument("--device", default="cpu", help="cpu or CUDA GPU index")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    counts = audit_dataset(args.dataset, 60)
    config = replace(Config.load(ROOT / "config/default.json"), confidence_threshold=.05)
    detector = LaptopDetector(args.weights, config, device=args.device, cpu_threads=args.threads)
    if list(detector.names.values()) != CLASSES:
        raise ValueError("Expected a fine-tuned bottle/cup/box model in that order")
    detector.warmup()
    import cv2
    import torch
    import ultralytics
    records = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    cached, timings = {"val": [], "test": []}, []

    def predict_split(split):
        for row in records:
            if row["split"] != split:
                continue
            image = cv2.imread(str(args.dataset / row["image"]))
            if image is None:
                raise ValueError(f"Cannot decode {row['image']}")
            start = time.perf_counter()
            predictions = detector.predict(image)
            elapsed = (time.perf_counter() - start) * 1000
            if split == "test":
                timings.append(elapsed)
            cached[split].append({**row, "predictions": [vars(d) for d in predictions]})

    predict_split("val")
    grid = [.15, .20, .25, .30, .35, .40, .50, .60]
    validation = {str(t): aggregate(cached["val"], t) for t in grid}
    threshold = max(grid, key=lambda t: (validation[str(t)]["macro_f1"], t))
    detector.config = replace(config, confidence_threshold=threshold)
    predict_split("test")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    # Use the native validator for standard AP curves. No new tuning uses test results.
    metrics = detector.model.val(data=str((args.dataset / "dataset.yaml").resolve()),
        split="test", imgsz=640, rect=False, batch=1, device=args.device, workers=0,
        conf=.001, iou=config.nms_iou_threshold, max_det=300, plots=True, save=False,
        save_json=False, cache=False, project=str(args.output.parent.resolve()),
        name="native-validator", exist_ok=True, verbose=False)
    ap = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        _, _, ap50, ap5095 = metrics.class_result(index)
        ap[detector.names[int(class_id)]] = {"ap50": float(ap50), "ap50_95": float(ap5095)}
    report = {"recorded_utc": datetime.now(timezone.utc).isoformat(),
        "model": "YOLO11n fine-tuned bottle/cup/box", "classes": CLASSES,
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256((args.dataset / "manifest.jsonl").read_bytes()).hexdigest(),
        "images_per_class": counts,
        "image_counts": dict(Counter(r["split"] for r in records)),
        "test_instances": dict(Counter(o["label"] for r in records if r["split"] == "test" for o in r["objects"])),
        "configuration": {"inference_size": 640, "device": args.device, "cpu_threads": args.threads,
                          "confidence_threshold": threshold, "nms_iou": config.nms_iou_threshold,
                          "matching_iou": .5, "ap_confidence_floor": .001},
        "selection": "Checkpoint selected on validation fitness; confidence chosen by validation macro F1, higher-threshold ties; no test-based tuning",
        "validation_grid": validation, "test": aggregate(cached["test"], threshold),
        "average_precision": {"map50": float(metrics.box.map50), "map50_95": float(metrics.box.map), "per_class": ap},
        "warm_detector_latency_ms": {"median": statistics.median(timings), "mean": statistics.mean(timings), "samples": len(timings)},
        "environment": {"python": platform.python_version(), "torch": torch.__version__, "ultralytics": ultralytics.__version__},
        "limitations": ["Small public-image test, particularly 11 box images; annotations may be incomplete.",
                        "Exact content hashes are disjoint; upstream pretraining and near-duplicate overlap are not audited.",
                        "Original test photos appeared in integration work; this remains an exploratory benchmark.",
                        "Native AP and fixed-threshold scores use separate matching/operating-point protocols.",
                        "Three-class fine-tuning replaces the COCO head; it does not preserve 80-class recognition.",
                        "No depth, full obstacle coverage, instruction accuracy, or physical avoidance measured.",
                        "Timing excludes decoding, camera, depth, UI, and Wi-Fi."]}
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".predictions.json").write_text(json.dumps(cached, indent=2), encoding="utf-8")
    runtime = replace(Config.load(ROOT / "config/default.json"), confidence_threshold=threshold)
    args.output.with_suffix(".config.json").write_text(json.dumps(runtime.to_dict(), indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "validation_grid"}, indent=2))


if __name__ == "__main__":
    main()
