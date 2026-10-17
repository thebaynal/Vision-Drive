"""Evaluate pretrained YOLO11n on the existing project test split, without training."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import platform
import shutil
import statistics
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from scripts.evaluate_objects import CLASSES, score
from visiondrive.config import Config
from visiondrive.laptop_detector import LaptopDetector

SUPPORTED = ("bottle", "cup")


def verify_records(dataset):
    """Check project split separation and exact test image bytes before inference."""
    manifest = dataset / "manifest.jsonl"
    records = [json.loads(line) for line in manifest.read_text(encoding="utf-8").splitlines()]
    seen = set()
    for row in records:
        digest = row["sha256"]
        if digest in seen:
            raise ValueError("Duplicate image hashes in the dataset; check split leakage")
        seen.add(digest)
    tests = [row for row in records if row["split"] == "test"]
    if not tests:
        raise ValueError("The manifest has no test images")
    for row in tests:
        image = (dataset / row["image"]).resolve()
        if not image.is_relative_to(dataset.resolve()):
            raise ValueError("Image path escapes the dataset")
        if hashlib.sha256(image.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError(f"Image hash mismatch: {row['image']}")
    return tests, hashlib.sha256(manifest.read_bytes()).hexdigest()


def fixed_metrics(rows, threshold):
    """Use the same class-aware, one-to-one IoU >= .5 scorer as the baseline."""
    supported_rows = [{**row, "objects": [o for o in row["objects"] if o["label"] in SUPPORTED]}
                      for row in rows]
    per_class = score(supported_rows, threshold)["per_class"]
    per_class = {name: per_class[name] for name in SUPPORTED}
    tp, fp, fn = (sum(v[key] for v in per_class.values()) for key in ("tp", "fp", "fn"))
    return {"per_class": per_class,
            "macro_f1": statistics.mean(v["f1"] for v in per_class.values()),
            "micro": {"tp": tp, "fp": fp, "fn": fn,
                      "precision": tp / (tp + fp) if tp + fp else 0.,
                      "recall": tp / (tp + fn) if tp + fn else 0.,
                      "f1": 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.}}


def prepare_native_dataset(dataset, records, directory, names):
    """Remap labels by name to COCO IDs, preserving all test images, even box-only ones."""
    ids = {name: class_id for class_id, name in names.items() if name in SUPPORTED}
    if set(ids) != set(SUPPORTED):
        raise ValueError("Expected a COCO model containing bottle and cup classes")
    images, labels = directory / "images/test", directory / "labels/test"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    for row in records:
        source = dataset / row["image"]
        shutil.copyfile(source, images / source.name)
        lines = []
        for target in row["objects"]:
            if target["label"] not in ids:
                continue
            x1, y1, x2, y2 = target["xyxy"]
            lines.append(f"{ids[target['label']]} {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} "
                         f"{x2-x1:.10f} {y2-y1:.10f}")
        (labels / (source.stem + ".txt")).write_text("\n".join(lines) + "\n", encoding="utf-8")
    path = directory / "dataset.yaml"
    # JSON is also valid YAML; using names for all 80 classes preserves model IDs.
    path.write_text(json.dumps({"path": directory.as_posix(), "train": "images/test",
                                "val": "images/test", "test": "images/test",
                                "names": names}), encoding="utf-8")
    return path, sorted(ids.values())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/laptop/yolo11n.pt")
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/bottle-cup-box")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/yolo-evaluation/results.json")
    parser.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    records, manifest_hash = verify_records(args.dataset)
    config = Config.load(args.config)
    detector = LaptopDetector(args.weights, config, cpu_threads=args.threads)
    detector.warmup()
    import cv2
    import torch
    import ultralytics

    rows, timings = [], []
    for row in records:
        image = cv2.imread(str(args.dataset / row["image"]))
        if image is None:
            raise ValueError(f"Cannot decode {row['image']}")
        start = time.perf_counter()
        detections = detector.predict(image)  # Actual laptop adapter; no depth or motor commands.
        timings.append((time.perf_counter() - start) * 1000)
        predictions = [{**vars(d), "class_id": CLASSES.index(d.label)}
                       for d in detections if d.label in SUPPORTED]
        rows.append({**row, "predictions": predictions})

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="yolo-labels-", dir=args.output.parent) as temporary:
        yaml, class_ids = prepare_native_dataset(args.dataset, records, Path(temporary), detector.names)
        # Separate AP evaluation sweeps confidence rather than reporting AP at runtime confidence.
        metrics = detector.model.val(data=str(yaml), split="test", imgsz=640, rect=False,
            batch=1, device="cpu", workers=0, conf=.001, iou=config.nms_iou_threshold,
            classes=class_ids, max_det=300, plots=False, save=False, save_json=False,
            cache=False, project=str(args.output.parent.resolve()), name="native-validator",
            exist_ok=True, verbose=False)
    ap_classes = {}
    for index, class_id in enumerate(metrics.box.ap_class_index):
        _, _, ap50, ap5095 = metrics.class_result(index)
        ap_classes[detector.names[int(class_id)]] = {"ap50": float(ap50), "ap50_95": float(ap5095)}

    report = {
        "recorded_utc": datetime.now(timezone.utc).isoformat(),
        "model": "YOLO11n COCO pretrained; no project fine-tuning",
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": manifest_hash,
        "split": "test", "image_count": len(records),
        "annotated_instances": dict(Counter(o["label"] for r in records for o in r["objects"])),
        "evaluated_classes": list(SUPPORTED), "excluded_classes": {"box": "No COCO box class"},
        "configuration": {"image_size": 640, "device": "cpu", "cpu_threads": args.threads,
                          "runtime_confidence": config.confidence_threshold,
                          "nms_iou": config.nms_iou_threshold, "matching_iou": .5,
                          "ap_confidence_floor": .001, "max_detections": 300},
        "fixed_threshold": fixed_metrics(rows, config.confidence_threshold),
        "average_precision": {"map50": float(metrics.box.map50),
                              "map50_95": float(metrics.box.map), "per_class": ap_classes},
        "warm_detector_latency_ms": {"median": statistics.median(timings),
                                     "mean": statistics.mean(timings), "samples": len(timings)},
        "environment": {"python": platform.python_version(), "platform": platform.platform(),
                        "processor": platform.processor(), "torch": torch.__version__,
                        "ultralytics": ultralytics.__version__},
        "limitations": [
            "Only bottle and cup annotations are scored; this is not an 80-class or all-obstacle benchmark.",
            "Box annotations are excluded, not silently mapped to another COCO category.",
            "Open Images labels may be incomplete; apparent false positives need manual audit.",
            "No overlap audit with upstream model pretraining data; not proven unseen to pretrained YOLO.",
            "The test split was previously used for integration examples; treat this as exploratory.",
            "Fixed runtime confidence was not selected on this test split; changing it requires validation data.",
            "AP uses the native Ultralytics validator, whose matching differs from the fixed-threshold scorer.",
            "No depth, region blockage, instruction accuracy, physical avoidance, or end-to-end latency measured.",
            "Timing includes the warm CPU detector adapter, excluding image decoding, depth, UI, camera, and Wi-Fi."
        ]
    }
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".predictions.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
