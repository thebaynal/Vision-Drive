"""Package the selected YOLO model and compact, verifiable submission evidence."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=ROOT / "runs/yolo11n-gpu-bottle-cup-box")
    parser.add_argument("--evaluation", type=Path, default=ROOT / "runs/yolo11n-finetuned-evaluation/results.json")
    args = parser.parse_args()
    weights = args.run / "weights/best.pt"
    report = json.loads(args.evaluation.read_text())
    training = json.loads((args.run / "visiondrive-manifest.json").read_text())
    digest = hashlib.sha256(weights.read_bytes()).hexdigest()
    if digest != report["weights_sha256"] or digest != training["weights_sha256"]:
        raise ValueError("Training, evaluation, and checkpoint hashes differ")
    if training["dataset_manifest_sha256"] != report["dataset_manifest_sha256"]:
        raise ValueError("Training and evaluation datasets differ")
    progress = [json.loads(line) for line in (args.run / "progress.jsonl").read_text().splitlines()]
    # The final validation callback repeats the last epoch; keep its training record.
    unique = {}
    for row in progress:
        unique.setdefault(row["epoch"], row)
    selected = max(unique.values(), key=lambda row: (row["fitness"], row["epoch"]))
    config = dict(training["configuration"])
    for key in ("model", "data", "project", "name", "save_dir"):
        config.pop(key, None)  # Do not publish machine-specific absolute paths.
    training = {**training, "configuration": config, "completed_epochs": max(unique),
                "selected_epoch": selected["epoch"], "selected_validation": selected["metrics"]}
    artifact = {"model": "YOLO11n", "task": "detect", "classes": report["classes"],
                "sha256": digest, "bytes": weights.stat().st_size,
                "source_url": "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt",
                "source_sha256": training["source_weights_sha256"],
                "dataset_manifest_sha256": training["dataset_manifest_sha256"],
                "training_size": config["imgsz"], "inference_size": report["configuration"]["inference_size"],
                "completed_epochs": training["completed_epochs"], "selected_epoch": training["selected_epoch"],
                "confidence_threshold": report["configuration"]["confidence_threshold"],
                "license": "AGPL-3.0; retain upstream Ultralytics licensing terms",
                "scope": "Three named classes; category-independent obstacles require the separate depth branch"}
    destination = ROOT / "weights/laptop"
    destination.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(weights, destination / "yolo11n-bottle-cup-box.pt")
    shutil.copyfile(ROOT / "weights/obstacle/LICENSE", destination / "LICENSE")
    (destination / "yolo11n-bottle-cup-box.json").write_text(json.dumps(artifact, indent=2) + "\n")
    (ROOT / "docs/yolo11-finetuned-evaluation.json").write_text(json.dumps(report, indent=2) + "\n")
    (ROOT / "docs/yolo11-training-configuration.json").write_text(json.dumps(training, indent=2) + "\n")
    shutil.copyfile(args.run / "results.csv", ROOT / "docs/yolo11-training-results.csv")
    shutil.copyfile(args.evaluation.with_suffix(".config.json"), ROOT / "config/yolo-finetuned.json")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    with (args.run / "results.csv").open(newline="") as stream:
        rows = [{k.strip(): float(v) for k, v in row.items()} for row in csv.DictReader(stream)]
    epochs = [r["epoch"] for r in rows]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.7), constrained_layout=True)
    for axis, loss in zip(axes[:2], ("box", "cls")):
        axis.plot(epochs, [r[f"train/{loss}_loss"] for r in rows], label="Training")
        axis.plot(epochs, [r[f"val/{loss}_loss"] for r in rows], label="Validation")
        axis.set_ylabel("Loss")
        axis.set_title("Bounding-box loss" if loss == "box" else "Classification loss")
    axes[2].plot(epochs, [r["metrics/mAP50(B)"] for r in rows], label="mAP@0.5")
    axes[2].plot(epochs, [r["metrics/mAP50-95(B)"] for r in rows], label="mAP@0.5:0.95")
    axes[2].set_title("Validation detection quality")
    axes[2].set_ylabel("mAP")
    axes[2].set_ylim(0, 1)
    for axis in axes:
        axis.axvline(training["selected_epoch"], color="#64748b", linestyle="--", linewidth=1)
        axis.set_xlabel("Epoch")
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
    fig.suptitle(f"YOLO11n fine-tuning — selected epoch {training['selected_epoch']}")
    fig.savefig(ROOT / "assets/yolo11-training-curves.png", dpi=160)
    plt.close(fig)
    print(json.dumps(artifact, indent=2))


if __name__ == "__main__":
    main()
