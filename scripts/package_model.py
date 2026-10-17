"""Package only the selected small three-class model and publish measured results."""

import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path
import shutil
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visiondrive.config import Config
from visiondrive.detector import ESPDetector


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run", type=Path, required=True)
    cli.add_argument("--evaluation", type=Path, required=True)
    cli.add_argument("--dataset", type=Path, default=ROOT / "datasets/bottle-cup-box")
    args = cli.parse_args()
    source = args.run / "weights/best.pt"
    if source.stat().st_size > 5 * 1024 * 1024:
        raise ValueError("Selected checkpoint exceeds the compact-artifact budget")
    detector = ESPDetector(source, ROOT / "external/esp-detection", Config())
    evaluation = json.loads(args.evaluation.read_text())
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if evaluation["weights_sha256"] != digest:
        raise ValueError("Evaluation does not match this checkpoint")
    origin = json.loads((args.run / "visiondrive-manifest.json").read_text())
    map_report = json.loads((args.run / "test-metrics.json").read_text())
    with (args.run / "results.csv").open(newline="") as handle:
        history = [{k.strip(): v for k, v in row.items()} for row in csv.DictReader(handle)]
    best = max(history, key=lambda row: .1*float(row["metrics/mAP50(B)"])+.9*float(row["metrics/mAP50-95(B)"]))
    records = [json.loads(line) for line in (args.dataset / "manifest.jsonl").read_text().splitlines()]
    counts = {split: {"images": sum(row["split"] == split for row in records),
                      "instances": dict(Counter(obj["label"] for row in records if row["split"] == split for obj in row["objects"]))}
              for split in ("train", "val", "test")}
    manifest = {
        "model": "ESPDet-Pico", "classes": detector.names, "input_size": [224, 224],
        "sha256": digest, "size_bytes": source.stat().st_size,
        "parameters": sum(p.numel() for p in detector.model.model.parameters()),
        "confidence_threshold": evaluation["confidence_threshold"],
        "upstream_repository": "https://github.com/espressif/esp-detection",
        "upstream_revision": origin["esp_revision"], "initialization_sha256": origin["source_sha256"],
        "license": "AGPL-3.0; see accompanying LICENSE and upstream notices",
        "training": {"epochs_completed": len(history), "best_epoch": int(float(best["epoch"])),
                     "seed": origin["configuration"].get("seed"), "batch": origin["configuration"].get("batch"),
                     "device": origin["configuration"].get("device"), "patience": origin["configuration"].get("patience"),
                     "elapsed_seconds": float(history[-1]["time"])},
        "dataset": {"source": "Open Images official validation pool", "project_splits": counts,
                    "note": "New project partitions; not official benchmark scores", "manifest_sha256": hashlib.sha256((args.dataset / "manifest.jsonl").read_bytes()).hexdigest()},
        "evaluation": {"mAP": map_report, "at_configured_threshold": evaluation["test"]},
        "limitations": ["Initial small public-image prototype", "Box recall is low; needs more representative data",
                        "No independent camera-course, ESP32, or physical-driving validation"]}
    output = ROOT / "weights/obstacle"
    output.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, output / "best.pt")
    shutil.copy2(ROOT / "external/esp-detection/LICENSE", output / "LICENSE")
    (output / "model.json").write_text(json.dumps(manifest, indent=2)+"\n", encoding="utf-8")
    public = {**evaluation, "mAP": map_report, "model": manifest}
    (ROOT / "docs/model-evaluation.json").write_text(json.dumps(public, indent=2)+"\n", encoding="utf-8")
    shutil.copy2(args.run / "results.csv", ROOT / "docs/training-results.csv")
    lines = ["# Bottle, cup, and box model report", "", "Recorded on 2026-10-16.", "",
             "**Initial prototype: box detection remains weak. These results do not establish reliable obstacle avoidance.**", "",
             "## Model and training", "",
             f"- ESPDet-Pico, {manifest['parameters']:,} parameters, 224 x 224 input; {manifest['size_bytes'] / 1048576:.2f} MiB checkpoint.",
             "- Exactly three output classes: bottle, cup, box. Other categories are excluded.",
             f"- CPU training completed {len(history)} epochs; validation selected epoch {manifest['training']['best_epoch']}.",
             f"- Confidence {evaluation['confidence_threshold']:.2f} selected using validation macro F1 only.",
             f"- Checkpoint SHA-256: `{digest}`.", "",
             "The upstream single-class weights supply transferable initialization only. Training rebuilds the symbolic architecture with a three-class head. The original checkpoint is not used by the active demo.", "",
             "## Data", "",
             "The official Open Images validation pool supplied 179 bottle images, 155 cup images, and 68 box images. Content was deduplicated before new project partitions were assigned. Source IDs, author/license metadata, original URLs, and annotations remain in the local data manifest.", "",
             "| Project split | Images | Bottle instances | Cup instances | Box instances |", "| --- | ---: | ---: | ---: | ---: |"]
    for split, row in counts.items():
        lines.append(f"| {split} | {row['images']} | {row['instances'].get('bottle',0)} | {row['instances'].get('cup',0)} | {row['instances'].get('box',0)} |")
    lines += ["", "These are project splits from one public source pool, not official Open Images benchmark results. The selected source has relatively few box images. Annotation completeness and camera-domain differences limit generalization.", "",
              "## Held-out results", "",
              f"Across 62 held-out images, mAP@0.5 = **{map_report['aggregate']['metrics/mAP50(B)']:.3f}** and mAP@0.5:0.95 = **{map_report['aggregate']['metrics/mAP50-95(B)']:.3f}**.", "",
              f"The following precision/recall/F1 values use confidence {evaluation['confidence_threshold']:.2f} and IoU >= 0.5, with class-aware, one-to-one matching. They differ from mAP and are not a single overall accuracy percentage.", "",
              "| Class | Precision | Recall | F1 | TP / FP / FN |", "| --- | ---: | ---: | ---: | --- |"]
    for label, row in evaluation["test"]["per_class"].items():
        lines.append(f"| {label} | {row['precision']:.3f} | {row['recall']:.3f} | {row['f1']:.3f} | {row['tp']} / {row['fp']} / {row['fn']} |")
    lines += ["", "Cups performed best in this run. Bottles still have substantial misses. Box recall is low and requires more varied, representative data. No test-set-driven retraining or threshold changes were used for this published result.", "",
              f"Warm single-image model.predict calls over the evaluation images had median {evaluation['median_inference_ms']:.1f} ms on this development CPU. This excludes cold start, camera buffering, and rendering; it is not end-to-end FPS or an ESP32 benchmark.", "",
              "## Reproduce and extend", "",
              "Use [the notebook](../notebooks/train_bottle_cup_box.ipynb) or [the run guide](laptop-esp32.md). [Machine-readable evaluation](model-evaluation.json), [training history](training-results.csv), and [model provenance](../weights/obstacle/model.json) are included. Raw images, full run checkpoints, and local environments stay outside Git.", "",
              "Next steps are more box images and camera-height-specific recordings, an independent course test, and separate INT8/ESP32 evaluation. No physical car behavior is validated here."]
    (ROOT / "docs/model-report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(f"Packaged {output / 'best.pt'} ({source.stat().st_size:,} bytes), report, history, and model license.")


if __name__ == "__main__":
    main()
