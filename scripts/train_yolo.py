"""Fine-tune YOLO11n for bottle/cup/box with validation-only checkpoint selection."""

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
CLASSES = ["bottle", "cup", "box"]


def audit_dataset(dataset, minimum):
    records = [json.loads(line) for line in (dataset / "manifest.jsonl").read_text().splitlines()]
    seen = set()
    for row in records:
        image = dataset / row["image"]
        digest = hashlib.sha256(image.read_bytes()).hexdigest()
        if digest != row["sha256"] or digest in seen:
            raise ValueError("Changed or duplicate image; training aborted")
        seen.add(digest)
    counts = {s: Counter(label for r in records if r["split"] == s
                        for label in {o["label"] for o in r["objects"]}) for s in ("train", "val", "test")}
    if any(counts["train"][name] < minimum for name in CLASSES):
        raise ValueError(f"Need at least {minimum} unique training images per class: {counts['train']}")
    if any(not counts[s][name] for s in ("val", "test") for name in CLASSES):
        raise ValueError("Every class needs validation and test examples")
    data = json.loads((dataset / "dataset.yaml").read_text())
    if data["names"] != CLASSES:
        raise ValueError("Dataset class names/order must be bottle, cup, box")
    return {s: dict(c) for s, c in counts.items()}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=ROOT / "datasets/yolo-bottle-cup-box-v3")
    parser.add_argument("--weights", type=Path, default=ROOT / "weights/laptop/yolo11n.pt")
    parser.add_argument("--output", type=Path, default=ROOT / "runs/yolo11n-gpu-bottle-cup-box")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--freeze", type=int, default=0, help="Number of pretrained layers to freeze; zero trains the full network")
    parser.add_argument("--min-train-images", type=int, default=1000)
    parser.add_argument("--device", default="0", help="CUDA GPU index, or cpu")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()
    if args.output.exists() or not args.weights.is_file():
        parser.error("Choose a new run folder and an existing trusted YOLO11n checkpoint")
    if args.min_train_images < 60:
        parser.error("The minimum must be at least 60 unique training images per class")
    if min(args.epochs, args.batch, args.threads) < 1 or args.imgsz < 224 or args.imgsz % 32:
        parser.error("Use positive epochs/batch/threads and an image size >=224 divisible by 32")
    counts = audit_dataset(args.dataset, args.min_train_images)
    os.environ.setdefault("YOLO_CONFIG_DIR", str(ROOT / ".cache/ultralytics"))
    os.environ["YOLO_AUTOINSTALL"] = "false"
    import torch
    import ultralytics
    import ultralytics.utils as utils
    import ultralytics.utils.torch_utils as torch_utils
    from ultralytics import YOLO
    if args.device != "cpu" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable in this Python environment. Select the CUDA-enabled training environment or use --device cpu.")
    utils.NUM_THREADS = torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    model = YOLO(str(args.weights), task="detect")
    start = time.monotonic()

    def progress(trainer):
        record = {"epoch": trainer.epoch + 1, "elapsed_seconds": time.monotonic() - start,
                  "metrics": trainer.metrics, "fitness": trainer.fitness,
                  "best_fitness": trainer.best_fitness}
        with (Path(trainer.save_dir) / "progress.jsonl").open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, default=float) + "\n")
        print("VISIONDRIVE_PROGRESS " + json.dumps(record, default=float), flush=True)

    def verify_head(trainer):
        if trainer.model.model[-1].nc != 3:
            raise RuntimeError("Fine-tuned detection head must contain exactly three classes")

    model.add_callback("on_fit_epoch_end", progress)
    model.add_callback("on_train_start", verify_head)
    model.train(data=str((args.dataset / "dataset.yaml").resolve()), device=args.device,
        imgsz=args.imgsz, epochs=args.epochs, patience=12, batch=args.batch,
        freeze=args.freeze, optimizer="AdamW", lr0=.001, weight_decay=.0005,
        cos_lr=True, warmup_epochs=3, seed=42, deterministic=True,
        amp=args.amp, workers=args.workers, cache=False, plots=True, close_mosaic=min(10, args.epochs),
        project=str(args.output.parent.resolve()), name=args.output.name, exist_ok=False)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights/best.pt"
    provenance = {"model": "YOLO11n fine-tuned bottle/cup/box", "classes": CLASSES,
        "source_weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "dataset_manifest_sha256": hashlib.sha256((args.dataset / "manifest.jsonl").read_bytes()).hexdigest(),
        "images_per_class": counts, "configuration": vars(model.trainer.args),
        "elapsed_seconds": time.monotonic() - start, "torch": torch.__version__,
        "ultralytics": ultralytics.__version__,
        "weights_sha256": hashlib.sha256(best.read_bytes()).hexdigest(),
        "selection": "Best native validation fitness only; no test-set checkpoint selection",
        "device": torch.cuda.get_device_name(int(args.device)) if args.device.isdecimal() else args.device,
        "note": "Transfer learning on project training only; checkpoint selected on validation. Test evaluation is separate."}
    (save_dir / "visiondrive-manifest.json").write_text(json.dumps(provenance, indent=2, default=str) + "\n")
    print(f"Selected checkpoint: {best}", flush=True)


if __name__ == "__main__":
    main()
