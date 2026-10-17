"""Fine-tune the registered ESP architecture with bounded project settings."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visiondrive.detector import register_esp


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--data", required=True, type=Path)
    cli.add_argument("--weights", type=Path, default=ROOT / "weights/pretrained/espdet_pico_224_224_cat.pt")
    cli.add_argument("--esp-repo", type=Path, default=ROOT / "external/esp-detection")
    cli.add_argument("--device", default="cpu")
    cli.add_argument("--epochs", type=int, default=100)
    cli.add_argument("--batch", type=int, default=16)
    cli.add_argument("--output", type=Path, default=ROOT / "runs/obstacle")
    cli.add_argument("--cpu-threads", type=int, default=2)
    cli.add_argument("--patience", type=int, default=20)
    cli.add_argument("--evaluate", action="store_true", help="Evaluate --weights on the test split without training")
    args = cli.parse_args()
    if not args.data.is_file() or not args.weights.is_file():
        cli.error("Data YAML and trusted pretrained checkpoint must exist. See docs/laptop-esp32.md.")
    if args.epochs < 1 or args.batch < 1:
        cli.error("Epochs and batch must be positive")
    YOLO = register_esp(args.esp_repo, cpu_threads=args.cpu_threads if args.device == "cpu" else None)
    import yaml
    data = yaml.safe_load(args.data.read_text(encoding="utf-8"))
    if data.get("names") not in (["obstacle"], {0: "obstacle"}, ["bottle", "cup", "box"], {0: "bottle", 1: "cup", 2: "box"}):
        cli.error("Dataset must contain bottle/cup/box in order, or one merged obstacle class")
    model = YOLO(str(args.weights.resolve()), task="detect")
    if not any(type(layer).__module__.startswith("nn.") for layer in model.model.modules()):
        cli.error("Checkpoint must use the ESP-Detection architecture")
    if args.evaluate:
        metrics = model.val(data=str(args.data.resolve()), imgsz=224, rect=False,
                            split="test", device=args.device, plots=False)
        report = {"aggregate": metrics.results_dict,
                  "per_class": {model.names[i]: float(value) for i, value in enumerate(metrics.box.maps)}}
        print(json.dumps(report, indent=2))
        output = args.weights.parent.parent / "test-metrics.json"
        output.write_text(json.dumps(report, indent=2), encoding="utf-8")
        return
    # The published checkpoint serializes a literal one-class head in its YAML.
    # Rebuild from the upstream symbolic `nc` architecture before transferring
    # weights, so the trainer can actually replace the head with three classes.
    architecture = args.esp_repo.resolve() / "cfg/models/espdet_pico.yaml"
    model = YOLO(str(architecture), task="detect")
    model.load(str(args.weights.resolve()))
    def verify_head(trainer):
        if trainer.model.model[-1].nc != len(trainer.data["names"]):
            raise RuntimeError("Detection head class count does not match the dataset")
    model.add_callback("on_train_start", verify_head)
    # Do not import the upstream generic 'train' module: our wrapper fixes all
    # project settings explicitly and leaves the upstream checkout untouched.
    result = model.train(data=str(args.data.resolve()), imgsz=224, rect=False,
                         epochs=args.epochs, batch=args.batch, patience=args.patience, seed=42,
                         device=args.device, workers=0, project=str(args.output.parent.resolve()),
                         name=args.output.name, exist_ok=False, close_mosaic=min(10, args.epochs),
                         plots=True, pretrained=True, amp=False)
    save_dir = Path(model.trainer.save_dir)
    manifest = {"source_weights": str(args.weights),
                "source_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
                "esp_revision": subprocess.check_output(["git", "-C", str(args.esp_repo), "rev-parse", "HEAD"], text=True).strip(),
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "configuration": model.trainer.args.__dict__,
                "metrics": result.results_dict if result is not None else {}}
    (save_dir / "visiondrive-manifest.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")
    print(f"Best checkpoint: {save_dir / 'weights/best.pt'}")
    print("Evaluate the held-out test split and your presentation footage before making accuracy claims.")


if __name__ == "__main__":
    main()
