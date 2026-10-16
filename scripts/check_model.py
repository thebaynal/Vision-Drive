"""Run genuine ESPDet inference on one image without requiring a webcam."""

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from visiondrive.config import Config
from visiondrive.decision import Controller
from visiondrive.detector import ESPDetector


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--image", type=Path, required=True)
    cli.add_argument("--weights", type=Path, required=True)
    cli.add_argument("--expected-label", default="obstacle")
    cli.add_argument("--esp-repo", type=Path, default=ROOT / "external/esp-detection")
    cli.add_argument("--output", type=Path)
    args = cli.parse_args()
    import cv2
    from visiondrive.render import render
    image = cv2.imread(str(args.image))
    if image is None:
        cli.error(f"Unable to read {args.image}")
    if args.output and args.output.exists():
        cli.error("Output already exists; choose another filename")
    config = Config(stable_frames=1, stale_after_ms=60000)
    detector = ESPDetector(args.weights, args.esp_repo, config, expected_label=args.expected_label)
    start = time.monotonic()
    detections = detector.predict(image)
    end = time.monotonic()
    decision = Controller(config).update(detections, 0, start, end)
    report = {"model": detector.name, "detections": [vars(d) for d in detections],
              "action": decision.action.value, "reason": decision.reason, "cold_inference_ms": (end-start)*1000,
              "note": "Single-image smoke check, not an accuracy or driving benchmark"}
    print(json.dumps(report, indent=2))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        canvas = render(image, decision, config, detector.name,
                        {"inference_ms": (end-start)*1000, "latency_ms": (end-start)*1000}, detector.label)
        if not cv2.imwrite(str(args.output), canvas):
            cli.error("Unable to save output image")
        args.output.with_suffix(".json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
