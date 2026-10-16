"""Explicitly download the laptop models; inference itself stays offline."""

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
DEPTH_REPO = "depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf"
DEPTH_REVISION = "8078d68a9c75a972131914f6afd0c1723be0da7f"
DEPTH_SHA256 = "e990eb82fbf11b05b7813261196a2b841bdcf5a05f64396724a8987fa90504a3"
YOLO_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
YOLO_SHA256 = "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1"


def fetch(url, path, expected=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.is_file():
        temporary = path.with_suffix(path.suffix + ".part")
        print(f"Downloading {path.name}...", flush=True)
        with urllib.request.urlopen(url, timeout=120) as response, temporary.open("wb") as out:
            while chunk := response.read(1024 * 1024):
                out.write(chunk)
        temporary.replace(path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if expected and digest != expected:
        raise ValueError(f"Hash mismatch for {path}; remove the invalid file and run setup again")
    return {"file": path.name, "source": url, "sha256": digest, "bytes": path.stat().st_size}


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--output", type=Path, default=ROOT / "weights/laptop")
    args = cli.parse_args()
    records = [fetch(YOLO_URL, args.output / "yolo11n.pt", YOLO_SHA256)]
    for name in ("config.json", "preprocessor_config.json", "model.safetensors", "README.md"):
        url = f"https://huggingface.co/{DEPTH_REPO}/resolve/{DEPTH_REVISION}/{name}"
        records.append(fetch(url, args.output / "depth-indoor" / name,
                             DEPTH_SHA256 if name == "model.safetensors" else None))
    (args.output / "sources.json").write_text(json.dumps({"depth_revision": DEPTH_REVISION,
        "files": records, "licenses": {"depth": "Apache-2.0", "yolo": "AGPL-3.0"}}, indent=2)+"\n")
    print("Laptop models ready. No ESP32 hardware is required for the visualization.")


if __name__ == "__main__":
    main()
