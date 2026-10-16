"""Fetch the pinned public upstream source and record the pretrained checkpoint."""

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess

ROOT = Path(__file__).resolve().parents[1]
REVISION = "28de5902a357d99dd9a5db0f52905255641c5a78"
URL = "https://github.com/espressif/esp-detection.git"
CHECKPOINT_SHA256 = "7afca1564f45c3087e39758049412337ff4cec6f63a8f7a5380d50999e4924c5"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=ROOT / "external/esp-detection")
    args = parser.parse_args()
    repo = args.repo.resolve()
    if not repo.exists():
        repo.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--no-checkout", URL, str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "checkout", "--detach", REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    if revision != REVISION:
        raise ValueError(f"Upstream is {revision}; expected {REVISION}. Use a separate clean checkout; existing files were not changed.")
    source = repo / "examples/cat_detection/espdet_pico_224_224_cat.pt"
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != CHECKPOINT_SHA256:
        raise ValueError("Upstream checkpoint hash differs from the recorded revision; not copying it")
    target = ROOT / "weights/pretrained/espdet_pico_224_224_cat.pt"
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists() and hashlib.sha256(target.read_bytes()).hexdigest() != digest:
        raise ValueError(f"Refusing to overwrite a different checkpoint at {target}")
    shutil.copy2(source, target)
    manifest = {"repository": URL, "revision": revision, "checkpoint": source.relative_to(repo).as_posix(),
                "sha256": digest, "source_class": "cat", "intended_use": "training initialization only; not accepted by the demo",
                "license_reference": f"https://github.com/espressif/esp-detection/blob/{revision}/LICENSE"}
    target.with_suffix(".json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2))
    print("Training initialization ready. The demo rejects this checkpoint; fine-tune bottle/cup/box first.")


if __name__ == "__main__":
    main()
