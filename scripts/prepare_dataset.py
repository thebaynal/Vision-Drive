"""Build a single-class YOLO dataset from provenance-preserving JSONL records."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import shutil

LABELS = {"bottle", "cup", "box"}


def prepare(manifests, output, multiclass=False):
    output = Path(output).resolve()
    if output.exists():
        raise ValueError("Output already exists; use a new directory to avoid stale labels")
    records, hashes = [], {}
    for manifest in manifests:
        manifest = Path(manifest).resolve()
        for line_no, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = row["split"]
            if split not in {"train", "val", "test"}:
                raise ValueError(f"{manifest}:{line_no}: split must be train, val, or test")
            image = (manifest.parent / row["image"]).resolve()
            if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                raise ValueError(f"Missing or unsupported image: {image}")
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            if digest in hashes:
                raise ValueError(f"Duplicate image content: {image} (already in {hashes[digest]})")
            hashes[digest] = split
            source = row.get("source", {})
            if not source.get("url") or not source.get("license"):
                raise ValueError(f"Image provenance must include url and license: {image}")
            lines = []
            for obj in row["objects"]:
                if obj["label"] not in LABELS:
                    raise ValueError(f"Unsupported original label: {obj['label']}")
                x1, y1, x2, y2 = obj["xyxy"]
                if not all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
                           for v in (x1, y1, x2, y2)):
                    raise ValueError("Box coordinates must be finite numbers")
                if not 0 <= x1 < x2 <= 1 or not 0 <= y1 < y2 <= 1:
                    raise ValueError(f"Invalid normalized box in {image}")
                class_id = ("bottle", "cup", "box").index(obj["label"]) if multiclass else 0
                lines.append(f"{class_id} {(x1+x2)/2:.8f} {(y1+y2)/2:.8f} {x2-x1:.8f} {y2-y1:.8f}")
            if not lines and row.get("verified_background") is not True:
                raise ValueError("Empty annotations require verified_background=true")
            records.append((row, image, digest, lines))
    if not all(any(row["split"] == split for row, *_ in records) for split in ("train", "val", "test")):
        raise ValueError("Provide nonempty train, val, and test splits")
    # Validate all metadata before creating the dataset.
    output.mkdir(parents=True)
    provenance = []
    counts = {s: 0 for s in ("train", "val", "test")}
    for row, image, digest, lines in records:
        split = row["split"]
        images, labels = output / "images" / split, output / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        labels.mkdir(parents=True, exist_ok=True)
        name = digest + image.suffix.lower()
        shutil.copy2(image, images / name)
        (labels / (digest + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
        provenance.append({**row, "image": f"images/{split}/{name}", "sha256": digest})
        counts[split] += 1
    (output / "manifest.jsonl").write_text("".join(json.dumps(row) + "\n" for row in provenance), encoding="utf-8")
    # JSON is valid YAML, avoiding another dependency in this preparation tool.
    data = {"path": output.as_posix(), "train": "images/train", "val": "images/val", "test": "images/test", "names": ["bottle", "cup", "box"] if multiclass else ["obstacle"]}
    (output / "dataset.yaml").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"Dataset ready: {output / 'dataset.yaml'}; images: {counts}")
    return counts


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifests", nargs="+", required=True, type=Path)
    cli.add_argument("--output", type=Path, default=Path("datasets/obstacles"))
    cli.add_argument("--multiclass", action="store_true", help="Keep bottle/cup/box as three named classes")
    args = cli.parse_args()
    try:
        prepare(args.manifests, args.output, args.multiclass)
    except (ValueError, KeyError, OSError) as exc:
        cli.error(str(exc))


if __name__ == "__main__":
    main()
