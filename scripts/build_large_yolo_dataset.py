"""Build a larger credited dataset with fixed old holdouts and duplicate filtering."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_object_dataset import LABELS
from scripts.prepare_dataset import prepare

SOURCE = "https://storage.googleapis.com/openimages/2018_04/train/"


def image_counts(records, split):
    return Counter(label for row in records if row["split"] == split
                   for label in {o["label"] for o in row["objects"]})


def filtered_annotations(cache):
    import requests
    path = cache / "target-boxes.csv"
    if path.exists():
        return path
    partial = path.with_suffix(".part")
    ids = [mid.encode() for mid in LABELS]
    count, selected = 0, 0
    local = cache / "train-annotations-bbox.csv"
    if local.exists():
        with local.open("rb") as source, partial.open("wb") as output:
            output.write(next(source))
            for line in source:
                if any(mid in line for mid in ids):
                    output.write(line)
        partial.replace(path)
        return path
    with requests.get(SOURCE + "train-annotations-bbox.csv", stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        with partial.open("wb") as output:
            lines = response.iter_lines(chunk_size=1024*1024)
            output.write(next(lines) + b"\n")
            for line in lines:
                count += 1
                if any(mid in line for mid in ids):
                    output.write(line + b"\n")
                    selected += 1
                if count % 1000000 == 0:
                    print(f"Source annotations scanned: {count:,}; target rows: {selected:,}", flush=True)
    partial.replace(path)
    return path


def filtered_metadata(cache, selected):
    import requests
    path = cache / "target-metadata.csv"
    if path.exists():
        return path
    partial = path.with_suffix(".part")
    local = cache / "train-images-boxable-with-rotation.csv"
    if local.exists():
        with local.open(newline="", encoding="utf-8-sig") as stream, partial.open("w", newline="", encoding="utf-8") as output:
            source = csv.DictReader(stream)
            writer = csv.DictWriter(output, fieldnames=source.fieldnames)
            writer.writeheader()
            for row in source:
                if row["ImageID"] in selected:
                    writer.writerow(row)
        partial.replace(path)
        return path
    with requests.get(SOURCE + "train-images-boxable-with-rotation.csv", stream=True,
                      timeout=(20, 120)) as response:
        response.raise_for_status()
        response.raw.decode_content = True
        source = csv.DictReader(io.TextIOWrapper(response.raw, encoding="utf-8-sig", newline=""))
        with partial.open("w", newline="", encoding="utf-8") as output:
            writer = csv.DictWriter(output, fieldnames=source.fieldnames)
            writer.writeheader()
            count = 0
            for index, row in enumerate(source, 1):
                if row["ImageID"] in selected:
                    writer.writerow(row)
                    count += 1
                if index % 250000 == 0:
                    print(f"Source metadata scanned: {index:,}; matching photos: {count:,}", flush=True)
    partial.replace(path)
    return path


def dhash(image):
    import cv2
    gray = cv2.cvtColor(cv2.resize(image, (9, 8)), cv2.COLOR_BGR2GRAY)
    value = 0
    for bit in (gray[:, 1:] > gray[:, :-1]).flat:
        value = (value << 1) | int(bit)
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "datasets/yolo-bottle-cup-box-v2")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/yolo-bottle-cup-box-v3")
    parser.add_argument("--train-per-class", type=int, default=1000)
    parser.add_argument("--val-per-class", type=int, default=150)
    parser.add_argument("--test-per-class", type=int, default=150)
    parser.add_argument("--workers", type=int, default=10)
    args = parser.parse_args()
    if args.output.exists() or min(args.train_per_class, args.val_per_class, args.test_per_class) < 60:
        parser.error("Use a new output folder and at least 60 photos/class in each partition")
    import cv2
    import numpy as np
    import requests
    records = [json.loads(line) for line in (args.base / "manifest.jsonl").read_text().splitlines()]
    records = [{**row, "image": (args.base / row["image"]).resolve().as_posix()} for row in records]
    hashes = {row["sha256"] for row in records}
    used_ids = {row["image_id"] for row in records}
    perceptual = []
    for row in records:
        image = cv2.imread(row["image"])
        if image is None:
            raise ValueError("Cannot decode a base image")
        perceptual.append(dhash(image))
    cache = ROOT / "data/raw/openimages-train-supplement"
    cache.mkdir(parents=True, exist_ok=True)
    groups = defaultdict(list)
    with filtered_annotations(cache).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if len(row["ImageID"]) != 16 or any(c not in "0123456789abcdef" for c in row["ImageID"]):
                continue
            if row["Confidence"] != "1" or any(row.get(k) == "1" for k in ("IsGroupOf", "IsDepiction")):
                continue
            box = [float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
            if 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1:
                groups[row["ImageID"]].append({"label": LABELS[row["LabelName"]], "xyxy": box})
    print("Available training-source photos:", dict(Counter(name for objects in groups.values()
          for name in {o["label"] for o in objects})), flush=True)
    metadata = {}
    with filtered_metadata(cache, set(groups)).open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row.get("OriginalURL") and row.get("License"):
                metadata[row["ImageID"]] = row
    image_dir = cache / "images"
    image_dir.mkdir(exist_ok=True)
    failures, duplicates, added = [], [], []
    rng = random.Random(20261017)
    pools = {}
    for name in LABELS.values():
        candidates = sorted(i for i in metadata if any(o["label"] == name for o in groups[i]))
        rng.shuffle(candidates)
        pools[name] = candidates

    def fetch(image_id):
        path = image_dir / (image_id + ".jpg")
        try:
            if not path.exists():
                for attempt in range(3):
                    try:
                        response = requests.get(f"https://open-images-dataset.s3.amazonaws.com/train/{image_id}.jpg",
                                                timeout=(15, 45))
                        response.raise_for_status()
                        payload = response.content
                        image = cv2.imdecode(np.frombuffer(payload, np.uint8), cv2.IMREAD_COLOR)
                        if image is None or min(image.shape[:2]) < 64:
                            raise ValueError("Unreadable or too-small image")
                        path.write_bytes(payload)
                        break
                    except Exception:
                        if attempt == 2:
                            raise
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Cached image cannot be decoded")
            return image_id, hashlib.sha256(path.read_bytes()).hexdigest(), dhash(image), None
        except Exception as error:
            return image_id, None, None, str(error)

    # Lock extra test/validation assignments before collecting extra training images.
    targets = [("test", args.test_per_class), ("val", args.val_per_class), ("train", args.train_per_class)]
    for split, target in targets:
        for label in sorted(LABELS.values(), key=lambda name: len(pools[name])):
            candidates = pools[label]
            cursor = 0
            while image_counts(records, split)[label] < target:
                needed = target - image_counts(records, split)[label]
                batch = []
                while cursor < len(candidates) and len(batch) < min(needed, 100):
                    image_id = candidates[cursor]
                    cursor += 1
                    if image_id not in used_ids:
                        batch.append(image_id)
                        used_ids.add(image_id)
                if not batch:
                    raise ValueError(f"Not enough unique source photos for {split}/{label}")
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    for image_id, digest, visual, error in pool.map(fetch, batch):
                        if error:
                            failures.append({"image_id": image_id, "error": error})
                            continue
                        if digest in hashes or any((visual ^ prior).bit_count() <= 3 for prior in perceptual):
                            duplicates.append(image_id)
                            continue
                        hashes.add(digest)
                        perceptual.append(visual)
                        meta = metadata[image_id]
                        records.append({"image": (image_dir / (image_id + ".jpg")).as_posix(),
                            "image_id": image_id, "split": split, "sha256": digest,
                            "objects": groups[image_id], "source": {"dataset": "Open Images",
                                "original_split": "train", "url": meta["OriginalURL"],
                                "author": meta.get("Author", ""), "license": meta["License"]}})
                        added.append(image_id)
                print(f"{split} unique class photos: {dict(image_counts(records, split))}; "
                      f"rejected duplicates={len(duplicates)}; failures={len(failures)}", flush=True)
    manifest = cache / "project-large-splits.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    prepare([manifest], args.output, multiclass=True)
    audit = {"source_urls": [SOURCE + "train-annotations-bbox.csv", SOURCE + "train-images-boxable-with-rotation.csv"],
        "seed": 20261017, "images_per_class": {s: dict(image_counts(records, s)) for s, _ in targets},
        "image_counts": dict(Counter(r["split"] for r in records)),
        "instance_counts": {s: dict(Counter(o["label"] for r in records if r["split"] == s for o in r["objects"])) for s, _ in targets},
        "added_images": len(added), "rejected_duplicate_ids": duplicates, "download_failures": failures,
        "deduplication": "Unique SHA-256; supplemental photos rejected at 64-bit dHash Hamming distance <=3 against all retained photos",
        "filtered_annotations_sha256": hashlib.sha256((cache / "target-boxes.csv").read_bytes()).hexdigest(),
        "filtered_metadata_sha256": hashlib.sha256((cache / "target-metadata.csv").read_bytes()).hexdigest(),
        "limitations": "No exhaustive manual annotation/near-duplicate audit; original base near-duplicates were not repartitioned."}
    (args.output / "preparation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2), flush=True)


if __name__ == "__main__":
    main()
