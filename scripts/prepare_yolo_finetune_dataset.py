"""Extend only the training split with licensed Open Images box photos."""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.build_object_dataset import LABELS, download_csv
from scripts.prepare_dataset import prepare

SOURCE_URLS = {
    "boxes.csv": "https://storage.googleapis.com/openimages/v5/test-annotations-bbox.csv",
    "metadata.csv": "https://storage.googleapis.com/openimages/2018_04/test/test-images-with-rotation.csv",
}


def image_counts(records, split):
    return Counter(label for row in records if row["split"] == split
                   for label in {o["label"] for o in row["objects"]})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=ROOT / "datasets/bottle-cup-box")
    parser.add_argument("--output", type=Path, default=ROOT / "datasets/yolo-bottle-cup-box-v2")
    parser.add_argument("--min-train-images", type=int, default=80)
    parser.add_argument("--download", action="store_true", help="Allow explicit public metadata/image downloads")
    args = parser.parse_args()
    if args.output.exists() or args.min_train_images < 60:
        parser.error("Choose a new output folder and a minimum of at least 60 training images per class")
    records = [json.loads(line) for line in (args.base / "manifest.jsonl").read_text().splitlines()]
    # Preserve every existing split assignment, including all held-out images.
    records = [{**r, "image": (args.base / r["image"]).resolve().as_posix()} for r in records]
    known = {r["sha256"] for r in records}
    missing = {name: max(0, args.min_train_images - image_counts(records, "train")[name])
               for name in LABELS.values()}
    print(f"Additional training images needed: {missing}", flush=True)
    cache = ROOT / "data/raw/openimages-test-supplement"
    cache.mkdir(parents=True, exist_ok=True)
    for name, url in SOURCE_URLS.items():
        if not (cache / name).is_file() and not args.download:
            parser.error("Source metadata is missing; use --download")
        download_csv(url, cache / name)
    groups = defaultdict(list)
    with (cache / "boxes.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["LabelName"] not in LABELS or row["Confidence"] != "1":
                continue
            if any(row.get(k) == "1" for k in ("IsGroupOf", "IsDepiction")):
                continue
            box = [float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
            if 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1:
                groups[row["ImageID"]].append({"label": LABELS[row["LabelName"]], "xyxy": box})
    metadata = {}
    with (cache / "metadata.csv").open(newline="", encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            if row["ImageID"] in groups and row.get("OriginalURL") and row.get("License"):
                metadata[row["ImageID"]] = row
    image_dir = cache / "images"
    image_dir.mkdir(exist_ok=True)
    rng = random.Random(20261017)
    failures, added = [], []

    def fetch(image_id):
        import requests
        import cv2
        import numpy as np
        path = image_dir / (image_id + ".jpg")
        try:
            if not path.exists():
                if not args.download:
                    raise ValueError("Image missing; explicit download permission required")
                response = requests.get(f"https://open-images-dataset.s3.amazonaws.com/test/{image_id}.jpg",
                                        timeout=(15, 45))
                response.raise_for_status()
                payload = response.content
                if cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
                    raise ValueError("Invalid downloaded image")
                path.write_bytes(payload)
            return image_id, hashlib.sha256(path.read_bytes()).hexdigest(), None
        except Exception as error:
            return image_id, None, str(error)

    for label in sorted(missing):
        candidates = sorted(i for i in metadata if any(o["label"] == label for o in groups[i]))
        rng.shuffle(candidates)
        while image_counts(records, "train")[label] < args.min_train_images and candidates:
            needed = args.min_train_images - image_counts(records, "train")[label]
            batch, candidates = candidates[:needed], candidates[needed:]
            with ThreadPoolExecutor(max_workers=6) as pool:
                for image_id, digest, error in pool.map(fetch, batch):
                    if error:
                        failures.append({"image_id": image_id, "error": error})
                        continue
                    if digest in known:
                        continue
                    known.add(digest)
                    meta = metadata[image_id]
                    new = {"image": (image_dir / (image_id + ".jpg")).as_posix(),
                           "split": "train", "objects": groups[image_id], "image_id": image_id,
                           "sha256": digest, "source": {"dataset": "Open Images",
                               "original_split": "test", "url": meta["OriginalURL"],
                               "license": meta["License"], "author": meta.get("Author", "")}}
                    records.append(new)
                    added.append(image_id)
            print(f"Training image counts: {dict(image_counts(records, 'train'))}", flush=True)
    counts = image_counts(records, "train")
    if any(counts[name] < args.min_train_images for name in LABELS.values()):
        raise ValueError(f"Not enough successfully downloaded unique images: {counts}")
    manifest = cache / "training-extension.jsonl"
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    prepare([manifest], args.output, multiclass=True)
    audit = {"minimum_train_images": args.min_train_images, "added_source_ids": added,
             "source_urls": SOURCE_URLS, "download_failures": failures,
             "images_per_class": {s: dict(image_counts(records, s)) for s in ("train", "val", "test")},
             "note": "Open Images official test photos supplement project TRAIN only; not an official benchmark."}
    (args.output / "preparation-audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
