"""Download a bounded, reproducible public bottle/cup/box dataset and prepare YOLO labels.

Uses the official Open Images validation source, partitioned into NEW project
train/validation/test sets. Results are not official Open Images benchmark results.
"""

import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import json
from pathlib import Path
import random
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
LABELS = {"/m/04dr76w": "bottle", "/m/02p5f1q": "cup", "/m/025dyy": "box"}
URLS = {
    "boxes.csv": "https://storage.googleapis.com/openimages/v5/validation-annotations-bbox.csv",
    "metadata.csv": "https://storage.googleapis.com/openimages/2018_04/validation/validation-images-with-rotation.csv",
}


def download_csv(url, path):
    import requests
    if path.is_file():
        print(f"Using cached metadata: {path.name}", flush=True)
        return
    temp = path.with_suffix(".part")
    print(f"Downloading {path.name}", flush=True)
    with requests.get(url, stream=True, timeout=(20, 120)) as response:
        response.raise_for_status()
        total, last = 0, 0
        with temp.open("wb") as output:
            for chunk in response.iter_content(1024 * 256):
                output.write(chunk)
                total += len(chunk)
                if total - last >= 4 * 1024 * 1024:
                    print(f"{path.name}: {total / 1048576:.1f} MB", flush=True)
                    last = total
    temp.replace(path)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--per-class", type=int, default=250)
    cli.add_argument("--output", type=Path, default=ROOT / "datasets/bottle-cup-box")
    cli.add_argument("--workers", type=int, default=8)
    args = cli.parse_args()
    if args.per_class < 20 or not 1 <= args.workers <= 16:
        cli.error("Use at least 20 images per class and 1-16 download workers")
    if args.output.exists():
        cli.error("Output exists. Choose a new directory; cached source downloads will be reused.")
    cache = ROOT / "data/raw/openimages-objects"
    cache.mkdir(parents=True, exist_ok=True)
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(download_csv, url, cache / name) for name, url in URLS.items()]
        for future in futures:
            future.result()
    groups = defaultdict(list)
    with (cache / "boxes.csv").open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row["LabelName"] not in LABELS or row["Confidence"] != "1":
                continue
            if any(row.get(name) == "1" for name in ("IsGroupOf", "IsDepiction")):
                continue
            coords = [float(row[key]) for key in ("XMin", "YMin", "XMax", "YMax")]
            if 0 <= coords[0] < coords[2] <= 1 and 0 <= coords[1] < coords[3] <= 1:
                groups[row["ImageID"]].append({"label": LABELS[row["LabelName"]], "xyxy": coords})
    rng = random.Random(42)
    pools = {}
    for label in LABELS.values():
        available = sorted(key for key, objects in groups.items() if any(obj["label"] == label for obj in objects))
        rng.shuffle(available)
        pools[label] = available[:args.per_class]
        print(f"Available {label}: {len(available)} images; selected {len(pools[label])}", flush=True)
        if len(pools[label]) < 20:
            raise ValueError(f"Not enough {label} data for even an initial disjoint training run")
    selected = set().union(*map(set, pools.values()))
    metadata = {}
    with (cache / "metadata.csv").open(newline="", encoding="utf-8-sig") as source:
        for row in csv.DictReader(source):
            if row["ImageID"] in selected:
                metadata[row["ImageID"]] = row
    image_dir = cache / "images"
    image_dir.mkdir(exist_ok=True)

    def fetch(image_id):
        import requests
        import cv2
        import numpy as np
        meta = metadata.get(image_id, {})
        if not meta.get("License") or not meta.get("OriginalURL"):
            return image_id, None, "Missing license/source metadata"
        image = image_dir / (image_id + ".jpg")
        try:
            if not image.exists():
                url = f"https://open-images-dataset.s3.amazonaws.com/validation/{image_id}.jpg"
                last_error = None
                for attempt in range(3):
                    try:
                        response = requests.get(url, timeout=(15, 45))
                        response.raise_for_status()
                        payload = response.content
                        if cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR) is None:
                            raise ValueError("Downloaded payload is not an image")
                        image.write_bytes(payload)
                        break
                    except Exception as exc:
                        last_error = exc
                        if attempt == 2:
                            raise last_error
                        time.sleep(attempt + 1)
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            return image_id, digest, None
        except Exception as exc:
            return image_id, None, str(exc)

    good, failures = {}, []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        pending = [pool.submit(fetch, image_id) for image_id in sorted(selected)]
        for index, future in enumerate(as_completed(pending), 1):
            image_id, digest, error = future.result()
            if error:
                failures.append({"image_id": image_id, "error": error})
            else:
                good[image_id] = digest
            if index % 25 == 0 or index == len(pending):
                print(f"Images: {index}/{len(pending)}, valid={len(good)}, failures={len(failures)}", flush=True)
    # Deduplicate content before splitting; group multi-label images with their
    # rarest present class so scarce cup/box samples have a chance in each split.
    hashes, stratified = set(), defaultdict(list)
    for image_id in sorted(good):
        if good[image_id] in hashes:
            continue
        hashes.add(good[image_id])
        present = {obj["label"] for obj in groups[image_id]}
        primary = min(present, key=lambda label: (len(pools[label]), label))
        stratified[primary].append(image_id)
    records, counts = [], Counter()
    for label in sorted(stratified):
        ids = stratified[label]
        rng.shuffle(ids)
        train_end, val_end = int(len(ids) * .70), int(len(ids) * .85)
        for i, image_id in enumerate(ids):
            split = "train" if i < train_end else "val" if i < val_end else "test"
            meta = metadata[image_id]
            records.append({"image": f"images/{image_id}.jpg", "split": split,
                            "objects": groups[image_id], "image_id": image_id, "source": {
                                "dataset": "Open Images", "original_split": "validation",
                                "url": meta["OriginalURL"], "license": meta["License"],
                                "author": meta.get("Author", "")}})
            counts[split] += 1
    manifest = cache / "project-split.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    (cache / "download-failures.json").write_text(json.dumps(failures, indent=2), encoding="utf-8")
    from scripts.prepare_dataset import prepare
    prepare([manifest], args.output, multiclass=True)
    print(f"Prepared project splits: {dict(counts)}. All are derived from the official validation source; "
          "this is not an official benchmark evaluation.", flush=True)


if __name__ == "__main__":
    main()
