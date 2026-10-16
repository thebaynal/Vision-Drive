"""Select a bounded Open Images split and produce a curated-data JSONL manifest."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import random
import urllib.request


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--classes", type=Path, required=True, help="Open Images boxable class description CSV")
    cli.add_argument("--annotations", type=Path, required=True, help="One split's bounding-box CSV")
    cli.add_argument("--metadata", type=Path, required=True, help="That split's image metadata CSV with OriginalURL, License, Author")
    cli.add_argument("--split", choices=["train", "val", "test"], required=True)
    cli.add_argument("--class-names", nargs=3, default=["Bottle", "Coffee cup", "Box"],
                     metavar=("BOTTLE_CLASS", "CUP_CLASS", "BOX_CLASS"))
    cli.add_argument("--limit", type=int, default=1500)
    cli.add_argument("--images-root", type=Path, default=Path("data/raw/images"))
    cli.add_argument("--output", type=Path, required=True)
    cli.add_argument("--download", action="store_true", help="Explicitly download selected public images")
    args = cli.parse_args()
    if args.limit < 1 or args.output.exists():
        cli.error("Limit must be positive and output must not already exist")
    with args.classes.open(encoding="utf-8-sig", newline="") as file:
        lookup = {name.casefold().strip(): mid for mid, name in csv.reader(file)}
    missing = [name for name in args.class_names if name.casefold() not in lookup]
    if missing:
        cli.error(f"Classes not found: {missing}. Audit class metadata; supply actual equivalent names, "
                  "or manually annotate public images with prepare_dataset.py's manifest format. No labels were substituted.")
    mapping = {lookup[name.casefold()]: label for name, label in zip(args.class_names, ("bottle", "cup", "box"))}
    grouped = {}
    with args.annotations.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if (row["LabelName"] not in mapping or float(row.get("Confidence", "1")) < 1
                    or any(row.get(k, "0") == "1" for k in ("IsGroupOf", "IsDepiction"))):
                continue
            box = [float(row[k]) for k in ("XMin", "YMin", "XMax", "YMax")]
            if not 0 <= box[0] < box[2] <= 1 or not 0 <= box[1] < box[3] <= 1:
                continue
            grouped.setdefault(row["ImageID"], []).append({"label": mapping[row["LabelName"]], "xyxy": box,
                                                         "original_class": row["LabelName"]})
    # Deterministic round-robin selection gives each source type a chance even
    # when one category dominates the annotation file.
    rng = random.Random(42)
    pools = {label: sorted(image_id for image_id, objs in grouped.items() if any(o["label"] == label for o in objs))
             for label in ("bottle", "cup", "box")}
    for values in pools.values():
        rng.shuffle(values)
    selected = set()
    while len(selected) < args.limit and any(pools.values()):
        for values in pools.values():
            while values and values[-1] in selected:
                values.pop()
            if values and len(selected) < args.limit:
                selected.add(values.pop())
    metadata = {}
    with args.metadata.open(encoding="utf-8-sig", newline="") as file:
        for row in csv.DictReader(file):
            if row["ImageID"] in selected:
                metadata[row["ImageID"]] = row
    absent = [image_id for image_id in selected if not metadata.get(image_id, {}).get("OriginalURL")
              or not metadata.get(image_id, {}).get("License")]
    if absent:
        cli.error(f"Missing URL/license metadata for {len(absent)} selected images; no manifest created")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.images_root.mkdir(parents=True, exist_ok=True)
    records = []
    source_split = "validation" if args.split == "val" else args.split
    for image_id in sorted(selected):
        if not all(c in "0123456789abcdef" for c in image_id.lower()):
            cli.error("Unexpected image ID")
        image = args.images_root / source_split / (image_id + ".jpg")
        if not image.exists() and args.download:
            image.parent.mkdir(parents=True, exist_ok=True)
            url = f"https://open-images-dataset.s3.amazonaws.com/{source_split}/{image_id}.jpg"
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            image.write_bytes(payload)
        if not image.exists():
            cli.error(f"Missing image {image}; supply images or explicitly use --download")
        meta = metadata[image_id]
        records.append({"image": os.path.relpath(image.resolve(), args.output.parent.resolve()), "split": args.split,
                        "image_id": image_id, "objects": grouped[image_id], "source": {
                            "dataset": "Open Images", "url": meta["OriginalURL"], "license": meta["License"],
                            "author": meta.get("Author", ""), "source_split": source_split},
                        "sha256": hashlib.sha256(image.read_bytes()).hexdigest()})
    args.output.write_text("".join(json.dumps(row) + "\n" for row in records), encoding="utf-8")
    print(f"Selected {len(records)}/{args.limit} images. Audit annotations before training. Manifest: {args.output}")


if __name__ == "__main__":
    main()
