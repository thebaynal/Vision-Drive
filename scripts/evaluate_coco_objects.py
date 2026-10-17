"""Prepare and measure unchanged YOLO11n on bottles/cups in all COCO val2017.

Download official annotations_trainval2017.zip and val2017.zip into
data/raw/coco2017 first. Commands: --stage prepare, then --stage evaluate.
No training, confidence tuning, image selection or photo redistribution occurs.
"""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import sys
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
DATA = ROOT / 'datasets/coco-val2017-objects'
RAW = ROOT / 'data/raw/coco2017'
MANIFEST = ROOT / 'data/public/coco-objects-manifest.jsonl'
OUTPUT = ROOT / 'outputs/evaluation/coco-objects'
PUBLISHED = ROOT / 'docs/results/yolo11n-coco-objects'
CONFIG = ROOT / 'config/coco-objects-evaluation.json'


def sha(path):
    with Path(path).open('rb') as source:
        return hashlib.file_digest(source, 'sha256').hexdigest()


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')


def prepare(config):
    with zipfile.ZipFile(RAW / 'annotations_trainval2017.zip') as archive:
        raw = archive.read('annotations/instances_val2017.json')
    if hashlib.sha256(raw).hexdigest() != config['annotation_sha256']:
        raise ValueError('Official validation annotation checksum mismatch')
    document = json.loads(raw)
    if len(document['images']) != config['expected_images']:
        raise ValueError('Expected all 5,000 validation images')
    licenses = {item['id']: item for item in document['licenses']}
    categories = {item['id']: item['name'] for item in document['categories']}
    if [categories[i] for i in config['coco_category_ids']] != ['bottle', 'cup']:
        raise ValueError('Original COCO category mapping required')
    DATA.mkdir(parents=True, exist_ok=True)
    (DATA / 'instances_val2017.json').write_bytes(raw)
    rows = []
    for image in sorted(document['images'], key=lambda r: r['id']):
        license_info = licenses[image['license']]
        rows.append(dict(image_id=image['id'], file_name=image['file_name'],
                         width=image['width'], height=image['height'],
                         source_url=image['coco_url'], original_page=image.get('flickr_url'),
                         license_id=image['license'], license_name=license_info['name'],
                         license_url=license_info['url']))
    manifest = ''.join(json.dumps(row, sort_keys=True) + '\n' for row in rows)
    if MANIFEST.exists() and MANIFEST.read_text(encoding='utf-8') != manifest:
        raise ValueError('Existing frozen manifest differs')
    MANIFEST.write_text(manifest, encoding='utf-8')
    if sha(MANIFEST) != config['manifest_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    print('Prepared all', len(rows), 'images; manifest SHA-256:', sha(MANIFEST), flush=True)


def fixed_counts(eval_images, category_id, max_detections=100):
    """Aggregate COCO's IoU=.50 matches, excluding ignored GT/detections."""
    tp = fp = fn = 0
    for row in eval_images:
        if row is None or row['category_id'] != category_id or row['maxDet'] != max_detections:
            continue
        if list(row['aRng']) != [0, 10000000000.0]:
            continue  # Only area='all'; other area bins repeat each image.
        matched = np.asarray(row['dtMatches'])[0] > 0
        valid = ~np.asarray(row['dtIgnore'])[0].astype(bool)
        truth_valid = ~np.asarray(row['gtIgnore']).astype(bool)
        truth_matched = np.asarray(row['gtMatches'])[0] > 0
        tp += int(np.sum(matched & valid))
        fp += int(np.sum(~matched & valid))
        fn += int(np.sum(truth_valid & ~truth_matched))
    return dict(tp=tp, fp=fp, fn=fn,
                precision=tp / (tp + fp) if tp + fp else 0.0,
                recall=tp / (tp + fn) if tp + fn else 0.0,
                f1=2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)


def coco_score(truth, predictions, config, fixed=False):
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    if predictions:
        detected = truth.loadRes(predictions)
    else:
        detected = COCO()
        detected.dataset = dict(images=truth.dataset['images'],
                                categories=truth.dataset['categories'], annotations=[])
        detected.createIndex()
    evaluator = COCOeval(truth, detected, 'bbox')
    evaluator.params.imgIds = sorted(truth.getImgIds())
    evaluator.params.catIds = config['coco_category_ids']
    evaluator.params.maxDets = [1, 10, config['max_detections']]
    if fixed:
        evaluator.params.iouThrs = np.array([config['matching_iou']])
    evaluator.evaluate()
    evaluator.accumulate()
    if not fixed:
        evaluator.summarize()
    return evaluator


def evaluate(config):
    import cv2
    from pycocotools.coco import COCO
    from visiondrive.evaluation.runner import environment, load_model
    if not MANIFEST.exists():
        raise ValueError('Run --stage prepare, review and freeze the manifest first')
    if sha(MANIFEST) != config['manifest_sha256']:
        raise ValueError('Frozen manifest checksum mismatch')
    rows = [json.loads(line) for line in MANIFEST.read_text(encoding='utf-8').splitlines()]
    if len(rows) != config['expected_images'] or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('Unique complete validation manifest required')
    if sha(DATA / 'instances_val2017.json') != config['annotation_sha256']:
        raise ValueError('Validation annotations changed')
    truth = COCO(str(DATA / 'instances_val2017.json'))
    if sorted(r['image_id'] for r in rows) != sorted(truth.getImgIds()):
        raise ValueError('Manifest must contain every validation image')
    images = DATA / 'images'
    images.mkdir(exist_ok=True)
    image_audit = []
    print('Verifying official validation images before inference...', flush=True)
    with zipfile.ZipFile(RAW / 'val2017.zip') as archive:
        for row in rows:
            path = images / row['file_name']
            # Write only known basenames; no recursive archive extraction.
            if path.name != row['file_name']:
                raise ValueError('Invalid image basename')
            raw = archive.read('val2017/' + row['file_name'])  # Verifies ZIP CRC.
            if not path.exists() or hashlib.sha256(raw).hexdigest() != sha(path):
                path.write_bytes(raw)
            decoded = cv2.imread(str(path))
            if decoded is None or decoded.shape[:2] != (row['height'], row['width']):
                raise ValueError(f'Image decode/dimensions mismatch: {path.name}')
            image_audit.append(dict(image_id=row['image_id'], sha256=sha(path)))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    PUBLISHED.mkdir(parents=True, exist_ok=True)
    with (PUBLISHED / 'image_checksums.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['image_id', 'sha256'])
        writer.writeheader()
        writer.writerows(image_audit)
    before = sha(ROOT / config['weights'])
    model = load_model(config)
    mapping = dict(zip(config['class_ids'], config['coco_category_ids']))
    predictions = []
    kwargs = dict(imgsz=config['image_size'], rect=False, conf=config['ap_confidence_floor'],
                  iou=config['nms_iou'], max_det=config['max_detections'],
                  classes=config['class_ids'], device=config['device'], half=config['half'],
                  verbose=False, save=False)
    for index, row in enumerate(rows):
        result = model.predict(str(images / row['file_name']), **kwargs)[0]
        for box, confidence, cid in zip(result.boxes.xyxy.cpu().tolist(),
                                       result.boxes.conf.cpu().tolist(),
                                       result.boxes.cls.cpu().tolist()):
            x1, y1, x2, y2 = box
            predictions.append(dict(image_id=row['image_id'], category_id=mapping[int(cid)],
                                    score=float(confidence), bbox=[x1, y1, x2 - x1, y2 - y1]))
        if (index + 1) % 250 == 0:
            print(f'Actual CUDA predictions: {index + 1}/{len(rows)} images', flush=True)
    save(OUTPUT / 'predictions.json', predictions)
    ap = coco_score(truth, predictions, config)
    fixed = coco_score(truth, [p for p in predictions if p['score'] >= config['confidence']], config, fixed=True)
    per_class = {}
    for index, (yolo_id, category_id) in enumerate(mapping.items()):
        precision = ap.eval['precision'][:, :, index, 0, -1]
        valid_mean = lambda values: float(values[values >= 0].mean())
        counts = fixed_counts(fixed.evalImgs, category_id, config['max_detections'])
        annotations = truth.loadAnns(truth.getAnnIds(catIds=[category_id]))
        per_class[model.names[yolo_id]] = dict(
            yolo_id=yolo_id, coco_category_id=category_id, **counts,
            ap50=valid_mean(precision[0]), ap75=valid_mean(precision[5]),
            ap50_95=valid_mean(precision),
            ground_truth=sum(not a['iscrowd'] for a in annotations),
            crowd_regions=sum(a['iscrowd'] for a in annotations),
            positive_images=len(truth.getImgIds(catIds=[category_id])))
    after = sha(ROOT / config['weights'])
    if before != after:
        raise ValueError('Checkpoint changed during evaluation')
    measured = dict(protocol=config['protocol'], configuration=config,
                    completed_at_utc=datetime.now(timezone.utc).isoformat(), images=len(rows),
                    manifest_sha256=sha(MANIFEST), weights_sha256=after, weights_unchanged=True,
                    image_archive_sha256=sha(RAW / 'val2017.zip'),
                    image_checksums_sha256=sha(PUBLISHED / 'image_checksums.csv'),
                    predictions_sha256=sha(OUTPUT / 'predictions.json'),
                    git_commit=subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=ROOT, text=True).strip(),
                    working_tree_dirty=bool(subprocess.check_output(['git', 'status', '--porcelain'], cwd=ROOT, text=True).strip()),
                    source_sha256=sha(Path(__file__)), environment=environment(config),
                    evaluator_version=importlib.metadata.version('pycocotools'), per_class=per_class,
                    limitations=[
                        'COCO validation is a known upstream benchmark, not an independent unseen dataset.',
                        'Only bottle and cup are scored; other 78 classes remain available but are unscored here.',
                        'Photo detection does not measure depth, movement instructions or collision avoidance.',
                        'COCO crowd/ignore rules and maxDets=100 apply; AP uses official 101-point COCO interpolation.',
                        'NMS=0.70 differs from the live application default=0.45; confidence was not tuned.'])
    save(PUBLISHED / 'metrics.json', measured)
    with (PUBLISHED / 'per_class_metrics.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=['class_name', *next(iter(per_class.values())).keys()])
        writer.writeheader()
        writer.writerows(dict(class_name=name, **values) for name, values in per_class.items())
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axis = plt.subplots(figsize=(9, 4.5), layout='constrained')
    keys = ['precision', 'recall', 'f1', 'ap50', 'ap50_95']
    x = np.arange(len(keys))
    for index, (name, values) in enumerate(per_class.items()):
        bars = axis.bar(x + (index - .5) * .36, [values[k] * 100 for k in keys], .36, label=name)
        axis.bar_label(bars, fmt='%.1f', fontsize=9, padding=3)
    axis.set_xticks(x, ['Precision', 'Recall', 'F1', 'AP50', 'AP50-95'])
    axis.set_ylim(0, 100)
    axis.set_ylabel('Percent')
    axis.set_title('Unchanged YOLO11n: bottle/cup on all 5,000 COCO val2017 images')
    axis.text(.5, -.18, 'P/R/F1: confidence ≥ 0.35, IoU ≥ 0.50; AP: official COCO evaluator',
              transform=axis.transAxes, ha='center', fontsize=9)
    axis.legend()
    fig.savefig(ROOT / 'assets/evaluation/coco-objects-performance.png', dpi=160)
    plt.close(fig)
    print(json.dumps(per_class, indent=2), flush=True)


if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--stage', choices=['prepare', 'evaluate'], required=True)
    cli.add_argument('--device', help='0 for CUDA, cpu for explicit CPU scoring')
    args = cli.parse_args()
    config = json.loads(CONFIG.read_text(encoding='utf-8'))
    if args.device:
        config['device'] = args.device
    (prepare if args.stage == 'prepare' else evaluate)(config)
