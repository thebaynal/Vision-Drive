"""Traceable Open Images indoor cohort, explicit COCO mapping and integrity checks.

Only images verified positive/negative for BOTH scored classes are admitted.
Unknown image-level labels are never silently interpreted as negative.
"""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import shutil
import urllib.request

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / 'data/raw/openimages-evaluation'
PEOPLE = {'/m/01g317', '/m/04yx4', '/m/03bt1vf', '/m/01bl7v', '/m/05r655'}
MAPPING = {**{mid: 0 for mid in PEOPLE}, '/m/01mzpv': 56}
PARENTS = {0: '/m/01g317', 56: '/m/01mzpv'}
NAMES = {0: 'person', 56: 'chair'}
COCO_NAMES = ('person,bicycle,car,motorcycle,airplane,bus,train,truck,boat,traffic light,fire hydrant,stop sign,parking meter,bench,bird,cat,dog,horse,sheep,cow,elephant,bear,zebra,giraffe,backpack,umbrella,handbag,tie,suitcase,frisbee,skis,snowboard,sports ball,kite,baseball bat,baseball glove,skateboard,surfboard,tennis racket,bottle,wine glass,cup,fork,knife,spoon,bowl,banana,apple,sandwich,orange,broccoli,carrot,hot dog,pizza,donut,cake,chair,couch,potted plant,bed,dining table,toilet,tv,laptop,mouse,remote,keyboard,cell phone,microwave,oven,toaster,sink,refrigerator,book,clock,vase,scissors,teddy bear,hair drier,toothbrush').split(',')
SOURCE = 'https://storage.googleapis.com/openimages/web/factsfigures_v7.html'
LICENSE = 'https://creativecommons.org/licenses/by/2.0/'


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as src:
        for chunk in iter(lambda: src.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def read_jsonl(path):
    return [json.loads(x) for x in Path(path).read_text(encoding='utf-8').splitlines() if x.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def fetch(url, path, expected=None):
    """Atomic cached public download. Never bypasses authentication/challenges."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        part = path.with_suffix(path.suffix + '.part')
        try:
            with urllib.request.urlopen(url, timeout=90) as response, part.open('wb') as out:
                shutil.copyfileobj(response, out)
            part.replace(path)
        except Exception as exc:
            raise RuntimeError(f'Download unavailable: {url}. Obtain the file through its official page; place it at {path}. {exc}') from exc
    sha = digest(path)
    if expected and sha != expected:
        raise ValueError(f'Checksum mismatch: {path}')
    return sha


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    union = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - intersection
    return intersection / union if union else 0.


def valid_box(box):
    return len(box) == 4 and all(np.isfinite(box)) and 0 <= box[0] < box[2] <= 1 and 0 <= box[1] < box[3] <= 1


def dhash(image):
    gray = cv2.resize(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), (9, 8), interpolation=cv2.INTER_AREA)
    return int.from_bytes(np.packbits(gray[:, 1:] > gray[:, :-1]).tobytes(), 'big')


def descendants(node, target):
    if node['LabelName'] == target:
        def flatten(n):
            return {n['LabelName']} | set().union(*(flatten(c) for c in n.get('Subcategory', [])))
        return flatten(node)
    return set().union(*(descendants(c, target) for c in node.get('Subcategory', [])))


def map_objects(boxes):
    """Hierarchy propagation; collapse only coincident parent/child annotations.

    Person/Man/Woman/Boy/Girl are all person. Duplicate labels for the SAME
    instance are removed at IoU>=.95 only across differing source categories.
    Same-category overlapping instances are retained.
    """
    objects, collapsed = [], 0
    for row in sorted(boxes, key=lambda x: (x['LabelName'] != '/m/01g317', x['LabelName'])):
        mid = row['LabelName']
        if mid not in MAPPING:
            continue
        box = [float(row[k]) for k in ('XMin', 'YMin', 'XMax', 'YMax')]
        if not valid_box(box):
            raise ValueError('Malformed source box')
        obj = {'class_id': MAPPING[mid], 'label': NAMES[MAPPING[mid]], 'xyxy': box,
               'source_class': mid, 'occluded': int(row['IsOccluded']), 'truncated': int(row['IsTruncated'])}
        if any(o['class_id'] == obj['class_id'] and o['source_class'] != mid and iou(o['xyxy'], box) >= .95 for o in objects):
            collapsed += 1
        else:
            objects.append(obj)
    return objects, collapsed


def ambiguous_hierarchy(objects):
    """Reject uncertain source parent/subclass instances, never guess a merge.

    After coincident boxes are collapsed, overlapping cross-source person boxes
    might be duplicate labels or distinct people. Whole-image exclusion avoids
    inventing instance identities; the criterion uses ground truth only.
    """
    return any(a['class_id'] == b['class_id'] == 0 and a['source_class'] != b['source_class']
               and iou(a['xyxy'], b['xyxy']) > .3
               for i,a in enumerate(objects) for b in objects[i+1:])


def acquire(output=ROOT / 'datasets/indoor-person-evaluation', public_manifest=ROOT / 'data/public/indoor-evaluation-manifest.jsonl'):
    """Restore a frozen manifest, or build a cohort before any model prediction."""
    output, public_manifest = Path(output), Path(public_manifest)
    if (output / 'manifest.jsonl').exists():
        report = validate(output)
        if not report['valid']:
            raise ValueError(report['errors'])
        return read_jsonl(output / 'manifest.jsonl')
    if public_manifest.exists():
        records = read_jsonl(public_manifest)
    else:
        hierarchy_path = CACHE / 'hierarchy.json'
        fetch('https://storage.googleapis.com/openimages/2018_04/bbox_labels_600_hierarchy.json', hierarchy_path)
        building_ids = descendants(json.loads(hierarchy_path.read_text()), '/m/0cgh4')
        records, exclusions, source_files = [], Counter(), []
        prior_image_ids = set(json.loads((ROOT / 'data/public/prior-image-exclusions.json').read_text(encoding='utf-8'))['image_ids'])
        for split, partition, existing in [('validation', 'dev', 'openimages-objects'), ('test', 'test', 'openimages-test-supplement')]:
            paths = {}
            urls = {'boxes.csv': f'https://storage.googleapis.com/openimages/v5/{split}-annotations-bbox.csv',
                    'metadata.csv': f'https://storage.googleapis.com/openimages/2018_04/{split}/{split}-images-with-rotation.csv',
                    'labels.csv': f'https://storage.googleapis.com/openimages/v5/{split}-annotations-human-imagelabels-boxable.csv'}
            for name, url in urls.items():
                cached = ROOT / f'data/raw/{existing}/{name}'
                paths[name] = cached if cached.exists() else CACHE / f'{split}-{name}'
                # Reuse already-downloaded official label CSVs.
                prior = CACHE / f'{split}-labels.csv'
                if name == 'labels.csv' and prior.exists():
                    paths[name] = prior
                sha = fetch(url, paths[name])
                source_files.append({'url': url, 'sha256': sha, 'bytes': paths[name].stat().st_size})
            groups = defaultdict(list)
            inside, bad = set(), set()
            with paths['boxes.csv'].open(encoding='utf-8-sig', newline='') as src:
                for b in csv.DictReader(src):
                    image_id = b['ImageID']
                    if b['LabelName'] in building_ids and b['IsInside'] == '1':
                        inside.add(image_id)
                    if b['LabelName'] in MAPPING:
                        groups[image_id].append(b)
                        if any(b[k] != '0' for k in ('IsGroupOf', 'IsDepiction', 'IsInside')):
                            bad.add(image_id)
            verified = defaultdict(dict)
            with paths['labels.csv'].open(encoding='utf-8-sig', newline='') as src:
                for label in csv.DictReader(src):
                    if label['ImageID'] in inside and label['LabelName'] in PARENTS.values() and label['Confidence'] in ('0', '1'):
                        verified[label['ImageID']][label['LabelName']] = int(label['Confidence'])
            metadata = {}
            with paths['metadata.csv'].open(encoding='utf-8-sig', newline='') as src:
                for meta in csv.DictReader(src):
                    if meta['ImageID'] in verified:
                        metadata[meta['ImageID']] = meta
            for image_id in sorted(inside):
                if image_id in prior_image_ids:
                    exclusions['prior_inspected_image'] += 1
                    continue
                labels = verified[image_id]
                if not all(mid in labels for mid in PARENTS.values()):
                    exclusions['unverified_scored_class'] += 1
                    continue
                if image_id in bad:
                    exclusions['target_group_depiction_inside_or_unknown_flag'] += 1
                    continue
                objects, collapsed = map_objects(groups[image_id])
                if ambiguous_hierarchy(objects):
                    exclusions['ambiguous_overlapping_person_hierarchy'] += 1
                    continue
                if any(bool([o for o in objects if o['class_id'] == cid]) != bool(labels[mid]) for cid, mid in PARENTS.items()):
                    exclusions['positive_negative_box_conflict'] += 1
                    continue
                m = metadata.get(image_id, {})
                if m.get('License') != LICENSE or not m.get('AuthorProfileURL') or not m.get('OriginalLandingURL'):
                    exclusions['missing_or_nonapproved_attribution'] += 1
                    continue
                records.append({'image_id': image_id, 'split': partition, 'image': f'images/{partition}/{image_id}.jpg',
                    'group_id': m['AuthorProfileURL'], 'objects': objects, 'verified': {str(k): labels[v] for k,v in PARENTS.items()},
                    'collapsed_hierarchy_boxes': collapsed, 'source': {'dataset': 'Open Images V5 boxes / V7 protocol',
                    'original_split': split, 'url': m['OriginalLandingURL'], 'original_image_url': m['OriginalURL'],
                    'download_url': f'https://open-images-dataset.s3.amazonaws.com/{split}/{image_id}.jpg',
                    'author': m['Author'], 'author_url': m['AuthorProfileURL'], 'title': m['Title'], 'license': m['License']},
                    'indoor_evidence': 'official Building-family box with IsInside=1', 'annotation_protocol': SOURCE})
            print(f'{split}: {sum(r["split"] == partition for r in records)} eligible images', flush=True)
        # Entire photographer groups from official test remain test-only.
        test_groups = {r['group_id'] for r in records if r['split'] == 'test'}
        keep = []
        for row in records:
            if row['split'] == 'dev' and row['group_id'] in test_groups:
                exclusions['development_photographer_in_test'] += 1
            else:
                keep.append(row)
        records = keep
        write_json(CACHE / 'selection.json', {'exclusions': dict(exclusions), 'source_files': source_files,
            'building_ids': sorted(building_ids), 'mapping': MAPPING, 'selected_before_pixel_deduplication': len(records)})
    def restore(row):
        path = output / row['image']
        sha = fetch(row['source']['download_url'], path, row.get('sha256'))
        image = cv2.imread(str(path))
        if image is None:
            raise ValueError(f'Unreadable image: {row["image_id"]}')
        return {**row, 'sha256': sha, 'dhash': f'{dhash(image):016x}', 'width': image.shape[1], 'height': image.shape[0],
                'mean_luminance': float(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).mean())}
    with ThreadPoolExecutor(max_workers=8) as pool:
        restored = list(pool.map(restore, records))
    # Test-first ordering prevents related development frames contaminating test.
    unique, removed = [], []
    for r in sorted(restored, key=lambda r: (r['split'] != 'test', r['image_id'])):
        duplicate = next((p for p in unique if r['sha256'] == p['sha256'] or (int(r['dhash'],16) ^ int(p['dhash'],16)).bit_count() <= 4), None)
        if duplicate:
            removed.append({'image_id': r['image_id'], 'duplicate_of': duplicate['image_id']})
        else:
            unique.append(r)
    if public_manifest.exists() and len(unique) != len(records):
        raise ValueError('Frozen dataset unexpectedly contains duplicate images')
    records = sorted(unique, key=lambda r: (r['split'], r['image_id']))
    content = ''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records)
    public_manifest.parent.mkdir(parents=True, exist_ok=True)
    public_manifest.write_text(content, encoding='utf-8')
    (output / 'manifest.jsonl').write_text(content, encoding='utf-8')
    for row in records:
        path = output / 'labels' / row['split'] / (row['image_id'] + '.txt')
        path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        for o in row['objects']:
            x1,y1,x2,y2 = o['xyxy']
            lines.append(f'{o["class_id"]} {(x1+x2)/2:.10f} {(y1+y2)/2:.10f} {x2-x1:.10f} {y2-y1:.10f}')
        path.write_text('\n'.join(lines) + ('\n' if lines else ''), encoding='utf-8')
    # Ultralytics requires a train key even for val(); it aliases development,
    # never used for training. Preserve the ORIGINAL 80 COCO IDs, not two IDs.
    for split in ('dev','test'):
        (output/f'{split}.txt').write_text(''.join(str((output/r['image']).resolve())+'\n' for r in records if r['split']==split),encoding='utf-8')
    write_json(output / 'dataset.yaml', {'path': str(output.resolve()), 'train': 'dev.txt', 'val': 'dev.txt', 'test': 'test.txt', 'names': COCO_NAMES})
    write_json(output / 'pixel_deduplication.json', {'removed': removed, 'threshold_hamming_64bit': 4})
    report = validate(output)
    write_json(output / 'dataset_validation.json', report)
    if not report['valid']:
        raise ValueError(report['errors'])
    print(f'Validated {len(records)} images', flush=True)
    return records


def validate(directory):
    directory = Path(directory).resolve()
    records = read_jsonl(directory / 'manifest.jsonl')
    errors, warnings, hashes, signatures, groups = [], [], {}, [], defaultdict(set)
    counts, objects = Counter(), Counter()
    for r in records:
        path = (directory / r['image']).resolve()
        if not path.is_relative_to(directory):
            errors.append(f'Image escapes dataset: {r["image_id"]}')
            continue
        image = cv2.imread(str(path))
        if image is None or digest(path) != r['sha256']:
            errors.append(f'Unreadable/changed image: {r["image_id"]}')
            continue
        if r['sha256'] in hashes:
            errors.append(f'Duplicate pixels: {r["image_id"]}')
        hashes[r['sha256']] = r['split']
        signature = dhash(image)
        for other, prior_split, prior in signatures:
            if (signature ^ other).bit_count() <= 4:
                errors.append(f'Near duplicate: {prior}/{r["image_id"]} ({prior_split}/{r["split"]})')
        signatures.append((signature, r['split'], r['image_id']))
        groups[r['group_id']].add(r['split'])
        counts[r['split']] += 1
        if r['split'] not in ('dev', 'test'):
            errors.append('Unexpected split')
        if not r['source'].get('license') or not r['source'].get('url') or not r['source'].get('author'):
            errors.append('Missing attribution')
        lines = []
        seen_boxes = set()
        if ambiguous_hierarchy(r['objects']):
            errors.append(f'Ambiguous parent/subclass overlap: {r["image_id"]}')
        for o in r['objects']:
            box = o['xyxy']; cid = o['class_id']
            if cid not in NAMES or not valid_box(box):
                errors.append(f'Invalid class/box: {r["image_id"]}')
                continue
            key = (cid, *box)
            if key in seen_boxes:
                errors.append(f'Duplicate annotation: {r["image_id"]}')
            seen_boxes.add(key)
            x1,y1,x2,y2 = box
            lines.append([cid,(x1+x2)/2,(y1+y2)/2,x2-x1,y2-y1])
            objects[f'{r["split"]}/{NAMES[cid]}'] += 1
        label = directory / 'labels' / r['split'] / (r['image_id']+'.txt')
        if not label.is_file():
            errors.append(f'Missing annotation: {r["image_id"]}')
        else:
            try:
                actual = [list(map(float, line.split())) for line in label.read_text().splitlines() if line.strip()]
                if len(actual) != len(lines) or (actual and (np.asarray(actual).shape != np.asarray(lines).shape or not np.allclose(actual, lines, atol=1e-8))):
                    errors.append(f'Conflicting converted annotation: {r["image_id"]}')
            except ValueError:
                errors.append(f'Malformed annotation: {r["image_id"]}')
        required={str(cid) for cid in r.get('required_class_ids',[0,56])}
        if set(r['verified']) != required or any(bool([o for o in r['objects'] if str(o['class_id']) == cid]) != bool(v) for cid,v in r['verified'].items()):
            errors.append(f'Incomplete or contradictory verification: {r["image_id"]}')
    if any(len(splits) != 1 for splits in groups.values()):
        errors.append('Photographer group crosses development/test')
    if not counts['dev'] or not counts['test']:
        errors.append('Both development and test images are required')
    group_type = 'recording session' if all(r.get('group_type') == 'recording session' for r in records) else 'Flickr photographer URL'
    warnings.extend(['Recording/session identities are supplied by annotators; physical site separation requires independent review.' if group_type == 'recording session' else 'Photographer grouping is a proxy; physical scene IDs are unavailable.',
                     'COCO pretraining overlap is not ruled out by source split or hashing within this repository.',
                     'Public photographs are not a low-mounted robot camera course; no metric depth or motion ground truth.'])
    return {'valid': not errors, 'images': len(records), 'splits': dict(counts), 'instances': dict(objects),
            'independent_group_proxy': group_type, 'groups': len(groups),
            'manifest_sha256': digest(directory / 'manifest.jsonl'), 'errors': errors, 'warnings': warnings}
