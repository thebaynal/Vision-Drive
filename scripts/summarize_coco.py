"""Count COCO class images/instances directly from the official annotation ZIP.

Uses the Python standard library; downloads no images and runs no training.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
from pathlib import Path
import zipfile


ROOT = Path(__file__).resolve().parents[1]
SOURCE_URL = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


def summarize(document):
    """Count each image once per class, retaining crowd records separately."""
    categories = {row['id']: row['name'] for row in document['categories']}
    image_ids = {row['id'] for row in document['images']}
    images, ordinary_images = defaultdict(set), defaultdict(set)
    instances, crowds = Counter(), Counter()
    for annotation in document['annotations']:
        category, image = annotation['category_id'], annotation['image_id']
        if category not in categories or image not in image_ids:
            raise ValueError('Annotation references an unknown category or image')
        images[category].add(image)
        instances[category] += 1
        if annotation.get('iscrowd', 0):
            crowds[category] += 1
        else:
            ordinary_images[category].add(image)
    rows = [dict(coco_category_id=category, class_name=categories[category],
                 images=len(images[category]), annotation_records=instances[category],
                 noncrowd_images=len(ordinary_images[category]),
                 noncrowd_instances=instances[category] - crowds[category],
                 crowd_records=crowds[category])
            for category in sorted(categories)]
    return dict(total_images=len(image_ids), total_annotation_records=sum(instances.values()),
                total_noncrowd_instances=sum(instances.values()) - sum(crowds.values()),
                total_crowd_records=sum(crowds.values()), classes=rows)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--archive', type=Path,
                     default=ROOT / 'data/raw/coco2017/annotations_trainval2017.zip')
    cli.add_argument('--output', type=Path, default=ROOT / 'data/public')
    args = cli.parse_args()
    summary = dict(dataset='COCO 2017 instance annotations', source_url=SOURCE_URL,
                   download_page='https://cocodataset.org/#download',
                   counting_rule='Distinct image IDs per class; image counts overlap across classes. '
                   'Annotation records include iscrowd=1 regions; noncrowd columns exclude these. '
                   'A crowd region is not a count of individual objects.',
                   scope='Dataset composition, not audited exposures in the upstream training run.',
                   splits={})
    with args.archive.open('rb') as source:
        summary['archive_sha256'] = hashlib.file_digest(source, 'sha256').hexdigest()
    with zipfile.ZipFile(args.archive) as archive:
        for split in ('train2017', 'val2017'):
            member = f'annotations/instances_{split}.json'
            raw = archive.read(member)
            result = summarize(json.loads(raw))
            result.update(annotation_member=member,
                          annotation_sha256=hashlib.sha256(raw).hexdigest())
            summary['splits'][split] = result
    train, val = (summary['splits'][split] for split in ('train2017', 'val2017'))
    if len(train['classes']) != 80 or len(val['classes']) != 80:
        raise ValueError('Expected the official 80-class COCO detection annotation files')
    rows = []
    for yolo_id, (left, right) in enumerate(zip(train['classes'], val['classes'])):
        if (left['coco_category_id'], left['class_name']) != (right['coco_category_id'], right['class_name']):
            raise ValueError('Train/validation category mappings differ')
        row = dict(yolo_id=yolo_id, coco_category_id=left['coco_category_id'],
                   class_name=left['class_name'])
        for prefix, item in (('train', left), ('val', right)):
            row.update({f'{prefix}_{key}': item[key] for key in
                        ('images', 'annotation_records', 'noncrowd_images',
                         'noncrowd_instances', 'crowd_records')})
        rows.append(row)
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / 'coco2017-class-counts.csv').open('w', encoding='utf-8', newline='') as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (args.output / 'coco2017-statistics.json').write_text(
        json.dumps(summary, indent=2) + '\n', encoding='utf-8')
    for split, result in summary['splits'].items():
        print(split, result['total_images'], 'images;',
              result['total_annotation_records'], 'annotation records')


if __name__ == '__main__':
    main()
