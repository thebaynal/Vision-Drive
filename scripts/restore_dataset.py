"""Restore the published midterm dataset's exact hashes and project splits."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", type=Path, default=ROOT/'data/public/bottle-cup-box-manifest.jsonl')
    cli.add_argument("--output", type=Path, default=ROOT/'datasets/bottle-cup-box')
    cli.add_argument("--download", action="store_true", help="Permit explicit public-image downloads")
    args = cli.parse_args()
    if args.output.exists():
        cli.error("Output already exists; choose --output with a new directory")
    records = [json.loads(line) for line in args.manifest.read_text(encoding='utf-8').splitlines()]
    cache = ROOT/'data/raw/openimages-objects'
    images = cache/'images'
    images.mkdir(parents=True, exist_ok=True)

    def restore(row):
        image_id = row['image_id']
        if len(image_id) != 16 or any(c not in '0123456789abcdef' for c in image_id):
            raise ValueError('Invalid image ID in manifest')
        source_split = row.get('source', {}).get('original_split', 'validation')
        if source_split not in {'train', 'validation', 'test'}:
            raise ValueError(f'Invalid source split: {source_split}')
        source_images = images if source_split == 'validation' else ROOT/'data/raw'/f'openimages-{source_split}-supplement'/'images'
        source_images.mkdir(parents=True, exist_ok=True)
        path = source_images/(image_id+'.jpg')
        if not path.exists():
            if not args.download:
                raise ValueError('Source images are missing; use --download to allow downloads')
            url = f'https://open-images-dataset.s3.amazonaws.com/{source_split}/{image_id}.jpg'
            with urllib.request.urlopen(url, timeout=60) as response:
                payload = response.read()
            if hashlib.sha256(payload).hexdigest() != row['sha256']:
                raise ValueError(f'Changed image content: {image_id}; no substitution made')
            path.write_bytes(payload)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row['sha256']:
            raise ValueError(f'Cached image hash differs: {image_id}')
        return {**row, 'image': path.resolve().as_posix()}

    with ThreadPoolExecutor(max_workers=8) as pool:
        restored = list(pool.map(restore, records))
    manifest = cache/'restored-midterm.jsonl'
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in restored), encoding='utf-8')
    from scripts.prepare_dataset import prepare
    prepare([manifest], args.output, multiclass=True)


if __name__ == '__main__':
    main()
