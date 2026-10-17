"""Check submission links, notebooks and published evidence without downloading models."""
import ast
import csv
import json
import math
from pathlib import Path
import re
import sys
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]


def anchors(text):
    values = set(re.findall(r'<a\s+[^>]*id=[\"\x27]([^\"\x27]+)', text))
    duplicates = {}
    for heading in re.findall(r'^#{1,6}\s+(.+?)\s*#*$', text, re.M):
        slug = re.sub(r'[^\w\- ]', '', heading.lower()).replace(' ', '-')
        number = duplicates.get(slug, 0)
        duplicates[slug] = number + 1
        values.add(slug + (f'-{number}' if number else ''))
    return values


def check_links(path, text):
    errors = []
    for link in re.findall(r'\]\(([^)]+)\)', text):
        link = link.split(' "')[0].strip('<>')
        parsed = urlsplit(link)
        if parsed.scheme or parsed.netloc:
            continue
        target = (path.parent / unquote(parsed.path)).resolve() if parsed.path else path
        if not target.exists():
            errors.append(f'{path.name}: missing {link}')
        elif parsed.fragment and target.suffix == '.md':
            if unquote(parsed.fragment) not in anchors(target.read_text(encoding='utf-8-sig')):
                errors.append(f'{path.name}: missing heading {link}')
    return errors


def check(root=ROOT):
    errors = []
    markdown = [root/'README.md', root/'THIRD_PARTY_NOTICES.md',
                *sorted((root/'docs').rglob('*.md')), *sorted((root/'data/public').glob('*.md'))]
    for path in markdown:
        errors.extend(check_links(path, path.read_text(encoding='utf-8-sig')))
    books = sorted((root/'notebooks').glob('*.ipynb'))
    for path in books:
        book = json.loads(path.read_text(encoding='utf-8'))
        for cell in book['cells']:
            source = ''.join(cell['source'])
            if cell['cell_type'] == 'markdown':
                errors.extend(check_links(path, source))
            else:
                try:
                    ast.parse(source)
                except SyntaxError as exc:
                    errors.append(f'{path.name}: invalid Python: {exc}')
                if any(out.get('output_type') == 'error' for out in cell.get('outputs', [])):
                    errors.append(f'{path.name}: saved execution error')
                if path.name == 'yolo11n_pretrained_evaluation.ipynb' and cell['execution_count'] is None:
                    errors.append(f'{path.name}: research cell has no saved execution')
    for folder in ('yolo11n', 'yolo11n-coco-objects'):
        directory = root/'docs/results'/folder
        measured = json.loads((directory/'metrics.json').read_text(encoding='utf-8'))
        classes = measured['per_class']
        for row in csv.DictReader((directory/'per_class_metrics.csv').open(encoding='utf-8')):
            name = row.get('class_name', row.get('class'))
            for key in ('tp', 'fp', 'fn', 'precision', 'recall', 'f1', 'ap50', 'ap75', 'ap50_95', 'ground_truth'):
                if not math.isfinite(float(row[key])) or abs(float(row[key]) - classes[name][key]) > 1e-10:
                    errors.append(f'{folder}: CSV/JSON mismatch for {name}/{key}')
        if not measured['weights_unchanged']:
            errors.append(f'{folder}: original checkpoint was changed')
    return errors, len(markdown), len(books)


if __name__ == '__main__':
    errors, documents, notebooks = check()
    for error in errors:
        print(error)
    print(f'Checked {documents} Markdown files, {notebooks} notebooks and both measured result tables.')
    raise SystemExit(bool(errors))
