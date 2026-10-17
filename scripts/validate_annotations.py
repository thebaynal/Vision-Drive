"""Validate evaluation pixels, labels, completeness, attribution and split leakage."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visiondrive.evaluation.data import ROOT, validate, write_json

if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--dataset', type=Path, default=ROOT / 'datasets/indoor-person-evaluation')
    args = cli.parse_args()
    report = validate(args.dataset)
    write_json(args.dataset / 'dataset_validation.json', report)
    print(report)
    raise SystemExit(0 if report['valid'] else 1)
