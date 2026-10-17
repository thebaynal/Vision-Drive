"""Evaluate original YOLO11n; never trains or changes the checkpoint."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visiondrive.evaluation.runner import load_config, evaluate
from visiondrive.evaluation.data import ROOT

if __name__ == '__main__':
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--device', help='0 for CUDA, cpu for explicit CPU evaluation')
    cli.add_argument('--config',type=Path,default=ROOT/'config/evaluation.yaml')
    args = cli.parse_args()
    result = evaluate(load_config(args.config,device=args.device))
    print(result['fixed_operating_point'])
    print(result['native_ap'])
