"""Model-driven visualization: detections -> regions -> rule -> instruction.

Uses the trained bottle/cup/box checkpoint. There is no cat-model fallback.
"""

import sys
from pathlib import Path
from demo import main as demo_main

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print("Logic visualization uses REAL bottle/cup/box inference; other categories are excluded.")
        print("Trained obstacles: python logic_demo.py --weights path/to/best.pt --expected-label obstacle\n")
    return demo_main(["--explain", *args])


if __name__ == "__main__":
    raise SystemExit(main())
