"""Model-driven visualization: detections -> regions -> rule -> instruction.

Uses pretrained YOLO11n, with optional metric depth.
"""

import sys
from demo import main as demo_main


def main(argv=None):
    args = list(sys.argv[1:] if argv is None else argv)
    if "--help" in args or "-h" in args:
        print("Logic visualization uses pretrained YOLO11n and its 80 COCO categories.")
        print("Object-only view: python logic_demo.py --detector coco\n")
    return demo_main(["--explain", *args])


if __name__ == "__main__":
    raise SystemExit(main())
