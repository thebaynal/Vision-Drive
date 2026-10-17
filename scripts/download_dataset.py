"""Download/restore the frozen, licensed indoor evaluation cohort (no training)."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from visiondrive.evaluation.data import acquire

if __name__ == '__main__':
    acquire()
