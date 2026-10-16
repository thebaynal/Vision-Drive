"""Explicit ESP-Detection adapter. No implicit downloads or model substitution."""

import importlib
import inspect
import os
from pathlib import Path
import sys
from .decision import Detection


def register_esp(repo, cpu_threads=None):
    repo = Path(repo).resolve()
    if not (repo / "nn" / "esp_tasks.py").is_file():
        raise ValueError(f"ESP-Detection not found at {repo}. Run python scripts/setup_model.py first.")
    # Registration follows the upstream API; all heavy imports stay optional.
    sys.path.insert(0, str(repo))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    cache = Path(__file__).resolve().parents[1] / ".cache"
    for variable, folder in (("YOLO_CONFIG_DIR", "ultralytics"), ("MPLCONFIGDIR", "matplotlib")):
        directory = cache / folder
        directory.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault(variable, str(directory))
    try:
        custom = importlib.import_module("nn.esp_tasks")
        tasks = importlib.import_module("ultralytics.nn.tasks")
        tasks.parse_model = custom.custom_parse_model
        # The pinned ESP head passes reg_max=1 to a newer Detect constructor,
        # while Ultralytics 8.3.112 has the older (nc, ch) API. The ESP head
        # replaces the regression branch itself, so discard only that known
        # argument for this subclass and preserve ordinary Detect behavior.
        detect = importlib.import_module("ultralytics.nn.modules.head").Detect
        original = detect.__init__
        if "reg_max" not in inspect.signature(original).parameters and not getattr(original, "_visiondrive_compat", False):
            def compatible_init(instance, *args, reg_max=None, **kwargs):
                if reg_max is not None and (reg_max != 1 or not type(instance).__module__.startswith("nn.modules")):
                    raise TypeError("Unsupported reg_max for this Ultralytics constructor")
                original(instance, *args, **kwargs)
            compatible_init._visiondrive_compat = True
            detect.__init__ = compatible_init
        if cpu_threads is not None:
            importlib.import_module("torch").set_num_threads(cpu_threads)
        return importlib.import_module("ultralytics").YOLO
    except ImportError as exc:
        raise RuntimeError("Model dependencies are unavailable. Install requirements-model.txt; see the run guide for tested versions.") from exc


class ESPDetector:
    def __init__(self, weights, repo, config, device="cpu", expected_label="obstacle", cpu_threads=2):
        weights = Path(weights).resolve()
        if not weights.is_file():
            raise ValueError(f"Checkpoint not found: {weights}.\n"
                             "The bottle/cup/box checkpoint requires training with scripts/train_model.py. "
                             "No replacement model was loaded.")
        if weights.suffix != ".pt":
            raise ValueError("Desktop ESPDet inference requires a trusted .pt checkpoint")
        if type(cpu_threads) is not int or cpu_threads < 1:
            raise ValueError("CPU thread count must be a positive integer")
        YOLO = register_esp(repo, cpu_threads=cpu_threads if device == "cpu" else None)
        self.model = YOLO(str(weights), task="detect")
        if not any(type(layer).__module__.startswith("nn.") for layer in self.model.model.modules()):
            raise ValueError("Checkpoint does not contain the ESP-Detection custom architecture")
        names = self.model.names
        names = list(names.values()) if isinstance(names, dict) else list(names)
        if names != ["bottle", "cup", "box"] or expected_label == "cat":
            raise ValueError(f"Unsupported classes {names}. Only the bottle/cup/box checkpoint is supported. "
                             "Cat, person, chair, and wall checkpoints are disabled.")
        self.names = names
        self.label = "objects"
        self.name = f"ESPDet-Pico / {', '.join(names)} / {device}"
        self.config, self.device = config, device

    def warmup(self):
        import numpy as np
        self.predict(np.zeros((224, 224, 3), dtype=np.uint8))

    def predict(self, image):
        result = self.model.predict(source=image, imgsz=224, rect=False,
                                    conf=self.config.confidence_threshold,
                                    iou=self.config.nms_iou_threshold,
                                    device=self.device, verbose=False, save=False)[0]
        # Ultralytics returns xyxyn in the ORIGINAL image coordinates, already
        # undoing the prediction letterbox. Do not inverse-transform it again.
        if result.boxes is None:
            return ()
        boxes = result.boxes.xyxyn.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        return tuple(Detection(*(max(0.0, min(1.0, v)) for v in box), float(score), int(label), self.names[int(label)])
                     for box, score, label in zip(boxes, scores, classes))
