# Run the desktop demo

The controller, OpenCV interface, webcam/video pipeline, ESPDet adapter, and data/training utilities are implemented. There are two deliberately distinct modes:

| Mode | What it demonstrates | Needs a model? |
| --- | --- | --- |
| `--simulate` | Synthetic obstacles and the real region/decision code | No |
| Default ESPDet mode | Actual neural-network predictions on your webcam/video | Yes; matching ESPDet checkpoint |

**Simulation is not object detection.** The banner and recordings label it explicitly. The upstream cat checkpoint is usable for a real-model smoke check, but bottles/cups/boxes require training. No bottle/cup/box checkpoint is bundled.

## 1. Quick preview, no ML installation

Use Python 3.10 or newer. From the repository root:

```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python demo.py --simulate
```

Space pauses/resumes; Q or Escape quits. The scenario cycles every three seconds through a clear center, center obstacle, side choices, all blocked, a near-center hazard, and a wide obstacle. The unmirrored image uses physical image-left and image-right consistently.

Record the preview explicitly:

```powershell
.venv\Scripts\python demo.py --simulate --save-video outputs/controller-demo.mp4
```

An annotated MP4 and matching `.jsonl` timestamp/decision log are written. Existing outputs are not overwritten. The recording is nominal 30 FPS; consult timestamps for actual wall-time behavior. No footage is saved unless you request recording or a snapshot.

For an automated, windowless preview:

```powershell
.venv\Scripts\python demo.py --simulate --headless --max-frames 720 --save-video outputs/scenarios.mp4
```

Headless simulation advances its scenario clock by 1/30 second per frame; processing may run faster than real time. Its scenario FPS is not a neural-inference benchmark.

## 2. Set up actual ESPDet inference

Use a separate environment for model dependencies. The supplied direct versions were exercised on **Windows / Python 3.13.5**: PyTorch 2.12.0 (CPU), torchvision 0.27.0, Ultralytics 8.3.112, NumPy 2.2.6, and OpenCV 4.12.0.88. This is a tested direct-version list, not a complete transitive lockfile or proof of compatibility on every platform.

For a clean setup:

```powershell
python -m venv .venv-model
.venv-model\Scripts\python -m pip install -r requirements-model.txt
.venv-model\Scripts\python scripts/setup_model.py
```

`setup_model.py` downloads the public ESP-Detection checkout at commit `28de5902a357d99dd9a5db0f52905255641c5a78`, verifies its revision, and copies its published cat checkpoint into `weights/pretrained/`. It records provenance and the checkpoint hash alongside the weights. Downloads happen only during explicit setup, not inference. Git must be installed.

On the current development computer, `.venv-inference` has already been prepared for smoke testing. It reuses the installed PyTorch/OpenCV/NumPy packages and keeps the additional model packages isolated. You can use `.venv-inference\Scripts\python` for the commands below on this computer; that environment is ignored by Git and is not a portable distribution.

The script does not modify an existing checkout at an unexpected revision. Use a clean location rather than discarding unrelated changes. Read upstream instructions before making any modifications there.

Only load trusted `.pt` checkpoints: PyTorch model files can contain executable serialized content. The loader registers upstream custom classes and confirms that the checkpoint uses an ESP architecture with the requested class name.

### Real model smoke check, no webcam needed

```powershell
.venv-model\Scripts\python scripts/check_model.py --image external/esp-detection/examples/cat_detection/cat.jpg --weights weights/pretrained/espdet_pico_224_224_cat.pt --expected-label cat --output outputs/cat-check.png
```

This runs actual cat inference and saves an annotated image/report. It is not bottle/cup/box detection, and the single-image timing includes cold-start work. A cat picture on a screen or webcam is an optional way to show the actual model path before training your own objects; recognition of a picture does not validate physical obstacle behavior.

For an explicit upstream cat webcam test:

```powershell
.venv-model\Scripts\python demo.py --source 0 --weights weights/pretrained/espdet_pico_224_224_cat.pt --expected-label cat
```

The default class is `obstacle`. A cat checkpoint passed without `--expected-label cat` is rejected, preventing it from being mislabeled as a general obstacle model.

The pinned upstream cat checkpoint stores its class name as `0`. The adapter recognizes that exact SHA-256 only when you explicitly select `--expected-label cat`; numeric labels on other checkpoints are rejected rather than assumed to mean obstacle.

## 3. Prepare a training dataset

The converter accepts reviewed JSONL manifests. Each line describes one image, its split, all bottle/cup/box objects, and provenance. Paths are relative to the manifest:

```json
{"image":"images/bottle.jpg","split":"train","objects":[{"label":"bottle","xyxy":[0.2,0.3,0.6,0.9]}],"source":{"url":"https://example.org/original-image","license":"actual source license","author":"actual author"}}
```

Use normalized `x1,y1,x2,y2` in the original image. Original labels must be `bottle`, `cup`, or `box`; all become class `0: obstacle`. A background image needs `objects: []` and `verified_background: true`. Supply nonempty `train`, `val`, and `test` splits. Duplicate image content is rejected, including across splits.

```powershell
.venv\Scripts\python scripts/prepare_dataset.py --manifests data/raw/train.jsonl data/raw/val.jsonl data/raw/test.jsonl --output datasets/obstacles
```

The output contains YOLO images/labels, `dataset.yaml`, and a provenance manifest. Use a new output directory for each build to avoid stale labels. Inspect missing labels before training; metadata validation cannot discover an unannotated object by itself.

### Public Open Images selection

Download class descriptions, a split's bounding-box annotations, and its image metadata using the [official download page](https://storage.googleapis.com/openimages/web/download_v7.html). Place these CSV files under ignored `data/raw/`. Large metadata/annotation downloads are deliberately not automatic.

```powershell
.venv\Scripts\python scripts/prepare_openimages.py --classes data/raw/classes.csv --annotations data/raw/train-boxes.csv --metadata data/raw/train-images.csv --split train --limit 1500 --output data/raw/train.jsonl --download
```

Repeat with the validation and test metadata, `--split val --limit 300` and `--split test --limit 300`, and separate output manifests. Without `--download`, selected images must already exist under `data/raw/images/<source-split>/`.

The selector audits the exact source classes, defaults to `Bottle`, `Coffee cup`, and `Box`, and fails if they are absent. Use `--class-names "actual bottle class" "actual cup class" "actual box class"` only after checking semantic equivalence. If Open Images does not have a suitable box class, manually annotate public box images in the manifest format; do not relabel an unrelated class to meet the quota. The converter also accepts fully manual manifests, so no particular public class taxonomy is required.

Selection is deterministic and approximately balanced. It preserves source splits, excludes group-of/depiction annotations, retains original labels, and reports actual counts. Metadata must contain `ImageID`, `OriginalURL`, and `License`; author metadata is retained where present. Image-specific usage terms still apply.

## 4. Fine-tune and evaluate

```powershell
.venv-model\Scripts\python scripts/train_model.py --data datasets/obstacles/dataset.yaml --device cpu
```

For an available CUDA runtime/GPU, supply `--device 0`. The supplied PyPI installation may use a CPU build; consult [PyTorch's official installer](https://pytorch.org/get-started/locally/) for a matching CUDA build if required. Check CUDA availability instead of assuming that an NVIDIA GPU makes the current wheel GPU-enabled.

Defaults: 224 × 224, up to 100 epochs, batch 16, seed 42, patience 20, one class. If memory is insufficient, explicitly retry with `--batch 8` or `--batch 4`. Training logs, the source hash, upstream revision, and settings are saved with the run. The terminal prints the actual best-checkpoint location.

Evaluate the trained checkpoint on the untouched test split:

```powershell
.venv-model\Scripts\python scripts/train_model.py --evaluate --data datasets/obstacles/dataset.yaml --weights runs/obstacle/weights/best.pt
```

This reports aggregate detector metrics. Per-original-type recall analysis and independent presentation-footage evaluation remain required before claiming bottle/cup/box performance; no accuracy results are bundled.

## 5. Run your trained obstacle model

```powershell
.venv-model\Scripts\python demo.py --source 0 --weights runs/obstacle/weights/best.pt
.venv-model\Scripts\python demo.py --source recordings/course.mp4 --weights runs/obstacle/weights/best.pt --config config/default.json
```

The model must expose exactly one class named `obstacle`. Default device is CPU; `--device 0` explicitly selects an available CUDA device. You can also place your trusted trained checkpoint at `weights/obstacle/best.pt` to use the default path.

CPU prediction defaults to two PyTorch threads; use `--cpu-threads N` to tune it. The model warms up before the source is opened, so cold initialization does not consume the start of a short video. The first warmup can take several seconds; no camera frames are being processed during that startup stage.

Video playback follows its source FPS, and the inference worker uses a latest-frame slot. Slow inference drops frames rather than building an old-frame backlog. Camera failures, inference errors, a paused input, completed video, or stale observations display STOP. At video completion, the window stays open for review until you quit; headless mode exits.

The display remains responsive independently of model work. Every movement suggestion needs three distinct matching results, while STOP is immediate. These are presentation rules; follow the hardware guide before treating them as physical-car controls.

## 6. Configure and test

### Model-to-instruction visualization

`logic_demo.py` is a separate launch command for a model-driven explanation view. It uses the same capture worker, detector, and controller as `demo.py`; only the presentation changes. It does not substitute manually positioned or synthetic obstacles for the model by default.

On this computer:

```powershell
.\run_logic_demo.cmd
.venv-inference\Scripts\python logic_demo.py
.venv-inference\Scripts\python logic_demo.py --source recordings/course.mp4
```

On a clean machine, follow model setup above and use `.venv-model\Scripts\python`. Setup downloads the explicitly identified upstream cat checkpoint used by this entry point's default. A webcam example using that checkpoint only recognizes cats; this is not a general-obstacle detector.

To explain your future trained obstacle model:

```powershell
.venv-model\Scripts\python logic_demo.py --source 0 --weights runs/obstacle/weights/best.pt --expected-label obstacle
```

The side panel traces four stages:

1. **Model detections:** accepted box count, class, and confidence threshold.
2. **Box/region overlap:** LEFT, MIDDLE, RIGHT occupancy and the near-center condition.
3. **Rule candidate:** the instruction suggested by the current valid observation.
4. **Final instruction:** the controller's actual output after input freshness and confirmation rules.

For example, a center obstacle can propose LEFT while the final output remains STOP during the first two confirming observations. Pause or stale input gates movement and is shown explicitly. The overlay remains a visualization of instructions, not physical motion.

The same presentation is available through `demo.py --explain` with its normal default obstacle-weight path. Recording, snapshots, pause, headless runs, and source options are shared. `--simulate` remains available only as an explicit synthetic layout test; it is never enabled automatically by `logic_demo.py`.

`config/default.json` contains validated region/threshold defaults. Unknown keys, non-finite values, invalid boundaries, and invalid streak/timeouts are rejected. Tune thresholds on validation footage, not the held-out test set. JSON has no comments.

```powershell
.venv\Scripts\python -m pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest -q
```

Tests cover region rules, overlaps, immediate stops, timestamp expiry, duplicate-frame rejection, pause behavior, worker isolation, data leakage checks, and simulation recording. Real-model checks require the separate model environment; they do not run in the lightweight CI job.

Troubleshooting:

- **Missing checkpoint:** run simulation, explicitly smoke-test cat weights, or train your obstacle model. No substitute model is silently downloaded.
- **Error mentions `weights/obstacle/best.pt`:** that is `demo.py`'s untrained custom-model default. For the existing real-model logic viewer, run `.\run_logic_demo.cmd`. It calls `logic_demo.py`, which explicitly uses the available pretrained cat checkpoint. It does not require a custom obstacle checkpoint.
- **Missing model packages:** install `requirements-model.txt` with a compatible interpreter.
- **Wrong class:** use the correct checkpoint; `--expected-label cat` is for a deliberate cat test only.
- **Camera unavailable:** close other camera applications, check Windows permissions, and try another numeric index.
- **STOP / stale input:** inspect displayed age/latency. Lowering the limit makes stopping stricter; increasing it does not improve model speed.
- **Existing output:** choose a new recording/snapshot filename; the app avoids overwriting previous work.
- **Non-GUI environment:** use `--headless` with a video, or with `--max-frames` for simulation/camera.

## 7. Current verification boundaries

The automated tests exercise actual rendering and video I/O with synthetic scenarios. Those tests do not establish trained-obstacle accuracy or physical driving performance. Record real-model environment and smoke-test results separately, then use the [hardware guide](hardware-implementation.md) when the camera board and chassis are available.
