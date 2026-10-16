# Run and reproduce VisionDrive

The active detector recognizes exactly **bottle, cup, and box**. Cats, people, chairs, and walls are excluded. The application visualizes movement suggestions; it does not drive hardware. The first trained model has weak box performance; inspect the [model report](model-report.md).

## 1. Start the model-driven visualization

On the development Windows computer:

```powershell
.\run_logic_demo.cmd
```

The launcher selects the prepared `.venv-inference` environment and opens camera index `0`. The view shows **real detections → region overlap → rule candidate → final instruction**. There is no cat-model fallback. Space pauses/resumes; Q/Escape quits.

```powershell
.\run_logic_demo.cmd --source recordings/course.mp4
```

### Fresh clone setup

The recorded local run used Python 3.13.5. Use an isolated environment with the tested direct versions in `requirements-model.txt`:

```powershell
python -m venv .venv-model
.venv-model\Scripts\python -m pip install -r requirements-model.txt
.venv-model\Scripts\python scripts/setup_model.py --source-only
.venv-model\Scripts\python logic_demo.py
```

The selected checkpoint is `weights/obstacle/best.pt`, with provenance and upstream license alongside it. External source downloads during explicit setup; inference does not download a substitute model. Environments and datasets are not bundled in Git.

Setup pins Espressif revision `28de5902a357d99dd9a5db0f52905255641c5a78` and refuses to silently replace a checkout at another revision. Only load trusted PyTorch checkpoints.

## 2. Input and recording options

```powershell
.\run_logic_demo.cmd --source 1
.\run_logic_demo.cmd --source recordings/course.mp4 --save-video outputs/annotated.mp4
.\run_logic_demo.cmd --config config/default.json
```

`--source` accepts a camera index or local video. The display is unmirrored. Optional recording creates an annotated MP4 and JSONL log with frame IDs, detections, instructions, and timestamps. Outputs are not overwritten. `--snapshot outputs/frame.png` saves the final view.

`demo.py` uses the same detector/controller with a simpler display; add `--explain` for the logic view. Both load the trained checkpoint by default. Another checkpoint must have exact ordered classes `bottle`, `cup`, `box`.

CPU inference defaults to two threads; tune with `--cpu-threads N`. `--device 0` selects an available CUDA device. An NVIDIA GPU does not make a CPU-only PyTorch wheel CUDA-enabled. Model warmup completes before the source opens.

## 3. Understand the decisions

- The model produces names, scores, and boxes in original image coordinates.
- Boxes mark left/middle/right regions blocked by overlap. A wide box can block several regions.
- A near-center hazard or all-blocked result produces immediate STOP.
- With the middle blocked, a side without accepted boxes produces a turn suggestion; the both-sides tie chooses left.
- Movement requires three distinct matching observations. The candidate can differ from the final instruction during confirmation.
- Pause, stale input, capture/inference failure, and video completion produce STOP.

No accepted box does not establish traversable space. Misses and false positives remain possible, particularly for boxes in this initial run.

## 4. Reproduce training in Jupyter

Open [the notebook](../notebooks/train_bottle_cup_box.ipynb) with a kernel using the model environment. It covers setup, data, fine-tuning, curves, confidence selection, test evaluation, and genuine predictions.

Switches default to inspecting existing artifacts. Enable downloads/training/evaluation to reproduce them. To use the virtual environment as a kernel, install/register `ipykernel` in that environment and select it in Jupyter.

## 5. Reproduce training from scripts

```powershell
.venv-model\Scripts\python scripts/setup_model.py
.venv-model\Scripts\python scripts/build_object_dataset.py --per-class 250
```

The public source yielded 179 bottle images, 155 cup images, and 68 box images. The run produced 280 training, 60 validation, and 62 test images. The source is the official Open Images **validation** pool, repartitioned into **new project splits**; these are not official benchmark scores.

The builder preserves IDs, authors and license/source URLs, deduplicates before splitting, and keeps all selected target boxes. Group-of/depiction boxes are excluded. Original annotations may omit another target object in an image; inspect the data before drawing strong conclusions.

```powershell
.venv-model\Scripts\python scripts/train_model.py --data datasets/bottle-cup-box/dataset.yaml --epochs 60 --batch 16 --patience 15 --device cpu --output runs/objects-reproduction
```

The original checkpoint initializes transferable layers only; its classification is replaced with a three-class head. The script rebuilds from symbolic `nc` and verifies the head/dataset class count. The initial run completed 55 epochs with early stopping; epoch 40 supplied the best checkpoint. Lower batch size if memory is insufficient. Runs increment instead of overwriting; use the printed checkpoint path.

## 6. Evaluate and calibrate

Use the actual path printed by training:

```powershell
.venv-model\Scripts\python scripts/evaluate_objects.py --weights runs/objects-reproduction/weights/best.pt --output runs/reproduction-evaluation.json
.venv-model\Scripts\python scripts/train_model.py --evaluate --data datasets/bottle-cup-box/dataset.yaml --weights runs/objects-reproduction/weights/best.pt
```

The first script selects confidence on **validation macro F1**, then reports test precision/recall/F1 at IoU >= 0.5. The second reports held-out mAP. These are different metrics. Test images are not used to choose weights or confidence. Add `--calibrate-config config/default.json` to deliberately save the validation-selected threshold.

See [model results](model-report.md), [stored evaluation](model-evaluation.json), and [training history](training-results.csv). Public-image testing does not establish course/webcam or embedded performance.

### Your own images

`prepare_dataset.py` accepts JSONL manifests with local images, provenance, normalized boxes, and train/val/test membership. Use bottle/cup/box labels and `--multiclass`. Background images need empty objects and `verified_background: true`. Keep recording sessions in one split to prevent adjacent-frame leakage.

## 7. Tests and synthetic preview

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
python demo.py --simulate
```

Simulation is an explicit controller test, not inference. For a bounded headless run add `--headless --max-frames 120`. Actual inference remains the default for the logic viewer.

## Troubleshooting

- **Missing checkpoint:** restore the bundled `weights/obstacle/best.pt` or reproduce training. No cat fallback is loaded.
- **Missing model/source packages:** use the model interpreter, install requirements, and run source setup.
- **Unsupported classes:** require bottle/cup/box in that order.
- **Camera unavailable:** check permissions, close other camera apps, or try another index/video.
- **STOP / stale input:** inspect timing. Increasing the timeout does not improve model speed.
- **No detections:** use known targets under good lighting. Camera-specific data may be needed, especially for boxes.
- **Existing output:** choose a new recording/snapshot filename.

Follow the [hardware guide](hardware-implementation.md) after selecting and measuring components.
