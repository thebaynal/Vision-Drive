# Run VisionDrive

The default viewer now uses **laptop YOLO11n + metric depth**, with optional
ESP32 Wi-Fi motor commands. Follow [the current setup guide](laptop-esp32.md)
for installation, calibration, notebook use, and receiver firmware.

```powershell
.\run_logic_demo.cmd
.\run_logic_demo.cmd --source path/to/video.mp4
.\run_logic_demo.cmd --avoid-distance 1.5 --stop-distance 0.45
```

For the original three-class ESPDet experiment:

```powershell
.venv-inference\Scripts\python -m pip install -r requirements-model.txt
.venv-inference\Scripts\python scripts/setup_model.py --source-only
.\run_logic_demo.cmd --detector esp
```

To reproduce its training, use [the training notebook](../notebooks/train_bottle_cup_box.ipynb).
The [model report](model-report.md) records its dataset and measured results.
That checkpoint has three trained categories; the current hybrid adds a broader
pretrained detector and category-independent depth cues.

For a synthetic controller-only preview:

```powershell
python -m pip install -r requirements.txt
python logic_demo.py --simulate
```

For an inference recording and decision log:

```powershell
.\run_logic_demo.cmd --source path/to/video.mp4 --save-video outputs/demo.mp4 --snapshot outputs/final.png
```

Output paths must be new. The MP4 is accompanied by JSONL records containing
source frames, timestamps, detections, estimated distances, region state,
instructions, model mode, and timings. Recordings and simulation never send
motor commands. Missing model files cause an actionable error instead of a
silent model substitution.
