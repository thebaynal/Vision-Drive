# Verification report

Verification date: **2026-10-16**. This report distinguishes executable checks from outstanding model and hardware evaluation.

## Desktop controller and presentation

- Environment: Windows, Python 3.13.5, NumPy 2.2.6, OpenCV 4.12.0, pytest 8.4.2.
- **55 tests passed** in the recorded local run. The suite covers configuration, geometry/rule priority, near-area boundaries, overlapping boxes, invalid detections, duplicate frames, stabilization, stale timestamps, paused/in-flight work, source failure/EOF, dataset conversion/leakage, adapter class checks, CLI behavior, and the model-driven explanation view.
- Simulation smoke checks exercise the real renderer and MP4 writer, with matching decision/timestamp JSONL records. These use explicitly synthetic boxes and do not test a neural network.
- The repository preview image is produced by the application in simulation mode, not a mockup or detection-accuracy example.
- Source compilation is checked separately with `python -m compileall -q demo.py visiondrive scripts`.

Run the lightweight checks with:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest -q
```

## ESP-Detection provenance

- Checkout: `28de5902a357d99dd9a5db0f52905255641c5a78`.
- Published 224 × 224 cat checkpoint SHA-256: `7afca1564f45c3087e39758049412337ff4cec6f63a8f7a5380d50999e4924c5`.
- Real-model validation must use the pinned compatibility environment, independently of synthetic/unit tests. See the run guide for the single-image smoke check and explicit cat test mode.

### Actual model smoke result

Passed on Windows / Python 3.13.5 with PyTorch 2.12.0 CPU, torchvision 0.27.0, Ultralytics 8.3.112, OpenCV 4.12.0.88, and NumPy 2.2.6. The isolated `.venv-inference` reused existing system packages; it did not replace them.

On the upstream reference cat image, the actual ESPDet-Pico checkpoint returned one accepted detection with confidence approximately **0.887**. The restored normalized box was approximately `[0.504, 0.148, 1.000, 0.829]`; the real controller selected **STOP / Near-center obstacle**. The rendered output was inspected locally. The upstream photo is not copied into this repository's published artwork.

Single-image cold inference took approximately **10.4 seconds** in that run. This includes first-prediction work and is not steady-state FPS. It does not establish real-time webcam performance. The one-second presentation freshness guard would reject a result this old in continuous operation; the still-image inspection tool uses a longer timeout so the prediction can be examined.

A separate same-image repetition check using two CPU threads measured approximately 9.3 seconds for the first prediction and **69.8 ms / 75.9 ms** for the next two. These two warm samples are not an end-to-end throughput benchmark. The application now defaults to two CPU threads and completes an explicit warmup before opening the source. Changing thread count requires remeasurement.

The real desktop pipeline was then exercised headlessly using a four-second, 15 FPS clip made by repeating the upstream reference image. It recorded **120 display frames**, with detections from **37 distinct source frames**, and ended with **STOP / VIDEO ENDED**. Median capture-to-result latency across those distinct results was approximately **131.9 ms**. The annotated MP4 and matching JSONL were verified locally. This integration check validates actual model/video/recording plumbing; a repeated still image is not representative obstacle-course footage or an accuracy test.

The separate `logic_demo.py` presentation was also checked against that real-model video path: **60 display frames** were recorded, **55** carrying detections. All records identified actual inference (`simulated: false`) and the logic presentation. The annotated snapshot was inspected for the four stages: model detections, region overlap, rule candidate, and final instruction. Default startup selects the explicitly labeled cat model, not simulation; tests also verify that an explicitly supplied trained model is preserved.

The source checkpoint serialized its class name as `0`. The adapter permits a `cat` alias only for the recorded SHA-256 and an explicit `--expected-label cat`. It never aliases numeric-class weights to `obstacle`.

## Not yet established

- A trained and evaluated bottle/cup/box checkpoint.
- Public-data download/curation at the intended full dataset size.
- Per-original-object-type recall and aggregate accuracy on a held-out obstacle dataset.
- Physical webcam permission/device behavior on the presenter's machine.
- A measured desktop processed-FPS target on a trained obstacle model.
- ONNX/INT8 accuracy parity, ESP32 memory/latency, wiring, motor control, or physical-car stopping behavior.

No simulated FPS, synthetic-rule result, or upstream published benchmark should be presented as evidence for these outstanding items.
