# Verification report

Recorded on **2026-10-16** for the bottle/cup/box prototype.

## Automated checks

**69 tests passed** on Windows / Python 3.13.5. Coverage includes region overlap, rule priority, near-stop boundaries, invalid boxes, unsupported categories, confirmation, freshness, pause/in-flight results, capture failures/EOF, data conversion/leakage, class-name preservation, CLI rendering/recording, evaluation matching, notebook code syntax, and packaged-model hash/config consistency.

Source compilation, Markdown/local-link checks, and whitespace checks were also performed. GitHub CI is configured for lightweight tests; a local pass is not a claim that a remote workflow has already completed.

## Actual training and evaluation

The initial run trained ESPDet-Pico on 280 images, selected weights using 60 validation images, and evaluated 62 untouched project test images. It completed 55 epochs with early stopping, selecting epoch 40. Confidence 0.35 was selected on validation macro F1 only.

The active classes are exactly bottle, cup, and box. Other category labels are rejected by the application. The source checkpoint is used for training initialization only; there is no cat fallback.

See [the model report](model-report.md), [machine-readable results](model-evaluation.json), and [model provenance](../weights/obstacle/model.json). Box recall is weak in the small held-out set and must not be presented as reliable obstacle detection.

## Notebook

All code cells in [the notebook](../notebooks/train_bottle_cup_box.ipynb) were executed in order in the prepared model environment with downloads/retraining disabled. This exercised the existing artifact/report inspection, training-history plot, and an actual model prediction. Code syntax and notebook structure are also tested. Outputs are cleared in the committed notebook for a clean, portable starting point.

## Desktop pipeline

The actual packaged model was exercised through `run_logic_demo.cmd` using a six-second clip assembled from real held-out object images. The recording contained **180 display frames**, detections labeled **bottle, cup, and box**, and ended with **STOP / VIDEO ENDED**. The log identified actual inference, not simulation. This verifies integration, not camera-domain accuracy.

Simulation remains an explicit separate mode for testing the controller and rendering. Its scenarios do not expand the model's classes or measure neural-network accuracy.

## Environment and outstanding work

The model checks used Windows / Python 3.13.5, PyTorch 2.12.0 CPU, torchvision 0.27.0, Ultralytics 8.3.112, NumPy 2.2.6, and OpenCV 4.12.0.88. The local model environment reuses existing system packages; it is not shipped in Git. A fresh installation uses the declared requirements and explicit upstream setup.

Still outstanding: larger representative data, stronger box performance, independent webcam/course testing, INT8 conversion/parity, ESP32 memory/latency, and physical-car behavior. Public-image inference timing is not end-to-end camera FPS or an embedded benchmark.
