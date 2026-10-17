# Third-party notices and provenance

This repository contains documentation, application/tooling code, tests, artwork, and the compact fine-tuned bottle/cup/box checkpoint. External source, initialization checkpoints, datasets, and full training runs remain ignored. Links identify dependencies; they do not transfer ownership to VisionDrive.

## Upstream components

| Component | Intended use | Source / license reference |
| --- | --- | --- |
| ESP-Detection / ESPDet-Pico | Detector architecture, custom modules, transfer-learning initialization, export workflow | [Repository](https://github.com/espressif/esp-detection), [upstream license](https://github.com/espressif/esp-detection/blob/main/LICENSE) |
| ESP-DL | Future ESP32-S3 inference | [Repository and applicable notices](https://github.com/espressif/esp-dl) |
| ESP-IDF | Future firmware and board integration | [Repository and applicable notices](https://github.com/espressif/esp-idf) |
| PyTorch and torchvision | Training and desktop inference | [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision) |
| Ultralytics / YOLO11n | Pretrained laptop detector and API used by ESP-Detection | [Repository and licensing information](https://github.com/ultralytics/ultralytics) |
| Depth Anything V2 Metric Indoor Small | Pretrained laptop depth model; Apache-2.0 | [Model and license](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf) |
| Hugging Face Transformers | Depth preprocessing and model implementation | [Repository and license](https://github.com/huggingface/transformers) |
| Arduino-ESP32 | Wi-Fi motor receiver firmware platform | [Repository and license](https://github.com/espressif/arduino-esp32) |
| OpenCV | Image capture, video, and presentation overlay | [Repository and license](https://github.com/opencv/opencv) |
| Open Images | Initial training and evaluation image sources | [Download and attribution information](https://storage.googleapis.com/openimages/web/download_v7.html) |
| Shields.io | Remote README status badges | [Service](https://shields.io/) |

ESP-Detection identifies its upstream license as AGPL-3.0. Retain its applicable notices and review the actual pinned dependencies and licenses when introducing code or distributing a trained application. Do not label third-party code or weights as an original VisionDrive model architecture.

## Provenance records

For each source dependency, record repository URL, exact revision, dependency versions, license files, and any local modifications. For a checkpoint, record its download URL, SHA-256 hash, original task/classes, source license, and fine-tuning configuration. Do not infer weight usage terms solely from the repository name.

For every dataset image, retain:

- Dataset version, image ID, original source URL, author and license metadata where supplied.
- Original object labels, bounding boxes, and the mapping to `obstacle`.
- Split assignment and any annotation edits.
- Retrieval date and file hash for reproducibility.

Dataset annotation terms and individual image terms can differ. Preserve attribution and validate redistribution rights before publishing images or presentation recordings. Prefer manifests and download instructions over committing the dataset.

## Model setup provenance

`scripts/setup_model.py` pins ESP-Detection revision `28de5902a357d99dd9a5db0f52905255641c5a78`. The upstream 224 × 224 cat checkpoint has SHA-256 `7afca1564f45c3087e39758049412337ff4cec6f63a8f7a5380d50999e4924c5`; setup calculates and records its actual hash. It is an initialization/smoke-test model, not a bottle/cup/box detector. The adapter uses the upstream registration interface without copying its model implementation into this repository.

## Packaged fine-tuned model

`weights/laptop/yolo11n-bottle-cup-box.pt` is the selected YOLO11n derivative
fine-tuned for bottle, cup, and box. Its provenance is recorded in the adjacent
JSON file; training/evaluation settings and measured results are in the YOLO11
report. It retains Ultralytics' AGPL-3.0 terms, with the license text alongside
the checkpoint. It is separate from the original ESPDet checkpoint below.

`weights/obstacle/best.pt` is the selected three-class fine-tuned checkpoint. Its class mapping, initialization hash, dataset/split summary, validation-selected confidence, measured results, and artifact hash are recorded in `weights/obstacle/model.json`. The upstream AGPL-3.0 license is preserved as `weights/obstacle/LICENSE`. The upstream architecture and initialization are not original VisionDrive inventions.

The model report describes the limited public-data run and weak box performance. No raw dataset images or original pretrained weights are bundled in Git. The original cat checkpoint is training initialization only and is rejected by the active demo.

## Project licensing

No project-wide redistribution license is declared. This file is an attribution inventory, not a replacement for an upstream license. Keep third-party notices and the packaged model licenses intact.

## Laptop models and sample images

`scripts/setup_laptop_models.py` downloads YOLO11n from the official Ultralytics
v8.3.0 assets release and metric indoor depth from Hugging Face revision
`8078d68a9c75a972131914f6afd0c1723be0da7f`. Both checkpoints are SHA-256 checked.
Downloaded weights/configuration and their generated source manifest remain
ignored in `weights/laptop/`. The YOLO checkpoint retains upstream AGPL-3.0
terms; the small depth model is Apache-2.0. These are upstream pretrained models,
not trained from scratch by the group.

The actual model-output screenshots in `assets/laptop-demo.png` and
`assets/unknown-obstacle-demo.png` include modified public images. Authors,
original image URLs, CC BY 2.0 license links, source IDs, and modifications
are recorded in [sample-result attribution](docs/sample-results.md).
The group's fixed dataset annotations/splits and individual attribution records
are published in [the manifest](data/public/bottle-cup-box-manifest.jsonl).
The expanded 435-image YOLO fine-tuning dataset has a separate
[version 2 manifest](data/public/yolo-bottle-cup-box-v2-manifest.jsonl), including
33 additional Open Images box photos assigned only to project training. Each
photo retains its author, original URL, license, and content hash.

`assets/dataset-preview.png` is a contact sheet of nine annotated public images.
Its author/source/license records are in [preview attribution](data/public/preview-sources.json).
The dataset release includes all original image files together with their
per-image attribution and CC BY 2.0 links. Preview images are resized and overlaid
with annotation labels; source dataset image bytes are unchanged.
