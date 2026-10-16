# Third-party notices and provenance

This milestone contains original project documentation and a concept illustration. It does not bundle third-party source code, checkpoints, or datasets. Links identify planned dependencies; they do not transfer their ownership to VisionDrive.

## Planned upstream components

| Component | Intended use | Source / license reference |
| --- | --- | --- |
| ESP-Detection / ESPDet-Pico | Detector architecture, custom modules, transfer-learning initialization, export workflow | [Repository](https://github.com/espressif/esp-detection), [upstream license](https://github.com/espressif/esp-detection/blob/main/LICENSE) |
| ESP-DL | Future ESP32-S3 inference | [Repository and applicable notices](https://github.com/espressif/esp-dl) |
| ESP-IDF | Future firmware and board integration | [Repository and applicable notices](https://github.com/espressif/esp-idf) |
| PyTorch and torchvision | Training and desktop inference | [PyTorch](https://github.com/pytorch/pytorch), [torchvision](https://github.com/pytorch/vision) |
| Ultralytics | API used by ESP-Detection | [Repository and licensing information](https://github.com/ultralytics/ultralytics) |
| OpenCV | Image capture, video, and presentation overlay | [Repository and license](https://github.com/opencv/opencv) |
| Open Images | Initial training and evaluation image sources | [Download and attribution information](https://storage.googleapis.com/openimages/web/download_v7.html) |
| Shields.io | Remote README status badges | [Service](https://shields.io/) |

ESP-Detection identifies its upstream license as AGPL-3.0. Retain its applicable notices and review the actual pinned dependencies and licenses when introducing code or distributing a trained application. Do not label third-party code or weights as an original VisionDrive model architecture.

## Required records when implementation begins

For each source dependency, record repository URL, exact revision, dependency versions, license files, and any local modifications. For a checkpoint, record its download URL, SHA-256 hash, original task/classes, source license, and fine-tuning configuration. Do not infer weight usage terms solely from the repository name.

For every dataset image, retain:

- Dataset version, image ID, original source URL, author and license metadata where supplied.
- Original object labels, bounding boxes, and the mapping to `obstacle`.
- Split assignment and any annotation edits.
- Retrieval date and file hash for reproducibility.

Dataset annotation terms and individual image terms can differ. Preserve attribution and validate redistribution rights before publishing images or presentation recordings. Prefer manifests and download instructions over committing the dataset.

## Project licensing

No project-wide redistribution license is declared in this documentation milestone. This file is an attribution inventory, not a replacement for an upstream license. Add a project license deliberately when the implementation and incorporated dependencies are known, and keep third-party notices intact.
