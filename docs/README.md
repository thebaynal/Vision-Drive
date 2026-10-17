# Project materials

Start with the [project README](../README.md) for group members, specialization,
architecture, installation, and the live demonstration.

## Submission materials

| Material | File |
| --- | --- |
| Midterm requirements and evidence | [Midterm report](midterm-report.md) |
| Current dataset, provenance, and exact restore command | [Dataset card](../data/public/README.md) |
| YOLO11 fine-tuning settings and measured test results | [Fine-tuning report](yolo11-finetuning-report.md) |
| Training notebook | [YOLO11 notebook](../notebooks/train_yolo11.ipynb) |
| Dataset/training explanation to read aloud | [Presentation script](dataset-and-training-presentation-script.md) |
| Methodology and results text for the paper | [Paper section](paper-dataset-and-training.md) |
| Guidance for group documentation writers | [Writing guide](documentation-and-paper-guide.md) |
| Setup, recording, calibration, and ESP32 commands | [Run guide](laptop-esp32.md) |
| Hardware implementation handoff | [Hardware guide](hardware-implementation.md) |
| Actual inference examples and image attribution | [Sample results](sample-results.md) |
| Software tests and integration scope | [Verification report](verification.md) |
| Independent camera/course evaluation protocol | [Evaluation guide](evaluation-guide.md) |
| Third-party licenses and source attribution | [Notices](../THIRD_PARTY_NOTICES.md) |

## Comparison experiments

- [Unchanged pretrained YOLO11n](yolo11-evaluation.md): exploratory bottle/cup
  scores, with no generic box category.
- [ESPDet-Pico](model-report.md): earlier three-class baseline for future onboard
  ESP32 work. Its model, notebook, training history, and provenance are retained
  so earlier results remain reproducible. It is separate from the YOLO model.

Recordings, caches, raw image downloads, virtual environments, and intermediate
training checkpoints are excluded from Git. Selected model checkpoints and
compact evidence files belong in the submission; generated local work does not.
