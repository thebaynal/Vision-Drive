# Desktop implementation plan

> Status: implementation specification. The application, training scripts, configuration files, and tests described here have not been created in this milestone.

## 1. Outcome and boundaries

Build a local Windows desktop presentation that accepts webcam or recorded-video frames, detects supported obstacles with ESPDet-Pico, and displays stable movement suggestions derived from left/middle/right occupancy. The application must run offline after setup. It does not connect to motors, estimate metric depth, map a room, or navigate to a destination.

The initial obstacle set is bottles, cups, and boxes, merged into a single `obstacle` class. A label in the presentation must state: **Visual suggestion only — unrecognized obstacles may be missed.**

## 2. Environment and dependency management

1. Use an isolated Python 3.10 environment; do not modify the system Python installation.
2. Retrieve [ESP-Detection](https://github.com/espressif/esp-detection) as an external dependency under an ignored directory. Record the exact commit. Follow any instructions shipped with that checkout before changing its files.
3. Resolve and pin a compatible desktop set of PyTorch, torchvision, Ultralytics, OpenCV, and NumPy. Upstream requirements are a starting point; establish compatibility with an import, checkpoint-load, and inference smoke test before training.
4. Keep ONNX/ESP-PPQ/ESP-DL export dependencies in a separate later environment. Do not run the upstream all-in-one deployment script just to start the desktop demo.
5. Support CPU inference. Use CUDA for training when available; the development laptop has an RTX 3050 with 6 GB VRAM. Benchmark actual speed and memory, including preprocessing and rendering.
6. Record an environment lock or pinned requirements after validation, plus OS, GPU, driver, runtime, upstream commit, and Python version. Do not publish an untested requirements lock.

## 3. Data preparation

### Source and labels

Use a bounded public-image subset from [Open Images](https://storage.googleapis.com/openimages/web/download_v7.html). First audit its boxable class metadata and inspect examples; do not assume the words used in this project exactly match dataset labels.

Select bottle and cup equivalents. Include suitable box annotations when present; otherwise manually annotate public box images with retained source/license metadata. Do not substitute image-level labels for bounding boxes. Reject group-of, depiction, or ambiguous annotations that do not represent individual physical target objects for this demo.

Target 1,500 training, 300 validation, and 300 test images, balancing the three original object types as closely as available data permits. Keep original dataset split boundaries and deduplicate by image ID and content hash. If source counts cannot meet the target, report actual counts and the gap; do not reuse evaluation images as training data.

Convert the selected boxes to YOLO detection annotations: class ID `0`, followed by normalized center x, center y, width, and height. Clamp to valid image bounds, reject zero-area boxes, and retain all supported target boxes in each image. Merge labels to `obstacle`, retaining original type metadata for separate evaluation.

### Quality and provenance

- Preserve image IDs, original classes, source URLs, author/license metadata, hashes, annotation changes, and split membership in a manifest.
- Review a random sample from every object type and split, including small objects and occlusions.
- Audit missing target labels. Empty labels are allowed only for verified background images, not download or annotation failures.
- Keep images and generated datasets out of Git; publish preparation instructions and manifests only where terms permit.
- Independently record staged webcam/video scenes for evaluation, including an empty view, each object type, wide overlap, mixed objects, and near-area entry. Keep adjacent frames from the same recording session in one split if later used for fine-tuning.

## 4. Training and model verification

Start from the repository's published 224 × 224 ESPDet-Pico `.pt` checkpoint. Record its URL, hash, license reference, and source task. The documented cat model is not a pretrained general-obstacle detector. Verify that the custom ESP modules load and transferable weights initialize correctly; fail clearly rather than silently switching to training from scratch or a different architecture.

Initial training settings:

| Setting | Initial value |
| --- | --- |
| Input | 224 × 224, aspect-ratio-preserving preprocessing |
| Class count | 1: `obstacle` |
| Maximum epochs | 100 |
| Batch size | 16; reduce to 8, then 4 only if GPU memory is exhausted |
| Early stopping patience | 20 epochs |
| Seed | 42 |
| Model selection | Best validation mAP checkpoint |

Use the upstream training API with explicit overrides; its training defaults are not the configuration for this project. Save complete settings and logs. Report detection precision, recall, mAP@0.5, and mAP@0.5:0.95 on the held-out test set, and recall by original object type. Do not tune thresholds on the test set.

Evaluate staged presentation footage separately. If a bottle/cup/box is repeatedly missed, report it and gather representative training data before claiming demo readiness. No accuracy threshold is claimed as met until results exist.

## 5. Application structure and interfaces

Separate capture, detector adapter, pure geometry/decision logic, stabilization, and rendering. The decision module must accept synthetic detections without importing PyTorch or opening a camera. Load the upstream custom modules before loading trusted ESPDet checkpoints; loading arbitrary untrusted checkpoints is outside the intended workflow.

### Planned CLI

These are future interfaces, not currently runnable commands:

```text
python demo.py --source 0
python demo.py --source path/to/video.mp4
python demo.py --source 0 --weights path/to/best.pt --config path/to/config.json
python demo.py --source path/to/video.mp4 --save-video path/to/annotated.mp4
```

Default source is camera index `0`; a numeric source selects that camera, otherwise interpret it as a video path. Default weights location is `weights/obstacle/best.pt`. Missing weights must explain that the model requires training; never auto-load a cat checkpoint as an obstacle detector.

The optional JSON configuration overrides validated built-in defaults. Reject unknown keys, non-finite numbers, and out-of-range geometry or thresholds with an actionable message.

### Internal contracts

- **Frame:** original unmirrored image, sequence ID, and monotonic capture timestamp. Recorded video also retains source presentation time.
- **Detection:** `xyxy` box normalized to the original frame, confidence in `[0,1]`, class ID `0`.
- **Decision:** `STOP`, `FORWARD`, `LEFT`, or `RIGHT`; reason; source frame ID/timestamp; three region flags; and near-center status.
- **Controller status:** starting, running, paused, source ended, or error. Non-running states produce STOP internally.

Use the library's verified inverse preprocessing transform when available. If implementing it explicitly, remove letterbox padding, undo scale, then normalize to the original image. Do not apply the transform twice. Keep the displayed frame unmirrored so regions match decision coordinates.

## 6. Geometry and rule specification

All coordinates refer to the original camera frame: origin at top-left, x increases rightward, y downward, normalized to `[0,1]`.

| Configuration key | Default |
| --- | --- |
| `confidence_threshold` | 0.50 |
| `nms_iou_threshold` | 0.45 |
| `left_boundary` | 1/3 |
| `right_boundary` | 2/3 |
| `decision_top` | 0.40 |
| `near_top` | 0.80 |
| `overlap_threshold` | 0.10 |
| `stable_frames` | 3 |
| `stale_after_ms` | 1000 |

Define the left, middle, and right decision rectangles between `decision_top` and `1`. For each valid, accepted obstacle box, compute intersection area with each rectangle divided by the full obstacle box area. A ratio greater than or equal to `overlap_threshold` marks that region blocked. Region flags are the OR across obstacles; a large box can block several regions. Zero-area boxes are ignored with a diagnostic count.

A near-center hazard exists when a detection that individually blocks the middle region has positive-area overlap with the middle rectangle between `near_top` and `1`. Boundary contact with zero area does not count. This is an image-based proximity heuristic, not a distance measurement.

Apply the following priority order:

1. Non-running state, capture/inference error, or stale input/result: STOP.
2. Near-center hazard: STOP.
3. Middle unblocked: FORWARD.
4. Middle blocked and left unblocked: LEFT, including when both sides are unblocked.
5. Middle blocked, left blocked, and right unblocked: RIGHT.
6. Otherwise: STOP.

The left tie-break is a deterministic presentation convention, not a computed shortest path. Instructions must be labeled as suggestions.

### Stabilization and freshness

Start with STOP. Apply candidate STOP immediately and clear the movement streak. A non-STOP candidate must appear in three consecutive newly processed frames before becoming the displayed instruction. A different candidate resets its streak; retain the previously accepted instruction while stabilizing, unless a STOP condition occurs.

Repeated inference on the same frame does not advance a streak. Freshness checks must use capture timestamps, not just inference completion. Stop if the newest valid result is older than one second. Keep the display loop responsive independently of slow inference so errors and stale-frame status can appear. A webcam capture worker should replace a one-frame queue with the latest frame rather than accumulate a backlog.

Video playback must respect source timing, dropping obsolete frames if necessary. Pausing produces STOP and PAUSED, freezes playback, and clears history. Resume requires fresh decisions. EOF produces STOP and VIDEO ENDED; do not loop silently. Camera loss produces STOP and a clear message; no automatic movement resumption without fresh stabilization.

## 7. Presentation and recording

Render overlays after inference: boxes/confidence, left/middle/right labels, occupancy tint, near-area outline, large suggestion banner, reason, model identity, processed FPS, and capture-to-decision latency. Include a visible inference-error/stale state and the limitation statement.

Space pauses or resumes; Q/Escape quits and releases capture, writer, workers, and windows. Optional output recording uses the annotated display frames; keep timestamps/decision records so dropped-frame recordings are not mistaken for latency evidence. Do not save camera footage unless `--save-video` was supplied.

Use a preflight check for weights, source, output directory, and loaded class mapping. No inference-time dependency or checkpoint download is allowed during the offline presentation.

## 8. Tests and acceptance

### Pure decision tests

- No detections; each blocked region; center blocked with one/both sides available; all blocked.
- One obstacle crossing a divider; a wide obstacle crossing all regions; exact overlap thresholds.
- High image objects excluded from occupancy; near-center entry; a close side obstacle with an unblocked middle.
- Invalid/zero-area boxes, low confidence, and scale-equivalent boxes at different frame resolutions.
- Three-frame transitions, immediate STOP, pause/resume reset, duplicated frames, and stale timestamps.
- Coordinate round-trip tests for portrait, square, and widescreen letterboxing.

### Integration and presentation checks

- Load the actual ESPDet checkpoint with the pinned custom architecture and run a known image.
- Verify webcam and video input, camera permissions, missing checkpoint, invalid video, EOF, and optional output creation.
- Replay independent scenes with bottles, cups, boxes, overlaps, and mixed placements; document misses and false positives.
- Verify a simulated stalled detector results in a visible STOP within the configured freshness timeout.
- Run with networking disabled after setup and verify no unexpected downloads.
- Target at least 10 processed FPS on the development laptop; report actual FPS and median/p95 capture-to-decision latency with model and source settings. A failed target must be reported rather than hidden by display FPS.

Completion requires working input/output, meaningful geometry tests, a trained checkpoint with provenance, held-out metrics, and an honest annotated recording. No hardware behavior is established by these tests.

## 9. Deliverables and next milestone

Deliver the application, tested dependency pins, configuration example, data preparation/training workflow, model manifest, decision tests, evaluation report, and demonstration recording. Keep large weights and recordings outside Git and distribute them with provenance through an appropriate release mechanism when ready.

Then follow the [hardware implementation guide](hardware-implementation.md), preserving the model/decision separation and verifying quantized-model behavior before connecting wheels.
