# Independent camera, depth and movement evaluation

Status: **pending real recordings and independent reference annotations**.
The public-photo detector score does not establish navigation performance.

## Collect and split recordings

Mount the intended camera rigidly at vehicle height. Record whole sessions with
consent and suitable usage rights. Include clear corridors, center/side obstacles,
moving people, furniture, unfamiliar boxes/walls, different distances, occlusion,
lighting, invalid input and dropped frames. Keep each physical site/session in
one partition; do not randomly split adjacent video frames. Freeze test sessions
before threshold calibration; develop on separate sessions/sites.

## Annotate and import

Use CVAT or another annotation editor. Draw every visible evaluated instance,
including small/occluded ones under a written visibility rule. Original COCO IDs
are person=0 and chair=56 for the current score. Other classes require extending
both the evaluator and the coverage protocol; generic boxes/walls cannot be
renamed to COCO categories. Mark verified empty frames explicitly and have a
second group member review completeness without seeing model predictions.

Export normalized XYXY annotations into a JSONL file beside the recording frames:

```json
{"image":"session_A/frame_001.jpg","split":"dev","session_id":"siteA_sessionA","exhaustive_classes":["person","chair"],"objects":[{"class_id":56,"xyxy":[0.1,0.2,0.4,0.9]}],"source":{"author":"Group recording","license":"Group-approved research use","url":"recording:siteA_sessionA"}}
```

Use different complete sessions for test. Empty `objects: []` still requires
`exhaustive_classes`. Then run:

```powershell
.venv-evaluation\Scripts\python scripts/import_camera_dataset.py recordings/camera-annotations.jsonl --output datasets/camera-evaluation
.venv-evaluation\Scripts\python scripts/validate_annotations.py --dataset datasets/camera-evaluation
```

The importer copies images, converts reviewed boxes to original-ID YOLO labels,
checks session separation, validates hashes and flags near duplicates. It does
not create labels from model output. Keep camera results separate from public
data; explicitly configure a new experiment and recording-level bootstrap.

To evaluate a reviewed camera dataset, copy `config/evaluation.yaml` to a new
configuration, set `dataset` and a separate `output` directory, and replace
`dataset_manifest_sha256` with that dataset's manifest checksum. Run
`python scripts/evaluate_pretrained.py --config config/camera-evaluation.yaml`.
Recording-session grouping is recorded by the importer. The automatic narrative
report is currently written for the public Open Images study; adapt its source,
licensing and annotation-method paragraphs before presenting a camera study.
Do not publish the public-dataset narrative as if it described your recordings.

## Depth reference measurements

Measure aligned optical-axis depth in meters using an independently calibrated
reference. Record camera intrinsics, RGB/depth alignment, valid masks and whether
values are z-depth or Euclidean ray distance. Never derive metric ground truth
from object size or YOLO confidence. With valid aligned arrays, report MAE, RMSE,
AbsRel and threshold ratios over a declared valid range, plus distance bins.
Do not silently scale-align a metric model to test ground truth. The module
`visiondrive.evaluation.system.depth_metrics` accepts already aligned arrays and
requires an explicit measurement-convention declaration.

## Acceptable movement actions

Before examining predictions, label each scenario with a **set** of acceptable
actions: for a blocked center, both LEFT and RIGHT may be acceptable if measured
clearance supports them. Separately annotate whether FORWARD is unsafe and
whether STOP is unnecessary. Account for stabilization and stale-input periods.

Record predicted action, acceptable actions, timestamp/latency and scenario ID.
Report acceptable-set correctness, unsafe-forward rate among independently
unsafe-forward scenarios, and false STOP rate among independently clear-to-move
scenarios. A multi-action reference cannot honestly be reduced to an arbitrary
single-label confusion matrix; export action versus acceptability tables instead.

`scripts/evaluate_scenarios.py` runs hand-authored controller fixtures only.
These software checks do not measure camera interpretation or physical success.
Actual collisions, stopping distance and motor behavior require separate physical
trials with existing STOP/watchdog protections retained.
