# Measuring VisionDrive performance

Evaluate object recognition, proximity, and movement instructions separately.
An object detector's mAP is not an obstacle-avoidance success rate.

## 1. Run the existing image test now

For the **new fine-tuned three-class YOLO11n** checkpoint, use:

```powershell
.venv-inference\Scripts\python scripts/evaluate_finetuned_yolo.py --weights weights/laptop/yolo11n-bottle-cup-box.pt
```

It uses the version 2 dataset, chooses confidence on validation only, and then
scores bottle/cup/box test detections. See its
[recorded report](yolo11-finetuning-report.md). The following original evaluator
is retained for the **unchanged pretrained 80-class YOLO model**; its two-class
aggregate has a different scope and cannot be directly ranked against the
fine-tuned three-class aggregate.

From the project folder in PowerShell:

```powershell
.venv-inference\Scripts\python scripts/evaluate_yolo.py
```

The existing laptop environment, local YOLO11n weights, and restored dataset are
required. On another computer, follow [installation](laptop-esp32.md) first. If
the dataset folder is missing, restore it explicitly:

```powershell
.venv-inference\Scripts\python scripts/restore_dataset.py --download
```

The evaluator uses the **62 project test images**, verifies their SHA-256 hashes,
and checks for duplicate hashes across project partitions. It does not train a
model, adjust a threshold, open a camera, or send motor commands.

Results are saved to:

- `runs/yolo-evaluation/results.json`: settings, provenance, metrics, and timing.
- `runs/yolo-evaluation/results.predictions.json`: predictions for manual review.

Fixed-threshold precision, recall, and F1 use the runtime settings from
`config/default.json`: confidence **0.35**, NMS IoU **0.45**, and class-aware,
one-to-one matching at IoU **0.5**. Micro metrics pool counts; macro F1 gives
each evaluated class equal weight. A duplicate detection is a false positive.

The native Ultralytics validator separately computes AP using confidence floor
0.001 and IoU thresholds 0.50 through 0.95. Its matching and confidence sweep
differ from the fixed-threshold scorer. Use the explicit `fixed_threshold`
fields when reporting performance at the displayed demo's confidence threshold.

**Scope:** YOLO11n has bottle and cup classes, but no generic box class. Only
bottle and cup are scored, using their original COCO IDs. All 62 test images are
kept, including images whose box annotations are omitted from this evaluation.
Predictions for the other COCO classes are excluded from these scores. This
does not establish performance for boxes or arbitrary obstacles.

This is an **exploratory public-image evaluation**. The subset's labels may be
incomplete for other visible target objects; manually audit apparent false
positives and record corrections with a dataset version. Overlap with upstream
pretraining data has not been audited. These images also appeared in earlier
integration work. They are not a fresh, independent test of the whole system.

Do not compare the two-class aggregate directly with ESPDet's published
three-class aggregate. A controlled comparison needs identical classes,
annotations, images, confidence-selection rules, and matching rules.

## 2. Collect a camera dataset for the actual task

Mount the camera at the intended vehicle height and keep its tilt fixed. Use
the actual indoor course, several lighting conditions, backgrounds, object
sizes, and near/far distances. Include empty paths, unusual objects, thin
obstacles, and transparent/reflective objects as challenging cases.

A manageable pilot is **120 short clips**, with 20 examples of each condition:

| Condition | Labels to record |
| --- | --- |
| Clear route | No near blockage; FORWARD allowed |
| Far object | Visible object outside the avoidance range; FORWARD allowed if otherwise clear |
| Center blocked, left clear | LEFT allowed; FORWARD unsafe |
| Center blocked, right clear | RIGHT allowed; FORWARD unsafe |
| All routes blocked or emergency near center | STOP required |
| Unknown object on route | Blocked regions and safe actions, regardless of object name |

Also include both sides clear and side-only obstacles. When both turns are
acceptable, label the set of acceptable instructions instead of marking one
arbitrary turn as the only correct answer. Account for the vehicle's intended
width and turning clearance when labeling; the current image thirds do not
guarantee a physically traversable route.

Assign whole recording sessions to training/development, validation, and test
partitions **before extracting frames**. An example target is 60/20/20, with
different sessions, arrangements, and preferably locations in each partition.
Record enough sessions to cover each condition in the test partition. Never
randomly split adjacent video frames across partitions. The training partition
can support future fine-tuning; pretrained inference does not require retraining.

For selected timestamps annotate:

- Visible obstacle bounding boxes, including an `unknown obstacle` label.
- Left, center, and right blockage at the configured near range.
- Actual distance in meters to the obstacle surface, measured with a tape.
- Acceptable instructions and whether forward motion would be unsafe.
- Clip/session ID, timestamp, lighting, camera pose, and split.

For distance labels, define the reference consistently, for example from the
camera to the visible front surface. Object depth uses an interior-crop
quantile, so visually check that the sampled region represents the measured
surface. Keep invalid/missing depth outputs in the failure count.

Because this model is pretrained, distinct local test sessions are the best
next step for testing the intended camera domain. Choose settings using
validation recordings, freeze them, and run the test clips once for the report.

## 3. Measure the whole vision system

| Metric | Calculation / protocol |
| --- | --- |
| Object precision | TP / (TP + FP), with class-aware IoU matching |
| Object recall | TP / (TP + FN), with the same matching |
| Object F1 | 2 × precision × recall / (precision + recall) |
| mAP | AP across classes and IoU thresholds; name both thresholds and dataset |
| Blocked-region precision/recall | Compare predicted and annotated blockage for each region |
| Near-obstacle recall | Fraction of annotated near hazards recognized as blockage |
| Depth MAE | Mean absolute difference between estimated and measured meters |
| Depth availability | Valid distance estimates / labeled distance samples |
| Instruction accuracy | Fraction of eligible observations whose instruction is in the allowed set |
| Unsafe-forward rate | FORWARD outputs during labeled unsafe-forward observations / all unsafe-forward observations |
| False-stop rate | STOP outputs during labeled clear-route observations / all clear-route observations |
| Response delay | Time from a hazard becoming visible to the required instruction |
| Processing rate | Actual processed observations / elapsed wall-clock seconds |
| End-to-end latency | Time from frame capture to the instruction or command being emitted |

For zero denominators report `N/A`, together with the raw counts. Report the
number of clips, independent sessions, labeled observations, and obstacle
instances. Break results down by condition; a mostly clear dataset can conceal
poor obstacle recall behind high overall instruction accuracy. Use independent
clips/sessions, rather than correlated frames, for uncertainty estimates.

Evaluate instruction transitions as well as stable scenes. The controller
requires three distinct model observations for confirmed movement, so annotate
an explicit transition interval and report startup/transition delay separately.
Do not remove missed hazards from safety-error counts during that interval.

The viewer holds visual results for up to five seconds, while the separate
motor controller uses a stricter freshness deadline. Record **visual instruction**
and **control action** separately; display refresh FPS does not equal neural
inference FPS. Camera failures and stale model observations belong in reliability
results, even if the held visual instruction still appears correct.

The existing evaluator times only the warmed YOLO detector adapter. It excludes
depth, file decoding, capture, drawing, and Wi-Fi. Measure the actual hybrid
pipeline separately for end-to-end claims. Physical stopping distance and
collision-free traversal require later hardware trials.

## 4. Report the result honestly

Use the [recorded YOLO11n evaluation](yolo11-evaluation.md) for the current
unchanged-pretrained experiment, the [fine-tuning report](yolo11-finetuning-report.md)
for the new three-class run, and [ESPDet report](model-report.md) for the original
fine-tuned baseline. Do not substitute Ultralytics' COCO benchmark numbers for
measurements on the project dataset.

For the final paper, state the model/version, dataset version and split,
confidence and NMS settings, matching rules, class scope, hardware, latency
scope, errors, and unmeasured items. Retain predictions and failed examples so
the group can explain both successful detections and limitations.
