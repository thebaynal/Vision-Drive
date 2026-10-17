# Fresh indoor person evaluation dataset

## Objective and measured scope

Evaluate original COCO-pretrained YOLO11n for **person localization** in real
indoor photographs. The final cohort contains **400 validated images** and
**923 annotated person instances**. Chair, furniture and all other model classes
are outside the primary numerical score; the live model keeps all 80 categories.

| Partition | Original source split | Images | Person-positive images | Verified negative images | Person instances |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Open Images validation | 80 | 60 | 20 | 191 |
| Final test | Open Images test | 320 | 240 | 80 | 732 |
| Total | Both | 400 | 300 | 100 | 923 |

One photo per photographer is retained; all 400 groups are distinct. All prior
inspected pilot and ESP baseline image IDs are excluded. The source-only
protocol and complete image hashes were committed as **`8a6f52a` before first
inference on this fresh test set**. No threshold or image selection was adjusted
using the final test results. The 300–500-image planning target is met, but it is
not a statistical minimum or proof of broad generalization.

These images were not captured by the group. No team camera recordings or
metric depth references were supplied. The 75/25 positive/negative stratification
is deliberate; precision depends on this sample prevalence.

## Candidate investigation

| Candidate | Suitability and decision |
| --- | --- |
| [SUN RGB-D](https://rgbd.cs.princeton.edu/), [official files](https://rgbd.cs.princeton.edu/data/) | Relevant RGB-D interiors and dedicated v2 2D box metadata. Homepage/README access failed; a definitive named image license was not verified. Not adopted or assigned an invented license. |
| [Roboflow interior-item-detection](https://universe.roboflow.com/prabhakar-egqij/interior-item-detection) | CC BY 4.0 listing, 57 heterogeneous labels. Generic Table and duplicate vocabulary require curation; exhaustive annotation and upstream independence unresolved. |
| [Roboflow-100 furniture](https://universe.roboflow.com/roboflow-100/furniture-ngpea) | CC BY 4.0; Chair/Table/Sofa. Generic Table is not automatically COCO dining table; completeness/upstream overlap unresolved. No export restriction bypassed. |
| [Mendeley indoor v2](https://doi.org/10.17632/3ggxwf2vpr.2), Nafiz Fahad | CC BY 4.0; seven classes, VOC archive. Official file metadata obtained: 453,984,831 bytes, SHA-256 3d5a25726f045f230eccbabfc9e3ef4db3916464dcda4d7f48102e7a9ababd20. Archive download returned an access challenge and was stopped. No successful image acquisition or completeness claim. |
| [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | Relevant dense indoor instance labels/depth. Named redistribution terms were not established from the author page. Not downloaded or assigned a guessed license. |
| Existing Open Images bottle/cup/box | Genuine historical baseline but narrow labels and incomplete coverage of other COCO classes. Preserved separately; original baseline image IDs excluded from the new cohort. |
| Open Images indoor Person | Selected for documented exhaustive person-positive boxes, verified negatives, explicit interior attributes and reproducible licensed access. The initial chair pilot exposed a taxonomy defect and was excluded from primary evidence. Public-photo viewpoint remains a limitation. |

Selection favors coverage and provenance over recycling convenient images. No synthetic images or augmentation are included.

## Source releases, licensing and attribution

Official **V5 bounding boxes**, human-verified boxable image labels and
**2018_04 image metadata** are interpreted using the
[V7 specification](https://storage.googleapis.com/openimages/web/download_v7.html).
The [source protocol](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
describes exhaustive instances for verified positive categories and verified
negative image labels. This selected cohort is not an official Open Images score.

Photos: **CC BY 2.0**, individually attributed to photographer, title and original
page. Annotations: **CC BY 4.0**, Open Images/Google. Cite Kuznetsova et al.,
*The Open Images Dataset V4*, IJCV 2020,
[paper](https://arxiv.org/abs/1811.00982). Retain author, source, license and
modification notices when showing or redistributing photos.

The [frozen manifest](../data/public/indoor-evaluation-manifest.jsonl) records
every source URL, author/license, source split, original category, normalized
box, pixel hash and dimensions. The [selection audit](../data/public/indoor-selection-audit.json)
pins official CSV URLs, byte sizes and SHA-256 values. These are acquisition
hashes, not publisher signatures. Downloaded image bytes total 131,371,983 bytes.

## Selection and annotation methodology

1. Require a Building-family annotation with `IsInside=1`, including appropriate
   House, Office building and other building descendants.
2. Require human-verified **Person presence or absence**; unknown never means absent.
3. Map Person `/m/01g317`, Man `/m/04yx4`, Woman `/m/03bt1vf`, Boy `/m/01bl7v`,
   Girl `/m/05r655` to original **COCO person=0**. Retain all 80 YAML names and
   apply native filter `[0]`, rather than inventing a custom head.
4. Propagate person subtypes. Collapse coincident boxes only across source
   categories at IoU≥.95. Remaining cross-source person overlaps at IoU>.30 cause
   whole-image exclusion because identity is ambiguous; no merge is guessed.
5. Exclude target group-of, depiction, inside-target/unknown flags, label-box
   conflicts, invalid geometry and missing approved attribution.
6. Source-only seeded selection uses 20261017, selects official test before
   development, admits one image per photographer, and takes 75% positive / 25%
   negative images in each partition. No model is imported by the selection script.

After excluding prior images, source filtering found 1,460 eligible official-test
and 484 eligible official-validation images. Candidate exclusions were:

| Reason | Official test | Official validation |
| --- | ---: | ---: |
| Prior inspected/baseline image | 191 | 66 |
| Unverified person presence/absence | 1,297 | 407 |
| Target attributes | 625 | 243 |
| Ambiguous person hierarchy | 105 | 31 |

These are sequential image-filter counts, not distinct physical objects.
The final sampling then selected 320 test and 80 development images. All 400
downloads validated; no exact/likely-near duplicate removal was needed.
Source categories outside person are counted separately in
[excluded-source-classes.csv](../data/public/indoor-excluded-source-classes.csv).
They are annotation records, not primary-score ground truth or absent objects.

Source protocol, structural checks and visual spot checks are not independent
exhaustive reannotation and cannot remove every source error. Annotated sheets
for all retained images are generated by the notebook.

## Pilot taxonomy correction and independence

The first 249-image person/chair pilot exposed the Stool/Chair mismatch during
figure QA. Its source Chair labels cannot establish complete COCO-chair coverage.
The pilot is exploratory and does not supply final test evidence. See the
[taxonomy audit](dataset_card.md#chair-stool-taxonomy) and [archived pilot manifest](../data/public/exploratory-indoor-manifest.jsonl).
The final person-only cohort is fresh and excludes every pilot image ID.

SHA-256 and 64-bit dHash (Hamming distance≤4) check internal duplicates. Official
splits and distinct photographer groups reduce project leakage. Physical scene
identities across photographers remain unknown. **Independence from COCO
pretraining is not established**: both datasets contain public photos and a
complete cross-dataset overlap audit was not performed. No camera-domain or
physical avoidance result is implied.

## Reconstruct and validate

```powershell
.venv-evaluation\Scripts\python scripts/download_dataset.py
.venv-evaluation\Scripts\python scripts/validate_annotations.py
```

Normal acquisition restores exactly the distributed frozen manifest, checks each
hash and reuses valid local data. `scripts/build_evaluation_cohort.py` documents
the original source-only selection; it is not necessary for normal restoration.
YOLO labels are normalized original-ID records. Local absolute image lists and
dataset YAML stay ignored; scoring regenerates lists strictly from the manifest.
No failed download is silently replaced. No YOLO training split is created;
Ultralytics' required `train` YAML key aliases development and is unused.


<a id="evaluation-methodology"></a>

## Evaluation methodology

The protocol evaluates unchanged original YOLO11n. It never trains or fine-tunes YOLO11n. Historical ESP results and earlier bottle/cup exploration remain separate.

### Reproducible inputs

- Official Ultralytics v8.3.0 asset `yolo11n.pt`, SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- [Configuration](../config/evaluation.yaml): FP32, batch 1, square 640 letterboxing, NMS IoU .70, max 300 detections.
- [Frozen real indoor cohort](dataset_card.md), source-verified person coverage.
- Person labels retain original ID 0 and all 80 YAML names; only person is primary-scored. No custom 0/1 head substitution.
- Environment, manifest/checkpoint/source hashes and Git commit/dirty state accompany actual runs.

### Operational P/R/F1

These are research operating-point measurements. Evaluation uses NMS IoU .70;
the live application's default NMS is .45. Do not describe these scores as an
end-to-end measurement of the live application's exact configuration.

Confidence≥.35; descending-confidence predictions; same-class one-to-one highest-available-IoU≥.50 matching. Unmatched prediction=FP; unmatched truth=FN. Duplicate predictions cannot both match one truth.

`P=TP/(TP+FP)`, `R=TP/(TP+FN)`, `F1=2TP/(2TP+FP+FN)`. Undefined ratios use zero, with sample counts shown. Micro pools counts. Macro P/R/F1 average corresponding class scores; macro F1 is not the harmonic mean of macro P/R.

### Average precision and matrices

Native Ultralytics `model.val()` uses the 80-name YAML, class filter [0], confidence floor .001, IoU thresholds .50:.05:.95. AP@75 is column 5. Only person is scored, so mAP equals person AP. This is not classification accuracy.

Native matching and its best-F1 P/R differ from our fixed operating-point matcher; never combine native P/R with fixed TP/FP/FN. Test PR/F1 maxima do not select a new runtime threshold.

The fixed-point detection confusion matrix shows TP, misses and unmatched predictions. Wrong-class detections contribute FP+FN. Off-diagonal class substitutions are not inferred. Row normalization includes the unmatched-prediction row. Native plots are retained separately.

### Timing and uncertainty

Ten warmups precede timing; CUDA synchronizes before/after measured calls. The first 100 test images in manifest order are decoded before measurement. Report pre/network/post and Python predict-call wall mean, median, population SD and p95. FPS=1000/mean milliseconds. Hardware, batch, precision and resolution are recorded. Model-only FPS excludes camera, depth, GUI, controller, UDP and motors.

1,000 seeded percentile bootstrap samples resample complete photographer groups (scene proxy). Report 95% intervals for micro P/R/F1 and group count; no AP interval claimed. Bootstrap cannot fix unknown shared scenes, sampling bias or pretraining overlap.

### Error analysis and limits

Condition recall uses source occlusion/truncation, source-pixel object area, full-image grayscale luminance and target counts. These observational bins are confounded, not causal lighting/perspective/clutter experiments. Actual FP/miss examples carry source attribution.

No metric depth reference or independent group camera recordings were supplied. Depth and physical navigation accuracy remain pending. Fixture-based movement results are software checks, not collision-avoidance evidence.


### Pilot correction and fresh frozen final test

Pilot figure QA found that source Stool and Chair are not exhaustive for COCO
chair. The pilot is exploratory; its scores do not validate furniture recognition.
The final person-only protocol was frozen as commit `8a6f52a` before inference
on a fresh 400-image cohort excluding every pilot and baseline image ID.
Its 80 development / 320 test photos have distinct photographers, source-verified
person presence/absence and a fixed 75% positive / 25% negative ratio. Source-only
selection used seed 20261017. No final test result selected thresholds or images.
See [the dataset card](dataset_card.md) and [taxonomy audit](dataset_card.md#chair-stool-taxonomy).


<a id="chair-stool-taxonomy"></a>

## Chair / Stool taxonomy audit

The initial experiment scored source Person and Chair annotations as COCO
person and chair. Figure QA exposed an important mismatch: **Open Images Stool
is separate from Chair**, while COCO chair predictions can legitimately cover
stools. Source Chair verification alone cannot guarantee completeness for the
broader COCO target concept.

For example, image **`3da6cd80dd411d74`** has verified **Chair=0, Stool=1** and
four source Stool boxes. The initial scorer counted chair predictions on the
stools as false positives. This is an annotation/taxonomy limitation, not
evidence that those model detections are necessarily wrong.

Only eight of the 249 cohort images have Stool verification: six positive
(two development, four test), two negative (one per partition). The remaining
241 are unknown. Removing only the known positives cannot establish complete
COCO-chair ground truth for the rest. No reliable ignored-region protocol can
be built from unknown regions, so **chair is excluded from primary scoring**.

The 249-image pilot and its intermediate person-only rescoring are not final
test evidence. A **fresh 400-image person-only cohort** excludes every pilot and
baseline image ID. Its protocol, source-only selection and image hashes were
frozen in commit `8a6f52a` before first inference. Confidence/NMS/matching
thresholds were retained; no fresh test score changed selection or configuration.

The earlier two-class measurements are preserved in
`docs/results/exploratory-person-chair/` as **exploratory, incomplete-taxonomy
results**. They must not be quoted as reliable chair performance or substituted
for the primary person result. Source annotations remain available for audit.

Future furniture evaluation requires either an exhaustively reannotated
COCO-compatible camera dataset or a source with verified coverage for the full
chair concept, including appropriate stool instances and reviewed ignored cases.


<a id="independent-camera-evaluation"></a>

## Independent camera evaluation

Status: **pending real recordings and independent reference annotations**.
The public-photo detector score does not establish navigation performance.

### Collect and split recordings

Mount the intended camera rigidly at vehicle height. Record whole sessions with
consent and suitable usage rights. Include clear corridors, center/side obstacles,
moving people, furniture, unfamiliar boxes/walls, different distances, occlusion,
lighting, invalid input and dropped frames. Keep each physical site/session in
one partition; do not randomly split adjacent video frames. Freeze test sessions
before threshold calibration; develop on separate sessions/sites.

### Annotate and import

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

### Depth reference measurements

Measure aligned optical-axis depth in meters using an independently calibrated
reference. Record camera intrinsics, RGB/depth alignment, valid masks and whether
values are z-depth or Euclidean ray distance. Never derive metric ground truth
from object size or YOLO confidence. With valid aligned arrays, report MAE, RMSE,
AbsRel and threshold ratios over a declared valid range, plus distance bins.
Do not silently scale-align a metric model to test ground truth. The module
`visiondrive.evaluation.system.depth_metrics` accepts already aligned arrays and
requires an explicit measurement-convention declaration.

### Acceptable movement actions

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
