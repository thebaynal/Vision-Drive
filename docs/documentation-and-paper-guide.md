# Documentation and paper guide

Use the instructor's paper template and citation style. This guide identifies
the current evidence; it is not a substitute for the group's own analysis.

## Project identity

**VisionDrive — Computer Vision–Based Obstacle Detection and Avoidance.**
Confirm the instructor's officially assigned specialization wording.

Group members: Sean Xander Aquino; Blessie Faith Bongalos; Joshua Jericho Barja;
John Renzzo Montenegro; John Carlo Nas; Divino Al Ricafort.

The laptop runs original YOLO11n and metric indoor depth. Three-region logic
displays movement recommendations. The optional ESP32 receives expiring commands;
it does not run those neural networks. Physical driving remains unvalidated.

## Keep the experiments separate

| Study | Evidence and permitted interpretation |
| --- | --- |
| Current pretrained YOLO11n | 400 fresh indoor photos, 80 dev / 320 test, 923 person annotations; person-only localization score; no YOLO fine-tuning |
| Supplementary bottle/cup benchmark | All 5,000 COCO val2017 images, 1,013 bottle and 895 cup non-crowd instances; official COCO evaluation; known upstream validation data |
| Historical ESPDet-Pico | 402 bottle/cup/box images with 280/60/62 training/validation/test; separate training and measured baseline |
| Hybrid demo | Real object/depth inference and displayed instructions; not an avoidance-accuracy measurement |

Do not call mAP classification accuracy, attribute ESP results to YOLO, or imply
that the group photographed public source images. The live detector retains 80
classes. The primary fresh indoor score covers person only; a separate COCO validation benchmark measures bottle and cup. Source Stool/Chair
coverage does not support a reliable COCO-chair score.

## Evidence to cite

- [Official YOLO11 software citation, formatted reference and BibTeX](../README.md#yolo11n-citation): cite **Jocher and Qiu (2024)** for YOLO11n.
- [Dataset, methodology and source licenses](dataset_card.md).
- [Measured report, graphs and failure examples](evaluation_results.md).
- [Executed research notebook](../notebooks/yolo11n_pretrained_evaluation.ipynb).
- [Measured JSON](results/yolo11n/metrics.json), [per-class CSV](results/yolo11n/per_class_metrics.csv) and [source attribution](results/yolo11n/dataset_provenance.csv).
- [Historical ESP methodology and results](model-report.md).
- [Application and hardware guide](laptop-esp32.md).
- [Verification and remaining limitations](verification.md).
- [Third-party sources and screenshot credits](../THIRD_PARTY_NOTICES.md).

## Suggested paper structure

1. **Introduction:** forward-camera obstacle interpretation, intended environment and research question.
2. **Related work:** cite original model/dataset sources; distinguish upstream methods from group contributions.
3. **System methodology:** capture, YOLO, depth, floor heuristic, three regions, confirmation and optional ESP32 communication.
4. **Dataset:** public origins, licenses, source labels, positive/negative sampling, photographer grouping, exclusions and frozen splits.
5. **Experiment:** unchanged checkpoint hash, original class ID 0, 640 input, confidence/IoU/NMS settings, hardware, software and timing scope.
6. **Results:** actual P/R/F1, TP/FP/FN, AP50/AP75/AP50–95, intervals, condition breakdowns and attributed examples.
7. **Discussion:** misses, false positives, sampling bias, viewpoint mismatch, uncertain COCO overlap and one-class scope.
8. **Conclusion and future work:** independent camera sessions, reference depth, acceptable-action labels and physical trials.

Use the generated [12-part research report](evaluation_results.md) for measured
values. Keep model-only timing separate from hybrid processing and live driving.
The evaluation NMS (.70) differs from the live default (.45).

<a id="midterm-evidence"></a>

## Midterm evidence: “Can the System SEE and RECOGNIZE?”

| Requirement | Submission evidence | Qualification |
| --- | --- | --- |
| Training / validation / testing | Historical ESP 280/60/62 split and training notebook | Public source images; physical scene independence unverified |
| Dataset appropriate to specialization | Fresh 400-image indoor person study and preserved baseline | No universal-obstacle coverage claim |
| Computer vision and image processing | Actual YOLO/depth demo, letterboxing, proximity and floor processing | Metric distance accuracy pending references |
| Model development and evaluation | Historical ESP training; current unchanged YOLO evaluation | The current study does not retrain YOLO |
| Unseen situations | Fresh test images excluded from prior inspected pilot, frozen before inference | Camera-domain testing and COCO overlap audit remain pending |
| Working visual system | Demo notebook, video workflow and verification records | No physical car success rate |

Discuss acceptance of the specialization scope with the instructor. Do not
invent field trials, dataset collection sessions, distance labels or scores.

## Figures and citations

Use actual evaluation graphs in `assets/evaluation/`, the architecture in the
README and historical curves only when discussing ESP training. Give each figure
a caption, model/configuration, dataset scope and applicable photo attribution.
Retain the provided author/source/license records. Cite Open Images, Ultralytics,
Depth Anything V2 and Espressif from the linked original sources; do not invent
bibliography entries or substitute published benchmarks for local measurements.

## Before submission

Verify that reported values match the CSV/JSON, titles distinguish current and
historical studies, table sample counts are correct, limitations are visible,
and every teammate can explain the data source and the person-only score.


<a id="presentation-script"></a>

## Presentation script

“VisionDrive treats the camera as the vehicle's forward view. The laptop runs
original YOLO11n plus a separate metric-depth model. Three-region logic explains
FORWARD, LEFT, RIGHT or STOP. The optional ESP32 receives expiring commands.

We did not train or fine-tune YOLO11n. We evaluated its existing person category
on 400 real indoor public photos: 80 development and 320 test, containing 923
source-annotated people. Every image has verified person presence or absence.
Each photo has a different photographer; all earlier inspected images are excluded.
We froze the source-only protocol in Git before inference on the fresh test set.

The photos and original annotations came from Open Images. Individual photos
are credited under CC BY 2.0; annotations are CC BY 4.0. We selected building
interiors, checked annotation geometry and hashes, and excluded group-of,
ambiguous person labels and unverified images. We used original COCO ID 0.

Our pilot found that Open Images separates Stool from Chair. Scoring source
Chair alone could mark genuine COCO chair detections on stools as false
positives, so furniture is outside the primary score. The live detector retains
all 80 categories, with separate depth cues for unnamed nearby surfaces.

On the fresh 320-image test set, precision was 79.19 percent,
recall 63.93 percent and F1 70.75 percent at confidence .35 and IoU .50.
AP at .50 was 72.12 percent; AP over .50 through .95 was 50.94 percent.
Only one category is scored, so mAP equals person AP. This is object localization,
not ordinary classification accuracy or collision-avoidance success.

The notebook records actual predictions, per-class counts, uncertainty intervals,
failure examples and synchronized GPU timing. Public photos do not prove
independence from COCO pretraining or performance on our car's camera.
Independent camera sessions, reviewed labels, calibrated depth and physical
trials remain necessary. Existing STOP and watchdog protections remain enabled.”

## Paper wording for the bottle/cup benchmark

In addition to the fresh indoor person study, we evaluated the unchanged
COCO-pretrained YOLO11n on all 5,000 COCO val2017 images for bottle and cup.
The official human annotations contain 1,013 non-crowd bottle instances and
895 non-crowd cup instances. COCO's official evaluator handles crowd regions;
all images without these objects remain included to measure false positives.
The manifest, original class mappings and scoring thresholds were frozen before
inference. No training, fine-tuning or confidence optimization was performed.

| Object | Ground-truth instances | Precision | Recall | F1 | AP50 | AP75 | AP50–95 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle | 1,013 | 77.86% | 32.28% | 45.64% | 48.32% | 35.73% | 32.06% |
| Cup | 895 | 70.94% | 39.55% | 50.79% | 49.48% | 38.44% | 35.61% |

These are measured localization results on a known upstream benchmark. They
must not be described as independent unseen-data accuracy, classification
accuracy, or physical obstacle-avoidance success. Cite Jocher and Qiu (2024)
for YOLO11n, Lin et al. (2014) for COCO, and the project's
[measured report](evaluation_results.md#bottle-and-cup-detection-additional-measured-benchmark)
for the values. The executed notebook's Section 13 reproduces this experiment.
