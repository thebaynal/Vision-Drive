# Bottle, cup, and box model report

Recorded on 2026-10-16.

**Initial prototype: box detection remains weak. These results do not establish reliable obstacle avoidance.**

## Model and training

- ESPDet-Pico, 361,365 parameters, 224 x 224 input; 0.95 MiB checkpoint.
- Exactly three output classes: bottle, cup, box. Other categories are excluded.
- CPU training completed 55 epochs; validation selected epoch 40.
- Confidence 0.35 selected using validation macro F1 only.
- Checkpoint SHA-256: `e5075b5a8fba62ab40c6580225451820006f9dc4f7463629dbe1c34d024de39c`.

The upstream single-class weights supply transferable initialization only. Training rebuilds the symbolic architecture with a three-class head. The original checkpoint is not used by the active demo.

## Data

The official Open Images validation pool supplied 179 bottle images, 155 cup images, and 68 box images. Content was deduplicated before new project partitions were assigned. Source IDs, author/license metadata, original URLs, and annotations remain in the local data manifest.

| Project split | Images | Bottle instances | Cup instances | Box instances |
| --- | ---: | ---: | ---: | ---: |
| train | 280 | 294 | 136 | 73 |
| val | 60 | 48 | 28 | 15 |
| test | 62 | 43 | 25 | 26 |

These are project splits from one public source pool, not official Open Images benchmark results. The selected source has relatively few box images. Annotation completeness and camera-domain differences limit generalization.

## Held-out results

Across 62 held-out images, mAP@0.5 = **0.469** and mAP@0.5:0.95 = **0.357**.

The following precision/recall/F1 values use confidence 0.35 and IoU >= 0.5, with class-aware, one-to-one matching. They differ from mAP and are not a single overall accuracy percentage.

| Class | Precision | Recall | F1 | TP / FP / FN |
| --- | ---: | ---: | ---: | --- |
| bottle | 0.559 | 0.442 | 0.494 | 19 / 15 / 24 |
| cup | 0.760 | 0.760 | 0.760 | 19 / 6 / 6 |
| box | 0.222 | 0.154 | 0.182 | 4 / 14 / 22 |

Cups performed best in this run. Bottles still have substantial misses. Box recall is low and requires more varied, representative data. No test-set-driven retraining or threshold changes were used for this published result.

Warm single-image model.predict calls over the evaluation images had median 25.9 ms on this development CPU. This excludes cold start, camera buffering, and rendering; it is not end-to-end FPS or an ESP32 benchmark.

## Reproduce and extend

This is the historical ESPDet experiment. Current YOLO research evaluates the
unchanged pretrained model separately. For ESPDet, use its notebook below;
the [recorded training configuration](baseline-training-configuration.json) and
[training curves](../assets/baseline-training-curves.png) support the methodology.

Use [the notebook](../notebooks/train_bottle_cup_box.ipynb) or [the run guide](laptop-esp32.md). [Machine-readable evaluation](model-evaluation.json), [training history](training-results.csv), and [model provenance](../weights/obstacle/model.json) are included. Raw images, full run checkpoints, and local environments stay outside Git.

Next steps are more box images and camera-height-specific recordings, an independent course test, and separate INT8/ESP32 evaluation. No physical car behavior is validated here.


<a id="historical-methodology"></a>

## Historical methodology for the paper

This section describes **ESPDet-Pico only**, not the current YOLO11n study.
The group selected Open Images Bottle, Coffee cup and Box annotations, retained
their original attribution and converted them into bottle/cup/box project labels.
The 402 images were partitioned into 280 training, 60 validation and 62 test
images. These project partitions came from the official validation source pool;
they are not official Open Images benchmark partitions or a verified scene split.

Training rebuilt the ESPDet-Pico detection head for three classes and transferred
compatible upstream initialization weights. CPU training completed 55 epochs;
validation selected epoch 40 and confidence .35. The held-out test was evaluated
after selection. The upstream single-class initialization is not an active cat
detector in the published three-class artifact.

The model remains weak on boxes. Do not claim physical obstacle-avoidance success,
onboard INT8 performance or that this dataset trained YOLO11n/Depth Anything.
Full reproducible settings, training commands, citations and figures remain in
the [training notebook](../notebooks/train_bottle_cup_box.ipynb),
[configuration](baseline-training-configuration.json),
[training CSV](training-results.csv), [measured JSON](model-evaluation.json),
[checkpoint provenance](../weights/obstacle/model.json) and
[source manifest](../data/public/bottle-cup-box-manifest.jsonl).
The older long-form paper draft and read-aloud training script are recoverable
from commit `a98b1d6`; they duplicated this evidence and are no longer separate
submission documents.


<a id="earlier-yolo-exploration"></a>

## Earlier YOLO exploration

Recorded on **2026-10-17**. This is an exploratory evaluation of the pretrained
object detector on the existing project images, not a whole-system driving test.

### Dataset and protocol

- All **62 test images** in the existing Open Images project subset were used.
- Evaluated annotations: **43 bottles and 25 cups**.
- The **26 box annotations were excluded** because COCO YOLO11n has no generic
  box class. Images containing boxes were retained.
- Model: unchanged pretrained YOLO11n, input **640 × 640**, CPU, two threads.
- Runtime confidence **0.35** and class-aware NMS IoU **0.45** came from the
  existing runtime config; no test-driven confidence tuning was performed.
- Fixed-threshold scores use class-aware, confidence-ordered, one-to-one matching
  at bounding-box IoU >= **0.5**. Duplicate detections count as false positives.
- AP was evaluated separately with the native Ultralytics 8.3.112 validator,
  confidence floor **0.001**, NMS IoU **0.45**, and a maximum of 300 detections.
- Exact test image hashes and project split hash uniqueness were checked.

### Measured results

| Class | TP | FP | FN | Precision | Recall | F1 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Bottle | 19 | 8 | 24 | 70.37% | 44.19% | 54.29% |
| Cup | 19 | 10 | 6 | 65.52% | 76.00% | 70.37% |
| Combined, micro average | 38 | 18 | 30 | **67.86%** | **55.88%** | **61.29%** |

Macro F1, giving bottle and cup equal weight: **62.33%**.

| Class scope | AP@0.5 / mAP@0.5 | AP@0.5:0.95 / mAP@0.5:0.95 |
| --- | ---: | ---: |
| Bottle | 49.75% | 36.77% |
| Cup | 80.36% | 70.94% |
| Mean across the two evaluated classes | **65.05%** | **53.86%** |

The native AP validator uses a confidence sweep and its own matching procedure.
Its console precision/recall summarize a different operating point. The first
table reports the fixed runtime threshold using the project's explicit scorer.

The warm CPU detector adapter had median latency **90.35 ms/image**, mean
**94.03 ms/image**, across 62 observations after one warmup. This excludes image
decoding, depth estimation, capture, UI, and Wi-Fi. It is not the hybrid system's
end-to-end latency or display frame rate.

Environment: Windows 11, Python 3.13.5, PyTorch 2.12.0+cpu, Ultralytics 8.3.112.
See [the machine-readable result](yolo11-evaluation.json) for hashes, full
precision values, settings, and environment metadata.

### Interpretation and limitations

At the runtime threshold, the detector matched **38 of 68 annotated target
instances**, missed 30, and produced 18 unmatched bottle/cup predictions. Bottle
recall remains low. These results do not justify a claim that every obstacle is
detected or that the car always chooses a safe direction.

Only bottle and cup outputs were scored. Other COCO outputs, box handling by
the depth branch, and unnamed obstacles are outside this evaluation. Annotation
completeness has not received a comprehensive manual audit; some unmatched
predictions may correspond to visible objects missing from the annotations.

The local partitions are hash-disjoint, but overlap with upstream pretraining
data has not been audited. Test images were previously used for integration
examples, so treat this as exploratory rather than a fresh independent course
benchmark. This two-class aggregate and ESPDet's original three-class aggregate
cover different scopes and should not be used for a direct ranking.

Depth error, near-obstacle recall, blocked-region accuracy, instruction accuracy,
unsafe-forward/false-stop rates, hybrid latency, and physical collision avoidance
remain unmeasured. Follow the [evaluation guide](dataset_card.md#evaluation-methodology) to collect
independent camera recordings and measured distances for those results.

### Reproduce

From the project folder, with the existing inference environment and data:

```powershell
.venv-inference\Scripts\python scripts/evaluate_yolo.py
```

The script writes `runs/yolo-evaluation/results.json` and the corresponding
`results.predictions.json`. Repeated timings vary with system load. The script
does not retrain or change runtime configuration and does not send motor commands.

Suggested paper statement:

> In an exploratory evaluation on 62 project test images, pretrained YOLO11n
> achieved micro precision of 67.86%, recall of 55.88%, and F1 of 61.29% for
> bottle and cup detections at confidence 0.35 and matching IoU 0.5. Across these
> two classes, mAP@0.5 was 65.05% and mAP@0.5:0.95 was 53.86%. Generic boxes were
> excluded because they are not a COCO output category. These results do not
> measure depth quality or whole-system obstacle avoidance.
