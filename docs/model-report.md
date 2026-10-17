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

This is the earlier ESPDet experiment. The current presentation and paper
describe the newer YOLO11 fine-tuning run. For ESPDet, use its notebook below;
the [recorded training configuration](baseline-training-configuration.json) and
[training curves](../assets/baseline-training-curves.png) support the methodology.

Use [the notebook](../notebooks/train_bottle_cup_box.ipynb) or [the run guide](laptop-esp32.md). [Machine-readable evaluation](model-evaluation.json), [training history](training-results.csv), and [model provenance](../weights/obstacle/model.json) are included. Raw images, full run checkpoints, and local environments stay outside Git.

Next steps are more box images and camera-height-specific recordings, an independent course test, and separate INT8/ESP32 evaluation. No physical car behavior is validated here.
