# YOLO11n project-image evaluation

Recorded on **2026-10-17**. This is an exploratory evaluation of the pretrained
object detector on the existing project images, not a whole-system driving test.

## Dataset and protocol

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

## Measured results

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

## Interpretation and limitations

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
remain unmeasured. Follow the [evaluation guide](evaluation-guide.md) to collect
independent camera recordings and measured distances for those results.

## Reproduce

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
