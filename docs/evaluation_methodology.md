# Pretrained YOLO11n evaluation methodology

The protocol evaluates unchanged original YOLO11n. It never trains or fine-tunes YOLO11n. Historical ESP results and earlier bottle/cup exploration remain separate.

## Reproducible inputs

- Official Ultralytics v8.3.0 asset `yolo11n.pt`, SHA-256 `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
- [Configuration](../config/evaluation.yaml): FP32, batch 1, square 640 letterboxing, NMS IoU .70, max 300 detections.
- [Frozen real indoor cohort](dataset_card.md), source-verified person coverage.
- Person labels retain original ID 0 and all 80 YAML names; only person is primary-scored. No custom 0/1 head substitution.
- Environment, manifest/checkpoint/source hashes and Git commit/dirty state accompany actual runs.

## Operational P/R/F1

These are research operating-point measurements. Evaluation uses NMS IoU .70;
the live application's default NMS is .45. Do not describe these scores as an
end-to-end measurement of the live application's exact configuration.

Confidence≥.35; descending-confidence predictions; same-class one-to-one highest-available-IoU≥.50 matching. Unmatched prediction=FP; unmatched truth=FN. Duplicate predictions cannot both match one truth.

`P=TP/(TP+FP)`, `R=TP/(TP+FN)`, `F1=2TP/(2TP+FP+FN)`. Undefined ratios use zero, with sample counts shown. Micro pools counts. Macro P/R/F1 average corresponding class scores; macro F1 is not the harmonic mean of macro P/R.

## Average precision and matrices

Native Ultralytics `model.val()` uses the 80-name YAML, class filter [0], confidence floor .001, IoU thresholds .50:.05:.95. AP@75 is column 5. Only person is scored, so mAP equals person AP. This is not classification accuracy.

Native matching and its best-F1 P/R differ from our fixed operating-point matcher; never combine native P/R with fixed TP/FP/FN. Test PR/F1 maxima do not select a new runtime threshold.

The fixed-point detection confusion matrix shows TP, misses and unmatched predictions. Wrong-class detections contribute FP+FN. Off-diagonal class substitutions are not inferred. Row normalization includes the unmatched-prediction row. Native plots are retained separately.

## Timing and uncertainty

Ten warmups precede timing; CUDA synchronizes before/after measured calls. The first 100 test images in manifest order are decoded before measurement. Report pre/network/post and Python predict-call wall mean, median, population SD and p95. FPS=1000/mean milliseconds. Hardware, batch, precision and resolution are recorded. Model-only FPS excludes camera, depth, GUI, controller, UDP and motors.

1,000 seeded percentile bootstrap samples resample complete photographer groups (scene proxy). Report 95% intervals for micro P/R/F1 and group count; no AP interval claimed. Bootstrap cannot fix unknown shared scenes, sampling bias or pretraining overlap.

## Error analysis and limits

Condition recall uses source occlusion/truncation, source-pixel object area, full-image grayscale luminance and target counts. These observational bins are confounded, not causal lighting/perspective/clutter experiments. Actual FP/miss examples carry source attribution.

No metric depth reference or independent group camera recordings were supplied. Depth and physical navigation accuracy remain pending. Fixture-based movement results are software checks, not collision-avoidance evidence.


## Pilot correction and fresh frozen final test

Pilot figure QA found that source Stool and Chair are not exhaustive for COCO
chair. The pilot is exploratory; its scores do not validate furniture recognition.
The final person-only protocol was frozen as commit `8a6f52a` before inference
on a fresh 400-image cohort excluding every pilot and baseline image ID.
Its 80 development / 320 test photos have distinct photographers, source-verified
person presence/absence and a fixed 75% positive / 25% negative ratio. Source-only
selection used seed 20261017. No final test result selected thresholds or images.
See [the dataset card](dataset_card.md) and [taxonomy audit](taxonomy-audit.md).
