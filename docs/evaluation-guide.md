# Evaluation entry points

Use [the executable notebook](../notebooks/yolo11n_pretrained_evaluation.ipynb) for the current unchanged-pretrained YOLO11n experiment. Installation and exact commands are in the [README](../README.md).

- [Methodology](evaluation_methodology.md): mapping, matching, AP, timing and uncertainty.
- [Dataset card](dataset_card.md): sources, licenses, completeness and leakage.
- [Results](evaluation_results.md): actual measured evidence.
- [Camera protocol](camera-evaluation.md): recordings, annotation and movement/depth references.

The earlier `scripts/evaluate_yolo.py` and [exploratory report](yolo11-evaluation.md) score bottle/cup on the original small dataset. They are historical exploration, not the primary indoor evaluation. ESPDet uses `scripts/evaluate_objects.py` and [its report](model-report.md).
