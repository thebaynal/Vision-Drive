# Acceptance audit

| Requested phase | Delivered evidence |
| --- | --- |
| Inspect and preserve | Pre-change inventory, initial tests, recoverable commit 4d1e0cf; [cleanup ledger](repository-inventory.md) |
| Acquire real indoor data | 400 validated Open Images interiors; [candidate/source/license comparison](dataset_card.md), per-image attribution and source CSV hashes |
| Validate annotations | Original COCO person=0, verified positive/negative coverage, source hierarchy handling, invalid/ambiguous/group exclusions, exact/near-duplicate and photographer checks |
| Avoid project leakage | Fresh cohort excludes all pilot and ESP baseline image IDs; 80/320 official-source split; one photo per photographer; protocol frozen at 8a6f52a before inference |
| Complete notebook | [12 sections](../notebooks/yolo11n_pretrained_evaluation.ipynb), 11 code cells executed end to end; actual native model.val and predictions |
| Genuine metrics | Fixed P/R/F1, TP/FP/FN, AP50/AP75/AP50–95, class table, 95% grouped bootstrap intervals and explicit single-class scope |
| Diagnostics | Raw/normalized matrices, PR/F1 curves, distribution/AP charts, GT/prediction/miss/FP evidence, source attribution |
| Timing | Synchronized 100-frame YOLO timing after 10 warmups; separate 20-image hybrid processing benchmark |
| Optional components | Aligned-depth metric utility, acceptable-set movement scorer, controller fixture checks; real depth/motion accuracy pending reference data |
| Clean outputs | Local outputs/evaluation/yolo11n plus compact docs/results/yolo11n and assets/evaluation; data, weights and caches ignored |
| Academic reporting | [12-part measured report](evaluation_results.md), methodology/card/taxonomy/camera guides and updated portfolio README |
| Verification | 139 Python tests, compiled native watchdog tests, exact reconstruction, real-model video EOF STOP, image/graph and link checks |

**Scope decisions:** Chair was excluded after pilot QA found a Stool/Chair taxonomy defect. The final person-only cohort is fresh, rather than reusing the inspected pilot as untouched test evidence. This is an object-localization study, not a claim about all 80 categories, all obstacles or physical navigation.

**Pending external evidence:** independent team camera sessions, aligned metric-depth ground truth, measured movement-action references and physical hardware trials. These are explicitly reported as pending; no measurements were invented. Pretraining overlap remains uncertain.

