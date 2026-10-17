# Repository inventory and cleanup ledger

Inspection checkpoint: `4d1e0cf` on `codex/pretrained-evaluation`. This commit
preserves the preceding unfinished fine-tuning experiment. No new training is
authorized by the current evaluation protocol.

## File use before structural changes

| Files | Dependencies and consumers | Decision |
| --- | --- | --- |
| `demo.py`, `logic_demo.py`, `run_logic_demo.cmd` | CLI, camera/video, visualization | Preserve entry points |
| `visiondrive/laptop_detector.py`, `depth.py` | Lazy Torch/Ultralytics/Transformers, config, detections | Preserve functioning model adapters |
| `decision.py`, `instructions.py`, `pipeline.py` | Shared safety, stale observations, three regions | Preserve algorithms |
| `esp_link.py`, `firmware/visiondrive_receiver/` | UDP protocol, expiring commands, motor watchdog | Preserve safeguards |
| `detector.py`, `weights/obstacle/`, ESP training scripts | Historical ESPDet baseline | Preserve measured research |
| `scripts/evaluate_yolo.py`, `docs/yolo11-evaluation.*` | Exploratory existing bottle/cup evaluation | Preserve with scope limitation |
| `scripts/train_yolo.py`, `build_large_yolo_dataset.py`, `prepare_yolo_finetune_dataset.py`, `evaluate_finetuned_yolo.py`, `package_yolo_model.py` | Unfinished separate fine-tuning workflow; tests and old notebook reference it | Retire from active workflow; recoverable in checkpoint |
| `notebooks/train_yolo11.ipynb` | Incomplete fine-tuning launcher | Replace with requested pretrained evaluation notebook |
| `data/public/yolo-bottle-cup-box-v2-*` | Unfinished fine-tuning preparation, not baseline | Retire; original baseline manifest retained |
| `tests/test_yolo_training.py` | Tests retired experiment only | Replace with evaluation validation/matching fixtures |
| `tests/test_artifacts.py` | Includes assertion for never-published fine-tuned weights | Remove that invalid expectation; validate real artifacts |
| `docs/`, `data/public/README.md`, root README | Research provenance and instructions | Consolidate active workflow; retain baseline records |
| `.cache/`, `.venv*/`, `runs/`, `data/raw/`, `datasets/`, downloaded laptop weights | Local environments, downloads, prior run files | Ignored; no bulk deletion |

Imports were inspected using Python AST across 29 application/script modules;
the detailed local inventory is `.cache/dependency-inventory.json`. Existing
tests initially returned 133 passed and one failure: a test expected an
unpublished fine-tuned checkpoint. This is not an application safety failure.

## Change ledger

The tables above record intended removals before execution. The final Git diff
and verification report record actual changes. Historical training runs remain
local and ignored; neither their validation scores nor older exploratory scores
are substituted for the new primary evaluation. No flat application module is
relocated merely to create an aesthetic directory hierarchy.

### Applied cleanup

Removed the five unfinished YOLO fine-tuning scripts listed above, their launcher
notebook, their training-only tests and two V2 preparation manifests. Replaced
the earlier training-GPU requirements file with separate CPU/GPU evaluation
definitions. Removed the test expecting an unpublished checkpoint and its
`.gitignore` exceptions. All are recoverable from `4d1e0cf`.

Modified README, dataset index, docs index/run guide/writing guide and laptop
notebook to remove broken fine-tuned-model instructions. Historical ESP records,
checkpoint, curves, notebook, scripts and license remain. The earlier bottle/cup
YOLO report remains explicitly exploratory. New research code lives in
`visiondrive/evaluation/`; no live decision, depth, rendering, communication or
firmware algorithm was changed.

Added the frozen indoor manifest, acquisition/validation/import/evaluation tools,
the executed pretrained notebook, compact measured outputs and graphs, and
methodology/dataset/results/camera/taxonomy documentation. Primary chair scoring
was excluded after taxonomy QA; its genuine two-class pilot is preserved as
exploratory evidence, not promoted to a valid furniture benchmark.
