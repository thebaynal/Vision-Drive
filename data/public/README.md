# Dataset preview and download

## Current pretrained YOLO11n evaluation

**400 fresh real indoor photos: 80 development / 320 test; 923 person annotations.**
Person is the only primary-scored class. Original YOLO11n remains unchanged.
Every photo has a different photographer; prior inspected images are excluded.
The protocol was frozen in Git before first inference on these images.

- [Dataset card, source comparison and licenses](../../docs/dataset_card.md).
- [Frozen labels and individual attribution](indoor-evaluation-manifest.jsonl).
- [Source CSV checksums and selection audit](indoor-selection-audit.json).
- [Excluded source category counts](indoor-excluded-source-classes.csv).
- [Executable notebook](../../notebooks/yolo11n_pretrained_evaluation.ipynb).
- [Measured results](../../docs/evaluation_results.md).
- [Pilot Chair/Stool limitation](../../docs/dataset_card.md#chair-stool-taxonomy).

```powershell
.venv-evaluation\Scripts\python scripts/download_dataset.py
.venv-evaluation\Scripts\python scripts/validate_annotations.py
```

Restoration checks exact hashes and reuses local data. Photos are CC BY 2.0;
source annotations are CC BY 4.0. This is a project cohort, not an official score.
Raw images are not committed. The old 249-image pilot manifest is retained only
for audit; its chair score is exploratory and does not validate furniture recognition.

## Supplementary COCO bottle/cup benchmark

The [5,000-image manifest](coco-objects-manifest.jsonl) includes every COCO
val2017 image, with source pages and individual license metadata. It supports
the separate bottle/cup evaluation, not the fresh person-only study above.
See the [dataset and reproduction protocol](../../docs/dataset_card.md#secondary-bottle-and-cup-benchmark).
Raw photos are downloaded from [COCO](https://cocodataset.org/#download) and kept
outside Git. [COCO class counts](coco2017-class-counts.csv) describe the original
training and validation distributions.

## Historical ESPDet-Pico baseline: preview and download

The original bottle/cup/box experiment retains **402 images** (280 train, 60
validation, 62 test) and 688 object instances. Its source is Open Images, with new
project partitions. It is separate from current pretrained YOLO evaluation.

![Historical annotated training/validation/test examples](../../assets/dataset-preview.png)

- [Download page](https://github.com/thebaynal/Vision-Drive/releases/tag/dataset-v1.0.0).
- [Dataset ZIP](https://github.com/thebaynal/Vision-Drive/releases/download/dataset-v1.0.0/VisionDrive-bottle-cup-box-dataset.zip).
- [Per-image manifest and attribution](bottle-cup-box-manifest.jsonl).
- [Preview image credits](preview-sources.json).
- [Original training notebook](../../notebooks/train_bottle_cup_box.ipynb).
- [Measured ESPDet results](../../docs/model-report.md).

These photos were gathered from their attributed public sources, not captured by
the group. Keep author, original page, license and transformation notices when
presenting them.
