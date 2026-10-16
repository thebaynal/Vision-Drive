# VisionDrive baseline dataset card

The group prepared **402 images / 688 annotated instances** for the original
bottle/cup/box ESPDet experiment. The current YOLO-plus-depth laptop demo uses
upstream pretrained models; this card does not describe their full training data.

| Split | Images | Bottle instances | Cup instances | Box instances |
| --- | ---: | ---: | ---: | ---: |
| Training | 280 | 294 | 136 | 73 |
| Validation | 60 | 48 | 28 | 15 |
| Testing | 62 | 43 | 25 | 26 |

Source: Open Images official validation image/box pool, repartitioned into
new **project** splits after content deduplication and stratification. These
are not the official Open Images benchmark partitions. The source annotations
map Bottle → bottle, Coffee cup → cup, and Box → box. Labels are class IDs
0/1/2 with normalized bounding boxes. Boxes marked depiction/group are excluded.

[The fixed manifest](bottle-cup-box-manifest.jsonl) records each image's
content hash, ID, relative prepared path, project split, source URL, author,
license, and original boxes. Image files are downloaded separately rather than
bundled in Git. Annotations and individual images retain their upstream terms;
see [attribution](../../THIRD_PARTY_NOTICES.md).

## Reconstruct the prepared splits

After installing the model requirements:

```powershell
.venv-inference\Scripts\python scripts/restore_dataset.py --download
```

This restores the exact published split, verifies image SHA-256 values, and
creates ignored `datasets/bottle-cup-box/` images, YOLO labels, and dataset YAML.
It refuses to overwrite an existing output dataset. Use `--output` for another
location. Existing verified source images in the local cache are reused.
Public image availability can change; a missing or changed image fails rather
than silently substituting another one.

The original sampling/training procedure remains in
[the training notebook](../../notebooks/train_bottle_cup_box.ipynb).

## Unseen situations and limitations

The three sets contain distinct file content. A random image-level split does
not prove different capture sessions, different real-world objects, or absence
from upstream models' pretraining. The old test images were used for the
published baseline evaluation and later integration illustrations. They must
not be presented as a fresh, prospective test for the newer hybrid system.

For the vehicle-view evaluation, collect new recordings at the intended camera
height and keep whole sessions/locations in one split. Reserve object instances,
lighting conditions, backgrounds, distances, angles, and obstacle arrangements
that do not occur in training. Include verified empty floor, clutter, walls,
small/thin objects, partial occlusion, and near/far transitions. Audit labels so
an unannotated obstacle is not incorrectly treated as background. Reserve a new
test session before threshold/model tuning and report its results separately.

The existing dataset is small, has relatively few box images, and is not a
comprehensive annotation set for arbitrary obstacles or drivable floor. It
supports the baseline training milestone; it does not establish universal
obstacle detection or real-car avoidance accuracy.
