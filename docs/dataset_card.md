# Fresh indoor person evaluation dataset

## Objective and measured scope

Evaluate original COCO-pretrained YOLO11n for **person localization** in real
indoor photographs. The final cohort contains **400 validated images** and
**923 annotated person instances**. Chair, furniture and all other model classes
are outside the primary numerical score; the live model keeps all 80 categories.

| Partition | Original source split | Images | Person-positive images | Verified negative images | Person instances |
| --- | --- | ---: | ---: | ---: | ---: |
| Development | Open Images validation | 80 | 60 | 20 | 191 |
| Final test | Open Images test | 320 | 240 | 80 | 732 |
| Total | Both | 400 | 300 | 100 | 923 |

One photo per photographer is retained; all 400 groups are distinct. All prior
inspected pilot and ESP baseline image IDs are excluded. The source-only
protocol and complete image hashes were committed as **`8a6f52a` before first
inference on this fresh test set**. No threshold or image selection was adjusted
using the final test results. The 300–500-image planning target is met, but it is
not a statistical minimum or proof of broad generalization.

These images were not captured by the group. No team camera recordings or
metric depth references were supplied. The 75/25 positive/negative stratification
is deliberate; precision depends on this sample prevalence.

## Candidate investigation

| Candidate | Suitability and decision |
| --- | --- |
| [SUN RGB-D](https://rgbd.cs.princeton.edu/), [official files](https://rgbd.cs.princeton.edu/data/) | Relevant RGB-D interiors and dedicated v2 2D box metadata. Homepage/README access failed; a definitive named image license was not verified. Not adopted or assigned an invented license. |
| [Roboflow interior-item-detection](https://universe.roboflow.com/prabhakar-egqij/interior-item-detection) | CC BY 4.0 listing, 57 heterogeneous labels. Generic Table and duplicate vocabulary require curation; exhaustive annotation and upstream independence unresolved. |
| [Roboflow-100 furniture](https://universe.roboflow.com/roboflow-100/furniture-ngpea) | CC BY 4.0; Chair/Table/Sofa. Generic Table is not automatically COCO dining table; completeness/upstream overlap unresolved. No export restriction bypassed. |
| [Mendeley indoor v2](https://doi.org/10.17632/3ggxwf2vpr.2), Nafiz Fahad | CC BY 4.0; seven classes, VOC archive. Official file metadata obtained: 453,984,831 bytes, SHA-256 3d5a25726f045f230eccbabfc9e3ef4db3916464dcda4d7f48102e7a9ababd20. Archive download returned an access challenge and was stopped. No successful image acquisition or completeness claim. |
| [NYU Depth V2](https://cs.nyu.edu/~fergus/datasets/nyu_depth_v2.html) | Relevant dense indoor instance labels/depth. Named redistribution terms were not established from the author page. Not downloaded or assigned a guessed license. |
| Existing Open Images bottle/cup/box | Genuine historical baseline but narrow labels and incomplete coverage of other COCO classes. Preserved separately; original baseline image IDs excluded from the new cohort. |
| Open Images indoor Person | Selected for documented exhaustive person-positive boxes, verified negatives, explicit interior attributes and reproducible licensed access. The initial chair pilot exposed a taxonomy defect and was excluded from primary evidence. Public-photo viewpoint remains a limitation. |

Selection favors coverage and provenance over recycling convenient images. No synthetic images or augmentation are included.

## Source releases, licensing and attribution

Official **V5 bounding boxes**, human-verified boxable image labels and
**2018_04 image metadata** are interpreted using the
[V7 specification](https://storage.googleapis.com/openimages/web/download_v7.html).
The [source protocol](https://storage.googleapis.com/openimages/web/factsfigures_v7.html)
describes exhaustive instances for verified positive categories and verified
negative image labels. This selected cohort is not an official Open Images score.

Photos: **CC BY 2.0**, individually attributed to photographer, title and original
page. Annotations: **CC BY 4.0**, Open Images/Google. Cite Kuznetsova et al.,
*The Open Images Dataset V4*, IJCV 2020,
[paper](https://arxiv.org/abs/1811.00982). Retain author, source, license and
modification notices when showing or redistributing photos.

The [frozen manifest](../data/public/indoor-evaluation-manifest.jsonl) records
every source URL, author/license, source split, original category, normalized
box, pixel hash and dimensions. The [selection audit](../data/public/indoor-selection-audit.json)
pins official CSV URLs, byte sizes and SHA-256 values. These are acquisition
hashes, not publisher signatures. Downloaded image bytes total 131,371,983 bytes.

## Selection and annotation methodology

1. Require a Building-family annotation with `IsInside=1`, including appropriate
   House, Office building and other building descendants.
2. Require human-verified **Person presence or absence**; unknown never means absent.
3. Map Person `/m/01g317`, Man `/m/04yx4`, Woman `/m/03bt1vf`, Boy `/m/01bl7v`,
   Girl `/m/05r655` to original **COCO person=0**. Retain all 80 YAML names and
   apply native filter `[0]`, rather than inventing a custom head.
4. Propagate person subtypes. Collapse coincident boxes only across source
   categories at IoU≥.95. Remaining cross-source person overlaps at IoU>.30 cause
   whole-image exclusion because identity is ambiguous; no merge is guessed.
5. Exclude target group-of, depiction, inside-target/unknown flags, label-box
   conflicts, invalid geometry and missing approved attribution.
6. Source-only seeded selection uses 20261017, selects official test before
   development, admits one image per photographer, and takes 75% positive / 25%
   negative images in each partition. No model is imported by the selection script.

After excluding prior images, source filtering found 1,460 eligible official-test
and 484 eligible official-validation images. Candidate exclusions were:

| Reason | Official test | Official validation |
| --- | ---: | ---: |
| Prior inspected/baseline image | 191 | 66 |
| Unverified person presence/absence | 1,297 | 407 |
| Target attributes | 625 | 243 |
| Ambiguous person hierarchy | 105 | 31 |

These are sequential image-filter counts, not distinct physical objects.
The final sampling then selected 320 test and 80 development images. All 400
downloads validated; no exact/likely-near duplicate removal was needed.
Source categories outside person are counted separately in
[excluded-source-classes.csv](../data/public/indoor-excluded-source-classes.csv).
They are annotation records, not primary-score ground truth or absent objects.

Source protocol, structural checks and visual spot checks are not independent
exhaustive reannotation and cannot remove every source error. Annotated sheets
for all retained images are generated by the notebook.

## Pilot taxonomy correction and independence

The first 249-image person/chair pilot exposed the Stool/Chair mismatch during
figure QA. Its source Chair labels cannot establish complete COCO-chair coverage.
The pilot is exploratory and does not supply final test evidence. See the
[taxonomy audit](taxonomy-audit.md) and [archived pilot manifest](../data/public/exploratory-indoor-manifest.jsonl).
The final person-only cohort is fresh and excludes every pilot image ID.

SHA-256 and 64-bit dHash (Hamming distance≤4) check internal duplicates. Official
splits and distinct photographer groups reduce project leakage. Physical scene
identities across photographers remain unknown. **Independence from COCO
pretraining is not established**: both datasets contain public photos and a
complete cross-dataset overlap audit was not performed. No camera-domain or
physical avoidance result is implied.

## Reconstruct and validate

```powershell
.venv-evaluation\Scripts\python scripts/download_dataset.py
.venv-evaluation\Scripts\python scripts/validate_annotations.py
```

Normal acquisition restores exactly the distributed frozen manifest, checks each
hash and reuses valid local data. `scripts/build_evaluation_cohort.py` documents
the original source-only selection; it is not necessary for normal restoration.
YOLO labels are normalized original-ID records. Local absolute image lists and
dataset YAML stay ignored; scoring regenerates lists strictly from the manifest.
No failed download is silently replaced. No YOLO training split is created;
Ultralytics' required `train` YAML key aliases development and is unused.
