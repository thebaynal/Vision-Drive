# Current pretrained evaluation — presentation script

“VisionDrive treats the camera as the vehicle's forward view. The laptop runs
original YOLO11n plus a separate metric-depth model. Three-region logic explains
FORWARD, LEFT, RIGHT or STOP. The optional ESP32 receives expiring commands.

We did not train or fine-tune YOLO11n. We evaluated its existing person category
on 400 real indoor public photos: 80 development and 320 test, containing 923
source-annotated people. Every image has verified person presence or absence.
Each photo has a different photographer; all earlier inspected images are excluded.
We froze the source-only protocol in Git before inference on the fresh test set.

The photos and original annotations came from Open Images. Individual photos
are credited under CC BY 2.0; annotations are CC BY 4.0. We selected building
interiors, checked annotation geometry and hashes, and excluded group-of,
ambiguous person labels and unverified images. We used original COCO ID 0.

Our pilot found that Open Images separates Stool from Chair. Scoring source
Chair alone could mark genuine COCO chair detections on stools as false
positives, so furniture is outside the primary score. The live detector retains
all 80 categories, with separate depth cues for unnamed nearby surfaces.

On the fresh 320-image test set, precision was 79.19 percent,
recall 63.93 percent and F1 70.75 percent at confidence .35 and IoU .50.
AP at .50 was 72.12 percent; AP over .50 through .95 was 50.94 percent.
Only one category is scored, so mAP equals person AP. This is object localization,
not ordinary classification accuracy or collision-avoidance success.

The notebook records actual predictions, per-class counts, uncertainty intervals,
failure examples and synchronized GPU timing. Public photos do not prove
independence from COCO pretraining or performance on our car's camera.
Independent camera sessions, reviewed labels, calibrated depth and physical
trials remain necessary. Existing STOP and watchdog protections remain enabled.”
