# Can the System SEE and RECOGNIZE?

**Project:** VisionDrive — Camera-Based Obstacle Detection and Avoidance

**Group members:** Sean Xander Aquino; Blessie Faith Bongalos; Joshua Jericho
Barja; John Renzzo Montenegro; John Carlo Nas; Divino Al Ricafort.

**Working specialization:** Computer Vision–Based Obstacle Detection and
Avoidance. The official assigned wording has not yet been supplied.

## Midterm scope

The supplied requirement prioritizes computer vision, image processing, dataset
preparation, and model development. It requires separate training, validation,
and testing sets, preparation with unseen situations in mind, and a working
system that detects, recognizes, or interprets relevant visual information.

VisionDrive demonstrates a forward-camera view interpreted as a vehicle's
path. Its working laptop application recognizes objects, estimates depth,
extracts nearby surface obstacles, and explains a three-region movement
suggestion. ESP32 motor reception is additional integration work; a connected
or self-driving physical vehicle has not been demonstrated.

## Requirement-to-evidence mapping

The current unchanged pretrained YOLO11n study adds a separate
[400-image indoor evaluation](dataset_card.md),
[executed notebook](../notebooks/yolo11n_pretrained_evaluation.ipynb) and
[measured report](evaluation_results.md). Its 80 development / 320 test images
are evaluation partitions; no YOLO training occurs. The original three-way
ESPDet dataset below remains the evidence for the rubric's training split.

| Requirement | Actual evidence | Remaining qualification |
| --- | --- | --- |
| Appropriate prepared dataset | 402 annotated bottle/cup/box images; [dataset card](../data/public/README.md) | Initial specialization subset; not arbitrary obstacles |
| Training / validation / testing | 280 / 60 / 62 images in the [fixed manifest](../data/public/bottle-cup-box-manifest.jsonl) | Image-disjoint; session/scene disjointness not verified |
| Image processing | Resizing, letterboxing, normalization, RGB conversion, depth/floor processing, connected components | Camera height and distance calibration pending |
| Model development | Original ESPDet fine-tuning, confidence calibration, test evaluation, [training notebook](../notebooks/train_bottle_cup_box.ipynb) | Original box recall is weak |
| Working vision system | Actual laptop YOLO11n + metric-depth inference, instruction viewer, [sample output](sample-results.md) | Broader pretrained model; not trained on group dataset |
| Prepare for unseen situations | Session-based evaluation protocol below; fixed existing split | Independent new course data still needs collection |
| Measured results | [Baseline detector metrics](model-report.md), [current checks](verification.md) | No new hybrid avoidance-accuracy score yet |

The artifacts satisfy the existence of a working visual pipeline and a prepared
three-way baseline split. A claim of reliable performance on unseen vehicle
scenarios requires additional data and evaluation. Instructor acceptance of
the baseline dataset's scope should be checked against the official specialization.

## Computer vision and model methodology

The original experiment rebuilt ESPDet-Pico with three output classes, transferred
initialization weights, and fine-tuned at 224×224. Training used 280 images;
validation used 60 for checkpoint/threshold selection; the selected model was
evaluated on 62 project test images. Training completed 55 epochs with early
stopping and selected epoch 40. Confidence 0.35 was selected on validation only.

The current demonstration uses pretrained YOLO11n at 640×640 plus Depth
Anything V2 Metric Indoor Small. Both run on the laptop. The system estimates
distances for recognized object boxes and extracts unnamed nearby surfaces from
the depth map. A robust inverse-depth plane rejects likely floor. Without a
valid floor fit, nearby surfaces remain conservative obstacles.

Distant detections remain visible without blocking the controller. A nearby
middle obstacle causes a turn toward a usable side; if both sides are usable,
the controller chooses the lower occupied-area side. Emergency center clearance,
no usable side, stale/invalid input, or a paused/ended source produces STOP.
These instructions are deterministic rules, not learned motor commands.

## Dataset preparation and unseen situations

The public source supplied 179 bottle images, 155 cup images, and 68 box images.
Content hashes were deduplicated before new 70/15/15 project partitions were
assigned. Original class names, boxes, hashes, source URLs, authors, and licenses
were retained. The source was Open Images' validation pool; these partitions
are not official benchmark train/validation/test sets.

For the next data collection, reserve entire sessions before extracting frames:

| Split | Planned use |
| --- | --- |
| Training | Several course sessions, supported obstacle types, verified empty-floor examples |
| Validation | Different sessions for confidence/distance/occupancy tuning |
| Testing | New object instances, held-out lighting/backgrounds, unseen arrangements; no tuning |

Avoid adjacent frames from one clip crossing split boundaries. Audit missing
labels and include walls, unfamiliar objects, occlusion, small/thin obstacles,
near/far transitions, and scenes with no obstacles. Measure per-obstacle recall,
false stops on clear floor, estimated-distance error, turn/stop correctness,
and capture-to-decision latency. Record camera placement and course conditions.

The existing test set was evaluated for the original detector and later reused
for integration examples. The new hybrid needs a separate prospective course
test; no guarantee of disjointness from its upstream pretraining is made.

## Results and limitations

The original ESPDet baseline achieved mAP@0.5 **0.469** and mAP@0.5:0.95
**0.357**. Recall at confidence 0.35 was **0.442 bottle**, **0.760 cup**, and
**0.154 box**. These numbers do not belong to the new hybrid model.

The hybrid has run on real recorded images with actual labels, depth estimates,
and instruction overlays. Rule tests cover far-center continuation, nearby
center turning, choice of clearer side, and emergency stop. Synthetic depth
tests exercise unknown obstacles and floor/wall handling. These are integration
and algorithm checks, not a measured real-course accuracy percentage.

Transparent surfaces, mirrors, thin structures, poor lighting, camera movement,
and non-flat floors may cause misses or false stops. Learned meter estimates
need calibration; three image regions do not describe vehicle width or turn
clearance. The ESP32 receiver has native protocol/watchdog tests, but no physical
motor or board validation is claimed.

## Submission materials

- [Dataset and training presentation script](dataset-and-training-presentation-script.md)
- [Dataset and training section for the paper](paper-dataset-and-training.md)
- [Repository overview and group information](../README.md)
- [Dataset card](../data/public/README.md) and [split/annotation manifest](../data/public/bottle-cup-box-manifest.jsonl)
- [Original training notebook](../notebooks/train_bottle_cup_box.ipynb)
- [Current laptop-model notebook](../notebooks/laptop_esp32_demo.ipynb)
- [Original model report](model-report.md), [machine-readable evaluation](model-evaluation.json), [training history](training-results.csv)
- [Sample results](sample-results.md) and [verification report](verification.md)
- [Installation/running/Wi-Fi guide](laptop-esp32.md) and [hardware handoff](hardware-implementation.md)

No presentation-slide URL or separately submitted demonstration-video URL has
been provided. Link those materials here when available; do not claim they exist.
