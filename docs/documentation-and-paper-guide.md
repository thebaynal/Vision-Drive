# Guide for the documentation and research-paper team

Use this guide to write about **VisionDrive** accurately. Follow the instructor's
required paper template and citation style; adapt the suggested structure below
to chapter-based or IMRaD formats. This is a writing guide, not a completed paper.

## 1. Project identity and scope

**Suggested title:** VisionDrive: Camera-Based Obstacle Detection and Movement
Instruction Using Laptop Vision and an ESP32 Controller

**Group members:**

- Sean Xander Aquino
- Blessie Faith Bongalos
- Joshua Jericho Barja
- John Renzzo Montenegro
- John Carlo Nas
- Divino Al Ricafort

**Working specialization:** Computer Vision–Based Obstacle Detection and
Avoidance. Confirm the instructor's exact assigned wording before submission.

The midterm asks **“Can the System SEE and RECOGNIZE?”** Prioritize computer
vision, image processing, dataset preparation, model development, interpretation,
and evaluation. Separate later motor integration from the demonstrated vision
system. No physical board, motor, or autonomous-driving validation is claimed.

The camera represents the vehicle's forward view. A laptop runs YOLO11n object
detection and an indoor metric-depth model. Nearby surfaces can affect decisions
without known class names. Three image regions guide FORWARD, LEFT, RIGHT, or
STOP suggestions. Optional Wi-Fi commands connect to an ESP32 motor receiver.
**The current neural networks run on the laptop, not on the ESP32.**

## 2. Keep the two model experiments separate

| Item | Original training experiment | Current demonstration |
| --- | --- | --- |
| Detector | ESPDet-Pico | YOLO11n |
| Input | 224×224 | 640×640 YOLO; depth resize configured separately |
| Recognized classes | Bottle, cup, box | 80 pretrained COCO categories |
| Additional obstacle cues | None | Metric-depth estimates and nearby surfaces |
| Project training | Fine-tuning on the prepared group dataset | Upstream pretrained weights; no project-specific fine-tuning of these models |
| Execution | Desktop baseline; future onboard research | Laptop perception, optional ESP32 commands |
| Published detector metrics | Three-class baseline available | Exploratory bottle/cup scores available; independent course evaluation pending |

The current detector's separate [YOLO11n report](yolo11-evaluation.md) records
an exploratory two-class evaluation. Use the [evaluation guide](evaluation-guide.md)
for independent camera tests. Neither the detector scores nor the original
baseline measure the hybrid system's depth or instruction accuracy.

Do not attribute the baseline's mAP or recall to the hybrid. Do not say the
402-image group dataset trained YOLO11n or the depth model. The group's work
includes dataset preparation, baseline fine-tuning, integration, region logic,
visualization, communication, and evaluation; it did not invent these networks.

## 3. Read these sources before writing

| Information | Repository source |
| --- | --- |
| Read-aloud dataset/training explanation | [Presentation script](dataset-and-training-presentation-script.md) |
| Formal dataset/training section for the paper | [Paper section](paper-dataset-and-training.md) |
| Recorded settings and plotted training history | [Configuration](baseline-training-configuration.json), [curves](../assets/baseline-training-curves.png) |
| Overview, names, installation, running | [README](../README.md) |
| Midterm requirement mapping | [Midterm report](midterm-report.md) |
| Architecture, distance rules, Wi-Fi, limitations | [Laptop/ESP32 guide](laptop-esp32.md) |
| Dataset preparation, split counts, unseen-situation limits | [Dataset card](../data/public/README.md) |
| Image IDs, hashes, labels, splits, licenses, authors | [Fixed manifest](../data/public/bottle-cup-box-manifest.jsonl) |
| Original training and measured performance | [Model report](model-report.md), [evaluation JSON](model-evaluation.json) |
| Training/validation curves data | [Training CSV](training-results.csv) |
| Actual model screenshots and attribution | [Sample results](sample-results.md) |
| Software checks and unfinished validation | [Verification report](verification.md) |
| Current integration-check counts and timing | [Smoke-test JSON](laptop-smoke-results.json) |
| Baseline model provenance | [Model manifest](../weights/obstacle/model.json) |
| Hardware status and firmware handoff | [Hardware guide](hardware-implementation.md) |
| Upstream models and licenses | [Third-party notices](../THIRD_PARTY_NOTICES.md) |

Inspect the [training notebook](../notebooks/train_bottle_cup_box.ipynb) and
[current model walkthrough](../notebooks/laptop_esp32_demo.ipynb). Ask the
implementation team to explain any step you cannot trace to an artifact. Record
the final paper's software version with `git rev-parse HEAD`.

## 4. Suggested paper structure

### Title page and abstract

Confirm the title, specialization, group names, course, instructor, institution,
and submission date. Obtain missing administrative information rather than
inventing it. Write the abstract last: summarize the problem, camera/laptop/ESP32
division, vision methods, tests actually performed, results, and main limitation.
Describe a working prototype without implying a tested self-driving car.

### Introduction, problem statement, and objectives

Explain why a forward-camera system needs to interpret blocking obstacles and
why a few named object classes are insufficient for the broader objective.
State a testable problem: how can a controlled indoor vision prototype recognize
objects or nearby surfaces and produce understandable movement instructions?

Separate implemented objectives—dataset preparation, baseline training, visual
interpretation, proximity rules, explainable instructions—from future physical
avoidance objectives. Identify educational relevance and the intended environment.

### Related work and background

Explain object detection, bounding boxes, confidence, nonmaximum suppression,
transfer learning, monocular depth, dataset splits, and temporal confirmation
only as needed to understand the design. Cite original papers, official model
cards, and documentation. Distinguish upstream architectures from the group's
implementation contributions. Avoid unsupported claims that a chosen model is
universally the best or recognizes every object.

### Methodology and system design

Describe the implemented workflow in order:

1. Capture an unmirrored forward frame or recorded frame, retaining timestamps.
2. Resize/letterbox for detection; normalize and convert channels for depth.
3. Run laptop detection and metric-depth estimation.
4. Restore boxes to image coordinates and estimate object proximity.
5. Reject likely flat-floor depth pixels and extract connected nearby surfaces.
6. Exclude distant obstacles from blockage; calculate region occupancy.
7. Apply deterministic rules and confirm movement across fresh observations.
8. Display the explanation; optionally send expiring Wi-Fi commands to the ESP32.

Copy actual settings from [the config](../config/default.json). Initial defaults
are estimated 1.5 m approach distance, estimated 0.45 m emergency center distance,
three matching processed observations, and a 1,000 ms motor freshness limit. These are
demo values, not calibrated stopping distances. Expired outputs stop immediately;
fresh observations within the allowed capture gap can continue confirmation.
True outages, hazards, invalid input, and changed directions interrupt confirmation
as applicable. This avoids confusing a display timeout with a new observation.

The display now has a separate, bounded five-second hold for the last visual
instruction while the next frame is analyzed. It explicitly labels that state;
input failure or a stalled model still stops the preview. Only the stricter
motor controller supplies ESP32 commands. Describe these as separate outputs,
not as a five-second permission for a physical car to move on an old observation.

If describing hardware, explain STOP handshakes, sequence checking, command
expiry, and the receiver's independent watchdog. Separate native C++ tests from
an Arduino board build, upload, Wi-Fi range test, or physical motor test.

### Dataset preparation and unseen situations

The original dataset came from Open Images' validation source and was
repartitioned into new project sets after deduplication. It is not the official
Open Images training/validation/test benchmark.

| Project set | Images | Bottle instances | Cup instances | Box instances |
| --- | ---: | ---: | ---: | ---: |
| Training | 280 | 294 | 136 | 73 |
| Validation | 60 | 48 | 28 | 15 |
| Testing | 62 | 43 | 25 | 26 |
| Total | 402 | 385 | 189 | 114 |

Explain class mapping, YOLO labels, normalized coordinates, exclusion of
depiction/group annotations, content-hash deduplication, and preserved provenance.
The approximate image distribution is 70% / 15% / 15%; report exact counts too.

Distinct hashes do not establish different scenes/sessions or independence from
upstream pretraining. For the next collection, reserve entire sessions and new
physical object instances for testing. Vary lighting, distance, angle, background,
occlusion, and arrangements. Include verified empty floors and difficult objects.
Never put adjacent frames from one clip into different splits. Audit missing
labels so unlabeled obstacles are not treated as background. Distinguish this
planned protocol from data collection already completed.

### Baseline training and evaluation

ESPDet fine-tuning rebuilt a three-class head and transferred initialization
weights. Training completed 55 epochs with early stopping; validation selected
epoch 40 and confidence 0.35. Explain why checkpoint/threshold selection uses
validation and test evaluation happens separately. These test photos were later
used for illustrations; they are not an independent test of the newer hybrid.

### Results and discussion

Use the original detector's measured table with its exact scope:

| Class | Precision | Recall | F1 |
| --- | ---: | ---: | ---: |
| Bottle | 0.559 | 0.442 | 0.494 |
| Cup | 0.760 | 0.760 | 0.760 |
| Box | 0.222 | 0.154 | 0.182 |

These use confidence 0.35 and IoU ≥ 0.5 with class-aware one-to-one matching on
62 project test images. Baseline mAP@0.5 is 0.469; mAP@0.5:0.95 is 0.357.
Explain weak box recall. Neither mAP nor recall is a driving-accuracy percentage.

Present current hybrid screenshots as qualitative integration examples. Its
recorded check processed 11 observations from a six-second clip, produced
179 display frames, and ended in STOP / VIDEO ENDED. Median combined inference
was about 477 ms on the development CPU, excluding capture/rendering. Display
frames are not inference FPS. This clip showed FORWARD and STOP; controller
tests cover LEFT/RIGHT, but there is no measured physical-turn success rate.

The current suite has 123 Python tests plus native C++ watchdog/protocol
assertions. These measure software behavior, not model accuracy. Recheck the
[verification report](verification.md) before freezing test counts in the paper.

### Limitations, conclusion, and future work

Discuss depth error, floor-fit failures, false stops, thin/transparent obstacles,
lighting, occlusion, camera mounting, latency, and the difference between image
regions and real chassis/turning clearance. Include failures, not just successes.

Conclude at the level demonstrated: a visual interpretation/instruction prototype,
baseline training evidence, and a documented motor-control interface. Future
work includes new scene-disjoint data, annotation review, calibration, independent
hybrid evaluation, board integration, and measured physical stopping/turning.
Onboard ESP32 inference is a separate future experiment.

## 5. Figures and tables to prepare

- Architecture: camera → laptop detection/depth → controller → visualization /
  Wi-Fi → ESP32 → motor driver.
- Dataset distribution with image and instance counts.
- Correctly attributed annotated examples, with split names.
- Baseline training/validation curves from the CSV, with labeled axes.
- Baseline precision/recall/F1 and clearly defined mAP.
- Actual screenshots with predicted labels, estimated distances, instructions,
  and limitations; use [the supplied attribution](sample-results.md).
- A new unseen-course results table only after collecting measurements.

Label synthetic examples as synthetic. Never present simulation as model
inference or a diagram as a screenshot. Retain source credits when moving
images into slides or the paper.

## 6. Claims checklist

| Supported wording | Wording that overstates the evidence |
| --- | --- |
| Laptop vision with an optional ESP32 receiver | The hybrid runs onboard an ESP32 |
| Unnamed nearby surfaces can influence steering | The system detects everything |
| Learned distance estimates need calibration | The camera measures exact distance |
| The baseline achieved the reported test metrics | The hybrid has the baseline's measured accuracy |
| Tests verify software behavior | 123 tests are 123 successful driving trials |
| Receiver/watchdog code is native-tested | A physical car is proven to stop safely |
| Image-hash-disjoint splits were prepared | All test scenes and pretraining sources are independent |
| Image occupancy guides a suggestion | The turning path is proven collision-free |

## 7. Work allocation and final review

Choose owners within the group; this guide does not assign roles to members.
Suggested responsibilities are dataset/provenance, methodology/architecture,
results/evaluation, citations/formatting, and a final technical review. Maintain
one shared version of tables and figure captions.

Before submission, confirm the official specialization, instructor's format,
roles, deadlines, slide/video links, final commit, and evaluation sessions actually
performed. Have implementers verify every numerical table and hardware/model
claim. Mark unknowns as not measured. Never invent accuracy, timing, electrical
specifications, author lists, or participant information.

## 8. Primary references to consult

- [Depth Anything V2 paper](https://arxiv.org/abs/2406.09414)
- [Metric Indoor Small model card](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf)
- [YOLO11 documentation](https://docs.ultralytics.com/models/yolo11/)
- [Espressif ESP-Detection](https://github.com/espressif/esp-detection)
- [Open Images downloads and documentation](https://storage.googleapis.com/openimages/web/download_v7.html)
- [Transformers depth documentation](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2)
- [Arduino-ESP32 documentation](https://docs.espressif.com/projects/arduino-esp32/en/latest/)

Build citations from actual source metadata using the required style. Do not
invent a YOLO11 paper, DOI, publication date, or author list. Cite documentation
as a web source when that is what you used. Record access dates for changing
documentation and image source/license links for figures.
