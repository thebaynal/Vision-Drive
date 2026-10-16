# Verification report

Recorded on **2026-10-16** for the laptop-vision / ESP32-controller prototype.

## Automated checks

**123 Python tests passed** on Windows / Python 3.13.5. Checks cover region
overlap, near/far gating, clearer-side selection, emergency stop, unique-frame
confirmation, stopping forward motion while confirming a turn, invalid/stale
observations, capture/pause/EOF, data conversion, split/hash integrity, preserved
class names, CLI recording, notebook syntax, and model artifact consistency.

Known-depth-field tests verify flat-floor rejection, unknown center obstacles,
near/far walls, emergency obstacles, and invalid-map failure. These test the
depth-processing algorithm using controlled arrays; they are not a neural
depth-accuracy benchmark. Broad-detector adapter tests verify that a far object
does not stop solely because its box reaches the lower image edge.

Timing regressions reproduce 600 ms inference under the 1,000 ms freshness
deadline. Expired outputs still STOP, but fresh, distinct observations within
the allowed capture gap can confirm FORWARD/LEFT/RIGHT instead of restarting
the confirmation count at 1/3 forever. True capture outages, already-stale
results, pause, hazards, and changed candidates still gate or reset movement.

The viewer now has a bounded five-second visual-result hold, independent of
the one-second motor controller. Tests verify continuous preview FORWARD/LEFT/
RIGHT with 600, 1,200, and 2,000 ms inference cadences, immediate failure/hazard
stops, eventual preview expiry, and that the application sends STOP to the motor
link even when its held visual instruction remains FORWARD. Both controllers
receive identical observations, including one-shot input iterators.

An actual-model integration check added an artificial 800 ms delay after each
prediction on a repeated real-image clip. Recorded inference was at least
1,270 ms. After initial confirmation, the viewer held FORWARD for 179 consecutive
display frames without STOP flicker; across all 300 display frames the separate
motor controller remained STOP because the observations exceeded its one-second
deadline. The maximum held frame age was about 2,775 ms. This deliberately
slowed test verifies output separation, not model accuracy or physical driving.

Actual localhost UDP tests cover initial STOP acknowledgement, command sending,
perception expiry, acknowledgement loss/replay, and final STOP. Windows sandbox
network isolation prevented loopback communication inside the sandbox; the
network tests passed outside it. No ESP32 was connected during these tests.

The shared firmware command state compiled with MinGW g++ 8.1.0 using C++11,
`-Wall -Wextra -Werror`, and its native executable passed assertions for initial
STOP, motion, malformed/unauthorized/replayed/out-of-order commands, expiry,
rearming, ownership, disconnect, and `millis()` wraparound. GitHub Actions is
configured to run the Python tests and native C++ checks. A configured workflow
is not a claim that a remote run has completed.

Python compilation and local Markdown/notebook links were checked. Both
notebooks' code cells were executed with real inference enabled and downloads/
retraining disabled; saved outputs remain cleared for portability. The published
dataset reconstruction script also restored and hash-verified all 402 cached
source images into the exact 280/60/62 project split.

## Real model integration

YOLO11n and Depth Anything V2 Metric Indoor Small ran on actual public images
through the same video/worker/controller/viewer pipeline. The six-second source
clip produced **179 display frames / 11 processed observations** and ended in
**STOP / VIDEO ENDED**. Object labels, unnamed near surfaces, estimated distances,
region states, and instructions were recorded as actual inference, not simulation.

Median combined inference time across those processed observations was
**477 ms** on this development CPU. A separate warm repeated-image check ran
about **415–430 ms** per image. These timings exclude camera acquisition and
rendering; they do not establish live end-to-end FPS or motor responsiveness.
Three processed observations can take more than one second at this rate.

The clip contained FORWARD and STOP outcomes. LEFT/RIGHT behavior is verified
in controller/known-depth tests; this clip is not evidence of real turning.
The source consists of previously used object-test photographs, assembled with
padding; it is not an independent vehicle-height camera course. Ground removal
and conservative near-surface false positives require live validation.

See [sample outputs and attribution](sample-results.md) and
[machine-readable smoke results](laptop-smoke-results.json).

## Baseline training evidence

The group's original ESPDet run used 280 train / 60 validation / 62 test images.
It completed 55 epochs, selected epoch 40, and chose confidence 0.35 using
validation macro F1 only. Test mAP@0.5 was 0.469; box recall was weak at 0.154.
See [the original model report](model-report.md). These detector metrics do
not measure the current hybrid, unseen-course performance, or physical driving.

The published [dataset manifest](../data/public/bottle-cup-box-manifest.jsonl)
has 402 unique image hashes and preserved original annotations/attribution.
Hash separation does not establish scene/session or upstream-pretraining
independence. See [the dataset card](../data/public/README.md).

## Environment and remaining validation

The prepared environment uses PyTorch 2.12.0 CPU, torchvision 0.27.0,
Ultralytics 8.3.112, Transformers 4.57.6, NumPy 2.2.6, OpenCV 4.12.0.88,
and Python 3.13.5. Direct requirements and explicit model-download scripts are
included; local environments, downloaded large weights, and raw images are not.

Still unvalidated: physical ESP32 build/upload, Wi-Fi range/latency, motor GPIOs,
driver braking/coasting, live vehicle camera operation, distance calibration,
independent unseen-session recall/false-stop/steering accuracy, chassis clearance,
and stopping distance. No universal-obstacle or reliable autonomous-driving claim
is made. These limitations are included in the [midterm report](midterm-report.md).
