# Sample results

Recorded on 2026-10-16 using real pretrained-model inference. No board or motors were connected.

## Actual bottle recognition and distance-aware continuation

![Actual model output](../assets/laptop-demo.png)

Actual bottle recognition and distance-aware continuation. A distant bottle remains visible without blocking the center.

Instruction in this recorded frame: **FORWARD** — No nearby center obstacle detected. Distances are learned estimates and labels may be wrong; this is an integration example, not a calibrated course test.

Source image: Open Images ID `487815c82c579830`, author **Kasia**, [original image](https://farm1.staticflickr.com/2097/2258189812_98f8dcc2f1_o.jpg), [image license](https://creativecommons.org/licenses/by/2.0/). VisionDrive resized the image and added model predictions/instruction overlays. The file hash, split, and annotations are retained in the [dataset manifest](../data/public/bottle-cup-box-manifest.jsonl).

## Actual depth-based near surfaces on a box scene

![Actual model output](../assets/unknown-obstacle-demo.png)

Actual depth-based near surfaces on a box scene. No box class is required; the conservative result is STOP.

Instruction in this recorded frame: **STOP** — All three regions blocked. Distances are learned estimates and labels may be wrong; this is an integration example, not a calibrated course test.

Source image: Open Images ID `995243986ca2dd84`, author **Justin Henry**, [original image](https://farm5.staticflickr.com/2357/1940024041_e7bc88652f_o.jpg), [image license](https://creativecommons.org/licenses/by/2.0/). VisionDrive resized the image and added model predictions/instruction overlays. The file hash, split, and annotations are retained in the [dataset manifest](../data/public/bottle-cup-box-manifest.jsonl).

## Interpretation and measurement limits

The six-second input produced 179 display frames and 11 unique processed observations. The application ended in STOP / VIDEO ENDED. Median combined inference time was 477 ms on this development CPU, excluding camera capture/rendering. The application drops old frames to avoid an inference backlog. These counts and timing are integration evidence, not an avoidance-accuracy score.

LEFT/RIGHT behavior is covered by controller and known-depth-field tests. This particular public-image clip produced FORWARD and STOP; it is not evidence of successful physical turning. See [machine-readable smoke results](laptop-smoke-results.json), [verification](verification.md), and [the original detector report](model-report.md). The old three-class metrics must not be attributed to the hybrid.
