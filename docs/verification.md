# Verification report

Recorded on **2026-10-17** for the fresh pretrained evaluation.

- **127 Python tests passed** on Windows/Python 3.13.5 after the YOLO11n repository pivot. Tests for retired functionality were removed; coverage includes the camera/video pipeline, region logic, depth processing, stabilization, stale-input STOP, ESP32 UDP expiry/acknowledgement safeguards, dataset validation, mappings, matching, bootstrap grouping, independent-camera import and COCO image/instance/crowd counting. The CLI rejects the removed backend and guides missing-checkpoint users to the YOLO setup script.
- COCO class counts were computed from the official 2017 annotation ZIP: 118,287 training images and 5,000 validation images. All 80 category names match the project's original YOLO class mapping. The exported CSV separates distinct images, non-crowd instances and crowd regions; the statistics JSON preserves source SHA-256 hashes. README and notebook citation/provenance links were checked; notebook code and saved inference outputs were unchanged.
- Native firmware command-state tests compiled with MinGW g++ 8.1.0, C++11, `-Wall -Wextra -Werror -static`, and passed. MinGW's bin directory was added to PATH for its runtime. UDP loopback tests ran outside sandbox network isolation.
- The full **12-section notebook executed all 11 code cells successfully** in 100.9 seconds using RTX 3050 CUDA inference. It made genuine predictions and generated the measured report, tables and plots. Only textual repository-path prefixes in saved outputs were redacted as `<repo>` for portability.
- The fresh **400-image dataset reconstructed exactly** from the frozen manifest into a separate cache: identical manifest, hashes, converted labels and no validation errors. Original source images total 131,371,983 bytes. No previously inspected image is in this cohort.
- The source-only protocol and pixel hashes were committed as **8a6f52a before first fresh-test inference**. Results retain that commit, dirty-tree disclosure, source digests, configuration, environment and checkpoint checksum. Uncommitted documentation was present during execution.
- Original YOLO11n SHA-256 remained unchanged. No YOLO training or fine-tuning ran under this protocol.
- Actual YOLO + metric depth + controller + explanation rendering were benchmarked separately over 20 fresh-cohort images. This is serial software processing, excluding capture, GUI, asynchronous scheduling, UDP and motors.
- A saved-video workflow smoke test used a clearly identified **12-public-photo slideshow**, not newly recorded camera footage. It processed 74 observations over 192 display frames with actual hybrid inference and ended in **STOP / VIDEO ENDED**. It verifies software integration, not navigation accuracy.
- Controller fixture checks covered clear/center/side/all-blocked/emergency/stale/invalid cases with acceptable action sets. They are explicitly labelled software fixtures. Moving-obstacle and low-light reference evaluation remains pending real recordings.
- Decision rules, instruction freshness, depth processing, pipeline, UDP and firmware safeguards retain their behavior.

The dataset source protocol and visual spot checks do not constitute independent exhaustive reannotation. Pilot chair scoring was rejected because source Stool/Chair coverage does not match COCO chair; those numbers remain exploratory. The final primary score is person only, on a fresh cohort frozen after the pilot's methodological correction.

No ESP32 board, motors or physical course was connected. Live webcam capture was not tested in this run; the existing saved-video workflow was tested. Metric depth accuracy, independent car-camera generalization, end-to-end live-camera FPS, stopping distance and physical collision avoidance remain unmeasured. COCO pretraining overlap is not ruled out.



## Repository scope

YOLO11n is the only supported object detector. Its original checkpoint, measured
person and bottle/cup results, source manifests, graph data and executable
evaluation notebook are retained. The laptop can run object-only or object-plus-
depth modes. The optional ESP32 remains a command receiver.

The cleanup removed 26 obsolete model/training/support files. The README,
paper guide, demo notebook and third-party attribution now describe the current
system. The repository contains eight focused Markdown documents. Historical
versions remain available through Git history.

Previously inspected image IDs are retained in `prior-image-exclusions.json`
so frozen person-cohort exclusion checks remain reproducible. Two remaining
application screenshot photos retain their source/license/hash records in
`assets/application-source-attribution.json`; the unrelated dataset is removed.
Measured JSON/CSV results and all executed notebook code/output cells are unchanged.

Both remaining demo modes were smoke-tested with real GPU inference on a
90-frame slideshow made from two official COCO photos. Object-only and hybrid
modes each wrote 91 display records, terminated with STOP / VIDEO ENDED, and
ran without hardware commands. These are software integration checks, not
physical driving or recognition-accuracy measurements.

## Bottle/cup benchmark verification

- All 5,000 official COCO validation images decoded at the annotated dimensions;
  ZIP CRC verification and per-image SHA-256 checks completed.
- Manifest/configuration were committed in `127dfbe` before this experiment's
  inference. The original YOLO11n hash was unchanged. No training ran.
- Actual RTX 3050 FP32 inference completed for every image. Official pycocotools
  2.0.11 scored bottle and cup, retaining crowd/ignore handling. Fixed-count
  fixtures verify ignored crowd matches and repeated area bins are excluded.
- The new Section 13 executed its inference/scoring cell successfully. The main
  notebook now contains 13 sections and 12 executed code cells. Earlier person
  study code and outputs are preserved. The saved table and chart are generated
  from genuine predictions; they are not published upstream values.
- This is known COCO validation data, not an independent vehicle-camera trial.
  No physical movement, calibrated depth or instruction accuracy is established.
