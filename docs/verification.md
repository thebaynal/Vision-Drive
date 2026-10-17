# Verification report

Recorded on **2026-10-17** for the fresh pretrained evaluation.

- **139 Python tests passed** on Windows/Python 3.13.5. Coverage includes the existing camera/video pipeline, region logic, depth processing, stabilization, stale-input STOP, ESP32 UDP expiry/acknowledgement safeguards, dataset validation, mappings, matching, bootstrap grouping and independent-camera import.
- Native firmware command-state tests compiled with MinGW g++ 8.1.0, C++11, `-Wall -Wextra -Werror -static`, and passed. MinGW's bin directory was added to PATH for its runtime. UDP loopback tests ran outside sandbox network isolation.
- The full **12-section notebook executed all 11 code cells successfully** in 100.9 seconds using RTX 3050 CUDA inference. It made genuine predictions and generated the measured report, tables and plots. Only textual repository-path prefixes in saved outputs were redacted as `<repo>` for portability.
- The fresh **400-image dataset reconstructed exactly** from the frozen manifest into a separate cache: identical manifest, hashes, converted labels and no validation errors. Original source images total 131,371,983 bytes. No image from the prior inspected pilot or ESP baseline is in this cohort.
- The source-only protocol and pixel hashes were committed as **8a6f52a before first fresh-test inference**. Results retain that commit, dirty-tree disclosure, source digests, configuration, environment and checkpoint checksum. Uncommitted documentation was present during execution.
- Original YOLO11n SHA-256 remained unchanged. No YOLO training or fine-tuning ran under this protocol.
- Actual YOLO + metric depth + controller + explanation rendering were benchmarked separately over 20 fresh-cohort images. This is serial software processing, excluding capture, GUI, asynchronous scheduling, UDP and motors.
- A saved-video workflow smoke test used a clearly identified **12-public-photo slideshow**, not newly recorded camera footage. It processed 74 observations over 192 display frames with actual hybrid inference and ended in **STOP / VIDEO ENDED**. It verifies software integration, not navigation accuracy.
- Controller fixture checks covered clear/center/side/all-blocked/emergency/stale/invalid cases with acceptable action sets. They are explicitly labelled software fixtures. Moving-obstacle and low-light reference evaluation remains pending real recordings.
- Existing decision, instruction freshness, depth, pipeline, UDP and firmware source files were unchanged by this refactor.

The dataset source protocol and visual spot checks do not constitute independent exhaustive reannotation. Pilot chair scoring was rejected because source Stool/Chair coverage does not match COCO chair; those numbers remain exploratory. The final primary score is person only, on a fresh cohort frozen after the pilot's methodological correction.

No ESP32 board, motors or physical course was connected. Live webcam capture was not tested in this run; the existing saved-video workflow was tested. Metric depth accuracy, independent car-camera generalization, end-to-end live-camera FPS, stopping distance and physical collision avoidance remain unmeasured. COCO pretraining overlap is not ruled out.

Historical CPU/baseline measurements are retained in their original model reports and smoke records. They are not substituted for the current GPU experiment.


## Documentation cleanup

The 2026-10-17 documentation consolidation reduced tracked Markdown documents
from 24 to nine. Detailed dataset/method/camera/taxonomy material now lives in
`dataset_card.md`; hardware instructions in `laptop-esp32.md`; midterm and
presentation material in `documentation-and-paper-guide.md`. Historical baseline
evidence remains in `model-report.md` and its original JSON/CSV/notebook files.
Screenshot attribution is consolidated in the root third-party notices.

Redundant indexes, internal acceptance/cleanup ledgers and duplicated historical
prose were removed. Their original versions are recoverable at commit `a98b1d6`.
No model checkpoint, source manifest, measured JSON/CSV, graph, notebook, runtime
algorithm or firmware safeguard was deleted. Existing research results refer to
their recorded evaluation commit; documentation-link/export changes do not
represent a new experiment.
