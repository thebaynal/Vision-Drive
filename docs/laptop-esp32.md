# Laptop vision and ESP32 motor control

The camera is the vehicle's forward view. **The laptop runs perception and
chooses instructions; the ESP32 receives short commands and drives the motors.**
Neither the camera nor the neural network needs to be attached to the ESP32.
For a moving car, rigidly mount the camera on the vehicle and send its live
video to the laptop. A laptop webcam is enough for a handheld presentation.

```mermaid
flowchart LR
  A[Forward camera] --> B[Laptop: YOLO11n + metric depth]
  B --> C[Nearby obstacles and three-region clearance]
  C --> D[FORWARD / LEFT / RIGHT / STOP]
  D -->|Wi-Fi UDP + expiring commands| E[ESP32 receiver]
  E --> F[Motor driver and wheels]
  E --> G[Independent stop watchdog]
```

## Run the visualization now

No board is required. On the prepared development computer:

```powershell
.\run_logic_demo.cmd
```

For a fresh checkout:

```powershell
python -m venv .venv-inference
.venv-inference\Scripts\python -m pip install -r requirements-laptop.txt
.venv-inference\Scripts\python scripts/setup_laptop_models.py
.\run_logic_demo.cmd
```

Setup downloads about 105 MB of model files into ignored `weights/laptop/`.
The script pins the depth revision and verifies both checkpoint hashes. The
viewer reads local files and never downloads substitute models during inference.
The PC requires the PyTorch runtime; the ESP32 requires only its receiver firmware.

```powershell
# Record a visualization from saved footage; no commands go to the car.
.\run_logic_demo.cmd --source path/to/video.mp4 --save-video outputs/demo.mp4
# Set approach/emergency thresholds in estimated meters.
.\run_logic_demo.cmd --avoid-distance 1.5 --stop-distance 0.45
# Explicit alternative backends:
.\run_logic_demo.cmd --detector coco  # Known categories, image proximity only
.\run_logic_demo.cmd --detector esp   # Original tiny bottle/cup/box model
```

Use [the demo notebook](../notebooks/laptop_esp32_demo.ipynb) to inspect model
setup, depth, detections, and instructions in stages. The original
[training notebook](../notebooks/train_bottle_cup_box.ipynb) remains available
for the earlier ESPDet experiment. The current [pretrained evaluation notebook](../notebooks/yolo11n_pretrained_evaluation.ipynb)
and [research report](evaluation_results.md) evaluate unchanged YOLO11n.
Its 80 COCO categories do not provide universal obstacle detection.

## What makes an obstacle?

Default YOLO11n recognizes 80 COCO categories. **Recognized names are optional for
obstacle reasoning:** a separate indoor metric-depth model estimates distance
for image surfaces. Nearby connected surfaces in the decision area can be
labeled `depth obstacle` or `near surface` even if YOLO cannot name them. This
adds a path for boxes, walls, furniture, and unfamiliar shapes to affect the
controller. Detection of any particular object is still not guaranteed.

On a flat course, the program fits a perspective ground plane in inverse depth
to the lower view and removes matching floor pixels. If it cannot establish a
floor, it retains nearby surfaces conservatively instead of inventing empty
space. Reflective/transparent surfaces, thin legs, floors with changing slopes,
camera vibration, occlusion, and poor lighting remain difficult.

Depth components use a geometry acceptance flag internally (`confidence=1`),
not a calibrated neural confidence. The overlay deliberately omits a probability
for those components. Actual object boxes retain their model confidence.

## Distance and steering

Defaults in [the config](../config/default.json) are **initial demo values**:

| Observation in the vehicle's forward decision area | Instruction |
| --- | --- |
| Center objects farther than estimated 1.5 m | FORWARD, after confirmation |
| Center object within estimated 1.5 m; left usable | LEFT |
| Center object within estimated 1.5 m; right usable | RIGHT |
| Both sides usable | Choose the lower occupied-area side; equal scores choose left |
| No usable side | STOP |
| Center object within estimated 0.45 m | STOP immediately |
| Invalid/stale vision, pause, end of input | STOP immediately |

Far detections remain visible but do not contribute to blockage or occupied
area. Overlapping obstacle boxes are unioned so they do not count the same space
twice. Movement requires three matching **processed observations**; a newly
blocked center stops forward movement while a turn is being confirmed. This
is reactive steering, not target following, mapping, or destination navigation.

The visualization and motor commands have **independent freshness limits**.
The viewer retains the last visual instruction while analyzing the next frame,
up to a default five-second frame age, and labels that hold explicitly. It does
not emit alternating FORWARD/STOP merely because normal CPU inference takes
longer than the motor deadline. Camera/inference failure, pause, end of input,
or exceeding the display deadline still stops the preview. Set the bounded
display deadline with `--preview-timeout-ms 5000` (1,000–10,000 ms allowed).

The separate motor controller keeps `stale_after_ms` from the configuration
(default 1,000 ms). Only its decision reaches the Wi-Fi sender. The footer
shows the sent ESP32 command independently of the visual instruction. Recorded
JSONL includes `control_action`, `control_reason`, and, when connected,
`esp_command`. A clear-looking preview never authorizes motors using stale data.

The side regions are image-space approximations, not proof that the car's full
width or turning arc fits. Mark a measured driving corridor after final camera
mounting and test it with the actual chassis dimensions.

The depth model produces learned meter estimates. It is not a measured range
sensor. Compare predictions with tape-measured distances at several positions,
then use `--depth-scale ACTUAL_DISTANCE/PREDICTED_DISTANCE` as an initial scale
correction. One scale factor cannot fix all errors. Check the entire course,
choose speed and thresholds from observed error/latency/stopping distance,
and keep these trials supervised.

With `--detector esp` or `--detector coco`, there is no depth model: the lower box
edge is a camera-specific proximity proxy. Boxes ending above 60% of image
height are considered distant; a center box reaching below 92% with at least
4% of frame area is an emergency. **These percentages are not meters.**

## Connect the ESP32 later

1. Open [visiondrive_receiver.ino](../firmware/visiondrive_receiver/visiondrive_receiver.ino)
   in Arduino IDE with the correct ESP32 board/core installed.
2. Copy [config.example.h](../firmware/visiondrive_receiver/config.example.h) to
   `config.h` beside the sketch. Actual credentials are ignored by Git.
3. Set the Wi-Fi SSID/password and generate a command token:

   ```powershell
   python -c "import secrets; print(secrets.token_hex(16))"
   ```

4. Paste that token into the receiver's `COMMAND_TOKEN`. Leave
   `ENABLE_MOTORS=false` first. Flash, open Serial Monitor at 115200, and note
   the ESP32 IP. Instructions can be inspected before wiring motors.
5. Put both devices on the same local Wi-Fi and run:

   ```powershell
   $env:VISIONDRIVE_ESP_TOKEN = "the_same_32_character_token"
   .\run_logic_demo.cmd --source 0 --esp-host 192.168.1.50
   ```

6. Configure six valid, free GPIOs for your board and compatible two-channel
   PWM/direction motor driver. Wire its standby/enable and power according to
   its datasheet. Enable motor output only after raised-wheel direction checks.
   No pin map is assumed, and this repository has no physical board validation.

The receiver uses a **350 ms default command expiry**, a STOP handshake, a
per-session sequence number, and a Wi-Fi-loss stop. It rejects malformed,
unauthorized, replayed, and out-of-order packets. Expiry requires another STOP
before movement can resume. A bounded packet loop keeps checking the watchdog.
The laptop sends at up to 20 Hz but stops refreshing motion once perception is
stale; it also stops issuing motion if acknowledgements disappear. Pausing or
closing the program sends STOP. UDP delivery is not guaranteed, so the board's
own expiry controls the fail-stop behavior.

Only a live numeric camera input can send commands. Saved videos and simulation
cannot command the board. A lost laptop or inference failure therefore cannot
leave a motion command running indefinitely. Motor stop is zero PWM/coast in
the supplied mapping; braking distance depends on the chosen driver/chassis.

The token filters unintended senders on a trusted local demo network. This UDP
protocol is not encrypted and does not provide cryptographic authentication.
Do not expose its port to the internet. If Windows prompts for network access,
allow the local private network used for the demo.

## What can be claimed to a professor?

> This is a camera-based obstacle-avoidance prototype. A laptop performs
> object detection, depth estimation, and movement decisions; an ESP32 is
> designed to receive expiring commands and control the motors. Detection is
> broader than our initial three-object model. Hardware driving, camera-distance
> calibration, and avoidance accuracy still require course testing.

The original ESPDet model remains an option for future onboard research.
The current YOLO-plus-depth system is **not an onboard ESP32 model**. A camera
connected to an ESP32 could stream to the laptop later; that streaming firmware
is separate from this motor receiver and is not included here.

## Model references

- [Ultralytics YOLO11](https://docs.ultralytics.com/models/yolo11/): pretrained category detector.
- [Depth Anything V2 indoor metric Small](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf): learned indoor depth, Apache-2.0.
- [Transformers depth model documentation](https://huggingface.co/docs/transformers/model_doc/depth_anything_v2): metric head and inference.
- [Espressif's COCO model measurements](https://github.com/espressif/esp-dl/blob/master/models/coco_detect/README.md): optional future onboard YOLO path; published upstream timing is not a VisionDrive measurement.


<a id="hardware-implementation"></a>

## Hardware implementation

The current architecture is **laptop vision → Wi-Fi instructions → ESP32 motor
control**. The ESP32 does not run YOLO, depth, or an onboard camera pipeline.
Follow [the setup/Wi-Fi guide](laptop-esp32.md) for installation and communication.

### Hardware to select

| Component | Requirement |
| --- | --- |
| Laptop | Runs the tested Python vision application |
| Forward camera | Rigidly mounted, unmirrored, clear view of the driving corridor; live video reaches laptop |
| ESP32 board | Wi-Fi and six suitable free motor-control GPIOs; no camera/ML accelerator required |
| Two-channel motor driver | Supports the supplied PWM + two-direction-input-per-wheel interface and motor current |
| Motors/chassis | Low-speed differential drive; measured width and turn radius |
| Power supply | Appropriate regulated board supply and motor supply, common ground, driver decoupling |

Actual board, motor, driver, supply, pin mapping, and physical behavior have
not been selected or validated. Choose GPIOs from the exact board's pinout;
avoid flash/PSRAM/camera pins and respect input-only or boot-related constraints.

### Firmware

Open [the receiver sketch](../firmware/visiondrive_receiver/visiondrive_receiver.ino).
Copy [config.example.h](../firmware/visiondrive_receiver/config.example.h) to
`config.h`, set Wi-Fi credentials and a generated token, and upload with the
correct Arduino-ESP32 board/core. Firmware uses the built-in WiFi/WiFiUDP APIs;
there is no external JSON or model dependency on the board.

`ENABLE_MOTORS=false` is the default. Instructions can be checked in Serial
Monitor before wiring motors. Configure all six valid unique GPIOs and the
driver's standby/enable wiring before enabling motor output. The supplied
mapping drives forward or makes a forward arc by stopping one wheel; STOP
sets PWM to zero and both direction inputs low. Whether that coasts/brakes
depends on the selected driver. No reverse maneuver is implemented.

### Bring-up order

1. Verify the vision display and camera orientation without a motor connection.
2. Flash the receiver, note its Wi-Fi IP, and inspect instructions with motor
   output disabled. Confirm pause/quit and network loss leave it stopped.
3. Configure driver/power/pins and test with raised wheels at low duty. Confirm
   forward/left/right polarity and the driver's actual coast/brake behavior.
4. Measure camera-distance prediction error, end-to-end latency, vehicle speed,
   turning clearance, and stopping/coasting distance on a controlled flat course.
5. Tune avoidance/emergency thresholds, freshness timeout, duty, and movement
   confirmation using validation sessions. Evaluate new held-out sessions once.

The board checks sequence numbers, STOP session handshakes, packet validity,
Wi-Fi state, and command expiry independently of the laptop. Command expiry is
350 ms by default. The sender refreshes commands at up to 20 Hz only while the
underlying observation remains fresh. Expiry requires a new STOP before motion
resumes. Motor-power removal must remain available during physical trials.

The longer (five-second default) visual-result hold is for the desktop preview
only. A separate controller enforces the motor freshness limit; the sender
cannot use the held preview action to extend a movement command.

### Validation status

The shared C++ command parser/watchdog has native host tests, including timeout,
replay, malformed packets, ownership, disconnect, and timer wraparound. The UDP
sender has localhost communication tests. **An Arduino board build, upload,
Wi-Fi range, GPIO operation, motor response, and physical driving have not been
validated.** Desktop/native checks do not establish those results.

The retained [ESPDet training notebook](../notebooks/train_bottle_cup_box.ipynb)
supports a separate future onboard research path. Moving inference onto an
ESP32-S3 would additionally require export, quantization, memory/latency profiling,
and prediction-parity checks; that work is not needed for the current split system.
