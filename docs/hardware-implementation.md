# Hardware implementation handoff

The current architecture is **laptop vision → Wi-Fi instructions → ESP32 motor
control**. The ESP32 does not run YOLO, depth, or an onboard camera pipeline.
Follow [the setup/Wi-Fi guide](laptop-esp32.md) for installation and communication.

## Hardware to select

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

## Firmware

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

## Bring-up order

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

## Validation status

The shared C++ command parser/watchdog has native host tests, including timeout,
replay, malformed packets, ownership, disconnect, and timer wraparound. The UDP
sender has localhost communication tests. **An Arduino board build, upload,
Wi-Fi range, GPIO operation, motor response, and physical driving have not been
validated.** Desktop/native checks do not establish those results.

The retained [ESPDet training notebook](../notebooks/train_bottle_cup_box.ipynb)
supports a separate future onboard research path. Moving inference onto an
ESP32-S3 would additionally require export, quantization, memory/latency profiling,
and prediction-parity checks; that work is not needed for the current split system.
