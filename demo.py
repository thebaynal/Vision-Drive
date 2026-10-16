"""Desktop entry point. Run --simulate to preview the controller without weights."""

import argparse
from collections import deque
from dataclasses import replace
import json
import os
from pathlib import Path
import sys
import time

from visiondrive.config import Config
from visiondrive.decision import EMPTY
from visiondrive.instructions import Instructions


ROOT = Path(__file__).resolve().parent


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", default="0", help="Camera index or video file")
    cli.add_argument("--detector", choices=["hybrid", "coco", "esp"], default="hybrid",
                     help="hybrid: laptop YOLO + depth; coco: YOLO only; esp: original tiny three-class model")
    cli.add_argument("--weights", type=Path, help="Override checkpoint for the selected detector")
    cli.add_argument("--depth-model", type=Path, default=ROOT / "weights/laptop/depth-indoor")
    cli.add_argument("--depth-size", type=int, default=392, help="Depth input size, divisible by 14")
    cli.add_argument("--depth-scale", type=float, default=1., help="Measured-distance / predicted-distance calibration factor")
    cli.add_argument("--avoid-distance", type=float, help="Start avoiding at this estimated distance (meters)")
    cli.add_argument("--stop-distance", type=float, help="Emergency center stop distance (meters)")
    cli.add_argument("--esp-host", help="ESP32 Wi-Fi IP; omit for visualization without hardware")
    cli.add_argument("--esp-port", type=int, default=4210)
    cli.add_argument("--command-ttl-ms", type=int, default=350)
    cli.add_argument("--preview-timeout-ms", type=int, default=5000,
                     help="Hold the visual interpretation for up to this frame age while analyzing (1000-10000 ms); motor expiry is unchanged")
    cli.add_argument("--esp-repo", type=Path, default=ROOT / "external/esp-detection")
    cli.add_argument("--config", type=Path, default=ROOT / "config/default.json")
    cli.add_argument("--device", default="cpu", help="cpu or CUDA device, e.g. 0")
    cli.add_argument("--cpu-threads", type=int, default=2, help="CPU inference threads; default 2")
    cli.add_argument("--expected-label", default="obstacle", choices=["obstacle"], help="Legacy merged-class label; bottle/cup/box models are recognized automatically")
    cli.add_argument("--simulate", action="store_true", help="Synthetic controller preview, NOT object detection")
    cli.add_argument("--explain", action="store_true", help="Show model detections -> region overlap -> rule -> instruction")
    cli.add_argument("--save-video", type=Path, help="Opt-in annotated MP4 plus timestamp/decision JSONL")
    cli.add_argument("--snapshot", type=Path, help="Opt-in final annotated PNG")
    cli.add_argument("--headless", action="store_true", help="No window; useful for video/simulation verification")
    cli.add_argument("--max-frames", type=int, default=0, help="Exit after N display frames; 0 means unlimited")
    return cli


def run(args):
    import cv2
    import numpy as np
    from visiondrive.pipeline import Pipeline
    from visiondrive.render import render
    from visiondrive.simulation import make_frame

    config = Config.load(args.config)
    config = replace(config, **{name: value for name, value in
        (("avoid_distance_m", args.avoid_distance), ("stop_distance_m", args.stop_distance)) if value is not None})
    if args.max_frames < 0:
        raise ValueError("--max-frames cannot be negative")
    camera = args.source.isdecimal()
    if args.headless and args.max_frames == 0 and (args.simulate or camera):
        raise ValueError("Headless webcam/simulation requires --max-frames")
    for output in (args.save_video, args.snapshot):
        if output and output.exists():
            raise ValueError(f"Output already exists: {output}. Choose a new filename.")
    if args.save_video and args.save_video.with_suffix(".jsonl").exists():
        raise ValueError("Recording JSONL already exists; choose a new output filename")
    pipeline = None
    esp_link = None
    writer = record = None
    canvas = None
    model_name, label = "Synthetic boxes / no neural network", "simulated obstacle"
    instructions = Instructions(config, args.preview_timeout_ms)
    preview_config = instructions.preview_config
    paused, count, generation, latest_id = False, 0, 0, -1
    elapsed, sim_id = 0., 0
    history = deque(maxlen=90)
    current_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    last_result = None
    started = previous = time.monotonic()
    try:
        if not args.simulate:
            # Fail model preflight before opening a camera.
            if args.detector == "esp":
                from visiondrive.detector import ESPDetector
                detector = ESPDetector(args.weights or ROOT / "weights/obstacle/best.pt", args.esp_repo,
                                       config, args.device, args.expected_label, args.cpu_threads)
            else:
                from visiondrive.laptop_detector import LaptopDetector
                detector = LaptopDetector(args.weights or ROOT / "weights/laptop/yolo11n.pt", config,
                    args.depth_model if args.detector == "hybrid" else None, args.device,
                    args.cpu_threads, args.depth_size, args.depth_scale)
            model_name, label = detector.name, detector.label
            if not camera and not Path(args.source).is_file():
                raise ValueError(f"Video not found: {args.source}")
            print("Warming up vision models before opening the camera/video...", flush=True)
            detector.warmup()
            capture = cv2.VideoCapture(int(args.source) if camera else args.source)
            if not capture.isOpened():
                capture.release()
                raise ValueError("Unable to open camera/video. Check source index and camera permissions.")
            raw_fps = capture.get(cv2.CAP_PROP_FPS)
            fps = raw_fps if 0 < raw_fps <= 240 else 30
            pipeline = Pipeline(capture, detector, camera, fps)
            pipeline.start()
        if args.esp_host:
            if args.simulate or not camera:
                raise ValueError("ESP32 commands require a live camera; recordings and simulation are visualization-only")
            from visiondrive.esp_link import ESPLink
            esp_link = ESPLink(args.esp_host, os.environ.get("VISIONDRIVE_ESP_TOKEN", ""),
                               args.esp_port, args.command_ttl_ms)
        print(f"VisionDrive: {model_name}")
        print("SPACE pause/resume; Q/ESC quit. Movement is a visual suggestion only.")
        print(f"Visual-result hold: {args.preview_timeout_ms} ms; separate motor freshness: {config.stale_after_ms} ms.")
        while True:
            now = time.monotonic()
            scene, status = "", "running"
            metrics = {"fps": 0., "latency_ms": 0., "inference_ms": 0., "age_ms": 0.}
            if args.simulate:
                if not paused:
                    elapsed += 1 / 30 if args.headless else now - previous
                    current_frame, detections, scene = make_frame(elapsed)
                    instructions.update(detections, sim_id, now, now)
                    sim_id += 1
                status = "PAUSED" if paused else "running"
                decision, control_decision = instructions.tick(now, status)
                metrics["fps"] = 30. if not paused else 0.
            else:
                status, result, raw, processed = pipeline.snapshot()
                if paused:
                    status = "PAUSED"
                if result is not None and (generation, latest_id) != (result.packet.generation, result.packet.frame_id):
                    generation, latest_id = result.packet.generation, result.packet.frame_id
                    last_result = result
                    current_frame = result.packet.image
                    if not paused and status == "running":
                        instructions.update(result.detections, latest_id, result.packet.captured_at, now)
                    history.append(result.completed_at)
                elif result is None and raw is not None and last_result is None:
                    current_frame = raw.image
                decision, control_decision = instructions.tick(now, status)
                if last_result is not None:
                    metrics.update(latency_ms=(last_result.completed_at - last_result.packet.captured_at) * 1000,
                                   inference_ms=last_result.inference_ms,
                                   age_ms=(now - last_result.packet.captured_at) * 1000)
                recent = [stamp for stamp in history if now - stamp <= 2]
                if len(recent) > 1 and recent[-1] > recent[0]:
                    metrics["fps"] = (len(recent) - 1) / (recent[-1] - recent[0])
            previous = now
            if status != "running":
                # Never leave a stale box overlay attached to a different frame.
                decision = replace(decision, observation=EMPTY)
            metrics["status"] = status
            metrics["preview_timeout_ms"] = args.preview_timeout_ms
            metrics["motor_freshness_ms"] = config.stale_after_ms
            metrics["analyzing"] = (not args.simulate and status == "running" and
                (last_result is None or now-last_result.completed_at > .15))
            metrics["control_action"] = control_decision.action.value
            metrics["control_reason"] = control_decision.reason
            if esp_link:
                # Only the strict controller can supply motor commands.
                # Never send the longer-lived visualization decision.
                esp_link.send(control_decision, config.stale_after_ms)
                metrics["esp_link"] = esp_link.status
                metrics["esp_command"] = esp_link.last_action.value
            metrics["metric_depth"] = not args.simulate and args.detector == "hybrid"
            metrics["avoid_distance_m"] = config.avoid_distance_m
            metrics["stop_distance_m"] = config.stop_distance_m
            renderer = render
            if args.explain:
                from visiondrive.explain import render_explanation
                renderer = render_explanation
            canvas = renderer(current_frame, decision, preview_config, model_name, metrics, label, args.simulate, scene)
            if args.save_video:
                if writer is None:
                    args.save_video.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.save_video), cv2.VideoWriter_fourcc(*"mp4v"),
                                             30, (canvas.shape[1], canvas.shape[0]))
                    if not writer.isOpened():
                        raise ValueError("Unable to create MP4; check output path and codec support")
                    record = args.save_video.with_suffix(".jsonl").open("x", encoding="utf-8")
                writer.write(canvas)
                record.write(json.dumps({"display_frame": count, "elapsed_s": now - started,
                    "source_frame": decision.frame_id, "captured_at": decision.captured_at,
                    "simulated": args.simulate, "status": status, "action": decision.action.value,
                    "presentation": "logic" if args.explain else "detection",
                    "reason": decision.reason, "blocked": decision.observation.blocked,
                    "occupancy": decision.observation.occupancy, "distant": decision.observation.distant,
                    "detector": args.detector if not args.simulate else "simulation",
                    "detections": [vars(d) for d in decision.observation.accepted],
                    "source_time_s": last_result.packet.source_time if last_result else None,
                    **metrics}) + "\n")
            count += 1
            if not args.headless:
                cv2.imshow("VisionDrive | camera-only movement suggestions", canvas)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if cv2.getWindowProperty("VisionDrive | camera-only movement suggestions", cv2.WND_PROP_VISIBLE) < 1:
                    break
                if key == ord(" "):
                    paused = not paused
                    instructions.reset("PAUSED" if paused else "Waiting for fresh observations")
                    latest_id = -1
                    if pipeline:
                        pipeline.set_paused(paused)
            if args.max_frames and count >= args.max_frames:
                break
            if args.headless and status != "running":
                break
            if not (args.headless and args.simulate):
                time.sleep(max(0, 1 / 30 - (time.monotonic() - now)))
        if args.snapshot and canvas is not None:
            args.snapshot.parent.mkdir(parents=True, exist_ok=True)
            if not cv2.imwrite(str(args.snapshot), canvas):
                raise ValueError("Unable to save snapshot")
        print(f"Finished: {count} display frames; last instruction {decision.action.value} ({decision.reason})")
        return 0
    finally:
        if esp_link:
            esp_link.close()
        if pipeline:
            pipeline.close()
        if writer:
            writer.release()
        if record:
            record.close()
        if not args.headless:
            cv2.destroyAllWindows()


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        return run(args)
    except KeyboardInterrupt:
        return 130
    except (ValueError, RuntimeError, ImportError, OSError) as exc:
        print(f"VisionDrive error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
