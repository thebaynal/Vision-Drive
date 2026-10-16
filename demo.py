"""Desktop entry point. Run --simulate to preview the controller without weights."""

import argparse
from collections import deque
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

from visiondrive.config import Config
from visiondrive.decision import Controller, EMPTY


ROOT = Path(__file__).resolve().parent


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source", default="0", help="Camera index or video file")
    cli.add_argument("--weights", type=Path, default=ROOT / "weights/obstacle/best.pt")
    cli.add_argument("--esp-repo", type=Path, default=ROOT / "external/esp-detection")
    cli.add_argument("--config", type=Path)
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
    writer = record = None
    canvas = None
    model_name, label = "Synthetic boxes / no neural network", "simulated obstacle"
    controller = Controller(config)
    paused, count, generation, latest_id = False, 0, 0, -1
    elapsed, sim_id = 0., 0
    history = deque(maxlen=90)
    current_frame = np.zeros((540, 960, 3), dtype=np.uint8)
    last_result = None
    started = previous = time.monotonic()
    try:
        if not args.simulate:
            from visiondrive.detector import ESPDetector
            # Fail model preflight before opening a camera.
            detector = ESPDetector(args.weights, args.esp_repo, config, args.device, args.expected_label, args.cpu_threads)
            model_name, label = detector.name, detector.label
            if not camera and not Path(args.source).is_file():
                raise ValueError(f"Video not found: {args.source}")
            print("Warming up ESPDet before opening the camera/video...", flush=True)
            detector.warmup()
            capture = cv2.VideoCapture(int(args.source) if camera else args.source)
            if not capture.isOpened():
                capture.release()
                raise ValueError("Unable to open camera/video. Check source index and camera permissions.")
            raw_fps = capture.get(cv2.CAP_PROP_FPS)
            fps = raw_fps if 0 < raw_fps <= 240 else 30
            pipeline = Pipeline(capture, detector, camera, fps)
            pipeline.start()
        print(f"VisionDrive: {model_name}")
        print("SPACE pause/resume; Q/ESC quit. Movement is a visual suggestion only.")
        while True:
            now = time.monotonic()
            scene, status = "", "running"
            metrics = {"fps": 0., "latency_ms": 0., "inference_ms": 0., "age_ms": 0.}
            if args.simulate:
                if not paused:
                    elapsed += 1 / 30 if args.headless else now - previous
                    current_frame, detections, scene = make_frame(elapsed)
                    controller.update(detections, sim_id, now, now)
                    sim_id += 1
                status = "PAUSED" if paused else "running"
                decision = controller.tick(now, status)
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
                        controller.update(result.detections, latest_id, result.packet.captured_at, now)
                    history.append(result.completed_at)
                elif result is None and raw is not None and last_result is None:
                    current_frame = raw.image
                decision = controller.tick(now, status)
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
            renderer = render
            if args.explain:
                from visiondrive.explain import render_explanation
                renderer = render_explanation
            canvas = renderer(current_frame, decision, config, model_name, metrics, label, args.simulate, scene)
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
                    controller.reset("PAUSED" if paused else "Waiting for fresh observations")
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
