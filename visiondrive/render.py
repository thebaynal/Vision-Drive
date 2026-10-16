"""OpenCV-only presentation, drawn after inference on an unmirrored frame."""

import textwrap
import cv2
import numpy as np
from .decision import Action

BG = (29, 20, 13)
TEXT = (241, 240, 234)
MUTED = (175, 164, 149)
TEAL = (181, 207, 77)
AMBER = (101, 189, 245)
RED = (123, 116, 247)


def text(canvas, words, x, y, scale=.55, color=TEXT, thickness=1):
    cv2.putText(canvas, str(words), (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)


def render(image, decision, config, model_name, metrics, label="obstacle", simulated=False, scene=""):
    width, height, header, footer, panel = 960, 540, 88, 100, 320
    canvas = np.full((header + height + footer, width + panel, 3), BG, dtype=np.uint8)
    text(canvas, "VISIONDRIVE", 24, 37, .85, TEAL, 2)
    text(canvas, "CAMERA-ONLY PERCEPTION / THREE-REGION CONTROLLER", 24, 64, .48, MUTED)
    mode = "SIMULATION - NOT DETECTION" if simulated else "LIVE MODEL INFERENCE"
    text(canvas, mode, 815, 40, .55, AMBER if simulated else TEAL, 1)
    h, w = image.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(image, (max(1, round(w * scale)), max(1, round(h * scale))))
    dx, dy = (width - resized.shape[1]) // 2, header + (height - resized.shape[0]) // 2
    canvas[dy:dy + resized.shape[0], dx:dx + resized.shape[1]] = resized
    rw, rh = resized.shape[1], resized.shape[0]
    overlay = canvas.copy()
    edges = (0., config.left_boundary, config.right_boundary, 1.)
    names = ("LEFT", "MIDDLE", "RIGHT")
    for i, blocked in enumerate(decision.observation.blocked):
        color = RED if blocked else TEAL
        x1, x2 = dx + round(edges[i] * rw), dx + round(edges[i + 1] * rw)
        cv2.rectangle(overlay, (x1, dy + round(config.decision_top * rh)), (x2, dy + rh - 1), color, -1)
        cv2.line(canvas, (x1, dy), (x1, dy + rh - 1), MUTED, 1)
    canvas = cv2.addWeighted(overlay, .13, canvas, .87, 0)
    # Redraw dividers/text on top of translucent tint.
    for i in range(3):
        x = dx + round(edges[i] * rw)
        text(canvas, names[i], x + 16, dy + 28, .55, TEXT, 2)
        if i:
            cv2.line(canvas, (x, dy), (x, dy + rh - 1), MUTED, 1)
    for det in decision.observation.accepted:
        a = (dx + int(det.x1 * rw), dy + int(det.y1 * rh))
        b = (dx + int(det.x2 * rw), dy + int(det.y2 * rh))
        cv2.rectangle(canvas, a, b, AMBER, 2)
        distance = f" ~{det.distance_m:.2f}m" if det.distance_m is not None else ""
        score = "" if det.label in ("depth obstacle", "near surface") else f" {det.confidence:.2f}"
        text(canvas, f"{det.label or label}{score}{distance}", a[0], max(dy + 45, a[1] - 8), .50, AMBER)
    if metrics.get("metric_depth"):
        text(canvas, f"Depth estimate: avoid <= {config.avoid_distance_m:g}m / stop <= {config.stop_distance_m:g}m",
             dx + 16, dy + rh - 16, .48, AMBER)
    else:
        approach_y = dy + round(config.approach_top * rh)
        cv2.line(canvas, (dx, approach_y), (dx + rw - 1, approach_y), AMBER, 1)
        text(canvas, "APPROACH / STEER (image proxy)", dx + 16, approach_y - 8, .42, AMBER)
        near_y = dy + round(config.near_top * rh)
        cv2.line(canvas, (dx, near_y), (dx + rw - 1, near_y), RED, 2)
        text(canvas, "EMERGENCY (image proxy)", dx + 16, min(dy + rh - 7, near_y + 24), .42, RED)
    px = width + 22
    text(canvas, "SUGGESTED ACTION", px, header + 30, .55, MUTED)
    color = RED if decision.action == Action.STOP else TEAL
    text(canvas, decision.action.value, px, header + 88, 1.35, color, 3)
    for j, line in enumerate(textwrap.wrap(decision.reason, width=29)):
        text(canvas, line, px, header + 122 + j * 22, .48)
    for i, (name, blocked) in enumerate(zip(names, decision.observation.blocked)):
        y = header + 230 + i * 37
        cv2.circle(canvas, (px + 6, y - 4), 5, RED if blocked else TEAL, -1)
        text(canvas, name, px + 24, y, .50)
        text(canvas, "BLOCKED" if blocked else "NO BOX", px + 135, y, .48, RED if blocked else TEAL)
    fps_label = "Scenario FPS" if simulated else "Processed FPS"
    text(canvas, f"{fps_label}   {metrics.get('fps', 0):.1f}", px, header + 380, .50)
    text(canvas, f"Capture -> result {metrics.get('latency_ms', 0):.0f} ms", px, header + 410, .48)
    text(canvas, f"Inference         {metrics.get('inference_ms', 0):.0f} ms", px, header + 440, .48)
    text(canvas, f"Result age        {metrics.get('age_ms', 0):.0f} ms", px, header + 470, .48)
    for j, line in enumerate(textwrap.wrap(model_name, width=41)):
        text(canvas, line, 24, header + height + 25 + j * 19, .44, MUTED)
    link_status = metrics.get("esp_link", "Visualization only / ESP32 not connected")
    text(canvas, link_status, 24, header + height + 76, .44, AMBER)
    text(canvas, "SPACE pause / resume    Q or ESC quit", 860, header + height + 76, .44, MUTED)
    if simulated and scene:
        text(canvas, scene, 550, header + height + 25, .44, AMBER)
    return canvas
