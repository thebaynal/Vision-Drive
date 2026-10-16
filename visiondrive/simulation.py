"""Labeled, synthetic scenarios for presenting controller behavior without weights."""

from .decision import Detection


def scenario(elapsed):
    center = Detection(.41, .48, .59, .74, .96)
    left = Detection(.08, .48, .25, .74, .94)
    right = Detection(.75, .48, .92, .74, .95)
    scenes = [
        ("Empty center", ()),
        ("Center blocked / left tie-break", (center,)),
        ("Center + left blocked", (center, left)),
        ("Center + right blocked", (center, right)),
        ("All three blocked", (center, left, right)),
        ("Near-center obstacle", (Detection(.41, .62, .59, .94, .98),)),
        ("Wide obstacle spans regions", (Detection(.12, .47, .89, .74, .97),)),
        ("Center clear again", ()),
    ]
    return scenes[int(elapsed // 3) % len(scenes)]


def make_frame(elapsed, width=960, height=540):
    import cv2
    import numpy as np
    name, detections = scenario(elapsed)
    image = np.full((height, width, 3), (34, 29, 22), dtype=np.uint8)
    horizon = int(height * .38)
    image[horizon:] = (43, 42, 33)
    for x in range(-width, width * 2, 120):
        cv2.line(image, (width // 2, horizon), (x, height), (60, 62, 52), 1)
    for fraction in (.48, .62, .80, .99):
        y = int(height * fraction)
        cv2.line(image, (0, y), (width, y), (60, 62, 52), 1)
    for det in detections:
        p = (int(det.x1 * width) + 8, int(det.y1 * height) + 8)
        q = (int(det.x2 * width) - 8, int(det.y2 * height) - 8)
        cv2.rectangle(image, p, q, (95, 143, 183), -1)
        cv2.line(image, p, q, (73, 108, 139), 2)
    return image, detections, name
