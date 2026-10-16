"""Explain existing model/controller results without changing their decisions."""

from dataclasses import dataclass
from .decision import Action, candidate


@dataclass(frozen=True)
class Explanation:
    candidate_action: Action
    candidate_reason: str
    input_available: bool
    stabilizing: bool


def explain(decision, status="running", age_ms=0., stale_after_ms=1000):
    available = (status == "running" and decision.frame_id >= 0
                 and age_ms < stale_after_ms
                 and decision.reason not in {"Input unavailable or stale", "Invalid frame timestamp"})
    action, reason = candidate(decision.observation)
    if not available:
        action, reason = Action.STOP, "Fresh model observations required"
    return Explanation(action, reason, available, available and decision.reason.startswith("Stabilizing "))


def render_explanation(image, decision, config, model_name, metrics, label="obstacle",
                       simulated=False, scene=""):
    """Keep model boxes in the camera view; trace them to the final instruction."""
    import cv2
    from .render import AMBER, BG, MUTED, RED, TEAL, TEXT, render, text
    import textwrap

    canvas = render(image, decision, config, model_name, metrics, label, simulated, scene)
    cv2.rectangle(canvas, (960, 88), (1279, 627), BG, -1)
    trace = explain(decision, metrics.get("status", "running"), metrics.get("age_ms", 0.), config.stale_after_ms)

    def card(top, bottom, heading):
        cv2.rectangle(canvas, (976, top), (1263, bottom), (46, 36, 25), -1)
        cv2.rectangle(canvas, (976, top), (1263, bottom), (77, 62, 44), 1)
        text(canvas, heading, 990, top + 23, .43, TEAL, 1)

    def arrow(y):
        cv2.arrowedLine(canvas, (1120, y), (1120, y + 13), TEAL, 1, cv2.LINE_AA, tipLength=.4)

    card(100, 198, "01  MODEL DETECTIONS")
    text(canvas, f"{len(decision.observation.accepted)} accepted / {label}", 990, 151, .51, TEXT, 1)
    text(canvas, f"{decision.observation.distant} distant / not blocking", 990, 174, .39, MUTED)
    arrow(204)

    card(223, 327, "02  BOX / REGION OVERLAP")
    for i, (name, blocked) in enumerate(zip(("LEFT", "MIDDLE", "RIGHT"), decision.observation.blocked)):
        x = 989 + i * 91
        color = RED if blocked else TEAL
        text(canvas, name, x, 273, .37, MUTED)
        text(canvas, "BLOCKED" if blocked else "NO NEAR", x, 294, .35, color, 1)
        text(canvas, f"{decision.observation.occupancy[i]:.0%} occupied", x, 309, .30, MUTED)
    text(canvas, "Near center: " + ("YES -> STOP" if decision.observation.near_center else "NO"),
         990, 322, .34, RED if decision.observation.near_center else MUTED)
    arrow(333)

    card(352, 450, "03  RULE CANDIDATE")
    text(canvas, trace.candidate_action.value if trace.input_available else "INPUT GATE", 990, 403, .67,
         RED if trace.candidate_action == Action.STOP else AMBER, 2)
    short_reason = {Action.FORWARD: "No nearby center obstacle detected",
                    Action.LEFT: "Center blocked; use clearer left",
                    Action.RIGHT: "Center blocked; use clearer right",
                    Action.STOP: trace.candidate_reason}[trace.candidate_action]
    if not trace.input_available:
        short_reason = "Fresh model observations required"
    text(canvas, short_reason, 990, 430, .37, MUTED)
    arrow(456)

    card(475, 614, "04  FINAL INSTRUCTION")
    color = RED if decision.action == Action.STOP else TEAL
    text(canvas, decision.action.value, 990, 533, .98, color, 2)
    reason = decision.reason
    if trace.stabilizing:
        reason += " / STOP until movement confirmed"
    for i, line in enumerate(textwrap.wrap(reason, width=37)[:2]):
        text(canvas, line, 990, 559 + i * 18, .37, TEXT)
    # Explain the stabilizer in context; the output still comes from Controller.
    if not trace.stabilizing:
        if not trace.input_available:
            text(canvas, f"Frame age {metrics.get('age_ms', 0):.0f} / {config.stale_after_ms} ms limit",
                 990, 602, .34, AMBER)
        else:
            text(canvas, "STOP immediate / movement confirmed", 990, 602, .34, MUTED)
    return canvas
