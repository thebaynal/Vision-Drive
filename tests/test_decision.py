import math
import pytest
from visiondrive.config import Config
from visiondrive.decision import Action, Controller, Detection, candidate, observe


LEFT = Detection(.06, .45, .26, .72)
CENTER = Detection(.4, .45, .6, .72)
RIGHT = Detection(.74, .45, .94, .72)


@pytest.mark.parametrize("boxes,expected", [
    ([], Action.FORWARD), ([LEFT], Action.FORWARD), ([RIGHT], Action.FORWARD),
    ([CENTER], Action.LEFT), ([CENTER, LEFT], Action.RIGHT),
    ([CENTER, RIGHT], Action.LEFT), ([LEFT, CENTER, RIGHT], Action.STOP),
    ([Detection(.4, .7, .6, .95)], Action.STOP),
    ([Detection(.1, .45, .9, .7)], Action.STOP),
])
def test_rule_table(boxes, expected):
    assert candidate(observe(boxes, Config()))[0] == expected


def test_overlap_uses_full_box_and_can_span_regions():
    obs = observe([Detection(.25, .5, .5, .75)], Config())
    assert obs.blocked == (True, True, False)
    assert not obs.near_center


def test_top_objects_and_confidence_filtered():
    obs = observe([Detection(.4, .05, .6, .2), Detection(.4, .5, .6, .7, .49)], Config())
    assert obs.blocked == (False, False, False)
    assert obs.ignored == 1


def test_near_boundary_requires_positive_area():
    assert not observe([Detection(.4, .5, .6, .8)], Config()).near_center
    assert observe([Detection(.4, .5, .6, .8001)], Config()).near_center
    assert not observe([Detection(.05, .7, .2, .95)], Config()).near_center


@pytest.mark.parametrize("det", [Detection(.5, .5, .5, .8), Detection(-.1, .4, .3, .6),
    Detection(.4, math.nan, .6, .7), Detection(.4, .5, .6, .7, 1.1), Detection(.4, .5, .6, .7, 1, 3)])
def test_invalid_boxes_ignored(det):
    obs = observe([det], Config())
    assert obs.ignored == 1 and not obs.accepted


def test_exact_overlap_threshold_is_inclusive():
    # Binary fractions avoid a decimal-rounding ambiguity at the boundary.
    cfg = Config(left_boundary=.25, right_boundary=.75, overlap_threshold=.5)
    assert observe([Detection(.125, .5, .375, .75)], cfg).blocked == (True, True, False)


def test_three_unique_observations_required_and_duplicate_is_not_evidence():
    ctl = Controller()
    assert ctl.update([], 0, 100, 100).action == Action.STOP
    assert ctl.update([], 0, 100, 100).action == Action.STOP
    assert ctl.update([], 1, 100.1, 100.1).action == Action.STOP
    assert ctl.update([], 2, 100.2, 100.2).action == Action.FORWARD


def test_stop_immediate_and_resume_needs_stabilization():
    ctl = Controller()
    for i in range(3):
        ctl.update([], i, 100 + i / 10, 100 + i / 10)
    hazard = [Detection(.4, .6, .6, .99)]
    assert ctl.update(hazard, 3, 100.3, 100.3).action == Action.STOP
    assert ctl.update([], 4, 100.4, 100.4).action == Action.STOP
    assert ctl.update([], 5, 100.5, 100.5).action == Action.STOP
    assert ctl.update([], 6, 100.6, 100.6).action == Action.FORWARD


def test_stale_inference_cannot_issue_movement():
    ctl = Controller(Config(stable_frames=1, stale_after_ms=100))
    assert ctl.update([], 0, 100, 100).action == Action.FORWARD
    assert ctl.tick(100.11).action == Action.STOP
    assert ctl.update([], 1, 100, 100.12).action == Action.STOP
    assert ctl.update([], 2, 101, 101).action == Action.FORWARD


def test_observation_gap_resets_partial_streak():
    ctl = Controller()
    ctl.update([], 0, 100, 100)
    ctl.update([], 1, 100.1, 100.1)
    assert ctl.update([], 2, 102, 102).action == Action.STOP
    assert ctl.streak == 1


def test_pause_and_reset():
    ctl = Controller(Config(stable_frames=1))
    ctl.update([], 0, 100, 100)
    assert ctl.tick(100.1, "PAUSED").action == Action.STOP
    ctl.reset()
    assert ctl.current.frame_id == -1


@pytest.mark.parametrize("timestamp", [math.nan, math.inf, 101])
def test_invalid_timestamps_stop(timestamp):
    assert Controller().update([], 0, timestamp, 100).action == Action.STOP
