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
    ([Detection(.4, .2, .6, .5)], Action.FORWARD),
    ([Detection(.4, .5, .6, .88)], Action.LEFT),
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
    assert not observe([Detection(.4, .5, .6, .92)], Config()).near_center
    assert observe([Detection(.4, .5, .6, .9201)], Config()).near_center
    assert not observe([Detection(.05, .7, .2, .95)], Config()).near_center


@pytest.mark.parametrize("det", [Detection(.5, .5, .5, .8), Detection(-.1, .4, .3, .6),
    Detection(.4, math.nan, .6, .7), Detection(.4, .5, .6, .7, 1.1), Detection(.4, .5, .6, .7, 1, -1)])
def test_invalid_boxes_ignored(det):
    obs = observe([det], Config())
    assert obs.ignored == 1 and not obs.accepted


@pytest.mark.parametrize("label", ["cat", "person", "chair", "wall"])
def test_controller_accepts_any_object_category(label):
    obs = observe([Detection(.4, .5, .6, .95, 1., 0, label)], Config())
    assert obs.blocked == (False, True, False)
    assert obs.ignored == 0


@pytest.mark.parametrize("class_id,label", [(0, "bottle"), (1, "cup"), (2, "box")])
def test_target_object_categories_can_block_regions(class_id, label):
    obs = observe([Detection(.4, .5, .6, .7, 1., class_id, label)], Config())
    assert obs.blocked == (False, True, False)


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


def test_estimated_distance_overrides_bottom_edge_for_near_and_far():
    far = Detection(.4, .4, .6, .99, distance_m=4.)
    obs = observe([far], Config())
    assert obs.distant == 1 and candidate(obs)[0] == Action.FORWARD
    nearby = Detection(.4, .4, .6, .99, distance_m=.8)
    assert candidate(observe([nearby], Config()))[0] == Action.LEFT
    emergency = Detection(.4, .4, .6, .7, distance_m=.3)
    assert candidate(observe([emergency], Config()))[0] == Action.STOP


def test_choose_side_with_more_free_space_when_both_are_unblocked():
    # A very wide upper box clips only a small part of the left decision area;
    # no single-region overlap passes the hit threshold, but occupancy differs.
    left_clutter = Detection(.01, .01, .32, .41, distance_m=.8)
    obs = observe([CENTER, left_clutter], Config())
    assert obs.blocked == (False, True, False)
    assert obs.occupancy[0] > obs.occupancy[2]
    assert candidate(obs)[0] == Action.RIGHT


def test_overlapping_boxes_do_not_inflate_side_occupancy():
    a = observe([LEFT], Config()).occupancy
    b = observe([LEFT, LEFT], Config()).occupancy
    assert a == b


def test_new_center_obstacle_stops_forward_while_confirming_a_turn():
    ctl = Controller()
    for i in range(3):
        ctl.update([],i,100+i*.1,100+i*.1)
    assert ctl.current.action == Action.FORWARD
    decision = ctl.update([CENTER],3,100.3,100.3)
    assert decision.action == Action.STOP and "Stabilizing LEFT" in decision.reason


@pytest.mark.parametrize("detections,expected", [([],Action.FORWARD),([CENTER],Action.LEFT),
    ([CENTER,LEFT],Action.RIGHT)])
def test_slow_but_fresh_results_can_confirm_movement_after_output_expiry(detections,expected):
    ctl = Controller(Config(stable_frames=3,stale_after_ms=1000))
    for index in range(3):
        captured = 100 + index*.6
        decision = ctl.update(detections,index,captured,captured+.6)
        assert decision.action == (expected if index==2 else Action.STOP)
        # Between results the old output must STOP, but a short capture gap
        # must not erase the evidence from genuinely distinct observations.
        assert ctl.tick(captured+1.05).action == Action.STOP
        assert ctl.streak == index+1
    assert ctl.update(detections,3,101.8,102.4).action == expected


def test_preserved_evidence_cannot_restart_after_a_real_capture_outage():
    ctl = Controller()
    ctl.update([],0,100.,100.6)
    ctl.update([],1,100.6,101.2)
    ctl.tick(101.7)
    assert ctl.streak == 2
    result = ctl.update([],2,103.,103.6)
    assert result.action == Action.STOP and ctl.streak == 1


def test_expiry_never_accepts_an_already_stale_result():
    ctl = Controller()
    ctl.update([],0,100.,100.6)
    ctl.update([],1,100.6,101.2)
    ctl.tick(101.7)
    assert ctl.update([],2,101.2,102.3).action == Action.STOP
    assert ctl.streak == 0


def test_pause_resets_evidence_even_after_output_expiry():
    ctl = Controller()
    ctl.update([],0,100.,100.6)
    ctl.update([],1,100.6,101.2)
    ctl.tick(101.7)
    ctl.tick(101.8,"PAUSED")
    assert ctl.streak == 0 and ctl.pending is None
    assert ctl.update([],2,101.2,101.85).action == Action.STOP


def test_changed_rule_cannot_reuse_the_previous_confirmation_streak():
    ctl = Controller()
    ctl.update([],0,100.,100.6)
    ctl.update([],1,100.6,101.2)
    ctl.tick(101.7)
    assert ctl.update([CENTER],2,101.2,101.8).action == Action.STOP
    assert ctl.streak == 1 and ctl.pending == Action.LEFT
