"""Geometry checks using known depth fields, not a model-accuracy benchmark."""
import numpy as np
import pytest
from visiondrive.config import Config
from visiondrive.decision import Action, candidate, observe
from visiondrive.depth import ground_plane, surface_detections


def floor():
    y, _ = np.mgrid[:120, :160]
    return (1 / (.12 + .60*y/120)).astype(np.float32)


def test_flat_floor_does_not_block_the_vehicle():
    depth = floor()
    assert ground_plane(depth) is not None
    detections, found = surface_detections(depth, Config())
    assert found and detections == ()


def test_unknown_center_shape_causes_turn_without_object_class():
    depth = floor()
    depth[52:88, 65:95] = .8
    detections, found = surface_detections(depth, Config())
    assert found and detections
    assert all(d.label == "depth obstacle" for d in detections)
    observation = observe(detections, Config())
    assert observation.blocked == (False, True, False)
    assert candidate(observation)[0] == Action.LEFT


@pytest.mark.parametrize("distance,expected", [(3., Action.FORWARD), (1., Action.STOP), (.3, Action.STOP)])
def test_front_facing_wall_cannot_be_fitted_away_as_floor(distance, expected):
    depth = np.full((120,160), distance, dtype=np.float32)
    detections, found = surface_detections(depth, Config())
    assert not found
    assert candidate(observe(detections, Config()))[0] == expected


def test_emergency_unknown_surface_stops():
    depth = floor()
    depth[52:88, 65:95] = .3
    detections, _ = surface_detections(depth, Config())
    assert candidate(observe(detections, Config()))[0] == Action.STOP


@pytest.mark.parametrize("value", [0., np.nan, np.inf, -.2])
def test_invalid_depth_fails_instead_of_returning_clear_path(value):
    with pytest.raises(ValueError, match="invalid"):
        surface_detections(np.full((32,32), value), Config())
