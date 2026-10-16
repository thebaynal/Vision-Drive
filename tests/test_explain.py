from visiondrive.config import Config
from visiondrive.decision import Action, Controller, Detection
from visiondrive.explain import explain


def test_view_distinguishes_rule_candidate_from_stabilized_output():
    ctl = Controller()
    decision = ctl.update([Detection(.4, .45, .6, .7)], 0, 100, 100)
    trace = explain(decision)
    assert trace.candidate_action == Action.LEFT
    assert decision.action == Action.STOP
    assert trace.stabilizing


def test_stale_or_paused_input_cannot_be_presented_as_a_movement_candidate():
    decision = Controller(Config(stable_frames=1)).update([], 0, 100, 100)
    for status, age in (("PAUSED", 0), ("running", 1000)):
        trace = explain(decision, status, age)
        assert not trace.input_available
        assert trace.candidate_action == Action.STOP


def test_default_program_uses_trained_objects_without_cat_fallback(monkeypatch):
    import logic_demo
    seen = []
    monkeypatch.setattr(logic_demo, "demo_main", lambda args: seen.extend(args) or 0)
    assert logic_demo.main([]) == 0
    assert "--explain" in seen and "--simulate" not in seen
    assert "cat" not in seen and not any("cat" in arg for arg in seen)


def test_explicit_trained_model_is_preserved(monkeypatch):
    import logic_demo
    seen = []
    monkeypatch.setattr(logic_demo, "demo_main", lambda args: seen.extend(args) or 0)
    logic_demo.main(["--weights", "trained.pt", "--expected-label", "obstacle"])
    assert seen.count("--weights") == 1
    assert seen[seen.index("--weights") + 1] == "trained.pt"
    assert "cat" not in seen
