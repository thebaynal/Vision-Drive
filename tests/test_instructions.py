"""The preview can wait for inference; motor commands cannot borrow that time."""

from types import SimpleNamespace
import pytest
from visiondrive.config import Config
from visiondrive.decision import Action, Detection
from visiondrive.instructions import Instructions

CENTER = Detection(.4,.45,.6,.72)
LEFT = Detection(.06,.45,.26,.72)


@pytest.mark.parametrize("latency", [.6,1.2,2.])
@pytest.mark.parametrize("detections,expected", [([],Action.FORWARD),([CENTER],Action.LEFT),
                                                           ([CENTER,LEFT],Action.RIGHT)])
def test_visual_instruction_does_not_flicker_between_consistent_results(latency,detections,expected):
    instructions = Instructions(Config())
    for frame in range(6):
        captured = 100+frame*latency
        completed = captured+latency
        instructions.update(detections,frame,captured,completed)
        for step in range(20):
            now = completed + (step/20)*latency
            preview, control = instructions.tick(now)
            assert preview.action == (expected if frame>=2 else Action.STOP)
            if now-captured >= 1.:
                assert control.action == Action.STOP


def confirmed_runtime():
    instructions = Instructions(Config())
    for frame in range(3):
        captured = 100+frame*.1
        instructions.update([],frame,captured,captured)
    return instructions


def test_visual_hold_is_bounded_and_does_not_extend_motor_deadline():
    instructions = confirmed_runtime()
    preview, control = instructions.tick(101.5)
    assert preview.action == Action.FORWARD and control.action == Action.STOP
    preview, control = instructions.tick(105.3)
    assert preview.action == control.action == Action.STOP
    assert instructions.preview.config.stale_after_ms == 5000
    assert instructions.control.config.stale_after_ms == 1000


@pytest.mark.parametrize("status", ["PAUSED","Camera unavailable","Inference error: failed","VIDEO ENDED"])
def test_input_failure_overrides_visual_hold_immediately(status):
    instructions = confirmed_runtime()
    preview, control = instructions.tick(100.3,status)
    assert preview.action == control.action == Action.STOP
    assert instructions.preview.streak == instructions.control.streak == 0


def test_new_emergency_observation_overrides_held_instruction():
    instructions = confirmed_runtime()
    instructions.tick(101.3)
    instructions.update([Detection(.4,.45,.6,.72,distance_m=.3)],3,101.3,101.5)
    preview, control = instructions.tick(101.5)
    assert preview.action == control.action == Action.STOP
    assert preview.reason == "Emergency center clearance"


def test_both_controllers_receive_identical_obstacles_from_one_shot_iterators():
    instructions = Instructions(Config(stable_frames=1))
    detections = (d for d in [Detection(.4,.45,.6,.72,distance_m=.3)])
    instructions.update(detections,0,100.,100.2)
    preview, control = instructions.tick(100.2)
    assert preview.action == control.action == Action.STOP
    assert preview.observation == control.observation
    assert control.observation.near_center


@pytest.mark.parametrize("timeout", [999,10001,True,1500.5])
def test_preview_timeout_must_be_explicit_and_bounded(timeout):
    with pytest.raises(ValueError,match="preview-timeout"):
        Instructions(Config(),timeout)


def test_application_sends_strict_control_even_when_preview_says_forward(monkeypatch):
    import demo
    runtime = confirmed_runtime()
    monkeypatch.setattr(demo,"Instructions",lambda *a:runtime)
    monkeypatch.setattr(demo.time,"monotonic",lambda:101.5)
    capture = SimpleNamespace(isOpened=lambda:True,get=lambda _:30)
    monkeypatch.setattr("cv2.VideoCapture",lambda *_:capture)
    detector = SimpleNamespace(name="Test model",label="objects",warmup=lambda:None)
    monkeypatch.setattr("visiondrive.laptop_detector.LaptopDetector",lambda *a:detector)
    pipeline = SimpleNamespace(start=lambda:None,close=lambda:None,
                             snapshot=lambda:("running",None,None,0))
    monkeypatch.setattr("visiondrive.pipeline.Pipeline",lambda *a:pipeline)
    sent = []
    link = SimpleNamespace(status="test link",last_action=Action.STOP,close=lambda:None,
                           send=lambda decision,deadline:sent.append((decision.action,deadline)))
    monkeypatch.setattr("visiondrive.esp_link.ESPLink",lambda *a:link)
    args = demo.parser().parse_args(["--headless","--max-frames","1","--esp-host","127.0.0.1"])
    assert demo.run(args) == 0
    assert runtime.preview.current.action == Action.FORWARD
    assert sent == [(Action.STOP,1000)]
