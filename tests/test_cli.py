import subprocess
import sys
from pathlib import Path
import cv2
import pytest


ROOT = Path(__file__).resolve().parents[1]


def test_removed_detector_backend_is_rejected():
    from demo import parser
    with pytest.raises(SystemExit) as error:
        parser().parse_args(['--detector', 'esp'])
    assert error.value.code == 2


def test_simulation_records_and_labels_itself(tmp_path):
    output = tmp_path / "demo.mp4"
    image = tmp_path / "preview.png"
    result = subprocess.run([sys.executable, str(ROOT / "demo.py"), "--simulate", "--headless",
                             "--max-frames", "8", "--save-video", str(output), "--snapshot", str(image)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    assert "Synthetic" in result.stdout
    assert image.exists() and output.stat().st_size > 0
    rows = output.with_suffix(".jsonl").read_text().splitlines()
    assert len(rows) == 8
    assert all('"simulated": true' in row for row in rows)
    cap = cv2.VideoCapture(str(output))
    ok, frame = cap.read()
    cap.release()
    assert ok and frame.shape[:2] == (728, 1280)


def test_missing_weights_error_precedes_camera_open(tmp_path):
    result = subprocess.run([sys.executable, str(ROOT / "demo.py"), "--detector", "coco", "--weights", str(tmp_path / "missing.pt"),
                             "--headless", "--max-frames", "1"], capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "Checkpoint not found" in result.stderr
    assert "setup_laptop_models.py" in result.stderr


def test_headless_requires_limit_for_infinite_input():
    result = subprocess.run([sys.executable, str(ROOT / "demo.py"), "--simulate", "--headless"],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert "--max-frames" in result.stderr


def test_logic_view_renders_and_records_with_explicit_synthetic_input(tmp_path):
    import json
    output = tmp_path / "logic.mp4"
    result = subprocess.run([sys.executable, str(ROOT / "logic_demo.py"), "--simulate", "--headless",
                             "--max-frames", "5", "--save-video", str(output)],
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr
    rows = [json.loads(line) for line in output.with_suffix(".jsonl").read_text().splitlines()]
    assert len(rows) == 5
    assert all(row["presentation"] == "logic" and row["simulated"] for row in rows)


def test_simulated_obstacles_cannot_send_motor_commands():
    result = subprocess.run([sys.executable,str(ROOT / "demo.py"),"--simulate","--headless",
        "--max-frames","1","--esp-host","127.0.0.1"],capture_output=True,text=True,timeout=30)
    assert result.returncode == 2 and "require a live camera" in result.stderr
