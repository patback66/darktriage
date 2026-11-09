"""Tests for the machine-readable analyze path used by the TUI."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from darktriage import cli
from darktriage.config import load_config
from darktriage.core.engine import assessment_from_dict, assessment_to_dict
from darktriage.core.exif_reader import Assessment


def _sharp_pattern() -> np.ndarray:
    grid = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (32, 32))
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _write_pair(directory: Path, stem: str, dt: str) -> None:
    image = Image.fromarray(_sharp_pattern())
    exif = Image.Exif()
    exif[0x9003] = dt
    exif[0x9291] = "000000"
    image.save(directory / f"{stem}.JPG", exif=exif)
    (directory / f"{stem}.NEF").write_bytes(b"RAW")


def test_assessment_dict_roundtrip(tmp_path):
    from datetime import datetime

    original = Assessment(
        raw_path=tmp_path / "A.NEF",
        jpg_path=tmp_path / "A.JPG",
        capture_time=datetime(2024, 1, 1, 12, 0, 0),
        burst_index=2,
        is_burst_winner=True,
        sharpness=1234.5,
        composition_score=7.5,
        mean_luminance=120.0,
        clipped_highlight_pct=4.2,
        ev_offset=0.3,
        rating=4,
        label="green",
    )
    restored = assessment_from_dict(assessment_to_dict(original))
    assert restored == original


def test_analyze_json_out_writes_assessments(tmp_path, capsys):
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    _write_pair(shoot, "A", "2024:01:01 12:00:00")
    _write_pair(shoot, "B", "2024:01:01 12:00:00")

    out = tmp_path / "assessments.json"
    code = cli.main(
        [
            "--config",
            str(tmp_path / "missing.toml"),
            "--threads",
            "1",
            "analyze",
            str(shoot),
            "--json-out",
            str(out),
            "--dry-run",
            "--max-dim",
            "128",
        ]
    )

    assert code == 0
    assert out.exists()
    payload = json.loads(out.read_text())
    assert len(payload) == 2
    assert {"raw_path", "rating", "sharpness", "is_burst_winner"} <= set(payload[0])
    err = capsys.readouterr().err
    assert "darktriage-progress discovered 2" in err
    assert "darktriage-progress scored" in err


def test_tui_analysis_command_isolates_process(tmp_path):
    from darktriage.tui.app import DarkTriageApp

    config = load_config(tmp_path / "missing.toml")
    app = DarkTriageApp(
        directory=str(tmp_path),
        config=config,
        max_dim=800,
        workers=3,
        use_embedded_preview=True,
        preview_cache_dir=str(tmp_path / "cache"),
        chunk_size=50,
    )
    command = app._analysis_command(tmp_path / "out.json")

    assert command[0] == __import__("sys").executable
    assert "analyze" in command
    assert "--json-out" in command
    assert "--dry-run" in command
    assert "--use-embedded-preview" in command
    assert "--threads" in command and "3" in command


def test_analyze_autotag_json_includes_tags(tmp_path, capsys):
    shoot = tmp_path / "shoot"
    shoot.mkdir()
    arr = np.full((120, 160, 3), (30, 140, 30), dtype=np.uint8)  # RGB green
    Image.fromarray(arr).save(shoot / "g.JPG", quality=95)
    (shoot / "g.NEF").write_bytes(b"RAW")

    out = tmp_path / "assessments.json"
    code = cli.main(
        [
            "--config", str(tmp_path / "missing.toml"),
            "--threads", "1",
            "analyze", str(shoot),
            "--json-out", str(out),
            "--dry-run", "--max-dim", "128", "--autotag",
        ]
    )
    assert code == 0
    payload = json.loads(out.read_text())
    assert payload and "nature" in payload[0]["tags"]
