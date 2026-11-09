"""Tests for the bounded parallel analysis engine (core/engine.py)."""

from __future__ import annotations

from datetime import datetime, timedelta

import numpy as np
from PIL import Image

from darktriage.config import load_config
from darktriage.core.engine import AnalysisOptions, analyze_directory, chunked
from darktriage.core.exif_reader import ImagePair


def _sharp_pattern() -> np.ndarray:
    grid = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (32, 32))
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def _write_pair(tmp_path, stem: str, dt: str, subsec: str) -> None:
    image = Image.fromarray(_sharp_pattern())
    exif = Image.Exif()
    exif[0x9003] = dt
    exif[0x9291] = subsec
    image.save(tmp_path / f"{stem}.JPG", exif=exif)
    (tmp_path / f"{stem}.NEF").write_bytes(b"RAW")


def test_chunked_is_bounded_and_lossless():
    chunks = list(chunked(range(10), 3))
    assert [len(c) for c in chunks] == [3, 3, 3, 1]
    assert [x for c in chunks for x in c] == list(range(10))


def test_analyze_directory_serial_rates_all(tmp_path):
    _write_pair(tmp_path, "A", "2024:01:01 12:00:00", "000000")
    _write_pair(tmp_path, "B", "2024:01:01 12:00:00", "400000")

    config = load_config(tmp_path / "missing.toml")
    options = AnalysisOptions(max_dim=256, workers=1, chunk_size=1)
    assessments = analyze_directory(
        str(tmp_path), config, options, threshold_seconds=2.0
    )

    assert len(assessments) == 2
    assert all(a.rating is not None for a in assessments)
    # Both shots share a burst; exactly one winner.
    assert sum(1 for a in assessments if a.is_burst_winner) == 1
    assert all(a.burst_index == 0 for a in assessments)


def test_analyze_directory_parallel(tmp_path):
    _write_pair(tmp_path, "A", "2024:01:01 12:00:00", "000000")
    _write_pair(tmp_path, "B", "2024:01:01 12:00:00", "400000")
    _write_pair(tmp_path, "C", "2024:01:01 12:05:00", "000000")

    config = load_config(tmp_path / "missing.toml")
    options = AnalysisOptions(max_dim=256, workers=2, chunk_size=2)
    assessments = analyze_directory(
        str(tmp_path), config, options, threshold_seconds=2.0
    )

    assert len(assessments) == 3
    assert all(a.rating is not None for a in assessments)


def test_analyze_uses_image_map_without_jpeg(tmp_path):
    """An explicit image_map (e.g. an embedded preview) enables scoring a RAW
    that has no matching JPG."""
    (tmp_path / "A.NEF").write_bytes(b"RAW")
    preview = tmp_path / "A_preview.jpg"
    Image.fromarray(_sharp_pattern()).save(preview)

    pairs = [
        ImagePair(
            raw_path=tmp_path / "A.NEF",
            jpg_path=None,
            capture_time=datetime(2024, 1, 1, 12, 0, 0),
            metadata_loaded=True,
            iso=None,
            orientation=1,
        )
    ]
    config = load_config(tmp_path / "missing.toml")
    options = AnalysisOptions(max_dim=256, workers=1)

    assessments = analyze_directory(
        str(tmp_path),
        config,
        options,
        pairs=pairs,
        image_map={tmp_path / "A.NEF": preview},
    )

    assert len(assessments) == 1
    assert assessments[0].jpg_path is None
    assert assessments[0].rating is not None
    assert assessments[0].sharpness is not None


def test_singleton_frames_are_not_burst_winners(tmp_path):
    # Two lone frames, 10 s apart -> two singleton bursts, no burst bonus.
    _write_pair(tmp_path, "A", "2024:01:01 12:00:00", "000000")
    _write_pair(tmp_path, "B", "2024:01:01 12:05:00", "000000")

    config = load_config(tmp_path / "missing.toml")
    assessments = analyze_directory(
        str(tmp_path), config, AnalysisOptions(max_dim=256, workers=1),
        threshold_seconds=2.0,
    )
    assert len(assessments) == 2
    assert all(a.is_burst_winner is False for a in assessments)
