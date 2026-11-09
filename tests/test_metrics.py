"""Tests for darktriage.core.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from darktriage.core.metrics import (
    ExposureReport,
    Score,
    check_exposure,
    compute_composition_score,
    compute_ev_offset,
    compute_tenengrad,
    prepare_proxy,
    score_image,
    score_images,
)


def _sharp_pattern() -> np.ndarray:
    grid = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (64, 64))
    return np.repeat(np.repeat(grid, 2, axis=0), 2, axis=1)


def test_prepare_proxy_downscales_long_edge(tmp_path):
    image = Image.new("L", (4000, 2000), 128).convert("RGB")
    path = tmp_path / "big.jpg"
    image.save(path)
    proxy = prepare_proxy(path, max_dim=1920)
    assert proxy.ndim == 2
    assert max(proxy.shape) <= 1920
    assert proxy.dtype == np.uint8


def test_prepare_proxy_keeps_small_images_unchanged(tmp_path):
    image = Image.new("L", (64, 64), 128).convert("RGB")
    path = tmp_path / "small.jpg"
    image.save(path)
    assert prepare_proxy(path, max_dim=1920).shape == (64, 64)


def test_compute_tenengrad_sharp_beats_flat():
    sharp = _sharp_pattern()
    flat = np.full((128, 128), 128, dtype=np.uint8)
    assert compute_tenengrad(sharp) > 0.0
    assert compute_tenengrad(sharp) > compute_tenengrad(flat)


def test_compute_composition_score_is_mass_ratio():
    arr = np.zeros((300, 300), dtype=np.uint8)
    arr[:, :100] = 255  # edge energy concentrated at the left third
    ratio = compute_composition_score(arr)
    assert ratio >= 0.0
    assert math.isfinite(ratio)


def test_check_exposure_counts_clipped_highlights():
    arr = np.full((100, 100), 200, dtype=np.uint8)
    arr[:, :30] = 255
    report = check_exposure(arr)
    assert isinstance(report, ExposureReport)
    assert report.clipped_highlight_pct == pytest.approx(30.0)


def test_compute_ev_offset_bounds():
    assert compute_ev_offset(128.0) == pytest.approx(0.0)
    assert compute_ev_offset(64.0) == pytest.approx(1.0)
    assert compute_ev_offset(0.0) == 2.0  # clamped to max
    assert compute_ev_offset(1000.0) == -2.0  # clamped to min
    assert -2.0 <= compute_ev_offset(255.0) <= 2.0


def test_score_image_includes_ev_offset(tmp_path):
    path = tmp_path / "sharp.jpg"
    Image.fromarray(_sharp_pattern()).save(path)
    score = score_image(path, max_dim=512)
    assert isinstance(score, Score)
    assert score.sharpness > 0.0
    assert -2.0 <= score.ev_offset <= 2.0


def test_score_images_parallel(tmp_path):
    paths = []
    for i in range(2):
        path = tmp_path / f"img{i}.jpg"
        Image.fromarray(_sharp_pattern()).save(path)
        paths.append(path)

    scores = score_images(paths, max_dim=256, workers=2)
    assert len(scores) == 2
    assert all(isinstance(s, Score) for s in scores)
    assert all(s.sharpness > 0.0 for s in scores)