"""Tests for darktriage.core.metrics."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from darktriage.core.metrics import (
    ExposureReport,
    Score,
    check_exposure,
    compute_composition_score,
    compute_tenengrad,
    prepare_proxy,
    score_image,
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
    proxy = prepare_proxy(path, max_dim=1920)
    assert proxy.shape == (64, 64)


def test_compute_tenengrad_sharp_beats_flat():
    sharp = _sharp_pattern()
    flat = np.full((128, 128), 128, dtype=np.uint8)
    assert compute_tenengrad(sharp) > 0.0
    assert compute_tenengrad(sharp) > compute_tenengrad(flat)


def test_compute_composition_score_in_range():
    arr = np.random.default_rng(0).integers(0, 256, (256, 256), dtype=np.uint8)
    score = compute_composition_score(arr)
    assert 0.0 <= score <= 1.0


def test_check_exposure_counts_clipped_highlights():
    arr = np.full((100, 100), 200, dtype=np.uint8)
    arr[:, :30] = 255  # 30% clipped
    report = check_exposure(arr)
    assert isinstance(report, ExposureReport)
    assert report.clipped_highlight_pct == pytest.approx(30.0)


def test_check_exposure_returns_zero_for_black():
    arr = np.zeros((64, 64), dtype=np.uint8)
    report = check_exposure(arr)
    assert report.clipped_highlight_pct == 0.0


def test_score_image_end_to_end(tmp_path):
    path = tmp_path / "sharp.jpg"
    Image.fromarray(_sharp_pattern()).save(path)
    score = score_image(path, max_dim=512)
    assert isinstance(score, Score)
    assert score.sharpness > 0.0
    assert 0.0 <= score.composition_score <= 1.0
    assert score.mean_luminance > 0.0