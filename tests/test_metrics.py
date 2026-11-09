"""Tests for darktriage.core.metrics."""

from __future__ import annotations

import math

import numpy as np
import pytest
from PIL import Image

from darktriage.core.metrics import (
    ExposureReport,
    ImageMetadata,
    Score,
    apply_noise_filter,
    apply_orientation,
    apply_region_mask,
    check_exposure,
    compute_composition_score,
    compute_ev_offset,
    compute_tenengrad,
    load_image_metadata,
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

    scores = score_images([(p, None, 1) for p in paths], max_dim=256, workers=2)
    assert len(scores) == 2
    assert all(isinstance(s, Score) for s in scores)
    assert all(s.sharpness > 0.0 for s in scores)


def test_load_image_metadata_reads_iso_and_orientation(tmp_path):
    path = tmp_path / "meta.jpg"
    image = Image.new("RGB", (32, 32), "gray")
    exif = Image.Exif()
    exif[0x8827] = 3200  # ISOSpeedRatings
    exif[0x0112] = 6      # orientation = rotate 90 CW
    image.save(path, exif=exif)

    meta = load_image_metadata(path)
    assert isinstance(meta, ImageMetadata)
    assert meta.iso == 3200
    assert meta.orientation == 6


def test_load_image_metadata_defaults_when_no_exif(tmp_path):
    path = tmp_path / "plain.jpg"
    Image.new("RGB", (16, 16), "gray").save(path)
    meta = load_image_metadata(path)
    assert meta.iso is None
    assert meta.orientation == 1


def test_apply_noise_filter_blurs_only_high_iso():
    rng = np.random.default_rng(0)
    arr = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    low = apply_noise_filter(arr, iso=200, iso_threshold=1600)
    high = apply_noise_filter(arr, iso=3200, iso_threshold=1600)
    assert np.array_equal(low, arr)  # no blur below threshold
    assert not np.array_equal(high, arr)  # blur applied above threshold


def test_apply_orientation_rotates_interchanges_shape():
    arr = np.zeros((10, 20), dtype=np.uint8)
    assert apply_orientation(arr, 6).shape == (20, 10)  # 90 CW swaps dims
    assert apply_orientation(arr, 3).shape == (10, 20)  # 180 keeps dims
    assert apply_orientation(arr, 1).shape == (10, 20)


def test_apply_region_mask_weights_center():
    arr = np.full((65, 65), 200, dtype=np.uint8)
    weighted = apply_region_mask(arr, fraction=0.6)
    assert weighted.shape == arr.shape
    assert weighted.dtype == np.float64
    # centre keeps full weight; far corners are suppressed
    assert weighted[32, 32] == pytest.approx(200.0)
    assert weighted[0, 0] < weighted[32, 32]