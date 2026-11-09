"""Tests for DarkTriage OpenCV metrics."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from darktriage.core.metrics import (
    Rating,
    analyze_image,
    composition_report,
    exposure_report,
    tenengrad_sharpness,
)

# A high-frequency (sharp) 64x64 checkerboard pattern.
_SHARP = np.tile(np.array([[0, 255], [255, 0]], dtype=np.uint8), (32, 32))
_SHARP = np.repeat(np.repeat(_SHARP, 2, axis=0), 2, axis=1)

# A smooth, low-frequency gradient (behaves like a soft image for Tenengrad).
_BLUR = np.tile(np.linspace(0, 255, 64, dtype=np.uint8), (64, 1))


def test_to_grayscale_handles_rgb_and_rgba():
    rgb = np.zeros((8, 8, 3), dtype=np.uint8)
    rgba = np.zeros((8, 8, 4), dtype=np.uint8)
    assert tenengrad_sharpness(rgb.astype(np.uint8)) == 0.0
    assert tenengrad_sharpness(rgba.astype(np.uint8)) == 0.0


def test_tenengrad_sharpness_ranks_sharp_above_blurred():
    sharp = tenengrad_sharpness(_SHARP.astype(np.uint8))
    blurred = tenengrad_sharpness(_BLUR.astype(np.uint8))
    assert sharp > 0.0
    assert sharp > blurred


def test_tenengrad_sharpness_is_resolution_independent():
    resized = np.array(Image.fromarray(_SHARP).resize((32, 32)), dtype=np.uint8)
    assert tenengrad_sharpness(resized) > 0.0


def test_exposure_report_counts_clipping_on_solid_white():
    white = np.full((64, 64), 255, dtype=np.uint8)
    report = exposure_report(white)
    assert report["highlight_ratio"] > 0.5
    assert report["clipped"] is True
    assert report["mean_luminance"] == 255.0


def test_exposure_report_counts_clipping_on_black():
    black = np.zeros((64, 64), dtype=np.uint8)
    report = exposure_report(black)
    assert report["highlight_ratio"] == 0.0
    assert report["underexposed"] > 0.99


def test_composition_report_returns_grid_samples():
    report = composition_report(_SHARP)
    assert len(report["cell_luminances"]) == 9
    assert len(report["node_luminances"]) == 4
    assert 0.0 <= report["composition_score"] <= 100.0


def test_analyze_image_returns_rating_and_label():
    metrics = analyze_image(_SHARP)
    assert "sharpness" in metrics
    assert "exposure" in metrics
    assert "composition" in metrics
    rating = metrics["rating"]
    assert isinstance(rating, Rating)
    assert 1 <= rating.rating <= 5
    assert rating.label in {"red", "yellow", "green", "purple"}


def test_analyze_image_accepts_numpy_array(tmp_path):
    image = Image.fromarray(_SHARP)
    path = tmp_path / "sharp.jpg"
    image.save(path)
    via_array = analyze_image(_SHARP)
    via_file = analyze_image(path)
    assert via_array["rating"].rating == via_file["rating"].rating


def test_analyze_image_rejects_unknown_types():
    with pytest.raises(TypeError):
        analyze_image(object())  # type: ignore[arg-type]
