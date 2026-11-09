"""Tests for data-driven threshold calibration (core/calibration.py)."""

from __future__ import annotations

from pathlib import Path

import pytest

from darktriage.config import load_config
from darktriage.core.calibration import (
    collect_scores,
    render_config,
    summarise,
    suggest_thresholds,
    write_config,
)
from darktriage.core.engine import AnalysisOptions
from darktriage.core.exif_reader import ImagePair
from darktriage.core.metrics import Score
from darktriage.core.previews import default_preview_cache_dir


def _score(sharpness: float, composition: float, clipped_pct: float) -> Score:
    return Score(
        sharpness=sharpness,
        composition_score=composition,
        mean_luminance=128.0,
        clipped_highlight_pct=clipped_pct,
    )


def test_summarise_percentiles():
    values = [float(i) for i in range(1, 101)]
    summary = summarise(values)
    assert summary.count == 100
    assert summary.minimum == 1.0
    assert summary.maximum == 100.0
    assert summary.p50 == pytest.approx(50.5, abs=0.6)


def test_summarise_empty_is_zero():
    summary = summarise([])
    assert summary.count == 0
    assert summary.minimum == 0.0
    assert summary.maximum == 0.0


def test_suggest_thresholds_ordering():
    scores = [_score(s, 1.0 + s / 10.0, 0.0) for s in range(10, 200, 10)]
    suggested = suggest_thresholds(scores)
    assert suggested["tenengrad_blur"] < suggested["tenengrad_sharp"]
    assert suggested["composition_min"] > 0
    assert 0.005 <= suggested["highlight_clip_percent"] <= 0.5


def test_suggest_thresholds_empty_is_sane():
    suggested = suggest_thresholds([])
    assert suggested["tenengrad_blur"] == 0.0
    assert suggested["tenengrad_sharp"] > suggested["tenengrad_blur"]
    assert suggested["highlight_clip_percent"] >= 0.005


def test_render_config_roundtrip(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    suggested = {
        "tenengrad_blur": 123.4,
        "tenengrad_sharp": 456.7,
        "composition_min": 2.75,
        "highlight_clip_percent": 0.02,
        "burst_interval_sec": 2.0,
    }
    target = tmp_path / "config.toml"
    target.write_text(render_config(config, suggested), encoding="utf-8")

    loaded = load_config(target)
    assert loaded.thresholds.tenengrad_blur == pytest.approx(123.4)
    assert loaded.thresholds.tenengrad_sharp == pytest.approx(456.7)
    assert loaded.thresholds.composition_min == pytest.approx(2.75)
    assert loaded.thresholds.highlight_clip_percent == pytest.approx(0.02)
    # untouched sections preserved
    assert loaded.tags.blur == config.tags.blur
    assert loaded.sensor.iso_blur_above == config.sensor.iso_blur_above
    assert loaded.safety.preserve_existing == config.safety.preserve_existing


def test_write_config_backs_up(tmp_path):
    target = tmp_path / "config.toml"
    target.write_text("original", encoding="utf-8")
    write_config(target, "new", backup=True)
    assert target.read_text() == "new"
    assert (tmp_path / "config.toml.bak").read_text() == "original"


def test_default_preview_cache_dir_is_stable_and_unique(tmp_path):
    one = tmp_path / "shoot-one"
    two = tmp_path / "shoot-two"
    one.mkdir()
    two.mkdir()
    a = default_preview_cache_dir(one)
    assert a == default_preview_cache_dir(one)
    assert a != default_preview_cache_dir(two)
    assert "darktriage" in a.parts and "previews" in a.parts


def test_collect_scores_disables_early_exit(tmp_path):
    # A flat image would normally early-exit (composition=0); calibration must
    # still return a real composition value, so collect_scores uses blur=0.
    from PIL import Image
    import numpy as np

    jpg = tmp_path / "flat.jpg"
    Image.fromarray(np.full((120, 120), 128, dtype=np.uint8)).save(jpg)
    pair = ImagePair(
        raw_path=tmp_path / "flat.NEF",
        jpg_path=jpg,
        metadata_loaded=True,
    )
    scores = collect_scores([pair], AnalysisOptions(max_dim=128, workers=1))
    assert len(scores) == 1
    assert scores[0].early_exit is False
