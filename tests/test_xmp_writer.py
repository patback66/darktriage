"""Tests for darktriage.core.xmp_writer."""

from __future__ import annotations

import shutil
import stat

import pytest

from darktriage.core.xmp_writer import (
    LABEL_GREEN,
    LABEL_PURPLE,
    LABEL_RED,
    LABEL_YELLOW,
    ExifToolNotFoundError,
    XmpWriter,
    derive_rating_and_label,
    xmp_sidecar_path,
)


def test_xmp_sidecar_path_appends_extension():
    assert xmp_sidecar_path("DSC_0001.NEF").name == "DSC_0001.NEF.xmp"


def test_derive_burst_winner_purple():
    decision = derive_rating_and_label(
        sharpness=300.0, composition_score=0.8, clipped_pct=0.0, is_burst_winner=True
    )
    assert (decision.rating, decision.label) == (5, LABEL_PURPLE)


def test_derive_crisp_keeper_green():
    decision = derive_rating_and_label(
        sharpness=300.0, composition_score=0.0, clipped_pct=0.0
    )
    assert (decision.rating, decision.label) == (4, LABEL_GREEN)


def test_derive_average_neutral():
    decision = derive_rating_and_label(
        sharpness=100.0, composition_score=0.3, clipped_pct=0.0
    )
    assert (decision.rating, decision.label) == (3, None)


def test_derive_soft_blur_red():
    decision = derive_rating_and_label(
        sharpness=10.0, composition_score=0.0, clipped_pct=0.0
    )
    assert (decision.rating, decision.label) == (1, LABEL_RED)


def test_derive_clipping_overrides_label_to_yellow():
    decision = derive_rating_and_label(
        sharpness=400.0, composition_score=0.9, clipped_pct=7.0
    )
    assert decision.rating == 4
    assert decision.label == LABEL_YELLOW


def test_writer_builds_exiftool_args():
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    result = writer.write("/photos/DSC.NEF", rating=5, label="Purple")
    args = result.command
    assert "-XMP-xmp:Rating=5" in args
    assert "-XMP-xmp:Label=Purple" in args
    assert "-overwrite_original" in args
    assert args[-1] == "/photos/DSC.NEF"


def test_writer_clears_label_when_none():
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    result = writer.write("/photos/DSC.NEF", rating=3, label=None)
    assert "-XMP-xmp:Label=" in result.command


def test_writer_accepts_pathlike(tmp_path):
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    target = tmp_path / "DSC.NEF"
    result = writer.write(target, rating=4, label="Green")
    assert result.raw_path == target
    assert result.success is True


def test_writer_executes_exiftool(tmp_path):
    script = tmp_path / "exiftool"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    writer = XmpWriter(exiftool=str(script), dry_run=False)
    result = writer.write("/photos/DSC.NEF", rating=4, label="Green")
    assert result.success is True


def test_writer_raises_when_exiftool_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ExifToolNotFoundError):
        XmpWriter(exiftool="exiftool")