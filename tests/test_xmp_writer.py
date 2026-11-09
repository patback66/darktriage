"""Tests for darktriage.core.xmp_writer (non-destructive safety model)."""

from __future__ import annotations

import shutil
import stat

import pytest

from darktriage.config import Tags, Thresholds
from darktriage.core.xmp_writer import (
    ExifToolNotFoundError,
    SidecarRequest,
    XmpWriter,
    backup_sidecar,
    derive_rating_and_label,
    has_darktable_history,
    read_sidecar_rating,
    should_write_xmp,
    xmp_sidecar_path,
)


def _thresholds() -> Thresholds:
    return Thresholds(
        tenengrad_blur=150.0,
        tenengrad_sharp=350.0,
        composition_min=1.2,
        highlight_clip_percent=0.05,
        burst_interval_sec=2.0,
    )


def _tags() -> Tags:
    return Tags(blur="red", keeper="green", portfolio="purple", exposure_warn="yellow")


def test_xmp_sidecar_path():
    assert xmp_sidecar_path("DSC.NEF").name == "DSC.NEF.xmp"


def test_should_write_new_sidecar(tmp_path):
    assert should_write_xmp(tmp_path / "DSC.NEF.xmp") is True


def test_should_write_preserves_existing_rating(tmp_path):
    xmp = tmp_path / "DSC.NEF.xmp"
    xmp.write_text('<rdf:Description xmp:Rating="5"/>')
    assert should_write_xmp(xmp) is False
    assert should_write_xmp(xmp, overwrite=True) is True


def test_should_write_preserves_darktable_history(tmp_path):
    xmp = tmp_path / "DSC.NEF.xmp"
    xmp.write_text("<x:xmpmeta><darktable:history operation='monochrome'/></x:xmpmeta>")
    assert should_write_xmp(xmp) is False


def test_should_write_allows_sidecar_without_edits(tmp_path):
    xmp = tmp_path / "DSC.NEF.xmp"
    xmp.write_text("<x:xmpmeta></x:xmpmeta>")
    assert should_write_xmp(xmp) is True


def test_read_sidecar_rating_and_history(tmp_path):
    rated = tmp_path / "rated.xmp"
    rated.write_text("<xmp:Rating>3</xmp:Rating>")
    assert read_sidecar_rating(rated) == 3
    assert has_darktable_history(rated) is False

    history = tmp_path / "history.xmp"
    history.write_text("<darktable:history enabled='1'/>")
    assert has_darktable_history(history) is True


def test_backup_sidecar(tmp_path):
    xmp = tmp_path / "DSC.NEF.xmp"
    xmp.write_text("original")
    backup = backup_sidecar(xmp)
    assert backup == tmp_path / "DSC.NEF.xmp.bak"
    assert backup.read_text() == "original"


def test_derive_burst_winner_purple():
    decision = derive_rating_and_label(
        sharpness=400.0,
        composition_score=2.0,
        clipped_fraction=0.0,
        is_burst_winner=True,
        thresholds=_thresholds(),
        tags=_tags(),
    )
    assert (decision.rating, decision.label) == (5, "purple")


def test_derive_crisp_keeper_green():
    decision = derive_rating_and_label(
        sharpness=400.0,
        composition_score=0.5,
        clipped_fraction=0.0,
        is_burst_winner=False,
        thresholds=_thresholds(),
        tags=_tags(),
    )
    assert (decision.rating, decision.label) == (4, "green")


def test_derive_average_neutral():
    decision = derive_rating_and_label(
        sharpness=200.0,
        composition_score=1.0,
        clipped_fraction=0.0,
        is_burst_winner=False,
        thresholds=_thresholds(),
        tags=_tags(),
    )
    assert (decision.rating, decision.label) == (3, None)


def test_derive_soft_blur_red():
    decision = derive_rating_and_label(
        sharpness=50.0,
        composition_score=0.5,
        clipped_fraction=0.0,
        is_burst_winner=False,
        thresholds=_thresholds(),
        tags=_tags(),
    )
    assert (decision.rating, decision.label) == (1, "red")


def test_derive_clipping_overrides_label_to_yellow():
    decision = derive_rating_and_label(
        sharpness=400.0,
        composition_score=2.0,
        clipped_fraction=0.10,
        is_burst_winner=True,
        thresholds=_thresholds(),
        tags=_tags(),
    )
    assert decision.rating == 5
    assert decision.label == "yellow"

def test_writer_targets_sidecar_only(tmp_path):
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    result = writer.write(tmp_path / "DSC.NEF", rating=5, label="red", ev_offset=0.5)
    args = result.command
    assert args[-1].endswith("DSC.NEF.xmp")
    assert "-XMP-xmp:Rating=5" in args
    assert "-XMP-xmp:Label=red" in args
    assert "-XMP-dc:Description=darktriage recommended EV offset: +0.50" in args
    assert "-overwrite_original" in args


def test_writer_skips_protected_sidecar(tmp_path):
    (tmp_path / "DSC.NEF.xmp").write_text('<rdf:Description xmp:Rating="4"/>')
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    result = writer.write(tmp_path / "DSC.NEF", rating=5, label="red")
    assert result.skipped is True
    assert result.command == []


def test_backup_before_overwrite(tmp_path):
    script = tmp_path / "exiftool"
    script.write_text("#!/bin/sh\nexit 0\n")
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)

    (tmp_path / "DSC.NEF.xmp").write_text('<rdf:Description xmp:Rating="4"/>')
    writer = XmpWriter(exiftool=str(script), dry_run=False, create_backup=True)
    result = writer.write(tmp_path / "DSC.NEF", rating=5, label="red", overwrite=True)
    assert result.success is True
    assert (tmp_path / "DSC.NEF.xmp.bak").exists()


def test_write_batch_single_invocation(tmp_path):
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    requests = [
        SidecarRequest(raw_path=tmp_path / "A.NEF", rating=5, label="red"),
        SidecarRequest(raw_path=tmp_path / "B.NEF", rating=4, label="green"),
    ]
    results = writer.write_batch(requests)
    assert len(results) == 2
    assert all(r.success for r in results)
    # One execution pass via an ExifTool argfile (-@), not one call per file.
    command = results[0].command
    assert command[0] == "/exiftool"
    assert command[1] == "-@"

    lines = writer.build_argfile_lines(requests)
    assert lines.count("-execute") == 1  # separator between the two groups only
    assert lines[-1].endswith("B.NEF.xmp")  # no trailing -execute
    assert "-XMP-xmp:Rating=5" in lines
    assert "-XMP-xmp:Rating=4" in lines
    assert any(line.endswith("A.NEF.xmp") for line in lines)
    assert any(line.endswith("B.NEF.xmp") for line in lines)


def test_write_batch_skips_protected(tmp_path):
    (tmp_path / "A.NEF.xmp").write_text('<rdf:Description xmp:Rating="5"/>')
    writer = XmpWriter(exiftool="/exiftool", dry_run=True)
    requests = [
        SidecarRequest(raw_path=tmp_path / "A.NEF", rating=3, label=None),
        SidecarRequest(raw_path=tmp_path / "B.NEF", rating=4, label="green"),
    ]
    results = writer.write_batch(requests)
    assert results[0].skipped is True
    assert results[1].skipped is False


def test_writer_raises_when_exiftool_missing(monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda name: None)
    with pytest.raises(ExifToolNotFoundError):
        XmpWriter(exiftool="exiftool")