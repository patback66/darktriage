"""Tests for DarkTriage XMP sidecar writer (mocks the exiftool binary)."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from darktriage.core.xmp_writer import (
    ExifToolError,
    ExifToolNotFoundError,
    LABEL_BY_NAME,
    XmpWriter,
    WriteResult,
)

FAKE_EXIFTOOL_CONTENT = """#!/bin/sh
echo "$*" >> "$SANDBOX/cmd.log"
"""


@pytest.fixture()
def fake_exiftool(tmp_path):
    script = tmp_path / "exiftool"
    script.write_text(FAKE_EXIFTOOL_CONTENT)
    script.chmod(script.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return script


@pytest.fixture()
def exiftool_env(tmp_path, fake_exiftool):
    env = os.environ.copy()
    env["PATH"] = f"{tmp_path}:{env.get('PATH', '')}"
    os.environ["SANDBOX"] = str(tmp_path)
    return env


def test_resolve_tool_raises_when_binary_missing():
    with pytest.raises(ExifToolNotFoundError):
        XmpWriter(exiftool="exiftool")


def test_write_rating_builds_command(fake_exiftool, exiftool_env, tmp_path):
    writer = XmpWriter(exiftool=str(fake_exiftool), dry_run=False, verbose=False)
    result = writer.write_rating(str(tmp_path / "photo.jpg"), rating=4)

    assert isinstance(result, WriteResult)
    assert result.rating == 4
    assert result.label is None
    assert result.success is True
    assert "-XmpRating=4" in result.command[-1] or "-XmpRating=4" in " ".join(result.command)

    log = tmp_path / "cmd.log"
    assert log.exists()
    assert "photo.jpg" in log.read_text()
    assert "XmpRating" in log.read_text()


def test_write_label_maps_color_to_darktable_value(fake_exiftool, exiftool_env, tmp_path):
    writer = XmpWriter(exiftool=str(fake_exiftool), dry_run=False)
    result = writer.write_label(str(tmp_path / "photo.jpg"), label="purple")

    assert result.label == "purple"
    assert result.rating is None
    log = tmp_path / "cmd.log"
    assert log.exists()
    assert "XmpLabel=4" in log.read_text()


@pytest.mark.parametrize("bad_value", [0, 6, -1, "gold"])
def test_rating_and_label_validations(bad_value):
    writer = XmpWriter(exiftool="/nonexistent/exiftool", dry_run=True)
    with pytest.raises(ValueError):
        writer.write_rating("x.jpg", rating=bad_value)  # type: ignore[arg-type]


def test_write_sidecar_combines_rating_and_label(fake_exiftool, exiftool_env, tmp_path):
    writer = XmpWriter(exiftool=str(fake_exiftool), dry_run=False)
    result = writer.write_sidecar(
        str(tmp_path / "photo.jpg"), rating=5, label="green"
    )

    assert result.rating == 5
    assert result.label == "green"
    log = tmp_path / "cmd.log"
    assert log.exists()
    assert "XmpRating=5" in log.read_text()
    assert "XmpLabel=3" in log.read_text()


def test_write_batch_applies_same_rating_to_many_files(fake_exiftool, exiftool_env, tmp_path):
    writer = XmpWriter(exiftool=str(fake_exiftool), dry_run=False)
    files = [tmp_path / f"img{i}.jpg" for i in range(3)]
    results = writer.write_batch(files, rating=3, label="yellow")

    assert len(results) == 3
    log = tmp_path / "cmd.log"
    assert log.exists()
    content = log.read_text()
    assert content.count("XmpRating=3") == 3
    assert content.count("XmpLabel=2") == 3


def test_dry_run_does_not_invoke_exiftool(fake_exiftool, tmp_path):
    writer = XmpWriter(exiftool=str(fake_exiftool), dry_run=True)
    result = writer.write_rating(str(tmp_path / "photo.jpg"), rating=2)
    assert result.success is True
    log = tmp_path / "cmd.log"
    assert not log.exists()
