"""Tests for embedded-preview extraction (ExifTool argfile batching)."""

from __future__ import annotations

import subprocess
from pathlib import Path

import pytest

from darktriage.core.previews import (
    DEFAULT_PREVIEW_TAGS,
    _valid_jpeg,
    extract_previews,
)

MINIMAL_JPEG = b"\xff\xd8\xff" + b"0" * 200

TAG_FLAGS = {f"-{tag}" for tag in DEFAULT_PREVIEW_TAGS}


def _fake_runner(tag_that_writes: str | None):
    """Return a runner that writes a minimal JPEG only for one tag (or all)."""

    def runner(command, **kwargs):
        tag = next(flag for flag in command if flag in TAG_FLAGS)
        pattern = command[command.index("-w!") + 1]
        outdir = Path(pattern).parent
        for arg in command[command.index("-w!") + 2:]:
            raw = Path(arg)
            if tag_that_writes is None or tag == tag_that_writes:
                (outdir / (raw.name + ".jpg")).write_bytes(MINIMAL_JPEG)
        return subprocess.CompletedProcess(command, 0, "", "")

    return runner


def test_valid_jpeg_detects_sof_magic(tmp_path):
    good = tmp_path / "a.jpg"
    good.write_bytes(MINIMAL_JPEG)
    bad = tmp_path / "b.jpg"
    bad.write_bytes(b"not a jpeg at all")
    assert _valid_jpeg(good) is True
    assert _valid_jpeg(bad) is False
    assert _valid_jpeg(tmp_path / "missing.jpg") is False


def test_extract_previews_primary_tag(tmp_path):
    raws = [tmp_path / "A.NEF", tmp_path / "B.NEF"]
    for raw in raws:
        raw.write_bytes(b"RAW")
    cache = tmp_path / "cache"

    result = extract_previews(
        raws, cache, runner=_fake_runner("-PreviewImage")
    )
    assert set(result) == set(raws)
    assert all(path.exists() for path in result.values())


def test_extract_previews_falls_back_to_next_tag(tmp_path):
    raws = [tmp_path / "A.NEF"]
    raws[0].write_bytes(b"RAW")
    cache = tmp_path / "cache"

    result = extract_previews(
        raws, cache, runner=_fake_runner("-JpgFromRaw")
    )
    assert raws[0] in result
    assert result[raws[0]].name == "A.NEF.jpg"


def test_extract_previews_missing_exiftool_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("darktriage.core.previews.shutil.which", lambda name: None)
    raws = [tmp_path / "A.NEF"]
    raws[0].write_bytes(b"RAW")
    assert extract_previews(raws, tmp_path / "cache") == {}


def test_extract_previews_empty_input(tmp_path):
    assert extract_previews([], tmp_path / "cache") == {}

