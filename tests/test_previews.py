"""Tests for embedded-preview extraction (ExifTool batching + caching)."""

from __future__ import annotations

import io
import subprocess
from pathlib import Path

import pytest
from PIL import Image

from darktriage.core.previews import (
    DEFAULT_PREVIEW_TAGS,
    _valid_jpeg,
    extract_previews,
)

TAG_FLAGS = {f"-{tag}" for tag in DEFAULT_PREVIEW_TAGS}


def _jpeg_bytes(edge: int) -> bytes:
    buffer = io.BytesIO()
    Image.new("RGB", (edge, edge), (128, 128, 128)).save(buffer, "JPEG")
    return buffer.getvalue()


REAL_JPEG = _jpeg_bytes(600)  # passes the 512px min-edge guard
TINY_JPEG = _jpeg_bytes(100)  # rejected by the min-edge guard


def _fake_runner(payload: bytes, tag_that_writes: str | None = None):
    """Return a runner writing ``payload`` for one tag (or all tags)."""

    def runner(command, **kwargs):
        tag = next(flag for flag in command if flag in TAG_FLAGS)
        pattern = command[command.index("-w!") + 1]
        outdir = Path(pattern).parent
        for arg in command[command.index("-w!") + 2:]:
            raw = Path(arg)
            if tag_that_writes is None or tag == tag_that_writes:
                (outdir / (raw.name + ".jpg")).write_bytes(payload)
        return subprocess.CompletedProcess(command, 0, "", "")

    return runner


def test_valid_jpeg_detects_sof_magic(tmp_path):
    good = tmp_path / "a.jpg"
    good.write_bytes(REAL_JPEG)
    bad = tmp_path / "b.jpg"
    bad.write_bytes(b"not a jpeg at all")
    assert _valid_jpeg(good) is True
    assert _valid_jpeg(bad) is False
    assert _valid_jpeg(tmp_path / "missing.jpg") is False


def test_valid_jpeg_rejects_too_small(tmp_path):
    tiny = tmp_path / "t.jpg"
    tiny.write_bytes(TINY_JPEG)
    assert _valid_jpeg(tiny, min_long_edge=0) is True
    assert _valid_jpeg(tiny, min_long_edge=512) is False


def test_extract_previews_primary_tag(tmp_path):
    raws = [tmp_path / "A.NEF", tmp_path / "B.NEF"]
    for raw in raws:
        raw.write_bytes(b"RAW")
    result = extract_previews(
        raws, tmp_path / "cache", runner=_fake_runner(REAL_JPEG)
    )
    assert set(result) == set(raws)
    assert all(p.exists() for p in result.values())


def test_extract_previews_falls_back_to_next_tag(tmp_path):
    raw = tmp_path / "A.NEF"
    raw.write_bytes(b"RAW")
    result = extract_previews(
        [raw],
        tmp_path / "cache",
        runner=_fake_runner(REAL_JPEG, tag_that_writes="-JpgFromRaw"),
    )
    assert raw in result
    assert result[raw].name == "A.NEF.jpg"


def test_extract_previews_rejects_tiny_previews(tmp_path):
    raw = tmp_path / "A.NEF"
    raw.write_bytes(b"RAW")
    result = extract_previews(
        [raw], tmp_path / "cache", runner=_fake_runner(TINY_JPEG)
    )
    assert result == {}  # 100px thumbnails are rejected as unscoreable


def test_extract_previews_reuses_cache(tmp_path):
    raw = tmp_path / "A.NEF"
    raw.write_bytes(b"RAW")
    cache = tmp_path / "cache"
    first = extract_previews([raw], cache, runner=_fake_runner(REAL_JPEG))
    assert raw in first

    def failing_runner(command, **kwargs):
        raise AssertionError("extraction should be skipped when cache is warm")

    second = extract_previews([raw], cache, runner=failing_runner)
    assert second.get(raw) == first[raw]


def test_extract_previews_parallel_workers(tmp_path):
    raws = [tmp_path / f"{i}.NEF" for i in range(5)]
    for raw in raws:
        raw.write_bytes(b"RAW")
    result = extract_previews(
        raws, tmp_path / "cache", runner=_fake_runner(REAL_JPEG), workers=3
    )
    assert set(result) == set(raws)


def test_extract_previews_missing_exiftool_returns_empty(tmp_path, monkeypatch):
    monkeypatch.setattr("darktriage.core.previews.shutil.which", lambda name: None)
    raw = tmp_path / "A.NEF"
    raw.write_bytes(b"RAW")
    assert extract_previews([raw], tmp_path / "cache") == {}


def test_extract_previews_empty_input(tmp_path):
    assert extract_previews([], tmp_path / "cache") == {}
