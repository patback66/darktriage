"""Tests for darktriage.core.exif_reader."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from PIL import Image

from darktriage.core.exif_reader import (
    DEFAULT_BURST_SECONDS,
    ImagePair,
    decode_capture_time,
    detect_bursts,
    discover_pairs,
    read_capture_time,
    select_burst_winners,
)


def _jpeg_with_exif(path: Path, dt: str | None = None, subsec: str | None = None) -> None:
    image = Image.new("RGB", (8, 8), "deepskyblue")
    exif = Image.Exif()
    if dt:
        exif[0x9003] = dt
    if subsec is not None:
        exif[0x9291] = subsec
    image.save(path, exif=exif)


def test_decode_capture_time_combines_sub_second():
    result = decode_capture_time("2024:01:01 12:00:00", subsec="123456")
    assert result == datetime(2024, 1, 1, 12, 0, 0, 123456)


def test_decode_capture_time_none_when_missing():
    assert decode_capture_time(None) is None


def test_decode_capture_time_sub_second_padded():
    result = decode_capture_time("2024:01:01 12:00:00", subsec="12")
    assert result.microsecond == 120000


def test_read_capture_time_from_jpeg(tmp_path):
    path = tmp_path / "sample.JPG"
    _jpeg_with_exif(path, dt="2024:01:01 12:00:00", subsec="123456")
    assert read_capture_time(path) == datetime(2024, 1, 1, 12, 0, 0, 123456)


def test_read_capture_time_returns_none_for_corrupt_header(tmp_path):
    path = tmp_path / "bad.jpg"
    path.write_bytes(b"not a real jpeg")
    assert read_capture_time(path) is None


def test_discover_pairs_matches_nef_with_jpg(tmp_path):
    (tmp_path / "A.NEF").write_bytes(b"raw")
    (tmp_path / "B.NEF").write_bytes(b"raw")
    _jpeg_with_exif(tmp_path / "A.JPG", dt="2024:01:01 12:00:00")  # B has no JPG

    pairs = discover_pairs(tmp_path)
    assert len(pairs) == 2
    by_stem = {p.raw_path.stem: p for p in pairs}
    assert by_stem["A"].jpg_path == tmp_path / "A.JPG"
    assert by_stem["A"].capture_time is not None
    assert by_stem["B"].jpg_path is None
    assert by_stem["B"].capture_time is None


def test_detect_bursts_groups_close_shots():
    base = datetime(2024, 1, 1, 12, 0, 0)
    pairs = [
        ImagePair(
            raw_path=Path(f"{i}.NEF"),
            jpg_path=None,
            capture_time=base + timedelta(seconds=i * 0.4),
        )
        for i in range(4)
    ]
    pairs.append(
        ImagePair(raw_path=Path("far.NEF"), jpg_path=None, capture_time=base + timedelta(seconds=30))
    )

    groups = detect_bursts(pairs, threshold_seconds=2.0)
    assert len(groups) == 2
    assert len(groups[0].images) == 4
    assert len(groups[1].images) == 1


def test_select_burst_winners_picks_highest_sharpness():
    base = datetime(2024, 1, 1, 12, 0, 0)
    a = ImagePair(raw_path=Path("a.NEF"), jpg_path=None, capture_time=base, sharpness=10.0)
    b = ImagePair(raw_path=Path("b.NEF"), jpg_path=None, capture_time=base + timedelta(seconds=0.2), sharpness=50.0)
    c = ImagePair(raw_path=Path("c.NEF"), jpg_path=None, capture_time=base + timedelta(seconds=0.4), sharpness=30.0)

    groups = detect_bursts([a, b, c], threshold_seconds=2.0)
    select_burst_winners(groups)

    assert groups[0].winner is b


def test_default_burst_threshold_is_two_seconds():
    assert DEFAULT_BURST_SECONDS == 2.0