"""Tests for DarkTriage EXIF parsing and burst detection."""

from __future__ import annotations

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest
from PIL import Image

from darktriage.core.exif_reader import (
    ExifParsingError,
    ExifRecord,
    group_burst_sequences,
    read_records,
)

NOW = datetime(2024, 1, 1, 12, 0, 0)


def _make_image(tmp_path: Path, *, exif: dict | None = None) -> Path:
    image = Image.new("RGB", (64, 64), "deepskyblue")
    path = tmp_path / "sample.jpg"
    pillow_exif = Image.Exif()
    if exif:
        pillow_exif.update(exif)
    image.save(path, exif=pillow_exif)
    return path


def test_read_records_extracts_capture_time(tmp_path):
    path = _make_image(tmp_path, exif={0x9003: "2024:01:01 12:00:00"})
    records = read_records([path])
    assert len(records) == 1
    assert records[0].capture_time == NOW


def test_read_records_handles_missing_exif(tmp_path):
    path = _make_image(tmp_path)
    records = read_records([path])
    assert records[0].capture_time is None
    assert records[0].camera_model is None


def test_record_to_dict_roundtrip(tmp_path):
    path = _make_image(tmp_path)
    record = read_records([path])[0]
    data = record.to_dict()
    assert data["path"] == str(path)
    assert data["capture_time"] is None


def test_group_burst_sequences_groups_close_shots(tmp_path):
    records = [
        ExifRecord(path=Path(f"f{i}.jpg"), capture_time=NOW + timedelta(seconds=i * 0.2))
        for i in range(5)
    ]
    clusters = group_burst_sequences(records, max_gap=timedelta(seconds=1.0))
    assert len(clusters) == 1
    assert len(clusters[0]) == 5


def test_group_burst_sequences_splits_on_larger_gap(tmp_path):
    offset = timedelta(seconds=2.0)
    records = [
        ExifRecord(path=Path("a.jpg"), capture_time=NOW),
        ExifRecord(path=Path("b.jpg"), capture_time=NOW + offset),
        ExifRecord(path=Path("c.jpg"), capture_time=NOW + offset + timedelta(seconds=0.5)),
    ]
    clusters = group_burst_sequences(records, max_gap=timedelta(seconds=1.0))
    assert len(clusters) == 2
    assert {r.path.name for r in clusters[0]} == {"a.jpg"}
    assert {r.path.name for r in clusters[1]} == {"b.jpg", "c.jpg"}


def test_read_records_logs_unreadable_file(tmp_path, caplog):
    bad = tmp_path / "nope.jpg"
    bad.write_bytes(b"this is not an image")
    with caplog.at_level("WARNING"):
        records = read_records([bad])
    assert any("Could not read EXIF" in r.message for r in caplog.records)
    assert len(records) == 1
    assert records[0].camera_model == "unreadable"
