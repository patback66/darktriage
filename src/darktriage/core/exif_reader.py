"""EXIF parsing and burst sequence detection for DarkTriage.

Reads camera metadata (capture time, lens, ISO, aperture, ...) from Pillow
image objects and groups RAW/JPEG files into burst sequences based on their
original capture timestamps.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

from PIL import Image
from PIL.ExifTags import IFD, TAGS

logger = logging.getLogger(__name__)

# TIFF/EXIF tag identifiers we care about.
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_DATETIME_DIGITIZED = 0x9004
_TAG_OFFSET_TIME_ORIGINAL = 0x9011
_TAG_CAMERA_MODEL = 0x0110
_TAG_ISOSPEED = 0x8827
_TAG_APERTURE = 0x829D
_TAG_EXPOSURE_TIME = 0x829A
_TAG_FOCAL_LENGTH = 0x920A
_TAG_SEQUENCE_NUMBER = 0x0132
_TAG_USER_COMMENT = 0x9286

# Shots whose capture times differ by less than this are treated as one burst.
# Raw clocks can stray by a few microseconds; the value is deliberately
# tolerant and is tunable through the CLI/TUI.
_DEFAULT_BURST_GAP = timedelta(seconds=1.0)


class ExifParsingError(RuntimeError):
    """Raised when an image cannot be decoded or carries no readable EXIF data."""


@dataclass(frozen=True)
class ExifRecord:
    """Metadata record for one file (used for scoring and burst grouping)."""

    path: Path
    capture_time: Optional[datetime] = None
    camera_model: Optional[str] = None
    iso: Optional[int] = None
    aperture: Optional[float] = None
    exposure_time: Optional[float] = None
    focal_length: Optional[float] = None
    sequence_number: Optional[int] = None
    user_comment: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "capture_time": self.capture_time.isoformat()
            if self.capture_time
            else None,
            "camera_model": self.camera_model,
            "iso": self.iso,
            "aperture": self.aperture,
            "exposure_time": self.exposure_time,
            "focal_length": self.focal_length,
            "sequence_number": self.sequence_number,
            "user_comment": self.user_comment,
        }


def _decode_datetime(value) -> Optional[datetime]:
    """Decode an EXIF date/time value to a naive UTC datetime."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    try:
        return datetime.fromisoformat(text)
    except ValueError:
        pass
    for fmt in ("%Y:%m:%d %H:%M:%S", "%Y:%m:%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _read_ifd(image: Image.Image, ifd_tag: int) -> dict:
    """Recursively decode an EXIF IFD into a nested {tag: value} dict."""
    try:
        exif = image.getexif()
        ifd = exif.get_ifd(ifd_tag)
    except Exception:
        return {}
    result: dict = {}
    for tag_id, value in ifd.items():
        name = TAGS.get(tag_id, hex(tag_id))
        if isinstance(value, dict):
            result[name] = _read_ifd(image, tag_id)
        else:
            result[name] = value
    return result


def parse_exif(image: Image.Image, *, read_interop: bool = True) -> dict:
    """Return a flattened EXIF metadata dictionary for a ``PIL.Image``.

    The 0th IFD is always decoded (DateTimeOriginal, model, ...). When
    ``read_interop`` is true, the Interoperability IFD is also decoded in
    case the RAW data stores its capture time there.
    """
    metadata: dict = {}
    try:
        exif = image.getexif()
    except Exception:
        exif = None

    if exif is not None:
        tag_map = (
            (_TAG_DATETIME_ORIGINAL, "datetime_original"),
            (_TAG_DATETIME_DIGITIZED, "datetime_digital"),
            (_TAG_OFFSET_TIME_ORIGINAL, "offset_time_original"),
            (_TAG_CAMERA_MODEL, "camera_model"),
            (_TAG_ISOSPEED, "iso"),
            (_TAG_APERTURE, "aperture"),
            (_TAG_EXPOSURE_TIME, "exposure_time"),
            (_TAG_FOCAL_LENGTH, "focal_length"),
            (_TAG_SEQUENCE_NUMBER, "sequence_number"),
            (_TAG_USER_COMMENT, "user_comment"),
        )
        for tag_id, name in tag_map:
            try:
                value = exif.get(tag_id)
            except Exception:
                value = None
            if value is not None:
                if name == "camera_model" and isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                metadata[name] = value

        if read_interop:
            try:
                interoperability = exif.get_ifd(IFD.INTEROPERABILITY)
                for tag_id, value in interoperability.items():
                    if tag_id == 0x0002:  # InteroperabilityIndex
                        metadata.setdefault(
                            "interoperability", {}
                        ).update({"interoperability_index": value})
            except Exception:
                pass
    return metadata


def read_records(paths: Iterable[Path]) -> List[ExifRecord]:
    """Read EXIF metadata for every image path, returning one record each."""
    records: List[ExifRecord] = []
    for path in sorted(paths):
        try:
            with Image.open(path) as image:
                image.load()
                metadata = parse_exif(image)
        except Exception as exc:
            logger.warning("Could not read EXIF for %s: %s", path, exc)
            records.append(
                ExifRecord(
                    path=path,
                    camera_model="unreadable",
                    user_comment=f"EXIF read failed: {exc}",
                )
            )
            continue

        record = _record_from_metadata(path, metadata)
        records.append(record)
    return records


def _record_from_metadata(path: Path, metadata: dict) -> ExifRecord:
    capture_time = _decode_datetime(metadata.get("datetime_original"))
    offset = _decode_datetime(metadata.get("offset_time_original"))

    # Offset time is a local offset (e.g. "+02:00"); apply it to the
    # unmodified capture time only when the original value is naive.
    if offset is not None and capture_time is not None and capture_time.tzinfo is None:
        capture_time = capture_time + offset

    return ExifRecord(
        path=path,
        capture_time=capture_time,
        camera_model=metadata.get("camera_model"),
        iso=metadata.get("iso"),
        aperture=metadata.get("aperture"),
        exposure_time=metadata.get("exposure_time"),
        focal_length=metadata.get("focal_length"),
        sequence_number=metadata.get("sequence_number"),
        user_comment=metadata.get("user_comment"),
    )


def _capture_time_key(record: ExifRecord):
    return record.capture_time.timestamp() if record.capture_time else 0.0


def group_burst_sequences(
    records: Sequence[ExifRecord],
    *,
    max_gap: Optional[timedelta] = None,
) -> List[List[ExifRecord]]:
    """Group sorted records into burst sequences by capture-time gap.

    Records are ordered by capture time; any two consecutive records whose
    timestamps differ by ``max_gap`` seconds or less are placed in the same
    burst.  The result is a list of clusters, sorted oldest to newest.
    """
    ordered = sorted(records, key=_capture_time_key)
    gap = max_gap or _DEFAULT_BURST_GAP
    clusters: List[List[ExifRecord]] = []
    current: List[ExifRecord] = []

    for record in ordered:
        if (
            current
            and record.capture_time is not None
            and current[-1].capture_time is not None
        ):
            gap_since_prev = record.capture_time - current[-1].capture_time
            if gap_since_prev > gap:
                clusters.append(current)
                current = []
        current.append(record)

    if current:
        clusters.append(current)

    return clusters
