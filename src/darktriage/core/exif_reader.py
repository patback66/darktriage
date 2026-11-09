"""EXIF parsing, image-pair discovery, and burst detection for DarkTriage."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

from PIL import Image

from darktriage.core.metrics import compute_tenengrad, prepare_proxy

logger = logging.getLogger(__name__)

# RAW and JPEG extensions recognised when pairing .NEF files with previews.
RAW_EXTENSIONS = {".nef", ".arw", ".cr2", ".cr3", ".raf", ".rw2", ".dng", ".orf"}
JPG_EXTENSIONS = {".jpg", ".jpeg"}

# EXIF tag identifiers (TIFF/0th IFD + EXIF sub-IFD).
_TAG_DATETIME_ORIGINAL = 0x9003
_TAG_SUBSEC_ORIGINAL = 0x9291
_TAG_OFFSET_TIME_ORIGINAL = 0x9011
_TAG_DATETIME = 0x0132

# Default burst-detection window in seconds.
DEFAULT_BURST_SECONDS = 2.0


class ExifParsingError(RuntimeError):
    """Raised when an image carries no usable capture timestamp."""


@dataclass
class ImagePair:
    """A raw file plus its matched camera JPEG preview."""

    raw_path: Path
    jpg_path: Optional[Path]
    capture_time: Optional[datetime] = None
    sharpness: Optional[float] = None


@dataclass
class BurstGroup:
    """A cluster of consecutive shots taken within the burst threshold."""

    index: int
    images: List[ImagePair] = field(default_factory=list)
    winner: Optional[ImagePair] = None


def _decode_datetime(value) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="replace")
    text = str(value).strip()
    if not text:
        return None
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


def _subsec_microseconds(value) -> int:
    if value is None:
        return 0
    if isinstance(value, bytes):
        value = value.decode("utf-8", errors="ignore")
    digits = "".join(ch for ch in str(value).strip() if ch.isdigit())
    if not digits:
        return 0
    # Normalise to exactly six digits (microseconds): pad short strings,
    # truncate over-long strings.
    digits = digits[:6].ljust(6, "0")
    return int(digits)


def decode_capture_time(value, subsec=None) -> Optional[datetime]:
    """Combine DateTimeOriginal with SubSecTimeOriginal into one datetime."""
    base = _decode_datetime(value)
    if base is None:
        return None
    return base.replace(microsecond=_subsec_microseconds(subsec))


def read_capture_time(jpg_path: Path) -> Optional[datetime]:
    """Read DateTimeOriginal + sub-second from a JPEG preview.

    Returns None for a missing/corrupt EXIF header (the caller decides how
    to handle the pair) instead of raising.
    """
    try:
        with Image.open(jpg_path) as image:
            exif = image.getexif()
    except Exception as exc:  # noqa: BLE001 - corrupt header -> None
        logger.warning("Cannot read EXIF from %s: %s", jpg_path, exc)
        return None

    dt = exif.get(_TAG_DATETIME_ORIGINAL) or exif.get(_TAG_DATETIME)
    subsec = exif.get(_TAG_SUBSEC_ORIGINAL)
    return decode_capture_time(dt, subsec)


def _is_macos_metadata(path: Path) -> bool:
    """True for hidden/AppleDouble files (``._NAME``, ``.DS_Store``, ...).

    These resource-fork sidecars appear on macOS/exFAT/network volumes and are
    not real images, but they often share the ``.JPG``/``.NEF`` extension.
    """
    name = path.name
    return name.startswith(".") or name.startswith("._")


def discover_pairs(target_dir: Path) -> List[ImagePair]:
    """Match .NEF/.RAW files with same-stem JPEG previews in a directory."""
    directory = Path(target_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    jpg_by_stem: Dict[str, Path] = {}
    for path in directory.iterdir():
        if (
            path.is_file()
            and path.suffix.lower() in JPG_EXTENSIONS
            and not _is_macos_metadata(path)
        ):
            jpg_by_stem.setdefault(path.stem, path)

    raw_paths = sorted(
        (
            p
            for p in directory.iterdir()
            if p.is_file()
            and p.suffix.lower() in RAW_EXTENSIONS
            and not _is_macos_metadata(p)
        ),
        key=lambda p: p.name,
    )

    pairs: List[ImagePair] = []
    for raw in raw_paths:
        jpg = jpg_by_stem.get(raw.stem)
        if jpg is None:
            logger.warning(
                "No matching JPEG preview for %s (skipped for scoring)", raw
            )
        pairs.append(
            ImagePair(
                raw_path=raw,
                jpg_path=jpg,
                capture_time=read_capture_time(jpg) if jpg else None,
            )
        )
    return pairs


def detect_bursts(
    pairs: List[ImagePair],
    *,
    threshold_seconds: float = DEFAULT_BURST_SECONDS,
) -> List[BurstGroup]:
    """Group capture-timed pairs into continuous-drive bursts.

    A pair with a missing/None capture time is excluded from grouping.
    Consecutive timed shots whose timestamps differ by more than
    ``threshold_seconds`` split into separate bursts.
    """
    timed = [p for p in pairs if p.capture_time is not None]
    ordered = sorted(timed, key=lambda p: p.capture_time)
    gap = timedelta(seconds=threshold_seconds)

    groups: List[List[ImagePair]] = []
    current: List[ImagePair] = []
    for pair in ordered:
        if current and (pair.capture_time - current[-1].capture_time) > gap:
            groups.append(current)
            current = []
        current.append(pair)
    if current:
        groups.append(current)

    return [BurstGroup(index=i, images=list(g)) for i, g in enumerate(groups)]


def _sharpness_of(pair: ImagePair) -> float:
    if pair.sharpness is not None:
        return pair.sharpness
    if pair.jpg_path is None:
        return 0.0
    try:
        return compute_tenengrad(prepare_proxy(pair.jpg_path))
    except Exception as exc:  # noqa: BLE001
        logger.warning(
            "Sharpness computation failed for %s: %s", pair.jpg_path, exc
        )
        return 0.0


def select_burst_winners(groups: List[BurstGroup]) -> None:
    """Flag the highest-Tenengrad frame in each burst as its winner."""
    for group in groups:
        best: Optional[ImagePair] = None
        best_score = -1.0
        for pair in group.images:
            score = _sharpness_of(pair)
            pair.sharpness = score
            if score > best_score:
                best_score = score
                best = pair
        group.winner = best
@dataclass
class Assessment:
    """Complete per-image analysis: pairing, burst, metrics and decision."""

    raw_path: Path
    jpg_path: Optional[Path]
    capture_time: Optional[datetime] = None
    burst_index: Optional[int] = None
    is_burst_winner: bool = False
    sharpness: Optional[float] = None
    composition_score: Optional[float] = None
    mean_luminance: Optional[float] = None
    clipped_highlight_pct: Optional[float] = None
    ev_offset: Optional[float] = None
    rating: Optional[int] = None
    label: Optional[str] = None
    error: Optional[str] = None