"""Extract small embedded JPEG previews from RAW files via ExifTool.

Decoding a full 24 MB camera Fine JPEG is entropy-bound (the whole scan must
be decoded regardless of output size).  Most NEF/CR2/ARW files embed a much
smaller JPEG preview (~1600-2000 px, 1-3 MB) that is plenty for triage
scoring.  Extracting those previews once, in batched ExifTool passes, cuts
both disk I/O and decoder work by roughly 5-10x.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Callable, Dict, List, Sequence

logger = logging.getLogger(__name__)

# Embedded-preview tags to try, most useful first.  PreviewImage is the
# standard camera-rendered preview; JpgFromRaw is the full developed frame;
# ThumbnailImage is the tiny fallback.
DEFAULT_PREVIEW_TAGS: Sequence[str] = (
    "PreviewImage",
    "JpgFromRaw",
    "ThumbnailImage",
)

# Raws per ExifTool invocation (keeps argv safely below OS limits).
_PREVIEW_CHUNK = 128


class PreviewExtractionError(RuntimeError):
    """Raised when previews cannot be extracted (e.g. exiftool missing)."""


def _valid_jpeg(path: Path) -> bool:
    """True if ``path`` is a non-trivial file starting with the JPEG SOI magic."""
    try:
        if not path.is_file() or path.stat().st_size < 128:
            return False
        with open(path, "rb") as handle:
            return handle.read(3) == b"\xff\xd8\xff"
    except OSError:
        return False


def _chunks(items: List[Path], size: int) -> List[List[Path]]:
    return [items[i : i + size] for i in range(0, len(items), size)]


def extract_previews(
    raws: Sequence[Path],
    cache_dir: Path,
    *,
    exiftool: str = "exiftool",
    tags: Sequence[str] = DEFAULT_PREVIEW_TAGS,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    chunk_size: int = _PREVIEW_CHUNK,
    verbose: bool = False,
) -> Dict[Path, Path]:
    """Extract one embedded JPEG preview per RAW into ``cache_dir``.

    Runs one batched ``exiftool -b -<tag> -w! <cache>/%f.%e.jpg`` pass per tag,
    retrying only the RAWs that are still missing a valid preview.  Returns a
    mapping ``raw_path -> preview_jpeg_path`` for successful extractions.

    ``runner`` is injectable for testing.  If ExifTool is unavailable an empty
    mapping is returned so callers fall back to the full JPEG.
    """
    if not raws:
        return {}
    if (
        exiftool == "exiftool"
        and runner is subprocess.run
        and shutil.which("exiftool") is None
    ):
        logger.warning("exiftool not found; cannot extract embedded previews")
        return {}

    cache_dir.mkdir(parents=True, exist_ok=True)
    pattern = str(cache_dir / "%f.%e.jpg")
    extracted: Dict[Path, Path] = {}
    pending: List[Path] = list(dict.fromkeys(Path(r) for r in raws))

    for tag in tags:
        if not pending:
            break
        for chunk in _chunks(pending, max(1, chunk_size)):
            command = [exiftool, "-q", "-b", f"-{tag}", "-w!", pattern]
            command += [str(raw) for raw in chunk]
            if verbose:
                logger.info("exiftool preview <- %s (%d files)", tag, len(chunk))
            try:
                runner(command, capture_output=True, text=True)
            except Exception as exc:  # noqa: BLE001
                logger.warning("preview extraction failed for tag %s: %s", tag, exc)

        still_missing: List[Path] = []
        for raw in pending:
            candidate = cache_dir / (raw.name + ".jpg")
            if _valid_jpeg(candidate):
                extracted[raw] = candidate
            else:
                still_missing.append(raw)
        pending = still_missing

    if verbose:
        logger.info("extracted %d/%d embedded previews", len(extracted), len(raws))
    return extracted
