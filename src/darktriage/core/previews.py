"""Extract small embedded JPEG previews from RAW files via ExifTool.

Decoding a full 24 MB camera Fine JPEG is entropy-bound (the whole scan must
be decoded regardless of output size).  Most NEF/CR2/ARW files embed a much
smaller JPEG preview (~1600-2000 px, 1-3 MB) that is plenty for triage
scoring.  Extracting those previews once, in batched (optionally parallel)
ExifTool passes, cuts both disk I/O and decoder work by roughly 5-10x.

Because extraction reads every RAW, it is itself I/O heavy on large folders;
results are cached under a directory and reused (``reuse_existing``) so repeat
runs skip extraction entirely.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Dict, List, Sequence

from PIL import Image

logger = logging.getLogger(__name__)

# Embedded-preview tags to try, most useful first.  PreviewImage is the
# standard camera-rendered preview; JpgFromRaw is the full developed frame;
# ThumbnailImage is the tiny fallback (usually rejected by the size guard).
DEFAULT_PREVIEW_TAGS: Sequence[str] = (
    "PreviewImage",
    "JpgFromRaw",
    "ThumbnailImage",
)

# Reject previews smaller than this long edge (e.g. 160px thumbnails) so we
# never score a postage-stamp image and call it sharp/blurred.
DEFAULT_MIN_PREVIEW_EDGE = 512

# Raws per ExifTool invocation (keeps argv safely below OS limits).
_PREVIEW_CHUNK = 128


class PreviewExtractionError(RuntimeError):
    """Raised when previews cannot be extracted (e.g. exiftool missing)."""


def _valid_jpeg(path: Path, min_long_edge: int = 0) -> bool:
    """True if ``path`` is a JPEG at least ``min_long_edge`` on its long side."""
    try:
        if not path.is_file() or path.stat().st_size < 128:
            return False
        with open(path, "rb") as handle:
            if handle.read(3) != b"\xff\xd8\xff":
                return False
        if min_long_edge > 0:
            with Image.open(path) as image:
                if max(image.size) < min_long_edge:
                    return False
        return True
    except OSError:
        return False
    except Exception:  # noqa: BLE001 - corrupt preview header
        return False


def _chunks(items: List[Path], size: int) -> List[List[Path]]:
    if size < 1:
        size = _PREVIEW_CHUNK
    return [items[i : i + size] for i in range(0, len(items), size)]


def _run_tag_chunk(
    exiftool: str,
    tag: str,
    pattern: str,
    chunk: List[Path],
    runner: Callable[..., subprocess.CompletedProcess],
) -> None:
    command = [exiftool, "-q", "-b", f"-{tag}", "-w!", pattern]
    command += [str(raw) for raw in chunk]
    try:
        runner(command, capture_output=True, text=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("preview extraction failed for tag %s: %s", tag, exc)


def extract_previews(
    raws: Sequence[Path],
    cache_dir: Path,
    *,
    exiftool: str = "exiftool",
    tags: Sequence[str] = DEFAULT_PREVIEW_TAGS,
    runner: Callable[..., subprocess.CompletedProcess] = subprocess.run,
    chunk_size: int = _PREVIEW_CHUNK,
    workers: int = 1,
    min_long_edge: int = DEFAULT_MIN_PREVIEW_EDGE,
    reuse_existing: bool = True,
    verbose: bool = False,
) -> Dict[Path, Path]:
    """Extract one embedded JPEG preview per RAW into ``cache_dir``.

    * ``reuse_existing`` skips RAWs that already have a valid cached preview
      (enable a persistent cache dir to make repeat runs free).
    * ``workers`` runs ExifTool chunks concurrently (I/O-bound, so threads are
      ideal) to shorten cold extraction on large folders.
    * ``min_long_edge`` rejects tiny thumbnails so we never score a 160px JPEG.

    Returns ``raw_path -> preview_jpeg_path`` for successful extractions.
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
    pending: List[Path] = []

    for raw in dict.fromkeys(Path(r) for r in raws):
        candidate = cache_dir / (raw.name + ".jpg")
        if reuse_existing and _valid_jpeg(candidate, min_long_edge):
            extracted[raw] = candidate
        else:
            pending.append(raw)

    concurrent = max(1, int(workers))
    for tag in tags:
        if not pending:
            break
        if verbose:
            logger.info(
                "exiftool preview <- %s (%d files, %d workers)",
                tag, len(pending), concurrent,
            )
        chunks = _chunks(pending, chunk_size)
        if concurrent > 1:
            with ThreadPoolExecutor(max_workers=concurrent) as pool:
                list(
                    pool.map(
                        lambda c: _run_tag_chunk(exiftool, tag, pattern, c, runner),
                        chunks,
                    )
                )
        else:
            for chunk in chunks:
                _run_tag_chunk(exiftool, tag, pattern, chunk, runner)

        still_missing: List[Path] = []
        for raw in pending:
            candidate = cache_dir / (raw.name + ".jpg")
            if _valid_jpeg(candidate, min_long_edge):
                extracted[raw] = candidate
            else:
                still_missing.append(raw)
        pending = still_missing

    if verbose:
        logger.info("extracted %d/%d embedded previews", len(extracted), len(raws))
    return extracted
