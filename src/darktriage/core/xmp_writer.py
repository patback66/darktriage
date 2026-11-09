"""ExifTool-based XMP sidecar writer for DarkTriage.

Writes non-destructive Darktable/Adobe XMP tags (Rating + Label) to
``<raw>.xmp`` sidecars via the ExifTool command line.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

logger = logging.getLogger(__name__)

# Darktable label names written to the XMP-xmp:Label tag.
LABEL_RED = "Red"
LABEL_YELLOW = "Yellow"
LABEL_GREEN = "Green"
LABEL_PURPLE = "Purple"

# Tunable thresholds for deriving a rating decision.  Tenengrad values are
# resolution-normalised by the 1920px proxy produced in metrics.prepare_proxy.
SHARPNESS_HIGH = 250.0
SHARPNESS_LOW = 40.0
COMPOSITION_STRONG = 0.6
CLIP_PCT_THRESHOLD = 5.0


class ExifToolNotFoundError(RuntimeError):
    """Raised when the exiftool binary cannot be located on PATH."""


@dataclass(frozen=True)
class RatingDecision:
    """Star rating and colour label derived from scored metrics."""

    rating: int
    label: Optional[str]  # None means no label


@dataclass(frozen=True)
class WriteResult:
    """Outcome of a single XMP sidecar write."""

    raw_path: Path
    rating: Optional[int]
    label: Optional[str]
    command: List[str]
    success: bool
    stderr: str = ""


def xmp_sidecar_path(raw_path: Union[str, Path]) -> Path:
    """Return the conventional ``<raw>.xmp`` sidecar path for a raw file."""
    return Path(str(raw_path) + ".xmp")


def derive_rating_and_label(
    *,
    sharpness: float,
    composition_score: float,
    clipped_pct: float,
    is_burst_winner: bool = False,
) -> RatingDecision:
    """Map scored metrics onto the DarkTriage rating/label rule set.

    Rules (evaluated in order):

    * burst winner + high sharpness + strong composition -> 5 stars, Purple
    * crisp focus / keeper -> 4 stars, Green
    * soft focus / blur -> 1 star, Red
    * otherwise -> 3 stars, no label
    * highlight clipping > 5% overrides the label to Yellow
    """
    if (
        is_burst_winner
        and sharpness >= SHARPNESS_HIGH
        and composition_score >= COMPOSITION_STRONG
    ):
        rating, label = 5, LABEL_PURPLE
    elif sharpness >= SHARPNESS_HIGH:
        rating, label = 4, LABEL_GREEN
    elif sharpness <= SHARPNESS_LOW:
        rating, label = 1, LABEL_RED
    else:
        rating, label = 3, None

    if clipped_pct > CLIP_PCT_THRESHOLD:
        label = LABEL_YELLOW

    return RatingDecision(rating=rating, label=label)


class XmpWriter:
    """Wrapper around the exiftool command line."""

    def __init__(
        self,
        exiftool: str = "exiftool",
        *,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.exiftool = self._resolve(exiftool)
        self.dry_run = dry_run
        self.verbose = verbose

    @staticmethod
    def _resolve(exiftool: str) -> str:
        if exiftool == "exiftool":
            resolved = shutil.which("exiftool")
            if resolved is None:
                raise ExifToolNotFoundError(
                    "exiftool not found on PATH. Install it (e.g. apt install "
                    "libimage-exiftool-perl) or pass its absolute path."
                )
            return resolved
        return exiftool

    @staticmethod
    def _build_args(rating: int, label: Optional[str]) -> List[str]:
        args = [f"-XMP-xmp:Rating={rating}"]
        # Write an empty label when none is requested to clear any previous
        # label left from an earlier run.
        args.append(f"-XMP-xmp:Label={label if label else ''}")
        args.append("-overwrite_original")
        return args

    def write(
        self,
        raw_path: Union[str, Path],
        rating: int,
        label: Optional[str] = None,
    ) -> WriteResult:
        """Write Rating + Label XMP tags for a single raw path."""
        raw_path = Path(raw_path)
        args = self._build_args(rating, label)
        command = [self.exiftool] + args + [str(raw_path)]

        if self.verbose:
            logger.info("exiftool: %s", " ".join(command))
        if self.dry_run:
            return WriteResult(
                raw_path=raw_path,
                rating=rating,
                label=label,
                command=command,
                success=True,
            )

        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning(
                "exiftool failed (%s) for %s: %s",
                proc.returncode,
                raw_path,
                proc.stderr.strip(),
            )
        return WriteResult(
            raw_path=raw_path,
            rating=rating,
            label=label,
            command=command,
            success=proc.returncode == 0,
            stderr=proc.stderr,
        )

    def write_decision(
        self, raw_path: Union[str, Path], decision: RatingDecision
    ) -> WriteResult:
        """Write a RatingDecision to ``raw_path``."""
        return self.write(raw_path, decision.rating, decision.label)