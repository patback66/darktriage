"""ExifTool sidecar writer for DarkTriage XMP metadata.

DarkTriage writes two XMP bits per image:

* ``XmpRating``  - star rating (1-5).
* ``XmpLabel``   - darktable color label (Red=Blur, Yellow=Overexposed,
                   Green=Keeper, Purple=Portfolio).

All writes are performed by invoking the ``exiftool`` binary through
``subprocess``.  The class is fully testable: pass an explicit path to a
fake ``exiftool`` binary (or use ``dry_run=True``) to verify the exact
command lines it constructs.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable, List, Optional

# Darktable label numbers: 1 = Red, 2 = Yellow, 3 = Green, 4 = Purple.
LABEL_BY_NAME = {
    "red": "1",
    "yellow": "2",
    "green": "3",
    "purple": "4",
}
LABEL_BY_VALUE = {v: k for k, v in LABEL_BY_NAME.items()}
LABEL_NAMES = sorted(LABEL_BY_NAME)

RATING_MIN = 1
RATING_MAX = 5

logger = logging.getLogger(__name__)


class ExifToolNotFoundError(RuntimeError):
    """Raised when the ``exiftool`` binary cannot be located on PATH."""


class ExifToolError(RuntimeError):
    """Raised when exiftool exits with a non-zero status."""


@dataclass(frozen=True)
class WriteResult:
    """Result of a single XMP write operation."""

    path: str
    rating: Optional[int]
    label: Optional[str]  # label name (red/yellow/green/purple)
    command: List[str]
    success: bool
    stderr: str = ""


class XmpWriter:
    """Wrapper around the ``exiftool`` command line."""

    def __init__(
        self,
        exiftool: str = "exiftool",
        *,
        dry_run: bool = False,
        verbose: bool = False,
    ):
        self.exiftool = self._resolve_tool(exiftool)
        self.dry_run = dry_run
        self.verbose = verbose

    @staticmethod
    def _resolve_tool(exiftool: str) -> str:
        """Return the resolved executable path for the given tool name."""
        if exiftool == "exiftool":
            resolved = shutil.which("exiftool")
            if resolved is None:
                raise ExifToolNotFoundError(
                    "exiftool was not found on PATH. Install it (e.g. "
                    "``apt install libimage-exiftool-perl`` or "
                    "``brew install exiftool``) or pass the full path."
                )
            return resolved
        return exiftool

    def _run(self, argv: List[str]) -> subprocess.CompletedProcess:
        """Execute exiftool with the supplied arguments."""
        command = [self.exiftool] + argv
        if self.verbose:
            logger.info("exiftool: %s", " ".join(command))
        if self.dry_run:
            return subprocess.CompletedProcess(
                args=command, returncode=0, stdout="", stderr=""
            )
        proc = subprocess.run(
            command, capture_output=True, text=True
        )
        if proc.returncode != 0:
            logger.warning(
                "exiftool failed (%s) for %s: %s",
                proc.returncode,
                " ".join(command),
                proc.stderr.strip(),
            )
        return proc

    def _validate(self, path: str, rating: Optional[int] = None, label: Optional[str] = None) -> None:
        if rating is not None:
            if isinstance(rating, bool) or not isinstance(rating, int):
                raise ValueError(
                    f"Rating must be an integer between {RATING_MIN} and {RATING_MAX}, got {rating!r}"
                )
            if not (RATING_MIN <= rating <= RATING_MAX):
                raise ValueError(
                    f"Rating must be between {RATING_MIN} and {RATING_MAX}, got {rating}"
                )
        if label is not None and label not in LABEL_BY_NAME:
            raise ValueError(
                f"Unknown label {label!r}; expected one of {LABEL_NAMES}"
            )

    def write_rating(
        self, path: str, rating: int
    ) -> WriteResult:
        """Write a star rating to ``path`` via exiftool."""
        path = str(path)
        self._validate(path, rating=rating)
        proc = self._run(["-XmpRating=%d" % rating, path])
        return WriteResult(
            path=path,
            rating=rating,
            label=None,
            command=proc.args,
            success=proc.returncode == 0,
            stderr=proc.stderr,
        )

    def write_label(self, path: str, label: str) -> WriteResult:
        """Write a Darktable color label to ``path`` via exiftool."""
        path = str(path)
        self._validate(path, label=label)
        value = LABEL_BY_NAME[label]
        proc = self._run(["-XmpLabel=%s" % value, path])
        return WriteResult(
            path=path,
            rating=None,
            label=label,
            command=proc.args,
            success=proc.returncode == 0,
            stderr=proc.stderr,
        )

    def write_sidecar(
        self,
        path: str,
        *,
        rating: Optional[int] = None,
        label: Optional[str] = None,
    ) -> WriteResult:
        """Write rating and/or label for a single file.

        Returns a single consolidated :class:`WriteResult` containing the
        rating, label (if any) and the full exiftool command line.
        """
        path = str(path)
        results = []
        if rating is not None:
            results.append(self.write_rating(path, rating))
        if label is not None:
            results.append(self.write_label(path, label))

        if not results:
            return WriteResult(
                path=path, rating=None, label=None, command=[], success=True
            )

        # Combine the individual results into one WriteResult.
        command: List[str] = []
        stderr = ""
        for result in results:
            command.extend(result.command)
            stderr += result.stderr
        rating = next((r.rating for r in results if r.rating is not None), None)
        label = next((r.label for r in results if r.label is not None), None)
        return WriteResult(
            path=path,
            rating=rating,
            label=label,
            command=command,
            success=all(r.success for r in results),
            stderr=stderr,
        )

    def write_batch(
        self,
        paths: Iterable[str],
        *,
        rating: Optional[int] = None,
        label: Optional[str] = None,
    ) -> List[WriteResult]:
        """Write the same rating/label across many files (batch commit)."""
        results = []
        for path in sorted(paths):
            results.append(self.write_sidecar(path, rating=rating, label=label))
        return results

    @classmethod
    def create_from_env(
        cls, *, dry_run: bool = False, verbose: bool = False
    ) -> "XmpWriter":
        """Build an XmpWriter honoring ``DARKTRIAGE_EXIFTOOL`` if set."""
        exiftool = shutil.which("exiftool") or \
            __import__("os").environ.get("DARKTRIAGE_EXIFTOOL", "exiftool")
        return cls(exiftool, dry_run=dry_run, verbose=verbose)
