"""Non-destructive ExifTool XMP sidecar writer for DarkTriage.

Writes Darktable/Adobe XMP tags (Rating, Label, Description) to ``<raw>.xmp``
sidecars only - raw binaries (.NEF/.JPG) are never passed to ExifTool for
writing.  By default, an existing sidecar that already carries a user rating
or darktable edit history is preserved unless overwrite is explicitly set.
"""

from __future__ import annotations

import html
import logging
import os
import re
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union

from darktriage.config import Tags, Thresholds

logger = logging.getLogger(__name__)

_XMP_RATING_XML_RE = re.compile(r"<xmp:Rating>\s*([0-5])\s*</xmp:Rating>")
_XMP_RATING_ATTR_RE = re.compile(r'xmp:Rating=["\']([0-5])["\']')
_DARKTABLE_HISTORY_RE = re.compile(r"<darktable:history")
_SUBJECT_BAG_RE = re.compile(r"<dc:subject>(.*?)</dc:subject>", re.DOTALL | re.IGNORECASE)
_SUBJECT_LI_RE = re.compile(r"<rdf:li[^>]*>(.*?)</rdf:li>", re.DOTALL | re.IGNORECASE)

# Adobe/darktable colour-label vocabulary (case-sensitive capitalised names).
_XMP_LABEL_NAMES = {
    "red": "Red",
    "yellow": "Yellow",
    "green": "Green",
    "blue": "Blue",
    "purple": "Purple",
}

# Minimal valid XMP pack used to create a brand-new sidecar file.
MINIMAL_XMP = (
    '<?xpacket begin="\uFEFF" id="W5M0MpCehiHzreSzNTczkc9d"?>\n'
    '<x:xmpmeta xmlns:x="adobe:ns:meta/">\n'
    ' <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">\n'
    '  <rdf:Description rdf:about=""\n'
    '    xmlns:dc="http://purl.org/dc/elements/1.1/"\n'
    '    xmlns:xmp="http://ns.adobe.com/xap/1.0/"/>\n'
    ' </rdf:RDF>\n'
    '</x:xmpmeta>\n'
    '<?xpacket end="w"?>\n'
)


class ExifToolNotFoundError(RuntimeError):
    """Raised when the exiftool binary cannot be located on PATH."""


@dataclass(frozen=True)
class RatingDecision:
    rating: int
    label: Optional[str]


@dataclass(frozen=True)
class WriteResult:
    raw_path: Path
    rating: Optional[int]
    label: Optional[str]
    command: List[str]
    success: bool
    stderr: str = ""
    skipped: bool = False


@dataclass(frozen=True)
class SidecarRequest:
    raw_path: Path
    rating: Optional[int]
    label: Optional[str]
    ev_offset: Optional[float] = None
    tags: Tuple[str, ...] = ()
    overwrite: bool = False


def xmp_sidecar_path(raw_path: Union[str, Path]) -> Path:
    """Return the conventional ``<raw>.xmp`` sidecar path for a raw file."""
    return Path(str(raw_path) + ".xmp")


def read_sidecar_rating(xmp_path: Union[str, Path]) -> Optional[int]:
    """Read the existing ``xmp:Rating`` (or None) from a sidecar."""
    try:
        text = Path(xmp_path).read_text(errors="replace")
    except OSError:
        return None
    match = _XMP_RATING_XML_RE.search(text) or _XMP_RATING_ATTR_RE.search(text)
    return int(match.group(1)) if match else None


def read_sidecar_subjects(xmp_path: Union[str, Path]) -> Tuple[str, ...]:
    """Read existing ``dc:subject`` keywords from a sidecar (may be empty)."""
    try:
        text = Path(xmp_path).read_text(errors="replace")
    except OSError:
        return ()
    bag = _SUBJECT_BAG_RE.search(text)
    if not bag:
        return ()
    return tuple(html.unescape(item.strip()) for item in _SUBJECT_LI_RE.findall(bag.group(1)))


def has_darktable_history(xmp_path: Union[str, Path]) -> bool:
    """Return True if the sidecar contains darktable edit history modules."""
    try:
        text = Path(xmp_path).read_text(errors="replace")
    except OSError:
        return False
    return bool(_DARKTABLE_HISTORY_RE.search(text))


def should_write_xmp(xmp_path: Union[str, Path], overwrite: bool = False) -> bool:
    """Return True if it is safe/allowed to write the sidecar.

    Non-destructive default: if a sidecar exists and already contains a user
    rating (>0) or darktable history entries, refuse to write unless
    ``overwrite=True``.
    """
    path = Path(xmp_path)
    if not path.exists():
        return True
    if overwrite:
        return True
    if has_darktable_history(path):
        return False
    rating = read_sidecar_rating(path)
    if rating is not None and rating > 0:
        return False
    return True


def backup_sidecar(xmp_path: Union[str, Path]) -> Optional[Path]:
    """Copy an existing sidecar to ``<file>.xmp.bak`` before modification."""
    path = Path(xmp_path)
    if not path.exists():
        return None
    backup = Path(str(path) + ".bak")
    shutil.copy2(path, backup)
    return backup


def derive_rating_and_label(
    *,
    sharpness: float,
    composition_score: float,
    clipped_fraction: float,
    is_burst_winner: bool,
    thresholds: Thresholds,
    tags: Tags,
) -> RatingDecision:
    """Map scored metrics onto the configured rating/label rule set."""
    if (
        is_burst_winner
        and sharpness >= thresholds.tenengrad_sharp
        and composition_score >= thresholds.composition_min
    ):
        rating, label = 5, tags.portfolio
    elif sharpness >= thresholds.tenengrad_sharp:
        rating, label = 4, tags.keeper
    elif sharpness <= thresholds.tenengrad_blur:
        rating, label = 1, tags.blur
    else:
        rating, label = 3, None

    if clipped_fraction > thresholds.highlight_clip_percent:
        label = tags.exposure_warn

    return RatingDecision(rating=rating, label=label)

class XmpWriter:
    """Wrapper around the exiftool command line (sidecar-only writes)."""

    def __init__(
        self,
        exiftool: str = "exiftool",
        *,
        dry_run: bool = False,
        create_backup: bool = True,
        verbose: bool = False,
        label_case: str = "title",
    ):
        self.exiftool = self._resolve(exiftool)
        self.dry_run = dry_run
        self.create_backup = create_backup
        self.verbose = verbose
        self.label_case = label_case

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
    def _description(ev_offset: Optional[float]) -> Optional[str]:
        if ev_offset is None:
            return None
        return f"darktriage recommended EV offset: {ev_offset:+.2f}"

    def _xmp_label(self, label: Optional[str]) -> str:
        """Render a configured colour label in the chosen XMP case style."""
        if not label:
            return ""
        if self.label_case == "asis":
            return label
        if self.label_case == "lower":
            return label.lower()
        # default "title": Adobe/darktable vocabulary ("Green", "Purple", ...)
        return _XMP_LABEL_NAMES.get(label.lower(), label)

    def _build_args(
        self, rating: int, label: Optional[str], ev_offset: Optional[float]
    ) -> List[str]:
        args = [f"-XMP-xmp:Rating={rating}"]
        # Write an empty label when none is requested to clear any stale label.
        args.append(f"-XMP-xmp:Label={self._xmp_label(label)}")
        description = self._description(ev_offset)
        if description:
            args.append(f"-XMP-dc:Description={description}")
        args.append("-overwrite_original")
        return args

    def _ensure_sidecar(self, xmp_path: Path) -> None:
        if not xmp_path.exists() and not self.dry_run:
            xmp_path.write_text(MINIMAL_XMP, encoding="utf-8")

    @staticmethod
    def _subject_args(tags: Tuple[str, ...], has_existing: bool) -> List[str]:
        """ExifTool args to add ``dc:subject`` keywords without clobbering."""
        args: List[str] = []
        for index, tag in enumerate(tags):
            op = "+=" if has_existing or index else "="
            args.append(f"-XMP-dc:Subject{op}{tag}")
        return args

    def write(
        self,
        raw_path: Union[str, Path],
        rating: int,
        label: Optional[str] = None,
        ev_offset: Optional[float] = None,
        *,
        overwrite: bool = False,
        tags: Tuple[str, ...] = (),
    ) -> WriteResult:
        """Write Rating/Label/Description (and optional subject tags)."""
        raw_path = Path(raw_path)
        xmp_path = xmp_sidecar_path(raw_path)

        if not should_write_xmp(xmp_path, overwrite=overwrite):
            logger.info("Preserving existing sidecar for %s", raw_path)
            return WriteResult(
                raw_path=raw_path,
                rating=rating,
                label=label,
                command=[],
                success=True,
                skipped=True,
            )

        existing_subjects = read_sidecar_subjects(xmp_path)
        new_subjects = tuple(
            tag for tag in dict.fromkeys(tags) if tag and tag not in existing_subjects
        )

        args = self._build_args(rating, label, ev_offset)
        if new_subjects:
            args.extend(self._subject_args(new_subjects, bool(existing_subjects)))
        command = [self.exiftool] + args + [str(xmp_path)]

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

        if self.create_backup and xmp_path.exists():
            backup_sidecar(xmp_path)
        self._ensure_sidecar(xmp_path)

        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            logger.warning(
                "exiftool failed (%s) for %s: %s",
                proc.returncode,
                xmp_path,
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

    def build_argfile_lines(self, requests: Iterable[SidecarRequest]) -> List[str]:
        """Build ExifTool -@ argfile lines for a batch of sidecar writes.

        ``-execute`` separates independent command groups and is emitted only
        *between* groups (a trailing ``-execute`` would create an empty final
        command that ExifTool rejects with "Nothing to do").
        """
        requests = list(requests)
        lines: List[str] = []
        for index, request in enumerate(requests):
            if index > 0:
                lines.append("-execute")
            lines.extend(
                self._build_args(request.rating, request.label, request.ev_offset)
            )
            if request.tags:
                sidecar = xmp_sidecar_path(request.raw_path)
                existing = read_sidecar_subjects(sidecar)
                new_tags = tuple(
                    tag
                    for tag in dict.fromkeys(request.tags)
                    if tag and tag not in existing
                )
                if new_tags:
                    lines.extend(self._subject_args(new_tags, bool(existing)))
            lines.append(str(xmp_sidecar_path(request.raw_path)))
        return lines

    def write_batch(
        self,
        requests: Iterable[SidecarRequest],
        *,
        on_complete: Optional[Callable[[], None]] = None,
    ) -> List[WriteResult]:
        """Write many sidecars in a single ExifTool execution pass.

        A temporary ``exiftool -@ argfile`` is generated (one argument per
        line, ``-execute`` between independent groups) so the whole batch is
        applied by one process without hitting OS argv-length limits.
        """
        requests = list(requests)
        runnable: List[SidecarRequest] = []
        runnable_paths = set()
        for request in requests:
            xmp_path = xmp_sidecar_path(request.raw_path)
            if not should_write_xmp(xmp_path, overwrite=request.overwrite):
                continue
            runnable.append(request)
            runnable_paths.add(xmp_path)

        success, stderr = True, ""
        command: List[str] = []
        if runnable:
            for request in runnable:
                xmp_path = xmp_sidecar_path(request.raw_path)
                if self.create_backup and not self.dry_run and xmp_path.exists():
                    backup_sidecar(xmp_path)
                self._ensure_sidecar(xmp_path)

            lines = self.build_argfile_lines(runnable)
            command = [self.exiftool, "-@", "<argfile>"]
            if self.verbose:
                logger.info("exiftool -@ argfile (%d lines)", len(lines))
            if not self.dry_run:
                fd, argfile = tempfile.mkstemp(suffix=".args", prefix="darktriage-")
                try:
                    with os.fdopen(fd, "w", encoding="utf-8") as handle:
                        handle.write("\n".join(lines) + "\n")
                    command = [self.exiftool, "-@", argfile]
                    proc = subprocess.run(command, capture_output=True, text=True)
                    success, stderr = proc.returncode == 0, proc.stderr
                    if not success:
                        logger.warning(
                            "exiftool batch failed (%s): %s",
                            proc.returncode,
                            stderr.strip(),
                        )
                finally:
                    try:
                        os.unlink(argfile)
                    except OSError:
                        pass
            if on_complete:
                on_complete()

        ordered: List[WriteResult] = []
        for request in requests:
            xmp_path = xmp_sidecar_path(request.raw_path)
            if xmp_path not in runnable_paths:
                ordered.append(
                    WriteResult(
                        raw_path=request.raw_path,
                        rating=request.rating,
                        label=request.label,
                        command=[],
                        success=True,
                        skipped=True,
                    )
                )
            else:
                ordered.append(
                    WriteResult(
                        raw_path=request.raw_path,
                        rating=request.rating,
                        label=request.label,
                        command=command,
                        success=success,
                        stderr=stderr,
                    )
                )
        return ordered
