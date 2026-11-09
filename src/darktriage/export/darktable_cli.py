"""Headless darktable-cli batch export for DarkTriage.

Reads each ``.NEF`` file's ``.xmp`` rating and renders the high-rated shots
to JPEG via ``darktable-cli``.  JPEG quality/compression is driven by the
selected darktable style; the output extension (``.jpg``) selects JPEG export.
"""

from __future__ import annotations

import logging
import re
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from darktriage.core.xmp_writer import xmp_sidecar_path

logger = logging.getLogger(__name__)

_XMP_RATING_XML = re.compile(r"<xmp:Rating>\s*([0-5])\s*</xmp:Rating>")
_XMP_RATING_ATTR = re.compile(r'xmp:Rating=["\']([0-5])["\']')


@dataclass(frozen=True)
class ExportJob:
    """One raw file queued for headless export."""

    nef: Path
    xmp: Path
    output: Path
    rating: int


@dataclass
class ExportResult:
    """Outcome of a batch export run."""

    jobs: List[ExportJob] = field(default_factory=list)
    commands: List[List[str]] = field(default_factory=list)
    skipped: List[Path] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    dry_run: bool = False

    @property
    def success(self) -> bool:
        return not self.errors


def read_xmp_rating(xmp_path: Path) -> Optional[int]:
    """Parse the integer xmp:Rating from a sidecar, if present."""
    try:
        text = Path(xmp_path).read_text(errors="replace")
    except OSError:
        return None
    match = _XMP_RATING_XML.search(text) or _XMP_RATING_ATTR.search(text)
    return int(match.group(1)) if match else None


def build_command(
    job: ExportJob,
    style_name: Optional[str] = None,
    darktable_cli: str = "darktable-cli",
) -> List[str]:
    """Build a darktable-cli invocation for one job (input [xmp] output)."""
    command = [darktable_cli, str(job.nef)]
    if job.xmp.exists():
        command.append(str(job.xmp))
    command.append(str(job.output))
    if style_name:
        command += ["--style", style_name]
    return command


def batch_export(
    target_dir,
    *,
    min_rating: int = 4,
    style_name: Optional[str] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    darktable_cli: str = "darktable-cli",
) -> ExportResult:
    """Export .NEF files rated >= ``min_rating`` using darktable-cli.

    Iterates every .NEF in ``target_dir``, reads the companion .xmp rating,
    and renders qualifying shots (JPEG, 95% quality controlled by the style)
    into ``output_dir`` (defaults to ``<target_dir>/export``).
    """
    directory = Path(target_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    output = Path(output_dir) if output_dir else directory / "export"
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    result = ExportResult(dry_run=dry_run)

    nef_files = sorted(directory.glob("*.NEF")) + sorted(directory.glob("*.nef"))
    for nef in nef_files:
        xmp = xmp_sidecar_path(nef)
        rating = read_xmp_rating(xmp)

        if rating is None or rating < min_rating:
            result.skipped.append(nef)
            continue

        job = ExportJob(
            nef=nef,
            xmp=xmp,
            output=output / (nef.stem + ".jpg"),
            rating=rating,
        )
        result.jobs.append(job)

        command = build_command(job, style_name, darktable_cli)
        result.commands.append(command)
        if dry_run:
            continue

        logger.info("darktable-cli: %s", " ".join(command))
        proc = subprocess.run(command, capture_output=True, text=True)
        if proc.returncode != 0:
            message = (
                f"darktable-cli failed for {nef} ({proc.returncode}): "
                f"{proc.stderr.strip()}"
            )
            logger.warning(message)
            result.errors.append(message)

    return result