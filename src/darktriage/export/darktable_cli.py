"""Headless darktable-cli batch export for DarkTriage.

Reads each .NEF file's .xmp rating and renders qualifying shots to JPEG in
an isolated output directory (never overwriting source RAW/JPEG files).
Supports chained styles and custom resolution.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

from darktriage.core.xmp_writer import read_sidecar_rating, xmp_sidecar_path

logger = logging.getLogger(__name__)

# Alias for callers who prefer the export-module name.
read_xmp_rating = read_sidecar_rating


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


def build_command(
    job: ExportJob,
    styles: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    darktable_cli: str = "darktable-cli",
) -> List[str]:
    """Build a darktable-cli invocation: input [xmp] output + options."""
    command = [darktable_cli, str(job.nef)]
    if job.xmp.exists():
        command.append(str(job.xmp))
    command.append(str(job.output))
    for style in styles or ():
        command += ["--style", style]
    if width is not None:
        command += ["--width", str(width)]
    if height is not None:
        command += ["--height", str(height)]
    return command


def batch_export(
    target_dir,
    *,
    min_rating: int = 4,
    styles: Optional[Sequence[str]] = None,
    width: Optional[int] = None,
    height: Optional[int] = None,
    output_dir: Optional[str] = None,
    dry_run: bool = False,
    darktable_cli: str = "darktable-cli",
) -> ExportResult:
    """Export .NEF files rated >= min_rating to an isolated output directory.

    Outputs land in ``output_dir`` (default ``<target_dir>/exported``), so the
    source RAW/JPEG files are never overwritten.  Multiple styles are applied
    in the order given; width/height set explicit output resolution.
    """
    directory = Path(target_dir)
    if not directory.is_dir():
        raise NotADirectoryError(f"Not a directory: {directory}")

    output = Path(output_dir) if output_dir else directory / "exported"
    if not dry_run:
        output.mkdir(parents=True, exist_ok=True)

    result = ExportResult(dry_run=dry_run)
    nef_files = sorted(directory.glob("*.NEF")) + sorted(directory.glob("*.nef"))

    for nef in nef_files:
        xmp = xmp_sidecar_path(nef)
        rating = read_sidecar_rating(xmp)

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

        command = build_command(job, styles, width, height, darktable_cli)
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