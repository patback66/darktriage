"""Headless batch export engine for DarkTriage.

Wraps ``darktable-cli`` and prepares a deterministic command plan before
executing it.  The renderer streams the source RAW/DNG images through the
selected darktable style and writes final JPEG/TIFF outputs into the
requested output directory.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportResult:
    """Outcome of a headless export run."""

    success: bool
    frames: int
    command: List[str]
    stderr: str = ""


def _iter_image_files(directory: Path) -> List[Path]:
    """Return the image files (RAW/JPEG) inside ``directory``, sorted."""
    extensions = {
        ".jpg", ".jpeg", ".png", ".tif", ".tiff",
        ".raf", ".arw", ".cr2", ".cr3", ".nrw", ".nef",
        ".rw2", ".dng", ".orf", ".rwl", ".raw",
    }
    return sorted(
        p for p in Path(directory).iterdir()
        if p.is_file() and p.suffix.lower() in extensions
    )


def build_command(
    raw_dir: str,
    style: str,
    output: str,
    *,
    dry_run: bool = False,
    darktable_cli: str = "darktable-cli",
) -> List[str]:
    """Build the darktable-cli command for a batch export.

    The CLI argument surface varies between darktable versions, so this
    helper keeps the invocation deterministic and easily overridden for
    testing.  ``--dry-run`` only affects logging, not the command shape.
    """
    files = [str(p) for p in _iter_image_files(Path(raw_dir))]
    command = [darktable_cli, "-o", str(output), "--style", style] + files
    return command


def run_export(
    raw_dir: str,
    style: str,
    output: str,
    *,
    dry_run: bool = False,
    darktable_cli: str = "darktable-cli",
) -> ExportResult:
    """Run (or plan) a headless batch export."""
    files = _iter_image_files(Path(raw_dir))
    frames = len(files)
    command = build_command(
        raw_dir, style, output, dry_run=dry_run, darktable_cli=darktable_cli
    )
    log.info("darktable-cli: %s", " ".join(command))

    if dry_run:
        return ExportResult(
            success=True, frames=frames, command=command, stderr=""
        )

    proc = subprocess.run(command, capture_output=True, text=True)
    log.info("darktable-cli exited with %s", proc.returncode)
    if proc.stderr.strip():
        log.debug(proc.stderr)
    return ExportResult(
        success=proc.returncode == 0,
        frames=frames,
        command=command,
        stderr=proc.stderr,
    )
