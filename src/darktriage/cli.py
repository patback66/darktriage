"""Command-line interface for DarkTriage.

Commands:

* ``darktriage analyze <dir>`` - headless scoring + non-destructive XMP writes
* ``darktriage tui <dir>``     - interactive Textual dashboard
* ``darktriage export <dir>``  - headless darktable-cli batch renderer
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Callable, List, Optional, Sequence

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)
from rich.table import Table

from darktriage.config import Config, load_config
from darktriage.core.engine import AnalysisOptions, analyze_directory
from darktriage.core.exif_reader import Assessment, discover_pairs
from darktriage.core.metrics import DEFAULT_PROXY_MAX_DIM
from darktriage.core.xmp_writer import (
    ExifToolNotFoundError,
    SidecarRequest,
    XmpWriter,
)
from darktriage.export.darktable_cli import (
    batch_export,
    read_sidecar_rating,
    xmp_sidecar_path,
)

__all__ = ["build_parser", "main"]

console = Console()
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darktriage",
        description="Headless RAW culling & batch post-processing engine for Darktable.",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")
    parser.add_argument(
        "--config", default=None, help="path to an alternate config.toml"
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=None,
        help="number of worker processes for scoring (default: CPU count)",
    )

    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="run headless scoring and generate XMP sidecars"
    )
    analyze.add_argument("dir", help="directory containing .NEF/.JPG pairs")
    analyze.add_argument(
        "--burst-threshold",
        type=float,
        default=None,
        help="burst grouping window in seconds (default from config)",
    )
    analyze.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_PROXY_MAX_DIM,
        help="proxy long-edge size in pixels (default: 1920)",
    )
    analyze.add_argument(
        "--overwrite",
        action="store_true",
        help="explicitly allow modifying existing .xmp sidecars",
    )
    analyze.add_argument(
        "--dry-run", action="store_true", help="preview ratings without writing files"
    )
    analyze.set_defaults(func=cmd_analyze)

    tui = sub.add_parser("tui", help="launch the interactive Textual dashboard")
    tui.add_argument("dir", help="directory containing .NEF/.JPG pairs")
    tui.add_argument(
        "--burst-threshold", type=float, default=None, help="burst window in seconds"
    )
    tui.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_PROXY_MAX_DIM,
        help="proxy long-edge size in pixels (default: 1920)",
    )
    tui.set_defaults(func=cmd_tui)

    export = sub.add_parser(
        "export", help="run the headless darktable-cli batch renderer"
    )
    export.add_argument("dir", help="directory containing .NEF files and .xmp sidecars")
    export.add_argument(
        "--style",
        action="append",
        default=None,
        dest="styles",
        help="darktable style to apply (repeatable for chained styles)",
    )
    export.add_argument(
        "--min-rating", type=int, default=4, help="minimum rating to export (default: 4)"
    )
    export.add_argument("--width", type=int, default=None, help="output width in pixels")
    export.add_argument("--height", type=int, default=None, help="output height in pixels")
    export.add_argument(
        "--output", default=None, help="output directory (default: <dir>/exported)"
    )
    export.add_argument(
        "--dry-run", action="store_true", help="plan exports without invoking darktable-cli"
    )
    export.set_defaults(func=cmd_export)

    return parser


def _analyze_directory(
    directory: str,
    config: Config,
    *,
    threshold_seconds: Optional[float] = None,
    max_dim: int = DEFAULT_PROXY_MAX_DIM,
    workers: Optional[int] = None,
    on_complete: Optional[Callable[[], None]] = None,
) -> List[Assessment]:
    """Score every .NEF/.JPG pair via the bounded parallel engine."""
    options = AnalysisOptions(
        max_dim=max_dim,
        center_fraction=config.sensor.region_center_fraction,
        blur_threshold=config.thresholds.tenengrad_blur,
        iso_threshold=config.sensor.iso_blur_above,
        workers=workers,
    )
    return analyze_directory(
        directory,
        config,
        options,
        threshold_seconds=threshold_seconds,
        on_complete=on_complete,
    )


def _render_assessment_table(assessments: List[Assessment], title: str) -> None:
    table = Table(title=title)
    table.add_column("file", style="cyan")
    table.add_column("rating")
    table.add_column("label")
    table.add_column("sharpness")
    table.add_column("clipped")
    table.add_column("ev")
    table.add_column("burst")

    for a in assessments:
        burst = "-"
        if a.burst_index is not None:
            burst = f"{a.burst_index}{'*' if a.is_burst_winner else ''}"

        if a.rating is None:
            table.add_row(a.raw_path.name, "-", "-", "-", "-", "-", burst)
        else:
            table.add_row(
                a.raw_path.name,
                str(a.rating),
                a.label or "",
                f"{a.sharpness:.1f}",
                f"{a.clipped_highlight_pct:.1f}%",
                f"{a.ev_offset:+.2f}" if a.ev_offset is not None else "-",
                burst,
            )
    console.print(table)


def cmd_analyze(args) -> int:
    jpg_count = len(
        [p for p in discover_pairs(args.dir) if p.jpg_path is not None]
    )

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        score_task = progress.add_task("Scoring images", total=jpg_count)

        def _advance_scoring() -> None:
            progress.update(score_task, advance=1)

        assessments = _analyze_directory(
            args.dir,
            args.config,
            threshold_seconds=args.burst_threshold,
            max_dim=args.max_dim,
            workers=getattr(args, "threads", None),
            on_complete=_advance_scoring,
        )
        progress.update(score_task, completed=jpg_count)
        progress.stop_task(score_task)
        progress.refresh()

    _render_assessment_table(assessments, f"DarkTriage analysis: {args.dir}")

    if args.dry_run or args.config.safety.dry_run:
        console.print("[dim]dry-run: no XMP sidecars written[/dim]")
        return 0

    requests = [
        SidecarRequest(
            raw_path=a.raw_path,
            rating=a.rating,
            label=a.label,
            ev_offset=a.ev_offset,
            overwrite=args.overwrite,
        )
        for a in assessments
        if a.rating is not None
    ]

    try:
        writer = XmpWriter(
            dry_run=False,
            create_backup=args.config.safety.create_backup,
            verbose=args.debug,
        )
    except ExifToolNotFoundError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return 1

    if not requests:
        console.print("[dim]no sidecars to write[/dim]")
        return 0

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        console=console,
        transient=True,
    ) as progress:
        write_task = progress.add_task("Writing XMP sidecars", total=len(requests))

        def _advance_writing() -> None:
            progress.update(write_task, advance=1)

        results = writer.write_batch(requests, on_complete=_advance_writing)
        progress.update(write_task, completed=len(requests))
        progress.stop_task(write_task)
        progress.refresh()

    written = sum(1 for r in results if r.success and not r.skipped)
    preserved = sum(1 for r in results if r.skipped)
    summary = f"[green]Wrote XMP sidecars for {written} file(s)[/green]"
    if preserved:
        summary += f", preserved {preserved} existing sidecar(s)"
    console.print(summary)
    return 0

def cmd_tui(args) -> int:
    try:
        from darktriage.tui.app import DarkTriageApp

        DarkTriageApp(
            directory=args.dir,
            config=args.config,
            threshold=args.burst_threshold,
            max_dim=args.max_dim,
            workers=getattr(args, "threads", None),
        ).run()
    except ImportError as exc:
        console.print(f"[red]Textual is required for the TUI: {exc}[/red]")
        return 1
    return 0


def cmd_export(args) -> int:
    directory = Path(args.dir)
    nef_files = sorted(directory.glob("*.NEF")) + sorted(directory.glob("*.nef"))

    # Pre-count eligible jobs so the progress bar is determinate.
    eligible = 0
    for nef in nef_files:
        rating = read_sidecar_rating(xmp_sidecar_path(nef))
        if rating is not None and rating >= args.min_rating:
            eligible += 1

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        MofNCompleteColumn(),
        TimeRemainingColumn(),
        console=console,
        transient=True,
    ) as progress:
        export_task = progress.add_task("Exporting frames", total=eligible)

        def _advance_export(job) -> None:  # noqa: ARG001
            progress.update(export_task, advance=1)

        result = batch_export(
            args.dir,
            min_rating=args.min_rating,
            styles=args.styles,
            width=args.width,
            height=args.height,
            output_dir=args.output,
            dry_run=args.dry_run,
            darktable_cli=args.config.darktable.cli_path,
            on_progress=_advance_export,
        )
        progress.update(export_task, completed=eligible)
        progress.stop_task(export_task)
        progress.refresh()

    console.print(
        f"Exported {len(result.jobs)} frame(s); "
        f"skipped {len(result.skipped)} below min-rating; "
        f"{len(result.errors)} error(s)."
    )
    for error in result.errors:
        console.print(f"[red]{error}[/red]")
    return 0 if result.success else 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.WARNING)
    args.config = load_config(args.config)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())