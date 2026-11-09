"""Command-line interface for DarkTriage.

Commands:

* ``darktriage analyze <dir>`` - headless scoring + XMP sidecar generation
* ``darktriage tui <dir>``     - interactive Textual dashboard
* ``darktriage export <dir>``  - headless darktable-cli batch renderer
"""

from __future__ import annotations

import argparse
import logging
from typing import List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from darktriage.core.exif_reader import (
    DEFAULT_BURST_SECONDS,
    Assessment,
    detect_bursts,
    discover_pairs,
    select_burst_winners,
)
from darktriage.core.metrics import DEFAULT_PROXY_MAX_DIM, score_image
from darktriage.core.xmp_writer import (
    ExifToolNotFoundError,
    XmpWriter,
    derive_rating_and_label,
)
from darktriage.export.darktable_cli import batch_export

__all__ = ["build_parser", "main"]

console = Console()

logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darktriage",
        description="Headless RAW culling & batch post-processing engine for Darktable.",
    )
    parser.add_argument("--debug", action="store_true", help="enable debug logging")

    sub = parser.add_subparsers(dest="command", required=True)

    analyze = sub.add_parser(
        "analyze", help="run headless scoring and generate XMP sidecars"
    )
    analyze.add_argument("dir", help="directory containing .NEF/.JPG pairs")
    analyze.add_argument(
        "--burst-threshold",
        type=float,
        default=DEFAULT_BURST_SECONDS,
        help="burst grouping window in seconds (default: 2.0)",
    )
    analyze.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_PROXY_MAX_DIM,
        help="proxy long-edge size in pixels (default: 1920)",
    )
    analyze.add_argument(
        "--dry-run", action="store_true", help="score but do not write XMP sidecars"
    )
    analyze.set_defaults(func=cmd_analyze)

    tui = sub.add_parser("tui", help="launch the interactive Textual dashboard")
    tui.add_argument("dir", help="directory containing .NEF/.JPG pairs")
    tui.add_argument(
        "--burst-threshold",
        type=float,
        default=DEFAULT_BURST_SECONDS,
        help="burst grouping window in seconds (default: 2.0)",
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
    export.add_argument("--style", default=None, help="darktable style to apply")
    export.add_argument(
        "--min-rating", type=int, default=4, help="minimum rating to export (default: 4)"
    )
    export.add_argument("--output", default=None, help="output directory for rendered JPEGs")
    export.add_argument(
        "--dry-run", action="store_true", help="plan exports without invoking darktable-cli"
    )
    export.set_defaults(func=cmd_export)

    return parser


def _analyze_directory(
    directory: str,
    *,
    threshold_seconds: float = DEFAULT_BURST_SECONDS,
    max_dim: int = DEFAULT_PROXY_MAX_DIM,
) -> List[Assessment]:
    """Score every .NEF/.JPG pair in ``directory`` and return assessments."""
    pairs = discover_pairs(directory)
    groups = detect_bursts(pairs, threshold_seconds=threshold_seconds)
    select_burst_winners(groups)

    burst_index_of = {}
    for group in groups:
        for pair in group.images:
            burst_index_of[id(pair)] = group.index
    winner_ids = {id(group.winner) for group in groups if group.winner}

    assessments: List[Assessment] = []
    for pair in pairs:
        assessment = Assessment(
            raw_path=pair.raw_path,
            jpg_path=pair.jpg_path,
            capture_time=pair.capture_time,
            burst_index=burst_index_of.get(id(pair)),
            is_burst_winner=id(pair) in winner_ids,
        )

        if pair.jpg_path is None:
            assessment.error = "no matching JPEG preview"
            assessments.append(assessment)
            continue

        try:
            score = score_image(pair.jpg_path, max_dim=max_dim)
        except Exception as exc:  # noqa: BLE001 - corrupt preview -> error entry
            assessment.error = f"scoring failed: {exc}"
            assessments.append(assessment)
            continue

        decision = derive_rating_and_label(
            sharpness=score.sharpness,
            composition_score=score.composition_score,
            clipped_pct=score.clipped_highlight_pct,
            is_burst_winner=assessment.is_burst_winner,
        )
        assessment.sharpness = score.sharpness
        assessment.composition_score = score.composition_score
        assessment.mean_luminance = score.mean_luminance
        assessment.clipped_highlight_pct = score.clipped_highlight_pct
        assessment.rating = decision.rating
        assessment.label = decision.label
        assessments.append(assessment)

    return assessments


def _render_assessment_table(assessments: List[Assessment], title: str) -> None:
    table = Table(title=title)
    table.add_column("file", style="cyan")
    table.add_column("rating")
    table.add_column("label")
    table.add_column("sharpness")
    table.add_column("clipped")
    table.add_column("burst")

    for a in assessments:
        burst = "-"
        if a.burst_index is not None:
            burst = f"{a.burst_index}{'*' if a.is_burst_winner else ''}"

        if a.rating is None:
            table.add_row(a.raw_path.name, "-", "-", "-", "-", burst)
        else:
            table.add_row(
                a.raw_path.name,
                str(a.rating),
                a.label or "",
                f"{a.sharpness:.1f}",
                f"{a.clipped_highlight_pct:.1f}%",
                burst,
            )
    console.print(table)


def cmd_analyze(args) -> int:
    assessments = _analyze_directory(
        args.dir,
        threshold_seconds=args.burst_threshold,
        max_dim=args.max_dim,
    )
    _render_assessment_table(assessments, f"DarkTriage analysis: {args.dir}")

    if args.dry_run:
        console.print("[dim]--dry-run given: no XMP sidecars written[/dim]")
        return 0

    try:
        writer = XmpWriter(verbose=args.debug)
    except ExifToolNotFoundError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return 1

    written = 0
    skipped = 0
    for assessment in assessments:
        if assessment.rating is None:
            skipped += 1
            continue
        result = writer.write(
            assessment.raw_path, assessment.rating, assessment.label
        )
        if result.success:
            written += 1
        else:
            skipped += 1

    summary = f"[green]Wrote XMP sidecars for {written} file(s)[/green]"
    if skipped:
        summary += f", skipped {skipped}"
    console.print(summary)
    return 0


def cmd_tui(args) -> int:
    assessments = _analyze_directory(
        args.dir, threshold_seconds=args.burst_threshold, max_dim=args.max_dim
    )
    try:
        from darktriage.tui.app import DarkTriageApp

        DarkTriageApp(assessments=assessments, directory=args.dir).run()
    except ImportError as exc:
        console.print(f"[red]Textual is required for the TUI: {exc}[/red]")
        return 1
    return 0


def cmd_export(args) -> int:
    result = batch_export(
        args.dir,
        min_rating=args.min_rating,
        style_name=args.style,
        output_dir=args.output,
        dry_run=args.dry_run,
    )

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
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())