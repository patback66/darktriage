"""Command-line interface for DarkTriage.

Commands:

* ``darktriage analyze <dir>`` - headless scoring + non-destructive XMP writes
* ``darktriage tui <dir>``     - interactive Textual dashboard
* ``darktriage export <dir>``  - headless darktable-cli batch renderer
"""

from __future__ import annotations

import argparse
import logging
from typing import Dict, List, Optional, Sequence

from rich.console import Console
from rich.table import Table

from darktriage.config import Config, load_config
from darktriage.core.exif_reader import (
    Assessment,
    detect_bursts,
    discover_pairs,
    select_burst_winners,
)
from darktriage.core.metrics import DEFAULT_PROXY_MAX_DIM, score_images
from darktriage.core.xmp_writer import (
    ExifToolNotFoundError,
    SidecarRequest,
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
) -> List[Assessment]:
    """Score every .NEF/.JPG pair and return per-file assessments.

    ISO and orientation metadata is read per preview and passed through the
    scoring pipeline (noise filtering, auto-rotation, central-region masking)
    so results are camera-configurable rather than hardcoded.
    """
    from darktriage.core.metrics import load_image_metadata, score_image

    pairs = discover_pairs(directory)
    gap = (
        threshold_seconds
        if threshold_seconds is not None
        else config.thresholds.burst_interval_sec
    )
    groups = detect_bursts(pairs, threshold_seconds=gap)

    # Build per-file metadata and parallel-score every preview.
    jpg_pairs = [p for p in pairs if p.jpg_path is not None]
    items = []
    for pair in jpg_pairs:
        meta = load_image_metadata(pair.jpg_path)
        items.append((pair.jpg_path, meta.iso, meta.orientation))
    scores = score_images(
        items,
        max_dim=max_dim,
        center_fraction=config.sensor.region_center_fraction,
        workers=workers,
    )
    score_by_path = {p.jpg_path: s for p, s in zip(jpg_pairs, scores)}

    # The sharpest frame in each burst is the burst winner (+1 star).
    winner_by_path: Dict[Path, bool] = {}
    for group in groups:
        best = max(
            group.images,
            key=lambda r: score_by_path[r.jpg_path].sharpness,
            default=None,
        )
        if best is not None:
            winner_by_path[best.jpg_path] = True

    burst_index_by_path: Dict[Path, int] = {}
    for gi, group in enumerate(groups):
        for pair in group.images:
            burst_index_by_path[pair.raw_path] = gi

    assessments: List[Assessment] = []
    for pair in pairs:
        assessment = Assessment(
            raw_path=pair.raw_path,
            jpg_path=pair.jpg_path,
            capture_time=pair.capture_time,
            burst_index=burst_index_by_path.get(pair.raw_path),
            is_burst_winner=pair.jpg_path in winner_by_path,
        )

        if pair.jpg_path is None:
            assessment.error = "no matching JPEG preview"
            assessments.append(assessment)
            continue

        score = score_by_path[pair.jpg_path]
        if score.error:
            assessment.error = score.error
            assessments.append(assessment)
            continue

        decision = derive_rating_and_label(
            sharpness=score.sharpness,
            composition_score=score.composition_score,
            clipped_fraction=score.clipped_highlight_pct / 100.0,
            is_burst_winner=assessment.is_burst_winner,
            thresholds=config.thresholds,
            tags=config.tags,
        )
        assessment.sharpness = score.sharpness
        assessment.composition_score = score.composition_score
        assessment.mean_luminance = score.mean_luminance
        assessment.clipped_highlight_pct = score.clipped_highlight_pct
        assessment.ev_offset = score.ev_offset
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
    assessments = _analyze_directory(
        args.dir,
        args.config,
        threshold_seconds=args.burst_threshold,
        max_dim=args.max_dim,
        workers=getattr(args, "threads", None),
    )
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

    results = writer.write_batch(requests)
    written = sum(1 for r in results if r.success and not r.skipped)
    preserved = sum(1 for r in results if r.skipped)
    summary = f"[green]Wrote XMP sidecars for {written} file(s)[/green]"
    if preserved:
        summary += f", preserved {preserved} existing sidecar(s)"
    console.print(summary)
    return 0

def cmd_tui(args) -> int:
    assessments = _analyze_directory(
        args.dir,
        args.config,
        threshold_seconds=args.burst_threshold,
        max_dim=args.max_dim,
        workers=getattr(args, "threads", None),
    )
    try:
        from darktriage.tui.app import DarkTriageApp

        DarkTriageApp(assessments=assessments, directory=args.dir, config=args.config).run()
    except ImportError as exc:
        console.print(f"[red]Textual is required for the TUI: {exc}[/red]")
        return 1
    return 0


def cmd_export(args) -> int:
    result = batch_export(
        args.dir,
        min_rating=args.min_rating,
        styles=args.styles,
        width=args.width,
        height=args.height,
        output_dir=args.output,
        dry_run=args.dry_run,
        darktable_cli=args.config.darktable.cli_path,
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
    args.config = load_config(args.config)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())