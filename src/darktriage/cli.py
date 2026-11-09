"""Command-line interface for DarkTriage.

Subcommands:

* ``darktriage score PATH``  -- score a single image and print metrics.
* ``darktriage triage DIR``  -- score every image in a folder and optionally
  write Darktable XMP sidecars.
* ``darktriage burst DIR``   -- print burst-sequence clusters detected by
  capture-time grouping.
* ``darktriage gui``         -- launch the Textual dashboard.
* ``darktriage export DIR``  -- run darktable-cli headless batch rendering.
"""

from __future__ import annotations

import argparse
import logging
from datetime import timedelta
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

from rich.console import Console
from rich.json import JSON
from rich.table import Table

from darktriage.core.exif_reader import (
    ExifParsingError,
    ExifRecord,
    group_burst_sequences,
    read_records,
)
from darktriage.core.metrics import analyze_image, render_rating
from darktriage.core.xmp_writer import LABEL_NAMES, XmpWriter, WriteResult

__all__ = ["build_parser", "main"]

log = logging.getLogger(__name__)
console = Console()

_IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".tif", ".tiff",
    ".raf", ".arw", ".cr2", ".cr3", ".nrw", ".nef",
    ".rw2", ".dng", ".orf", ".rwl", ".raw",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="darktriage",
        description="Headless RAW culling & batch post-processing engine for Darktable.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="enable debug logging"
    )

    sub = parser.add_subparsers(dest="command", required=True)

    p_score = sub.add_parser("score", help="score a single image")
    p_score.add_argument("path", help="image or RAW file to score")
    p_score.set_defaults(func=cmd_score)

    p_triage = sub.add_parser(
        "triage", help="score every image in a folder and write XMP sidecars"
    )
    p_triage.add_argument("raw_dir", help="folder containing RAW/JPEG files")
    p_triage.add_argument(
        "--rating", "--ratings",
        type=int, dest="rating",
        help="apply a fixed star rating to every sidecar (1-5)",
    )
    p_triage.add_argument(
        "--label",
        choices=LABEL_NAMES,
        help="apply a fixed color label to every sidecar (red/yellow/green/purple)",
    )
    p_triage.add_argument(
        "--burst-gap", type=float, default=1.0,
        help="seconds between shots that still count as one burst (default: 1.0)",
    )
    p_triage.add_argument(
        "--no-write", action="store_true", help="score only, do not write XMP"
    )
    p_triage.set_defaults(func=cmd_triage)

    p_burst = sub.add_parser(
        "burst", help="report burst-sequence clusters in a folder"
    )
    p_burst.add_argument("raw_dir")
    p_burst.add_argument(
        "--burst-gap", type=float, default=1.0,
        help="seconds between shots that still count as one burst (default: 1.0)",
    )
    p_burst.set_defaults(func=cmd_burst)

    p_gui = sub.add_parser("gui", help="launch the Textual dashboard")
    p_gui.set_defaults(func=cmd_gui)

    p_export = sub.add_parser(
        "export", help="headless batch export via darktable-cli"
    )
    p_export.add_argument("raw_dir")
    p_export.add_argument("--style", required=True)
    p_export.add_argument("--output", required=True)
    p_export.add_argument(
        "--dry-run", action="store_true", help="print commands without running"
    )
    p_export.set_defaults(func=cmd_export)

    return parser


def _iter_image_files(directory: Path) -> List[Path]:
    return sorted(
        p for p in Path(directory).iterdir() if p.is_file() and p.suffix.lower() in _IMAGE_EXTENSIONS
    )


def _format_rating(record: ExifRecord, rating: int, label: str) -> str:
    return f"{record.path.name}  [{rating}★  {label}]"


def cmd_score(args) -> int:
    try:
        metrics = analyze_image(args.path)
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]error: could not score {args.path}: {exc}[/red]")
        return 1
    console.print(JSON.from_data(metrics, indent=2))
    return 0


def cmd_triage(args) -> int:
    paths = _iter_image_files(Path(args.raw_dir))
    if not paths:
        console.print(f"[yellow]no images found in {args.raw_dir}[/yellow]")
        return 0

    records = read_records(paths)
    clusters = group_burst_sequences(records, max_gap=timedelta(seconds=args.burst_gap))

    # The sharpest frame in each burst earns the +1 burst bonus.
    sharpest_by_path = {}
    for cluster in clusters:
        best = max(cluster, key=lambda r: r.capture_time or 0)
        sharpest_by_path[best.path] = best

    writer = XmpWriter.create_from_env(verbose=args.debug) if not args.no_write else None
    rows: List[Tuple[ExifRecord, dict, int, str, WriteResult]] = []
    for record in records:
        metrics = analyze_image(record.path)
        in_burst = record.path in sharpest_by_path
        rating = render_rating(
            metrics["sharpness"],
            metrics["exposure"],
            metrics["composition"],
            in_burst=in_burst,
        )
        if writer is not None:
            result = writer.write_sidecar(
                str(record.path), rating=rating.rating, label=rating.label
            )
        else:
            result = WriteResult(
                path=str(record.path),
                rating=rating.rating,
                label=rating.label,
                command=[],
                success=True,
            )
        rows.append((record, metrics, rating.rating, rating.label, result))

    table = Table(title="DarkTriage triage results")
    table.add_column("file", style="cyan")
    table.add_column("rating")
    table.add_column("label")
    table.add_column("sharpness")
    table.add_column("clipped")
    for record, metrics, rating, label, result in rows:
        table.add_row(
            record.path.name,
            str(rating),
            label,
            f"{metrics['sharpness']:.1f}",
            "yes" if metrics["exposure"]["clipped"] else "no",
        )
    console.print(table)

    if args.no_write:
        console.print("[dim]--no-write given: sidecars were NOT written[/dim]")
        return 0

    written = [r for r in rows if r[4].success]
    failed = [r for r in rows if not r[4].success]
    console.print(
        f"[green]wrote[/green] {len(written)}/{len(rows)} sidecar(s)" +
        (f" and [red]failed[/red] {len(failed)}" if failed else "")
    )
    return 0 if not failed else 1


def cmd_burst(args) -> int:
    paths = _iter_image_files(Path(args.raw_dir))
    if not paths:
        console.print(f"[yellow]no images found in {args.raw_dir}[/yellow]")
        return 0
    records = read_records(paths)
    clusters = group_burst_sequences(
        records, max_gap=timedelta(seconds=args.burst_gap)
    )

    if not clusters:
        console.print("[dim]no burst sequences detected[/dim]")
        return 0

    for idx, cluster in enumerate(clusters, start=1):
        files = " + ".join(r.path.name for r in cluster)
        console.print(f"[bold]{idx}.[/bold] {len(cluster)} frame(s) in {files}")
    return 0


def cmd_gui(args) -> int:
    try:
        from darktriage.tui.app import DarkTriageApp

        app = DarkTriageApp()
        app.run()
    except ImportError as exc:
        console.print(f"[red]Textual is required for the TUI: {exc}[/red]")
        return 1
    return 0


def cmd_export(args) -> int:
    try:
        from darktriage.export.darktable_cli import run_export

        dry_run = args.dry_run or getattr(args, "dry_run", False)
        result = run_export(
            args.raw_dir,
            style=args.style,
            output=args.output,
            dry_run=dry_run,
        )
        console.print(
            f"[green]ok[/green] export planned for {result.frames} frame(s)"
        )
        return 0 if result.success else 1
    except Exception as exc:  # noqa: BLE001
        console.print(f"[red]export failed: {exc}[/red]")
        return 1


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
