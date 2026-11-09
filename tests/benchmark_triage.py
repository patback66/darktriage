#!/usr/bin/env python3
"""Standalone throughput / memory benchmark for the DarkTriage scoring stage.

The script synthesises a batch of camera-sized JPEG previews (Nikon D800E
frame: 7360x4912) plus matching empty ``.NEF`` files, embeds
``DateTimeOriginal`` EXIF so burst detection works, and then measures:

1. decode-only throughput of ``prepare_proxy()`` against a reference
   "full PIL decode + LANCZOS resize" path implemented inline here,
2. full ``score_image()`` throughput (including blur early-exit stats),
3. end-to-end ``analyze_directory()`` throughput through the parallel engine,
4. peak memory (``resource.getrusage`` max RSS + a ``tracemalloc`` probe).

Run directly::

    python3 tests/benchmark_triage.py
    python3 tests/benchmark_triage.py --images 8 --workers 2 --max-dim 1920

Benchmark a real camera folder (synthetic generation is skipped)::

    python3 tests/benchmark_triage.py --dir "/Volumes/Koken Pass/Photos/Hokkaido 2026"
    python3 tests/benchmark_triage.py --dir "/Volumes/Koken Pass/Photos/Hokkaido 2026" --images 50 --workers 4

Without ``--dir`` everything is synthetic and self-contained; no fixture files
are required.
"""

from __future__ import annotations

import argparse
import gc
import os
import resource
import shutil
import sys
import tempfile
import time
import tracemalloc
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from darktriage.config import load_config
from darktriage.core.engine import AnalysisOptions, analyze_directory
from darktriage.core.exif_reader import discover_pairs
from darktriage.core.metrics import (
    DEFAULT_PROXY_MAX_DIM,
    prepare_proxy,
    score_image,
    score_images,
)

# --------------------------------------------------------------------------- #
# Synthetic frame parameters
# --------------------------------------------------------------------------- #

# Nikon D800E sensor: 36.3 MP, 7360 x 4912 px.
D800E_WIDTH = 7360
D800E_HEIGHT = 4912
ASPECT = D800E_WIDTH / D800E_HEIGHT

# Every third frame is generated out-of-focus so the score_image() early-exit
# path gets exercised.
BLUR_EVERY = 3

# Burst shape: groups of BURST_LEN frames 0.4 s apart, groups 10 s apart.
BURST_LEN = 3
BURST_GAP_SEC = 10.0
BURST_STEP_SEC = 0.4

JPEG_QUALITY = 90
EXIF_TAG_DATETIME_ORIGINAL = 0x9003
EXIF_TAG_SUBSEC_ORIGINAL = 0x9291
EXIF_TAG_ISO = 0x8827

ISO_LOW = 800
ISO_HIGH = 3200

# JPEG preview extensions recognised in real-folder mode.
JPG_SUFFIXES = {".jpg", ".jpeg"}

Row = Tuple[str, str]


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class Timing:
    """Wall-clock result of timing ``count`` identical operations."""

    label: str
    count: int
    seconds: float

    @property
    def per_second(self) -> float:
        return self.count / self.seconds if self.seconds > 0 else 0.0

    @property
    def ms_per_item(self) -> float:
        return (self.seconds / self.count) * 1000.0 if self.count else 0.0


def measure(label: str, count: int, func: Callable[[], None]) -> Timing:
    """Run ``func`` once and return a :class:`Timing` for ``count`` items."""
    gc.collect()
    start = time.perf_counter()
    func()
    return Timing(label=label, count=count, seconds=time.perf_counter() - start)


def peak_rss_mb() -> float:
    """Peak RSS of this process in MB (``ru_maxrss`` is KB on Linux)."""
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    divisor = 1024.0 * 1024.0 if sys.platform == "darwin" else 1024.0
    return usage / divisor


def default_workers() -> int:
    """Conservative worker default: never more than 4, never more than CPUs."""
    return max(1, min(4, os.cpu_count() or 1))


def frame_size(long_edge: int) -> Tuple[int, int]:
    """Return a 3:2 ``(width, height)`` frame size for the requested long edge."""
    width = max(320, int(long_edge))
    height = max(213, int(round(width / ASPECT)))
    return width, height


# --------------------------------------------------------------------------- #
# Synthetic image generation
# --------------------------------------------------------------------------- #


def make_sharp_frame(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """High-detail grayscale frame: coarse structure + full-res sensor noise.

    The base texture is built at 1/8 scale and upscaled (cheap, memory-safe);
    the full-resolution noise layer keeps the high-frequency energy a real
    in-focus frame has, so Tenengrad stays well above the blur threshold.
    """
    coarse = rng.integers(
        0, 256, (max(2, height // 8), max(2, width // 8)), dtype=np.uint8
    )
    structure = cv2.resize(coarse, (width, height), interpolation=cv2.INTER_LINEAR)
    detail = rng.integers(0, 256, (height, width), dtype=np.uint8)
    frame = cv2.addWeighted(structure, 0.6, detail, 0.4, 0.0)
    del coarse, structure, detail
    return frame


def make_blur_frame(width: int, height: int, rng: np.random.Generator) -> np.ndarray:
    """Low-frequency grayscale frame: mimics an out-of-focus capture."""
    coarse = rng.integers(0, 256, (6, 8), dtype=np.uint8)
    return cv2.resize(coarse, (width, height), interpolation=cv2.INTER_CUBIC)


def capture_time_for(index: int) -> datetime:
    """Deterministic capture time that forms bursts of ``BURST_LEN`` frames."""
    base = datetime(2024, 6, 1, 12, 0, 0)
    offset = (index // BURST_LEN) * BURST_GAP_SEC + (index % BURST_LEN) * BURST_STEP_SEC
    whole = int(offset)
    micro = int(round((offset - whole) * 1_000_000))
    return base + timedelta(seconds=whole, microseconds=micro)


def build_exif(index: int) -> "Image.Exif":
    """EXIF block with DateTimeOriginal/SubSecTimeOriginal and an ISO rating."""
    stamp = capture_time_for(index)
    exif = Image.Exif()
    exif[EXIF_TAG_DATETIME_ORIGINAL] = stamp.strftime("%Y:%m:%d %H:%M:%S")
    exif[EXIF_TAG_SUBSEC_ORIGINAL] = f"{stamp.microsecond:06d}"
    exif[EXIF_TAG_ISO] = ISO_HIGH if index % 4 == 3 else ISO_LOW
    return exif


def write_frames(
    directory: Path,
    count: int,
    size: Tuple[int, int],
    seed: int,
    progress: bool = True,
) -> Tuple[List[Path], float]:
    """Write ``count`` synthetic JPEG/NEF pairs; return (jpg paths, seconds)."""
    width, height = size
    rng = np.random.default_rng(seed)
    jpg_paths: List[Path] = []
    start = time.perf_counter()
    for index in range(count):
        blurry = index % BLUR_EVERY == BLUR_EVERY - 1
        gray = (
            make_blur_frame(width, height, rng)
            if blurry
            else make_sharp_frame(width, height, rng)
        )
        rgb = cv2.cvtColor(gray, cv2.COLOR_GRAY2RGB)
        del gray
        jpg_path = directory / f"DSC_{index:04d}.JPG"
        Image.fromarray(rgb).save(
            jpg_path, format="JPEG", quality=JPEG_QUALITY, exif=build_exif(index)
        )
        del rgb
        # Empty RAW placeholder so discover_pairs() can match the pair.
        (directory / f"DSC_{index:04d}.NEF").write_bytes(b"")
        jpg_paths.append(jpg_path)
        if progress:
            print(".", end="", flush=True)
    if progress:
        print()
    return jpg_paths, time.perf_counter() - start


# --------------------------------------------------------------------------- #
# Reference implementation (the baseline the fast path is compared against)
# --------------------------------------------------------------------------- #


def reference_decode_proxy(jpg_path: Path, max_dim: int) -> np.ndarray:
    """Baseline: full PIL decode to RGB, LANCZOS thumbnail, then grayscale."""
    with Image.open(jpg_path) as image:
        rgb = image.convert("RGB")
        rgb.load()
        rgb.thumbnail((max_dim, max_dim), Image.LANCZOS)
        return np.asarray(rgb.convert("L"), dtype=np.uint8)


# --------------------------------------------------------------------------- #
# Benchmark phases
# --------------------------------------------------------------------------- #


def bench_decode(
    jpg_paths: Sequence[Path], max_dim: int
) -> Tuple[Timing, Tuple[int, int]]:
    """Time ``prepare_proxy()`` over every JPEG (with one warm-up call)."""
    shape = prepare_proxy(jpg_paths[0], max_dim=max_dim).shape

    def run() -> None:
        for path in jpg_paths:
            prepare_proxy(path, max_dim=max_dim)

    return measure("decode-only (prepare_proxy)", len(jpg_paths), run), shape


def bench_reference(
    jpg_paths: Sequence[Path], max_dim: int, samples: int
) -> Tuple[Timing, Tuple[int, int]]:
    """Time the inline full-decode baseline over ``samples`` JPEGs."""
    subset = list(jpg_paths[: max(1, min(samples, len(jpg_paths)))])
    shape = reference_decode_proxy(subset[0], max_dim).shape

    def run() -> None:
        for path in subset:
            reference_decode_proxy(path, max_dim)

    return measure("reference PIL decode + resize", len(subset), run), shape


@dataclass
class ScoringStats:
    """Aggregate result of a sequential ``score_image()`` sweep."""

    timing: Timing
    early_exits: int
    mean_sharpness: float
    mean_composition: float


def bench_scoring(jpg_paths: Sequence[Path], max_dim: int) -> ScoringStats:
    """Time ``score_image()`` sequentially; collect early-exit statistics."""
    score_image(jpg_paths[0], max_dim=max_dim)  # warm-up
    early_exits = 0
    sharpness_sum = 0.0
    composition_sum = 0.0

    def run() -> None:
        nonlocal early_exits, sharpness_sum, composition_sum
        for path in jpg_paths:
            score = score_image(path, max_dim=max_dim)
            if score.error:
                raise RuntimeError(f"score_image failed for {path}: {score.error}")
            if score.early_exit:
                early_exits += 1
            sharpness_sum += score.sharpness
            composition_sum += score.composition_score

    timing = measure("full scoring (score_image)", len(jpg_paths), run)
    count = max(1, len(jpg_paths))
    return ScoringStats(
        timing=timing,
        early_exits=early_exits,
        mean_sharpness=sharpness_sum / count,
        mean_composition=composition_sum / count,
    )


@dataclass
class EngineStats:
    """Aggregate result of one parallel ``analyze_directory()`` run."""

    timing: Timing
    pair_count: int
    burst_winners: int
    rated: int


def bench_engine(
    directory: Path,
    max_dim: int,
    workers: int,
    use_embedded_preview: bool = False,
) -> EngineStats:
    """Time a single full ``analyze_directory()`` pass (pairs + bursts + pool)."""
    config = load_config()
    options = AnalysisOptions(
        max_dim=max_dim,
        workers=workers,
        use_embedded_preview=use_embedded_preview,
    )
    pair_count = len(discover_pairs(directory))

    gc.collect()
    start = time.perf_counter()
    assessments = analyze_directory(str(directory), config, options)
    seconds = time.perf_counter() - start

    return EngineStats(
        timing=Timing("parallel analyze_directory", max(1, len(assessments)), seconds),
        pair_count=pair_count,
        burst_winners=sum(1 for a in assessments if a.is_burst_winner),
        rated=sum(1 for a in assessments if a.rating is not None),
    )


def discover_jpegs(directory: Path, limit: Optional[int] = None) -> List[Path]:
    """Return real, decodable JPEG previews in a folder, sorted by name.

    Skips macOS AppleDouble/resource-fork files (``._NAME``), hidden files, and
    any file whose header Pillow cannot identify, so a stray sidecar cannot
    abort the benchmark.  ``limit`` stops scanning once enough valid previews
    have been collected.
    """
    found: List[Path] = []
    for path in sorted(directory.iterdir(), key=lambda p: p.name):
        if limit is not None and len(found) >= limit:
            break
        if not path.is_file():
            continue
        if path.name.startswith("."):
            continue
        if path.suffix.lower() not in JPG_SUFFIXES:
            continue
        try:
            with Image.open(path) as image:
                image.size  # header-only validation; raises if not a real image
        except Exception:  # noqa: BLE001 - skip unreadable/corrupt files
            continue
        found.append(path)
    return found


def bench_parallel_scoring(
    jpg_paths: Sequence[Path], max_dim: int, workers: int
) -> EngineStats:
    """Parallel scoring phase for JPEG-only folders (no RAW pairs present)."""
    if not jpg_paths:
        return EngineStats(Timing("parallel score_images", 0, 0.0), 0, 0, 0)
    items = [(str(path), None, 1) for path in jpg_paths]
    gc.collect()
    start = time.perf_counter()
    scores = score_images(items, max_dim=max_dim, workers=workers)
    seconds = time.perf_counter() - start
    return EngineStats(
        timing=Timing("parallel score_images", len(scores), seconds),
        pair_count=0,
        burst_winners=0,
        rated=sum(1 for score in scores if score.error is None),
    )


def tracemalloc_probe(jpg_path: Path, max_dim: int) -> float:
    """Peak Python heap (MB) for a single decode + score, via ``tracemalloc``."""
    tracemalloc.start()
    try:
        prepare_proxy(jpg_path, max_dim=max_dim)
        score_image(jpg_path, max_dim=max_dim)
        gc.collect()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak / (1024 * 1024)


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #


def banner(text: str, char: str = "=") -> None:
    """Print a section header."""
    print(f"\n{char * 76}\n {text}\n{char * 76}")


def summary_table(rows: Sequence[Row]) -> None:
    """Print a simple two-column aligned summary table."""
    width = max(len(left) for left, _ in rows)
    for left, right in rows:
        print(f"  {left.ljust(width)}  {right}")


def rate_line(timing: Timing) -> str:
    """One-line throughput description for a :class:`Timing`."""
    return (
        f"wall {timing.seconds:7.3f} s | {timing.per_second:7.2f} img/s"
        f" | {timing.ms_per_item:8.1f} ms/img | n={timing.count}"
    )


# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark DarkTriage throughput and memory on a synthetic "
            "D800E-sized batch."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--images", type=int, default=20, help="number of synthetic image pairs"
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers(),
        help="worker processes for analyze_directory()",
    )
    parser.add_argument(
        "--max-dim",
        type=int,
        default=DEFAULT_PROXY_MAX_DIM,
        help="long-edge px used for proxy decode / scoring",
    )
    parser.add_argument(
        "--frame-dim",
        type=int,
        default=D800E_WIDTH,
        help="synthetic frame long edge in px (D800E native is 7360; lower it "
        "to trade fidelity for speed)",
    )
    parser.add_argument(
        "--ref-samples",
        type=int,
        default=8,
        help="images used for the reference full-decode baseline (cost control)",
    )
    parser.add_argument("--seed", type=int, default=1234, help="RNG seed")
    parser.add_argument(
        "--use-embedded-preview",
        action="store_true",
        help="engine phase scores each RAW's embedded JPEG preview instead of "
        "the Fine JPEG (much faster; needs exiftool)",
    )
    parser.add_argument(
        "--keep", action="store_true", help="keep the generated temp directory"
    )
    parser.add_argument(
        "--dir",
        type=Path,
        default=None,
        metavar="PATH",
        help=(
            "benchmark against real camera folders instead of synthetic frames "
            "(e.g. --dir \"/Volumes/Koken Pass/Photos/Hokkaido 2026\"). "
            "--images caps how many JPEG previews are sampled for phases 1-3; "
            "the parallel engine phase processes the whole folder."
        ),
    )
    args = parser.parse_args(argv)
    if args.images < 1:
        parser.error("--images must be >= 1")
    if args.workers < 1:
        parser.error("--workers must be >= 1")
    if args.max_dim < 64:
        parser.error("--max-dim must be >= 64")
    if args.frame_dim < 320:
        parser.error("--frame-dim must be >= 320")
    if args.dir is not None:
        if not args.dir.expanduser().is_dir():
            parser.error(f"--dir is not a directory: {args.dir}")
        args.dir = args.dir.expanduser().resolve()
    args.workers = min(args.workers, max(1, args.images))
    return args


def finish(
    args: argparse.Namespace,
    decode_timing: Timing,
    ref_timing: Timing,
    scoring: ScoringStats,
    engine: EngineStats,
    speedup: float,
    peak_mb: float,
    baseline_rss: float,
    heap_peak: float,
    overall_start: float,
) -> int:
    """Print the final summary table and return the process exit code."""
    early_pct = scoring.early_exits / max(1, scoring.timing.count) * 100.0
    banner("Summary")
    rows: List[Row] = [
        (
            "decode-only (prepare_proxy)",
            f"{decode_timing.per_second:7.2f} img/s"
            f" | {decode_timing.ms_per_item:8.1f} ms/img"
            f" | {decode_timing.seconds:6.2f} s",
        ),
        (
            "reference full PIL decode+resize",
            f"{ref_timing.per_second:7.2f} img/s"
            f" | {ref_timing.ms_per_item:8.1f} ms/img"
            f" | {ref_timing.seconds:6.2f} s (n={ref_timing.count})",
        ),
        (
            "full scoring (score_image)",
            f"{scoring.timing.per_second:7.2f} img/s"
            f" | {scoring.timing.ms_per_item:8.1f} ms/img"
            f" | {scoring.timing.seconds:6.2f} s",
        ),
        (
            f"parallel engine ({args.workers}"
            f" {'worker' if args.workers == 1 else 'workers'})",
            f"{engine.timing.per_second:7.2f} img/s"
            f" | {engine.timing.ms_per_item:8.1f} ms/img"
            f" | {engine.timing.seconds:6.2f} s",
        ),
        ("decode speedup vs reference", f"{speedup:.2f}x"),
        (
            "blur early-exit frames",
            f"{scoring.early_exits}/{scoring.timing.count} ({early_pct:.0f}%)",
        ),
        ("peak RSS (ru_maxrss, parent)", f"{peak_mb:.1f} MB"),
        ("peak RSS above baseline", f"{peak_mb - baseline_rss:+.1f} MB"),
        ("python heap peak (tracemalloc)", f"{heap_peak:.1f} MB (single-image probe)"),
        ("total wall time", f"{time.perf_counter() - overall_start:.1f} s"),
    ]
    summary_table(rows)
    print(
        "\n  note: ru_maxrss covers this process only; pool worker memory is"
        " excluded."
    )
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run every benchmark phase and print the report."""
    args = parse_args(argv)
    overall_start = time.perf_counter()

    work_dir: Optional[Path] = None
    try:
        if args.dir is not None:
            banner("DarkTriage throughput benchmark (real folder)")
            jpg_paths = discover_jpegs(args.dir, limit=args.images)
            if not jpg_paths:
                print(f"  no JPEG previews (.jpg/.jpeg) found in {args.dir}")
                return 2
            with Image.open(jpg_paths[0]) as image:
                real_width, real_height = image.size
            megapixels = real_width * real_height / 1e6
            summary_table(
                [
                    ("Python", sys.version.split()[0]),
                    ("source folder", str(args.dir)),
                    ("sampled JPEGs", f"{len(jpg_paths)} (cap --images {args.images})"),
                    ("frame size", f"{real_width}x{real_height} ({megapixels:.1f} MP)"),
                    ("proxy max_dim", f"{args.max_dim} px long edge"),
                    ("engine workers", str(args.workers)),
                ]
            )
            pairs = discover_pairs(args.dir)
            matched = sum(1 for pair in pairs if pair.jpg_path is not None)
            print(f"\n  discover_pairs: {len(pairs)} RAW, {matched} matched JPEG")
            source_dir = args.dir
            has_raw_pairs = matched > 0
            baseline_rss = peak_rss_mb()
        else:
            work_dir = Path(tempfile.mkdtemp(prefix="darktriage-bench-"))
            width, height = frame_size(args.frame_dim)
            megapixels = width * height / 1e6
            banner("DarkTriage throughput benchmark (synthetic D800E batch)")
            summary_table(
                [
                    ("Python", sys.version.split()[0]),
                    ("frame size", f"{width}x{height} ({megapixels:.1f} MP)"),
                    ("images", str(args.images)),
                    ("proxy max_dim", f"{args.max_dim} px long edge"),
                    ("engine workers", str(args.workers)),
                    ("temp dir", str(work_dir)),
                ]
            )
            baseline_rss = peak_rss_mb()
            print(f"\n[0/4] generating {args.images} JPEG+NEF pairs ", end="")
            jpg_paths, gen_seconds = write_frames(
                work_dir, args.images, (width, height), args.seed
            )
            total_mb = (
                sum(p.stat().st_size for p in work_dir.iterdir() if p.is_file())
                / (1024 * 1024)
            )
            print(
                f"      generated {len(jpg_paths)} pairs, {total_mb:.1f} MB on disk"
                f" | {gen_seconds:.2f} s"
                f" ({gen_seconds / len(jpg_paths) * 1000:.0f} ms/frame)"
            )
            pairs = discover_pairs(work_dir)
            matched = sum(1 for pair in pairs if pair.jpg_path is not None)
            timed = sum(1 for pair in pairs if pair.capture_time is not None)
            print(
                f"      discover_pairs: {len(pairs)} RAW, {matched} matched JPEG,"
                f" {timed} with DateTimeOriginal EXIF"
            )
            source_dir = work_dir
            has_raw_pairs = True

        # -- phase 1: decode-only vs reference -------------------------------
        print("\n[1/4] decode-only throughput: prepare_proxy() vs full PIL decode")
        decode_timing, decode_shape = bench_decode(jpg_paths, args.max_dim)
        print(f"      prepare_proxy  -> {decode_shape} {rate_line(decode_timing)}")
        ref_timing, ref_shape = bench_reference(
            jpg_paths, args.max_dim, args.ref_samples
        )
        print(f"      reference PIL  -> {ref_shape} {rate_line(ref_timing)}")
        speedup = (
            ref_timing.ms_per_item / decode_timing.ms_per_item
            if decode_timing.ms_per_item > 0
            else 0.0
        )
        print(
            f"      speedup: {speedup:.2f}x ({ref_timing.ms_per_item:.1f} ms ->"
            f" {decode_timing.ms_per_item:.1f} ms per image)"
        )

        # -- phase 2: full scoring -------------------------------------------
        print("\n[2/4] full scoring throughput: score_image()")
        scoring = bench_scoring(jpg_paths, args.max_dim)
        print(f"      {rate_line(scoring.timing)}")
        early_pct = scoring.early_exits / max(1, scoring.timing.count) * 100.0
        print(
            f"      blur early-exits: {scoring.early_exits}/{scoring.timing.count}"
            f" ({early_pct:.0f}%) | mean sharpness {scoring.mean_sharpness:,.1f}"
            f" | mean composition {scoring.mean_composition:.2f}"
        )

        # -- phase 3: memory probe -------------------------------------------
        print("\n[3/4] memory")
        heap_peak = tracemalloc_probe(jpg_paths[0], args.max_dim)
        print(f"      python heap peak (single-image probe): {heap_peak:.1f} MB")

        # -- phase 4: parallel engine ----------------------------------------
        workers_label = "worker" if args.workers == 1 else "workers"
        print(
            f"\n[4/4] parallel engine with {args.workers} {workers_label}"
        )
        if has_raw_pairs:
            engine = bench_engine(
                source_dir,
                args.max_dim,
                args.workers,
                use_embedded_preview=args.use_embedded_preview,
            )
            print(f"      {rate_line(engine.timing)}")
            print(
                f"      pairs {engine.pair_count} | burst winners"
                f" {engine.burst_winners} | rated {engine.rated}"
            )
            print(
                "      (whole-folder analyze_directory: pairing + EXIF + pool;"
                " this is the end-to-end number)"
            )
        else:
            engine = bench_parallel_scoring(jpg_paths, args.max_dim, args.workers)
            print(f"      {rate_line(engine.timing)}")
            print("      (JPEG-only folder: no .NEF pairs, using parallel score_images())")

        peak_mb = peak_rss_mb()
        return finish(
            args, decode_timing, ref_timing, scoring, engine, speedup,
            peak_mb, baseline_rss, heap_peak, overall_start,
        )
    finally:
        if work_dir is not None:
            if args.keep:
                print(f"  kept temp dir: {work_dir}")
            else:
                shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    raise SystemExit(main())



