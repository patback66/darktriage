"""Bounded-memory parallel batch analysis engine for DarkTriage.

Stage 2 (JPEG decode + metric math) runs across all CPU cores via a
``ProcessPoolExecutor``.  Image paths are consumed from a generator and
processed in fixed-size chunks with ``itertools.islice`` so a large directory
never materialises every in-flight future or image array at once.
"""

from __future__ import annotations

import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Tuple

from darktriage.config import Config
from darktriage.core.exif_reader import (
    Assessment,
    ImagePair,
    detect_bursts,
    discover_pairs,
)
from darktriage.core.metrics import Score, load_image_metadata, score_image
from darktriage.core.xmp_writer import derive_rating_and_label

logger = logging.getLogger(__name__)

DEFAULT_CHUNK_SIZE = 256
DEFAULT_MAX_DIM = 1920


@dataclass(frozen=True)
class AnalysisOptions:
    """Tunables for a batch analysis run (picklable for process workers)."""

    max_dim: int = DEFAULT_MAX_DIM
    center_fraction: float = 0.6
    blur_threshold: float = 150.0
    iso_threshold: int = 1600
    workers: Optional[int] = None
    chunk_size: int = DEFAULT_CHUNK_SIZE


Payload = Tuple[str, Optional[int], int, AnalysisOptions]


def chunked(iterable, size: int) -> Iterator[list]:
    """Yield lists of at most ``size`` items from an iterable (bounded memory)."""
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _score_payload(payload: Payload) -> Score:
    """Worker entry point: decode + score a single JPEG preview."""
    path, iso, orientation, options = payload
    return score_image(
        path,
        max_dim=options.max_dim,
        iso=iso,
        orientation=orientation,
        center_fraction=options.center_fraction,
        iso_threshold=options.iso_threshold,
        blur_threshold=options.blur_threshold,
    )


def _build_payload(pair: ImagePair, options: AnalysisOptions) -> Payload:
    meta = load_image_metadata(pair.jpg_path)
    return (str(pair.jpg_path), meta.iso, meta.orientation, options)


def score_pairs(
    pairs: List[ImagePair],
    options: AnalysisOptions,
    on_complete: Optional[Callable[[], None]] = None,
) -> Dict[Path, Score]:
    """Score every pair's JPEG preview, in bounded parallel chunks.

    Returns a mapping of jpg path -> Score.  Failures are captured as an
    error Score rather than aborting the batch.
    """
    scorable = [p for p in pairs if p.jpg_path is not None]
    if not scorable:
        return {}

    worker_count = max(1, options.workers or (os.cpu_count() or 1))
    results: Dict[Path, Score] = {}

    def _record(pair: ImagePair, score: Score) -> None:
        results[pair.jpg_path] = score
        if on_complete:
            on_complete()

    if worker_count == 1:
        for chunk in chunked(scorable, options.chunk_size):
            for pair in chunk:
                try:
                    _record(pair, _score_payload(_build_payload(pair, options)))
                except Exception as exc:  # noqa: BLE001
                    _record(pair, Score(0.0, 0.0, 0.0, 0.0, error=str(exc)))
        return results

    with ProcessPoolExecutor(max_workers=worker_count) as pool:
        for chunk in chunked(scorable, options.chunk_size):
            futures = {}
            for pair in chunk:
                try:
                    payload = _build_payload(pair, options)
                except Exception as exc:  # noqa: BLE001
                    _record(pair, Score(0.0, 0.0, 0.0, 0.0, error=str(exc)))
                    continue
                futures[pool.submit(_score_payload, payload)] = pair
            for future in as_completed(futures):
                pair = futures[future]
                try:
                    _record(pair, future.result())
                except Exception as exc:  # noqa: BLE001
                    _record(pair, Score(0.0, 0.0, 0.0, 0.0, error=str(exc)))
    return results


def analyze_directory(
    directory: str,
    config: Config,
    options: AnalysisOptions,
    *,
    threshold_seconds: Optional[float] = None,
    on_complete: Optional[Callable[[], None]] = None,
) -> List[Assessment]:
    """Full Stage 2 pipeline: pair, burst-detect, parallel-score, rate."""
    pairs = discover_pairs(directory)
    gap = (
        threshold_seconds
        if threshold_seconds is not None
        else config.thresholds.burst_interval_sec
    )
    groups = detect_bursts(pairs, threshold_seconds=gap)
    score_by_path = score_pairs(pairs, options, on_complete=on_complete)

    zero = Score(0.0, 0.0, 0.0, 0.0)

    def sharpness_of(pair: ImagePair) -> float:
        if pair.jpg_path is None:
            return 0.0
        return score_by_path.get(pair.jpg_path, zero).sharpness

    winner_by_path: Dict[Path, bool] = {}
    for group in groups:
        best = max(group.images, key=sharpness_of, default=None)
        if best is not None and best.jpg_path is not None:
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

        score = score_by_path.get(pair.jpg_path, zero)
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
