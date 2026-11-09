"""Bounded-memory parallel batch analysis engine for DarkTriage.

Stage 2 (decode + metric math) runs across all CPU cores via a
``ProcessPoolExecutor``.  Image paths are consumed from a generator and
processed in fixed-size chunks with ``itertools.islice`` so a large directory
never materialises every in-flight future or image array at once.

Optionally, embedded RAW previews (small JPEGs) can be used instead of the
full camera Fine JPEGs to cut decode/IO cost (~5-10x) on large batches.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass
from itertools import islice
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from darktriage.config import Config
from darktriage.core.exif_reader import (
    Assessment,
    ImagePair,
    detect_bursts,
    discover_pairs,
)
from darktriage.core.metrics import Score, load_image_metadata, score_image
from darktriage.core.previews import extract_previews
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
    use_embedded_preview: bool = False
    preview_cache_dir: Optional[str] = None


Payload = Tuple[str, Optional[int], int, AnalysisOptions]


def chunked(iterable, size: int) -> Iterator[list]:
    """Yield lists of at most ``size`` items from an iterable (bounded memory)."""
    try:
        size = int(size)
    except (TypeError, ValueError):
        size = DEFAULT_CHUNK_SIZE
    if size < 1:
        size = DEFAULT_CHUNK_SIZE
    iterator = iter(iterable)
    while True:
        chunk = list(islice(iterator, size))
        if not chunk:
            return
        yield chunk


def _score_payload(payload: Payload) -> Score:
    """Worker entry point: decode + score a single image (JPEG preview)."""
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


def _build_payload(
    pair: ImagePair, options: AnalysisOptions, image_path: Path
) -> Payload:
    """Assemble a worker payload, reusing cached EXIF when available."""
    if pair.metadata_loaded:
        iso, orientation = pair.iso, pair.orientation
    else:
        meta = load_image_metadata(image_path)
        iso, orientation = meta.iso, meta.orientation
    return (str(image_path), iso, orientation, options)


def _image_for(
    pair: ImagePair, image_map: Optional[Dict[Path, Optional[Path]]]
) -> Optional[Path]:
    """Resolve the image to score: explicit map entry, else the pair JPEG."""
    if image_map is not None and pair.raw_path in image_map:
        return image_map[pair.raw_path]
    return pair.jpg_path


def score_pairs(
    pairs: List[ImagePair],
    options: AnalysisOptions,
    on_complete: Optional[Callable[[], None]] = None,
    image_map: Optional[Dict[Path, Optional[Path]]] = None,
) -> Dict[Path, Score]:
    """Score every pair's chosen image, in bounded parallel chunks.

    Returns a mapping of *raw* path -> Score (raw paths are unique even when
    two raws share one JPEG preview).  Per-file failures are captured as an
    error Score; a crashed worker pool degrades to serial scoring for the
    remaining files rather than aborting the whole batch.
    """
    scorable = [p for p in pairs if _image_for(p, image_map) is not None]
    if not scorable:
        return {}

    worker_count = max(1, options.workers or (os.cpu_count() or 1))
    results: Dict[Path, Score] = {}

    def _record(pair: ImagePair, score: Score) -> None:
        results[pair.raw_path] = score
        if on_complete:
            on_complete()

    def _error(pair: ImagePair, exc: Exception) -> None:
        _record(pair, Score(0.0, 0.0, 0.0, 0.0, error=str(exc)))

    def _payload_for(pair: ImagePair) -> Payload:
        return _build_payload(pair, options, _image_for(pair, image_map))

    def _serial(pending: List[ImagePair]) -> None:
        for pair in pending:
            if pair.raw_path in results:
                continue
            try:
                _record(pair, _score_payload(_payload_for(pair)))
            except Exception as exc:  # noqa: BLE001
                _error(pair, exc)

    if worker_count == 1:
        _serial(scorable)
        return results

    try:
        with ProcessPoolExecutor(max_workers=worker_count) as pool:
            for chunk in chunked(scorable, options.chunk_size):
                futures = {}
                for pair in chunk:
                    if pair.raw_path in results:
                        continue
                    try:
                        payload = _payload_for(pair)
                    except Exception as exc:  # noqa: BLE001
                        _error(pair, exc)
                        continue
                    futures[pool.submit(_score_payload, payload)] = pair
                for future in as_completed(futures):
                    pair = futures[future]
                    try:
                        _record(pair, future.result())
                    except Exception as exc:  # noqa: BLE001
                        _error(pair, exc)
    except BrokenProcessPool:
        logger.warning(
            "Scoring worker pool terminated; falling back to serial scoring "
            "for %d remaining file(s)",
            len([p for p in scorable if p.raw_path not in results]),
        )
        _serial(scorable)

    return results


def _build_image_map(
    pairs: List[ImagePair], options: AnalysisOptions
) -> Tuple[Dict[Path, Optional[Path]], Optional[Path]]:
    """Resolve image-per-raw, extracting embedded previews when requested.

    Returns (image_map, temp_cache_dir_to_cleanup).
    """
    if not options.use_embedded_preview:
        return {p.raw_path: p.jpg_path for p in pairs}, None

    raws = [p.raw_path for p in pairs]
    cache_dir = (
        Path(options.preview_cache_dir)
        if options.preview_cache_dir
        else Path(tempfile.mkdtemp(prefix="darktriage-previews-"))
    )
    cleanup = None if options.preview_cache_dir else cache_dir

    previews = extract_previews(
        raws,
        cache_dir,
        workers=max(1, options.workers or 1),
        reuse_existing=True,
        verbose=False,
    )
    logger.info("embedded previews: %d/%d available", len(previews), len(raws))
    image_map: Dict[Path, Optional[Path]] = {}
    for pair in pairs:
        image_map[pair.raw_path] = previews.get(pair.raw_path) or pair.jpg_path
    return image_map, cleanup


def analyze_directory(
    directory: str,
    config: Config,
    options: AnalysisOptions,
    *,
    threshold_seconds: Optional[float] = None,
    on_complete: Optional[Callable[[], None]] = None,
    pairs: Optional[List[ImagePair]] = None,
    image_map: Optional[Dict[Path, Optional[Path]]] = None,
) -> List[Assessment]:
    """Full Stage 2 pipeline: pair, burst-detect, score, rate."""
    if pairs is None:
        pairs = discover_pairs(directory)
    gap = (
        threshold_seconds
        if threshold_seconds is not None
        else config.thresholds.burst_interval_sec
    )
    groups = detect_bursts(pairs, threshold_seconds=gap)

    cleanup_dir: Optional[Path] = None
    try:
        if image_map is None:
            image_map, cleanup_dir = _build_image_map(pairs, options)
        score_by_path = score_pairs(
            pairs, options, on_complete=on_complete, image_map=image_map
        )
    finally:
        if cleanup_dir is not None:
            shutil.rmtree(cleanup_dir, ignore_errors=True)

    zero = Score(0.0, 0.0, 0.0, 0.0)

    def sharpness_of(pair: ImagePair) -> float:
        return score_by_path.get(pair.raw_path, zero).sharpness

    # Winners are only meaningful for real bursts (>= 2 frames).  A lone frame
    # is not a "burst winner" (otherwise every singleton gets the +1 star).
    winner_raw: Set[Path] = set()
    for group in groups:
        if len(group.images) < 2:
            continue
        best = max(group.images, key=sharpness_of, default=None)
        if best is not None:
            winner_raw.add(best.raw_path)

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
            is_burst_winner=pair.raw_path in winner_raw,
        )
        score = score_by_path.get(pair.raw_path)
        if score is None:
            assessment.error = "no decodable preview"
            assessments.append(assessment)
            continue
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
