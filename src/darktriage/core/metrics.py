"""Computer-vision scoring primitives for DarkTriage.

Deterministic, order-independent scoring.  Camera JPEG previews are downscaled
in memory (Lanczos) to a long edge of 1920 px before metrics are computed.
Scoring can be parallelised across a process pool via score_images().
"""

from __future__ import annotations

import logging
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Long-edge size (px) proxies are resized to for fast, memory-bound scoring.
DEFAULT_PROXY_MAX_DIM = 1920

# Luminance (0-255) above which a pixel is considered a clipped highlight.
HIGHLIGHT_LUMINANCE = 250

# EV offset bounds and reference luminance used for exposure compensation.
MIN_EV_OFFSET = -2.0
MAX_EV_OFFSET = 2.0
EV_REFERENCE_LUMINANCE = 128.0


@dataclass(frozen=True)
class ExposureReport:
    mean_luminance: float
    clipped_highlight_pct: float


@dataclass(frozen=True)
class Score:
    sharpness: float
    composition_score: float
    mean_luminance: float
    clipped_highlight_pct: float
    ev_offset: float = 0.0
    error: Optional[str] = None


def prepare_proxy(
    jpg_path: Union[str, Path], max_dim: int = DEFAULT_PROXY_MAX_DIM
) -> np.ndarray:
    """Load a camera JPEG and downscale it in memory to max_dim long edge.

    Returns a single-channel uint8 grayscale matrix.  Resizing uses Lanczos.
    """
    with Image.open(jpg_path) as image:
        image = image.convert("L")
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > max_dim:
            scale = max_dim / float(long_edge)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, Image.LANCZOS)
        return np.asarray(image, dtype=np.uint8)


def compute_tenengrad(gray_img: np.ndarray) -> float:
    """Tenengrad micro-focus measure: mean(Gx^2 + Gy^2) via Sobel operators."""
    if gray_img.size == 0:
        return 0.0
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    energy_xy = grad_x * grad_x + grad_y * grad_y
    return float(energy_xy.mean())


def compute_composition_score(gray_img: np.ndarray) -> float:
    """Rule-of-Thirds mass ratio (off-centre grid energy / centre energy).

    The image is split into a 3x3 cell grid and Sobel gradient energy is
    summed per cell.  The returned ratio is unbounded (>= 0); a value above
    1.2 means the majority of edge energy sits away from the centre.
    """
    if gray_img.size == 0:
        return 0.0

    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    energy = grad_x * grad_x + grad_y * grad_y

    height, width = energy.shape
    cell_energy = np.zeros((3, 3), dtype=np.float64)
    for row in range(3):
        y0, y1 = row * height // 3, (row + 1) * height // 3
        for col in range(3):
            x0, x1 = col * width // 3, (col + 1) * width // 3
            cell_energy[row, col] = energy[y0:y1, x0:x1].sum()

    total = float(cell_energy.sum())
    centre = float(cell_energy[1, 1])
    if total <= 0.0:
        return 0.0
    off_centre = total - centre
    return float(off_centre / (centre + 1e-9))


def check_exposure(gray_img: np.ndarray) -> ExposureReport:
    """Return mean luminance and the percentage of clipped highlights (>250)."""
    if gray_img.size == 0:
        return ExposureReport(mean_luminance=0.0, clipped_highlight_pct=0.0)
    mean_luminance = float(gray_img.mean())
    clipped_pixels = int(np.count_nonzero(gray_img > HIGHLIGHT_LUMINANCE))
    clipped_highlight_pct = float(clipped_pixels) / float(gray_img.size) * 100.0
    return ExposureReport(
        mean_luminance=mean_luminance,
        clipped_highlight_pct=clipped_highlight_pct,
    )

def compute_ev_offset(mean_luminance: float) -> float:
    """Recommended exposure compensation: log2(128 / mean) bounded to [-2, 2].

    Positive values mean the frame is underexposed (increase exposure);
    negative values mean it is overexposed.
    """
    if mean_luminance <= 0.0:
        return MAX_EV_OFFSET
    ev = math.log2(EV_REFERENCE_LUMINANCE / mean_luminance)
    return float(max(MIN_EV_OFFSET, min(MAX_EV_OFFSET, ev)))


def score_image(
    jpg_path: Union[str, Path], max_dim: int = DEFAULT_PROXY_MAX_DIM
) -> Score:
    """One-shot scoring: proxy -> tenengrad -> composition -> exposure -> EV."""
    gray = prepare_proxy(jpg_path, max_dim=max_dim)
    exposure = check_exposure(gray)
    return Score(
        sharpness=compute_tenengrad(gray),
        composition_score=compute_composition_score(gray),
        mean_luminance=exposure.mean_luminance,
        clipped_highlight_pct=exposure.clipped_highlight_pct,
        ev_offset=compute_ev_offset(exposure.mean_luminance),
    )


def _score_jpeg(payload: Tuple[str, int]) -> Tuple[str, Score]:
    path, max_dim = payload
    try:
        return (path, score_image(path, max_dim=max_dim))
    except Exception as exc:  # noqa: BLE001 - corrupt preview -> error entry
        return (
            path,
            Score(
                sharpness=0.0,
                composition_score=0.0,
                mean_luminance=0.0,
                clipped_highlight_pct=0.0,
                ev_offset=0.0,
                error=str(exc),
            ),
        )


def score_images(
    jpg_paths: Iterable[Union[str, Path]],
    max_dim: int = DEFAULT_PROXY_MAX_DIM,
    workers: Optional[int] = None,
) -> List[Score]:
    """Score many JPEG previews in parallel via ProcessPoolExecutor.

    Results preserve input order.  ``workers`` defaults to min(CPU, N);
    a value of 1 (or a single image) runs serially in-process.
    """
    paths = [Path(p) for p in jpg_paths]
    if not paths:
        return []
    worker_count = workers or min((os.cpu_count() or 1), len(paths))
    payloads = [(str(p), max_dim) for p in paths]
    if worker_count <= 1 or len(paths) == 1:
        return [score for _, score in (_score_jpeg(pwd) for pwd in payloads)]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(_score_jpeg, payloads))
    return [score for _, score in results]