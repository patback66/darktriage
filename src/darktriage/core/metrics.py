"""Computer-vision scoring primitives for DarkTriage.

All scoring is deterministic and order-independent: the same input bytes
produce the same numbers every time.  Camera JPEGs are downscaled in memory
to a long edge of 1920 px (prepare_proxy) before any metric is computed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# Long-edge size (px) proxies are resized to for fast, memory-bound scoring.
DEFAULT_PROXY_MAX_DIM = 1920

# Luminance (0-255) above which a pixel is considered a clipped highlight.
HIGHLIGHT_LUMINANCE = 250


@dataclass(frozen=True)
class ExposureReport:
    """Exposure summary produced by check_exposure()."""

    mean_luminance: float
    clipped_highlight_pct: float


@dataclass(frozen=True)
class Score:
    """Aggregated scoring results for one camera JPEG proxy."""

    sharpness: float
    composition_score: float
    mean_luminance: float
    clipped_highlight_pct: float


def prepare_proxy(
    jpg_path: Union[str, Path], max_dim: int = DEFAULT_PROXY_MAX_DIM
) -> np.ndarray:
    """Load a camera JPEG and downscale it in memory to max_dim long edge.

    Returns a single-channel uint8 grayscale matrix (OpenCV-friendly) in
    which all luminance-based metrics are computed.  Resizing uses the
    Lanczos filter for clean high-frequency retention.
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
    """Return the Tenengrad micro-focus measure of a grayscale image.

    Tenengrad = mean(Gx ** 2 + Gy ** 2) where Gx/Gy are the OpenCV Sobel
    operators.  High values indicate crisp focus/micro contrast; very low
    values indicate motion blur or out-of-focus frames.
    """
    if gray_img.size == 0:
        return 0.0
    grad_x = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=3)
    energy_xy = grad_x * grad_x + grad_y * grad_y
    return float(energy_xy.mean())


def compute_composition_score(gray_img: np.ndarray) -> float:
    """Evaluate the Rule-of-Thirds composition heuristic for a grayscale image.

    The image is split into a 3x3 grid and the Sobel gradient *mass* (energy)
    is summed per cell.  A score in [0.0, 1.0] rewards frames whose edge
    energy sits away from the centre while staying balanced across the four
    corner regions - the Rule-of-Thirds power points.
    """
    if gray_img.size == 0:
        return 0.5

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
    if total <= 0.0:
        return 0.5

    fractions = cell_energy / total
    centre_fraction = float(fractions[1, 1])
    off_centre = 1.0 - centre_fraction

    # The four power points are the grid corners; a well-composed frame
    # spreads its energy across them rather than piling into one corner.
    corners = np.array(
        [
            fractions[0, 0],
            fractions[0, 2],
            fractions[2, 0],
            fractions[2, 2],
        ]
    )
    corner_sum = float(corners.sum())
    if corner_sum <= 0.0:
        corner_balance = 0.0
    else:
        coefficient_of_variation = float(
            corners.std() / max(float(corners.mean()), 1e-9)
        )
        corner_balance = float(
            np.clip(1.0 - coefficient_of_variation, 0.0, 1.0)
        )

    score = float(
        np.clip(off_centre * (0.35 + 0.65 * corner_balance), 0.0, 1.0)
    )
    return score


def check_exposure(gray_img: np.ndarray) -> ExposureReport:
    """Return mean luminance and the percentage of clipped highlights.

    A pixel is clipped when its luminance exceeds HIGHLIGHT_LUMINANCE (250).
    The returned percentage is relative to the whole image.
    """
    if gray_img.size == 0:
        return ExposureReport(mean_luminance=0.0, clipped_highlight_pct=0.0)
    mean_luminance = float(gray_img.mean())
    clipped_pixels = int(np.count_nonzero(gray_img > HIGHLIGHT_LUMINANCE))
    clipped_highlight_pct = (
        float(clipped_pixels) / float(gray_img.size) * 100.0
    )
    return ExposureReport(
        mean_luminance=mean_luminance,
        clipped_highlight_pct=clipped_highlight_pct,
    )


def score_image(
    jpg_path: Union[str, Path], max_dim: int = DEFAULT_PROXY_MAX_DIM
) -> Score:
    """One-shot convenience: proxy -> Tenengrad -> composition -> exposure."""
    gray = prepare_proxy(jpg_path, max_dim=max_dim)
    exposure = check_exposure(gray)
    return Score(
        sharpness=compute_tenengrad(gray),
        composition_score=compute_composition_score(gray),
        mean_luminance=exposure.mean_luminance,
        clipped_highlight_pct=exposure.clipped_highlight_pct,
    )