"""Computer-vision scoring primitives for DarkTriage.

Implements deterministic heuristics built on OpenCV:

* *Tenengrad sharpness* - sum/mean of squared Sobel gradient magnitudes.
* *Composition* - luminance and contrast sampled on a 3x3 rule-of-thirds grid
  (nine cells plus the four node intersections).
* *Exposure / highlight clipping* - luminance histogram and clipping ratio.
* *Rating* - deterministic 1-5 star score with a color label.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

import cv2
import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Thresholds (tunable through the TUI; values are tuned for 8-bit previews)
# ---------------------------------------------------------------------------

# Brightness above which a pixel counts as a clipped highlight.
HIGHLIGHT_THRESHOLD = 240

# Highlight clipping is reported when this fraction of pixels is above
# HIGHLIGHT_THRESHOLD.  Defaults to 5%.
CLIP_THRESHOLD_FRACTION = 0.05

# Tenengrad bands: mean squared Sobel gradient magnitude (per-pixel).
SHARPNESS_5 = 200.0
SHARPNESS_4 = 100.0
SHARPNESS_3 = 60.0
SHARPNESS_2 = 30.0
SHARPNESS_1 = 10.0

# Rule-of-thirds: weight given to node contrast when scoring composition.
COMPOSITION_NODE_WEIGHT = 0.6

# Exposure target luminance (0-255) used as the ideal middle grey.
EXPOSURE_TARGET = 128.0

# Burst bonus: the sharpest frame of a burst earns +1 star (capped at 5).
BURST_BONUS = 1

# Color label assignment depends on the dominant failure mode.
_LABEL_RED = ("red", "Red", "Blur")
_LABEL_YELLOW = ("yellow", "Yellow", "Overexposed")
_LABEL_GREEN = ("green", "Green", "Keeper")
_LABEL_PURPLE = ("purple", "Purple", "Portfolio")

_LABEL_BY_RATING = {
    5: _LABEL_GREEN,
    4: _LABEL_GREEN,
    3: _LABEL_YELLOW,
    2: _LABEL_RED,
    1: _LABEL_RED,
}

# Rule-of-thirds grid.
_GRID_COLS = 3
_GRID_ROWS = 3
_GRID_NODES = 4


logger = logging.getLogger(__name__)


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert an RGB/RGBA uint8 array to a single-channel uint8 array."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def tenengrad_sharpness(gray: np.ndarray) -> float:
    """Return the Tenengrad sharpness energy of a grayscale image.

    Energy is the *mean* squared Sobel gradient magnitude over the image,
    so it is resolution-independent and comparable across proxy images.
    Higher values indicate more high-frequency detail (crisp focus); very
    low values indicate motion blur or an out-of-focus frame.
    """
    if gray.size == 0:
        return 0.0
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.hypot(grad_x, grad_y)
    return float(np.mean(magnitude * magnitude))


def _cell_regions(gray: np.ndarray):
    """Yield (y0, y1, x0, x1) slices for a 3x3 rule-of-thirds grid."""
    height, width = gray.shape
    cell_h = height // _GRID_ROWS
    cell_w = width // _GRID_COLS
    for row in range(_GRID_ROWS):
        for col in range(_GRID_COLS):
            yield (
                row * cell_h,
                (row + 1) * cell_h,
                col * cell_w,
                (col + 1) * cell_w,
            )


def composition_report(gray: np.ndarray) -> dict:
    """Sample luminance and contrast on a 3x3 rule-of-thirds grid.

    Returns node luminances, cell statistics and a heuristic composition
    score in the range 0-100.  A healthy composition concentrates contrast
    at the rule-of-thirds intersections and maintains reasonable tonal
    separation between grid cells.
    """
    height, width = gray.shape
    cells = []
    for y0, y1, x0, x1 in _cell_regions(gray):
        region = gray[y0:y1, x0:x1]
        cells.append({"mean": float(region.mean()), "std": float(region.std())})

    # Four intersections of the rule-of-thirds lines (1/3 and 2/3 of each
    # image axis).
    fractions = (1.0 / 3.0, 2.0 / 3.0)
    node_positions = [(int(height * f), int(width * g)) for f in fractions for g in fractions]

    node_luminances: list[float] = []
    node_contrasts: list[float] = []
    for y, x in node_positions:
        patch = gray[max(0, y - 1) : y + 2, max(0, x - 1) : x + 2]
        node_luminances.append(float(patch.mean()))
        node_contrasts.append(float(patch.std()))

    cell_luminances = [c["mean"] for c in cells]
    cell_conts = [c["std"] for c in cells]
    mean_cell_contrast = float(np.mean(cell_conts)) if cell_conts else 0.0
    mean_node_contrast = float(np.mean(node_contrasts)) if node_contrasts else 0.0

    # Relative contrast at the grid intersections vs. the cells.  The
    # heuristic assumes well-composed photos put more edge energy at the
    # intersections than over the full cells.
    contrast_ratio = mean_node_contrast / (mean_cell_contrast + 1e-9)
    composition_score = float(100.0 * np.clip(contrast_ratio / 4.0, 0.0, 1.0))

    return {
        "grid_rows": _GRID_ROWS,
        "grid_cols": _GRID_COLS,
        "node_positions": node_positions,
        "node_luminances": node_luminances,
        "node_contrasts": node_contrasts,
        "cell_luminances": cell_luminances,
        "cell_contrasts": cell_conts,
        "contrast_ratio": float(contrast_ratio),
        "composition_score": composition_score,
    }


def exposure_report(gray: np.ndarray) -> dict:
    """Compute luminance statistics and highlight-clipping ratio."""
    if gray.size == 0:
        return {
            "mean_luminance": 0.0,
            "median_luminance": 0.0,
            "highlight_ratio": 0.0,
            "clipped": False,
            "underexposed": 0.0,
        }
    luminance = gray.astype(np.float32)
    mean_luminance = float(luminance.mean())
    median_luminance = float(np.median(luminance))
    histogram = np.bincount(
        luminance.astype(np.uint8).ravel(), minlength=256
    )
    total = float(luminance.size)
    highlight_ratio = float(histogram[HIGHLIGHT_THRESHOLD:].sum() / total)
    clipped = highlight_ratio >= CLIP_THRESHOLD_FRACTION
    return {
        "mean_luminance": mean_luminance,
        "median_luminance": median_luminance,
        "highlight_ratio": highlight_ratio,
        "clipped": clipped,
        "underexposed": float(histogram[:30].sum() / total),
    }


@dataclass(frozen=True)
class Rating:
    """Deterministic 1-5 star rating together with a Darktable color label."""

    rating: int
    label: str
    label_value: str
    in_burst: bool


def render_rating(
    sharpness: float,
    exposure: dict,
    composition: dict,
    *,
    in_burst: bool = False,
) -> Rating:
    """Convert raw metrics into a 1-5 star rating and Darktable label.

    Deterministic and order-independent.  ``in_burst`` grants +1 star (capped
    at 5) to the sharpest frame of a burst sequence.
    """
    rating = 3

    # Sharpness bands (higher is sharper).
    if sharpness >= SHARPNESS_5:
        rating += 2
    elif sharpness >= SHARPNESS_4:
        rating += 1
    elif sharpness >= SHARPNESS_3:
        pass
    elif sharpness >= SHARPNESS_2:
        rating -= 1
    else:
        rating -= 2

    # Highlight clipping penalty.
    if exposure["highlight_ratio"] > 0.10:
        rating -= 1
    elif exposure["highlight_ratio"] > CLIP_THRESHOLD_FRACTION:
        rating -= 1

    # Under-exposure penalty.
    if exposure["underexposed"] > 0.25:
        rating -= 1

    # Composition bonus (mild).
    if composition["composition_score"] >= 70.0:
        rating += 1
    elif composition["composition_score"] < 30.0:
        rating -= 1

    # Burst bonus for the sharpest frame in a sequence.
    if in_burst:
        rating += BURST_BONUS

    rating = max(1, min(5, rating))

    label_name, label_color, label_display = _LABEL_BY_RATING.get(rating, _LABEL_PURPLE)
    return Rating(rating=rating, label=label_name, label_value=label_color, in_burst=in_burst)


def analyze_image(
    source: Union[Path, str, np.ndarray, Image.Image]
) -> dict:
    """Score one image (path, array or PIL image) end-to-end.

    Returns a dictionary with: metric values, a ``rating`` :class:`Rating`
    object, and summary fields used for CLI/TUI presentation.
    """
    if isinstance(source, (str, Path)):
        with Image.open(source) as image:
            image.load()
            array = np.asarray(image.convert("RGB"))
    elif isinstance(source, Image.Image):
        array = np.asarray(source.convert("RGB"))
    elif isinstance(source, np.ndarray):
        array = source
    else:
        raise TypeError(f"Unsupported source type: {type(source)!r}")

    gray = to_grayscale(array)
    metrics = {
        "width": int(array.shape[1]),
        "height": int(array.shape[0]),
        "sharpness": tenengrad_sharpness(gray),
        "exposure": exposure_report(gray),
        "composition": composition_report(gray),
    }
    rating = render_rating(
        metrics["sharpness"],
        metrics["exposure"],
        metrics["composition"],
    )
    metrics["rating"] = rating
    return metrics
