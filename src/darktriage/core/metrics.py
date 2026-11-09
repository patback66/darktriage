"""Computer-vision scoring primitives for DarkTriage.

Deterministic, order-independent scoring.  Camera JPEG previews are downscaled
in memory (Lanczos) to a long edge of 1920 px before metrics are computed.
Scoring can be parallelised across a process pool via score_images().

Sensor metadata (ISO, EXIF orientation) is read from the preview and used to
make the scoring deterministic and camera-agnostic.
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


@dataclass(frozen=True)
class ImageMetadata:
    """Camera/proxy metadata read from the JPEG preview."""

    iso: Optional[int] = None
    orientation: int = 1  # 1 = normal; 2-8 = EXIF orientation tags


def load_image_metadata(jpg_path: Union[str, Path]) -> ImageMetadata:
    """Read ISO and EXIF orientation from a JPEG preview."""
    with Image.open(jpg_path) as image:
        image.load()
        exif = image.getexif()
        iso = None
        iso_value = exif.get(0x8827)  # ISOSpeedRatings
        if iso_value is not None:
            if isinstance(iso_value, (list, tuple)):
                iso = int(iso_value[0]) if iso_value else None
            elif isinstance(iso_value, int):
                iso = int(iso_value)
            elif isinstance(iso_value, bytes):
                iso = int(str(iso_value.decode("ascii", "ignore").split(",")[0]))
        orientation = exif.get(0x0112, 1)
        if isinstance(orientation, (tuple, list)):
            orientation = int(orientation[0]) if orientation else 1
        return ImageMetadata(iso=iso, orientation=int(orientation))


def to_grayscale(image: np.ndarray) -> np.ndarray:
    """Convert a numpy image array to a single-channel uint8 array."""
    if image.ndim == 2:
        return image.astype(np.uint8)
    if image.ndim == 3:
        if image.shape[2] == 3:
            return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        if image.shape[2] == 4:
            return cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
    raise ValueError(f"Unsupported image shape: {image.shape}")


def prepare_proxy(
    jpg_path: Union[str, Path], max_dim: int = DEFAULT_PROXY_MAX_DIM
) -> np.ndarray:
    """Load a camera JPEG and downscale it in memory to max_dim long edge.

    Returns a single-channel uint8 grayscale matrix.  Resizing uses Lanczos.
    """
    with Image.open(jpg_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        long_edge = max(width, height)
        if long_edge > max_dim:
            scale = max_dim / float(long_edge)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, Image.LANCZOS)
        return to_grayscale(np.asarray(image))


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
    clipped_highlight_pct = (
        float(clipped_pixels) / float(gray_img.size) * 100.0
    )
    return ExposureReport(
        mean_luminance=mean_luminance,
        clipped_highlight_pct=clipped_highlight_pct,
    )


def apply_orientation(gray: np.ndarray, orientation: int = 1) -> np.ndarray:
    """Apply an EXIF Orientation tag (1-8) to a grayscale matrix."""
    if orientation not in range(1, 9):
        return gray
    if orientation == 2:      # mirror horizontal
        return cv2.flip(gray, 1)
    if orientation == 3:      # rotate 180
        return cv2.rotate(gray, cv2.ROTATE_180)
    if orientation == 4:      # mirror vertical
        return cv2.flip(gray, 0)
    # 5-8 involve mirror + rotation, expressed with cv2/numpy ops
    if orientation == 5:      # mirror horizontal + rotate 270 CW
        return np.rot90(cv2.flip(gray, 1))
    if orientation == 6:      # rotate 90 CW
        return cv2.rotate(gray, cv2.ROTATE_90_CLOCKWISE)
    if orientation == 7:      # mirror vertical and rotate 90 CW
        return np.rot90(cv2.flip(gray, 0))
    if orientation == 8:      # rotate 270 CW
        return cv2.rotate(gray, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return gray


def apply_noise_filter(gray: np.ndarray, iso: Optional[int] = None,
                       iso_threshold: int = 1600) -> np.ndarray:
    """Apply a Gaussian blur if the ISO rating is above the threshold."""
    if iso is None or iso <= iso_threshold:
        return gray
    return cv2.GaussianBlur(gray, (3, 3), 0)


def apply_region_mask(gray: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    """Return a central-weighted copy of the image.

    A smooth mask (1.0 at the centre, 0.0 at the central-``fraction`` window
    boundary) down-weights lens-edge falloff so the focus/composition scores
    are driven by the sharpest central region.
    """
    h, w = gray.shape
    cx = (w - 1) / 2.0
    cy = (h - 1) / 2.0
    half_w = (w * fraction) / 2.0
    half_h = (h * fraction) / 2.0

    xx = np.arange(w, dtype=np.float64)
    yy = np.arange(h, dtype=np.float64)
    dx = np.clip(np.abs(xx - cx) / half_w, 0.0, 1.0)
    dy = np.clip(np.abs(yy - cy) / half_h, 0.0, 1.0)
    dist = np.sqrt(dx[None, :] ** 2 + dy[:, None] ** 2) / np.sqrt(2.0)
    weight = np.clip(1.0 - dist, 0.0, 1.0)
    return gray.astype(np.float64) * weight


def score_image(
    jpg_path: Union[str, Path],
    *,
    max_dim: int = DEFAULT_PROXY_MAX_DIM,
    iso: Optional[int] = None,
    orientation: int = 1,
    center_fraction: float = 0.6,
    iso_threshold: int = 1600,
) -> Score:
    """Score one JPEG through orientation correction, noise filtering and
    central-region weighting.

    Calling this directly gives full per-camera, per-file control.
    """
    gray = prepare_proxy(jpg_path, max_dim=max_dim)
    gray = apply_orientation(gray, orientation)
    exposure = check_exposure(gray)
    if iso is not None and iso > iso_threshold:
        gray = apply_noise_filter(gray, iso, iso_threshold)
    core = apply_region_mask(gray, center_fraction)
    return Score(
        sharpness=compute_tenengrad(core),
        composition_score=compute_composition_score(core),
        mean_luminance=exposure.mean_luminance,
        clipped_highlight_pct=exposure.clipped_highlight_pct,
        ev_offset=compute_ev_offset(exposure.mean_luminance),
    )


def score_item(
    item: Tuple[str, Optional[int], int], max_dim: int, center_fraction: float
) -> Score:
    path, iso, orientation = item
    return score_image(path, max_dim=max_dim, iso=iso, orientation=orientation,
                       center_fraction=center_fraction)


def score_images(
    items: Iterable[Tuple[str, Optional[int], int]],
    *,
    max_dim: int = DEFAULT_PROXY_MAX_DIM,
    center_fraction: float = 0.6,
    workers: Optional[int] = None,
) -> List[Score]:
    """Score several JPEG previews in parallel.

    ``items`` is a list of ``(path, iso, orientation)`` triples so that each
    file's metadata is respected.
    """
    paths = [(str(p), iso, orient) for p, iso, orient in items]
    if not paths:
        return []
    worker_count = workers or min((os.cpu_count() or 1), len(paths))
    if worker_count <= 1 or len(paths) == 1:
        return [
            score_image(path, max_dim=max_dim, iso=iso, orientation=orientation,
                       center_fraction=center_fraction)
            for path, iso, orientation in paths
        ]
    payloads = [(p, iso, orient, max_dim, center_fraction) for p, iso, orient in paths]
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        results = list(executor.map(_score_job, payloads))
    return results


def _score_job(payload: Tuple[str, Optional[int], int, int, float]) -> Score:
    path, iso, orientation, max_dim, center_fraction = payload
    return score_image(path, max_dim=max_dim, iso=iso, orientation=orientation,
                       center_fraction=center_fraction)


def compute_ev_offset(mean_luminance: float) -> float:
    """Recommended exposure compensation: log2(128 / mean) clamped to [-2, 2].

    Positive values mean the frame is underexposed (increase exposure);
    negative values mean it is overexposed.
    """
    if mean_luminance <= 0.0:
        return MAX_EV_OFFSET
    ev = math.log2(EV_REFERENCE_LUMINANCE / mean_luminance)
    return float(max(MIN_EV_OFFSET, min(MAX_EV_OFFSET, ev)))


analyze_image = score_image
