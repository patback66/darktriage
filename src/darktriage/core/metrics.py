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
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple, Union, cast

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
    early_exit: bool = False  # True when the blur early-exit path fired
    error: Optional[str] = None


@dataclass(frozen=True)
class ImageMetadata:
    """Camera/proxy metadata read from the JPEG preview."""

    iso: Optional[int] = None
    orientation: int = 1  # 1 = normal; 2-8 = EXIF orientation tags


def load_image_metadata(jpg_path: Union[str, Path]) -> ImageMetadata:
    """Read ISO and EXIF orientation from a JPEG preview.

    Only the EXIF header is read (Pillow parses the APP1 segment lazily);
    ``Image.load()`` is deliberately NOT called so we never full-decode the
    36 MP frame just to read metadata.
    """
    with Image.open(jpg_path) as image:
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


# OpenCV reduced-scale grayscale decode flags.  Decoding at 1/2, 1/4 or 1/8
# scale happens inside the JPEG DCT decoder and avoids materializing the full
# 36 MP sensor frame in memory.
_CV_REDUCED_FLAGS = {
    1: cv2.IMREAD_GRAYSCALE,
    2: cv2.IMREAD_REDUCED_GRAYSCALE_2,
    4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
    8: cv2.IMREAD_REDUCED_GRAYSCALE_8,
}

# Optional PyTurboJPEG backend (used only when the native lib is present).
try:  # pragma: no cover - depends on optional native dependency
    from turbojpeg import TJPF_GRAY, TurboJPEG

    _TURBOJPEG = TurboJPEG()
except Exception:  # noqa: BLE001 - optional dependency
    TJPF_GRAY = None
    _TURBOJPEG = None


def _choose_reduction(long_edge: int, max_dim: int) -> int:
    """Pick the DCT reduction (1/2/4/8) whose result is closest to max_dim.

    Choosing the closest result (rather than the smallest) keeps small images
    at full resolution while decoding 36 MP frames at 1/4 scale, and avoids the
    discontinuity where a 1 px size change would halve the proxy.
    """
    return min((1, 2, 4, 8), key=lambda f: abs(long_edge / float(f) - max_dim))


def _target_long_edge(long_edge: int, max_dim: int) -> int:
    """Canonical proxy long edge (never upscales past the source)."""
    return min(long_edge, max_dim)


def _resize_long_edge(array: np.ndarray, target: int) -> np.ndarray:
    """Resize so the long edge equals ``target`` (down- or up-scaling)."""
    long_edge = max(array.shape[:2])
    if long_edge == target or long_edge == 0:
        return array
    scale = target / float(long_edge)
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR
    return cv2.resize(
        array, None, fx=scale, fy=scale, interpolation=interpolation
    )


def _decode_with_turbojpeg(data: bytes, reduction: int) -> np.ndarray:
    """Decode to grayscale via PyTurboJPEG at the requested reduction."""
    scaling = {2: 0.5, 4: 0.25, 8: 0.125}.get(reduction, 1.0)
    array = np.asarray(
        _TURBOJPEG.decode(data, pixel_format=TJPF_GRAY, scaling_factor=scaling),
        dtype=np.uint8,
    )
    if array.ndim != 2:
        raise ValueError("TurboJPEG did not return a 2-D grayscale array")
    return array


def _decode_grayscale_fast(
    jpg_path: Union[str, Path], max_dim: int
) -> np.ndarray:
    """Decode a JPEG directly to a grayscale array at the canonical proxy size.

    Uses PyTurboJPEG when available, otherwise OpenCV's reduced-scale
    ``IMREAD_REDUCED_GRAYSCALE_*`` flags, falling back to Pillow on failure.
    The returned long edge always equals ``min(source_long_edge, max_dim)`` so
    the fast and Pillow paths produce identical proxy dimensions.
    """
    data = Path(jpg_path).read_bytes()

    # Header-only dimensions (no pixel decode) to pick the reduction factor.
    with Image.open(jpg_path) as image:
        width, height = image.size
    source_long_edge = max(width, height)
    reduction = _choose_reduction(source_long_edge, max_dim)
    target = _target_long_edge(source_long_edge, max_dim)

    array: Optional[np.ndarray] = None
    if _TURBOJPEG is not None:
        try:
            array = _decode_with_turbojpeg(data, reduction)
        except Exception:  # noqa: BLE001 - fall through to OpenCV
            array = None

    if array is None:
        array = cv2.imdecode(
            np.frombuffer(data, dtype=np.uint8), _CV_REDUCED_FLAGS[reduction]
        )
        if array is None:
            raise ValueError(f"OpenCV could not decode JPEG: {jpg_path}")
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)

    array = _resize_long_edge(array, target)
    return array.astype(np.uint8, copy=False)


def prepare_proxy(
    jpg_path: Union[str, Path], max_dim: int = DEFAULT_PROXY_MAX_DIM
) -> np.ndarray:
    """Fast-load a camera JPEG as a grayscale array at the canonical proxy size.

    Decodes at 1/2, 1/4 or 1/8 scale directly in the DCT domain (PyTurboJPEG /
    OpenCV reduced flags) to avoid full 36 MP decoding, falling back to a
    Pillow Lanczos resize if the fast path fails.  Both paths yield the same
    long edge (``min(source_long_edge, max_dim)``) so metrics are comparable.
    """
    try:
        return _decode_grayscale_fast(jpg_path, max_dim)
    except Exception as exc:  # noqa: BLE001 - fallback to Pillow
        logger.debug("Fast JPEG decode failed for %s (%s); using Pillow", jpg_path, exc)

    with Image.open(jpg_path) as image:
        image = image.convert("RGB")
        width, height = image.size
        long_edge = max(width, height)
        target = _target_long_edge(long_edge, max_dim)
        if long_edge != target and long_edge > 0:
            scale = target / float(long_edge)
            new_size = (
                max(1, int(round(width * scale))),
                max(1, int(round(height * scale))),
            )
            image = image.resize(new_size, Image.LANCZOS)
        return to_grayscale(np.asarray(image))


def central_roi(gray_img: np.ndarray, fraction: float = 0.6) -> np.ndarray:
    """Return the central ``fraction`` region-of-interest crop.

    ``fraction`` is clamped to [0.05, 1.0] to avoid degenerate/nonsensical
    crops from bad configuration.
    """
    fraction = min(1.0, max(0.05, float(fraction)))
    height, width = gray_img.shape[:2]
    crop_h = max(1, int(round(height * fraction)))
    crop_w = max(1, int(round(width * fraction)))
    y0 = (height - crop_h) // 2
    x0 = (width - crop_w) // 2
    return gray_img[y0 : y0 + crop_h, x0 : x0 + crop_w]


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
    blur_threshold: float = 150.0,
) -> Score:
    """Score one JPEG through orientation correction, noise filtering and
    central-region-of-interest focus analysis.

    Flow:
      1. Fast decode -> orientation correction.
      2. ISO > ``iso_threshold`` -> 3x3 Gaussian blur noise pre-filter.
      3. Sobel Tenengrad on the central ``center_fraction`` ROI crop.
      4. Early exit: if the ROI Tenengrad is below ``blur_threshold`` the
         frame is flagged as soft/blur and composition + highlight-histogram
         checks are skipped (only a cheap mean luminance is computed).
    """
    gray = prepare_proxy(jpg_path, max_dim=max_dim)
    gray = apply_orientation(gray, orientation)
    if iso is not None and iso > iso_threshold:
        gray = apply_noise_filter(gray, iso, iso_threshold)

    roi = central_roi(gray, center_fraction)
    sharpness = compute_tenengrad(roi)

    if sharpness < blur_threshold:
        # Early exit: skip the composition grid, but still compute exposure on
        # the full frame so clipping/EV stay correct for soft frames.
        mean_luminance = float(gray.mean())
        clipped = (
            float(np.count_nonzero(gray > HIGHLIGHT_LUMINANCE))
            / float(max(1, gray.size))
            * 100.0
        )
        return Score(
            sharpness=sharpness,
            composition_score=0.0,
            mean_luminance=mean_luminance,
            clipped_highlight_pct=clipped,
            ev_offset=compute_ev_offset(mean_luminance),
            early_exit=True,
        )

    exposure = check_exposure(gray)
    composition = compute_composition_score(roi)
    return Score(
        sharpness=sharpness,
        composition_score=composition,
        mean_luminance=exposure.mean_luminance,
        clipped_highlight_pct=exposure.clipped_highlight_pct,
        ev_offset=compute_ev_offset(exposure.mean_luminance),
    )


def score_item(
    item: Tuple[str, Optional[int], int],
    max_dim: int,
    center_fraction: float,
    blur_threshold: float = 150.0,
) -> Score:
    path, iso, orientation = item
    return score_image(
        path,
        max_dim=max_dim,
        iso=iso,
        orientation=orientation,
        center_fraction=center_fraction,
        blur_threshold=blur_threshold,
    )


def score_images(
    items: Iterable[Tuple[str, Optional[int], int]],
    *,
    max_dim: int = DEFAULT_PROXY_MAX_DIM,
    center_fraction: float = 0.6,
    workers: Optional[int] = None,
    blur_threshold: float = 150.0,
    on_complete: Optional[Callable[[], None]] = None,
) -> List[Score]:
    """Score several JPEG previews in parallel.

    ``items`` is a list of ``(path, iso, orientation)`` triples so that each
    file's metadata is respected.  ``on_complete``, if given, is called once
    for every file as it finishes (useful for driving a progress bar).
    Results are returned in input order.
    """
    paths = [(str(p), iso, orient) for p, iso, orient in items]
    if not paths:
        return []
    worker_count = workers or min((os.cpu_count() or 1), len(paths))
    if worker_count <= 1 or len(paths) == 1:
        results = [
            score_image(
                path,
                max_dim=max_dim,
                iso=iso,
                orientation=orientation,
                center_fraction=center_fraction,
                blur_threshold=blur_threshold,
            )
            for path, iso, orientation in paths
        ]
        if on_complete:
            for _ in results:
                on_complete()
        return results
    payloads = [
        (p, iso, orient, max_dim, center_fraction, blur_threshold)
        for p, iso, orient in paths
    ]
    results: List[Optional[Score]] = [None] * len(payloads)
    with ProcessPoolExecutor(max_workers=worker_count) as executor:
        future_to_index = {
            executor.submit(_score_job, payload): i
            for i, payload in enumerate(payloads)
        }
        for future in concurrent.futures.as_completed(future_to_index):
            idx = future_to_index[future]
            results[idx] = future.result()
            if on_complete:
                on_complete()
    return [cast(Score, r) for r in results]  # type: ignore[return-value]


def _score_job(
    payload: Tuple[str, Optional[int], int, int, float, float]
) -> Score:
    path, iso, orientation, max_dim, center_fraction, blur_threshold = payload
    return score_image(
        path,
        max_dim=max_dim,
        iso=iso,
        orientation=orientation,
        center_fraction=center_fraction,
        blur_threshold=blur_threshold,
    )


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
