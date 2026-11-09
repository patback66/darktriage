"""Deterministic, local subject auto-tagging for DarkTriage.

No LLM, no network.  Two cheap, explainable signals are combined on a small
colour proxy:

* **Faces** - OpenCV's bundled Haar cascade -> ``people`` / ``portrait`` /
  ``group``.
* **Scene colour/tonal statistics** - sky and vegetation fractions plus mean
  luminance -> ``landscape`` / ``nature`` / ``night`` (only when no faces).

Object classes such as ``car`` need a trained detector; supply one through the
optional DNN hook (see docs) - the heuristics here cannot recognise vehicles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_AUTOTAG_MAX_DIM = 640
# Minimum largest-face area (fraction of the proxy) to call a single face a portrait.
PORTRAIT_FACE_RATIO = 0.04

_CASCADE: Optional[cv2.CascadeClassifier] = None
_CASCADE_FAILED = False


@dataclass(frozen=True)
class SceneStats:
    sky_fraction: float
    green_fraction: float
    warm_fraction: float
    mean_luminance: float

    @property
    def night(self) -> bool:
        return self.mean_luminance < 50.0


@dataclass(frozen=True)
class SubjectTags:
    tags: Tuple[str, ...]
    face_count: int
    scene: SceneStats


def _load_cascade() -> Optional[cv2.CascadeClassifier]:
    global _CASCADE, _CASCADE_FAILED
    if _CASCADE is not None or _CASCADE_FAILED:
        return _CASCADE
    try:
        path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        cascade = cv2.CascadeClassifier(str(path))
        if cascade.empty():
            raise ValueError(f"empty cascade file: {path}")
        _CASCADE = cascade
    except Exception as exc:  # noqa: BLE001 - optional model
        logger.warning("Face cascade unavailable; face tags disabled: %s", exc)
        _CASCADE_FAILED = True
        _CASCADE = None
    return _CASCADE


def _decode_color_proxy(jpg_path, max_dim: int = DEFAULT_AUTOTAG_MAX_DIM) -> np.ndarray:
    data = Path(jpg_path).read_bytes()
    array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_REDUCED_COLOR_4)
    if array is None:
        array = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if array is None:
        raise ValueError(f"could not decode image for auto-tagging: {jpg_path}")
    long_edge = max(array.shape[:2])
    if long_edge > max_dim:
        scale = max_dim / float(long_edge)
        array = cv2.resize(array, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    return array


def detect_faces(gray: np.ndarray) -> list:
    """Return face rectangles (x, y, w, h); empty if the cascade is missing."""
    cascade = _load_cascade()
    if cascade is None or gray.size == 0:
        return []
    faces = cascade.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5, minSize=(24, 24)
    )
    return [tuple(int(v) for v in rect) for rect in faces]


def analyse_scene(bgr: np.ndarray) -> SceneStats:
    """Compute sky/vegetation/warm fractions and mean luminance from a BGR image."""
    if bgr.size == 0:
        return SceneStats(0.0, 0.0, 0.0, 0.0)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    hue, sat, val = cv2.split(hsv)
    sat_f = sat.astype(np.float32)
    val_f = val.astype(np.float32)

    total = float(hue.size) or 1.0
    sky_mask = (
        (hue >= 90) & (hue <= 130) & (sat >= 30) & (val >= 120)
    )
    # Sky is expected in the upper half of the frame.
    top_half = sky_mask[: max(1, sky_mask.shape[0] // 2), :]
    sky_fraction = float(top_half.mean())

    green_mask = (hue >= 35) & (hue <= 85) & (sat >= 40) & (val >= 40)
    green_fraction = float(green_mask.mean())

    warm_mask = ((hue <= 30) | (hue >= 160)) & (sat >= 50) & (val >= 40)
    warm_fraction = float(warm_mask.mean())

    return SceneStats(
        sky_fraction=sky_fraction,
        green_fraction=green_fraction,
        warm_fraction=warm_fraction,
        mean_luminance=float(val_f.mean()),
    )


def derive_tags(
    *, face_count: int, largest_face_ratio: float, scene: SceneStats
) -> Tuple[str, ...]:
    """Pure mapping from faces + scene stats to a subject-tag tuple."""
    tags: list = []
    if face_count >= 1:
        tags.append("people")
        if face_count == 1 and largest_face_ratio >= PORTRAIT_FACE_RATIO:
            tags.append("portrait")
        if face_count >= 2:
            tags.append("group")
    else:
        if scene.night:
            tags.append("night")
        if scene.green_fraction >= 0.30:
            tags.append("nature")
        if scene.sky_fraction >= 0.25:
            tags.append("landscape")
    # de-duplicate while preserving order
    return tuple(dict.fromkeys(tags))


class SubjectTagger:
    """Runs the face + scene heuristics on an image file."""

    def __init__(self, max_dim: int = DEFAULT_AUTOTAG_MAX_DIM):
        self.max_dim = max_dim
        _load_cascade()

    def analyze(self, image_path) -> SubjectTags:
        bgr = _decode_color_proxy(image_path, self.max_dim)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        faces = detect_faces(gray)
        scene = analyse_scene(bgr)
        height, width = gray.shape[:2]
        frame_area = float(height * width) or 1.0
        largest = max((w * h) / frame_area for (_, _, w, h) in faces) if faces else 0.0
        tags = derive_tags(
            face_count=len(faces), largest_face_ratio=largest, scene=scene
        )
        return SubjectTags(tags=tags, face_count=len(faces), scene=scene)
