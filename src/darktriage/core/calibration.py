"""Data-driven threshold calibration for DarkTriage ratings.

The rating bands (tenengrad_blur / tenengrad_sharp / composition_min /
highlight_clip_percent) depend on the source proxy (Fine JPEG vs embedded
preview, resolution, compression).  This module scores a folder with the blur
early-exit disabled, summarises the metric distributions, and suggests
thresholds from percentiles so the bands fit the actual images.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from darktriage.config import Config
from darktriage.core.engine import AnalysisOptions, build_image_map, score_pairs
from darktriage.core.exif_reader import ImagePair
from darktriage.core.metrics import Score


@dataclass(frozen=True)
class MetricSummary:
    """Percentile summary for one metric."""

    count: int
    minimum: float
    p10: float
    p25: float
    p50: float
    p75: float
    p90: float
    maximum: float

    def as_dict(self) -> Dict[str, float]:
        return {
            "min": self.minimum,
            "p10": self.p10,
            "p25": self.p25,
            "p50": self.p50,
            "p75": self.p75,
            "p90": self.p90,
            "max": self.maximum,
        }


def _clean(values: Sequence[float]) -> List[float]:
    return [float(v) for v in values if v is not None and np.isfinite(v)]


def summarise(values: Sequence[float]) -> MetricSummary:
    """Return a percentile summary for a metric (empty -> all zeros)."""
    data = _clean(values)
    if not data:
        return MetricSummary(0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    array = np.asarray(data, dtype=np.float64)
    p10, p25, p50, p75, p90 = np.percentile(array, [10, 25, 50, 75, 90])
    return MetricSummary(
        count=len(data),
        minimum=float(array.min()),
        p10=float(p10),
        p25=float(p25),
        p50=float(p50),
        p75=float(p75),
        p90=float(p90),
        maximum=float(array.max()),
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    data = _clean(values)
    if not data:
        return 0.0
    return float(np.percentile(np.asarray(data, dtype=np.float64), percentile))


def collect_scores(
    pairs: List[ImagePair],
    options: AnalysisOptions,
    on_complete=None,
    directory: Optional[str] = None,
) -> List[Score]:
    """Score pairs with early-exit disabled so every metric is populated.

    When ``options.use_embedded_preview`` is set, ``directory`` is used to
    locate/extract the embedded previews (same cache as `analyze`).
    """
    full_options = AnalysisOptions(
        max_dim=options.max_dim,
        center_fraction=options.center_fraction,
        blur_threshold=0.0,  # never early-exit -> composition/clipping for all
        iso_threshold=options.iso_threshold,
        workers=options.workers,
        chunk_size=options.chunk_size,
        use_embedded_preview=options.use_embedded_preview,
        preview_cache_dir=options.preview_cache_dir,
    )
    image_map = None
    if options.use_embedded_preview and directory is not None:
        image_map = build_image_map(pairs, full_options, directory)
    results = score_pairs(
        pairs, full_options, on_complete=on_complete, image_map=image_map
    )
    return [score for score in results.values() if not score.error]


def suggest_thresholds(
    scores: Sequence[Score],
    *,
    blur_percentile: float = 15.0,
    sharp_percentile: float = 75.0,
    composition_percentile: float = 60.0,
    clip_percentile: float = 85.0,
    burst_interval_sec: float = 2.0,
) -> Dict[str, float]:
    """Suggest rating thresholds from the metric distributions."""
    sharpness = [s.sharpness for s in scores]
    composition = [s.composition_score for s in scores]
    clipped = [s.clipped_highlight_pct / 100.0 for s in scores]

    blur = _percentile(sharpness, blur_percentile)
    sharp = _percentile(sharpness, sharp_percentile)
    if sharp <= blur:
        sharp = blur * 1.5 + 1.0
    comp = _percentile(composition, composition_percentile)
    clip = _percentile(clipped, clip_percentile)
    clip = min(max(clip, 0.005), 0.5)

    return {
        "tenengrad_blur": round(blur, 1),
        "tenengrad_sharp": round(sharp, 1),
        "composition_min": round(comp, 2),
        "highlight_clip_percent": round(clip, 4),
        "burst_interval_sec": float(burst_interval_sec),
    }


def _toml_value(value) -> str:
    """Render a scalar as TOML (json scalars are valid TOML scalars)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return json.dumps(float(value)) if isinstance(value, float) else json.dumps(value)
    return json.dumps(str(value))


def render_config(
    config: Config, thresholds: Optional[Dict[str, float]] = None
) -> str:
    """Render a complete config.toml, optionally overriding the thresholds."""
    override = thresholds or {}
    th = config.thresholds

    def threshold(key: str, default: float) -> float:
        return float(override.get(key, default))

    blocks = [
        "[safety]\n"
        + "\n".join(
            [
                f"preserve_existing = {_toml_value(config.safety.preserve_existing)}",
                f"create_backup = {_toml_value(config.safety.create_backup)}",
                f"dry_run = {_toml_value(config.safety.dry_run)}",
            ]
        ),
        "[thresholds]\n"
        + "\n".join(
            [
                f"tenengrad_blur = {_toml_value(threshold('tenengrad_blur', th.tenengrad_blur))}",
                f"tenengrad_sharp = {_toml_value(threshold('tenengrad_sharp', th.tenengrad_sharp))}",
                f"composition_min = {_toml_value(threshold('composition_min', th.composition_min))}",
                f"highlight_clip_percent = {_toml_value(threshold('highlight_clip_percent', th.highlight_clip_percent))}",
                f"burst_interval_sec = {_toml_value(threshold('burst_interval_sec', th.burst_interval_sec))}",
            ]
        ),
        "[tags]\n"
        + "\n".join(
            [
                f"blur = {_toml_value(config.tags.blur)}",
                f"keeper = {_toml_value(config.tags.keeper)}",
                f"portfolio = {_toml_value(config.tags.portfolio)}",
                f"exposure_warn = {_toml_value(config.tags.exposure_warn)}",
            ]
        ),
        "[darktable]\n"
        + "\n".join(
            [
                f"cli_path = {_toml_value(config.darktable.cli_path)}",
                f"config_dir = {_toml_value(config.darktable.config_dir)}",
                f"label_case = {_toml_value(config.darktable.label_case)}",
            ]
        ),
        "[sensor]\n"
        + "\n".join(
            [
                f"iso_blur_above = {_toml_value(config.sensor.iso_blur_above)}",
                f"region_center_fraction = {_toml_value(config.sensor.region_center_fraction)}",
            ]
        ),
    ]
    return "\n\n".join(blocks) + "\n"


def write_config(path: Path, text: str, *, backup: bool = True) -> Path:
    """Write a config file, backing up an existing one to ``.bak``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if backup and path.exists():
        shutil.copy2(path, Path(str(path) + ".bak"))
    path.write_text(text, encoding="utf-8")
    return path
