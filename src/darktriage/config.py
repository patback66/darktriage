"""Configuration management for DarkTriage.

Loads ``~/.config/darktriage/config.toml`` and merges it over the built-in
non-destructive defaults (``preserve_existing = true`` and
``create_backup = true``).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Union

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

logger = logging.getLogger(__name__)

CONFIG_DIR = "darktriage"
CONFIG_FILE = "config.toml"

DEFAULT_CONFIG_TEMPLATE = """\
[safety]
preserve_existing = true
create_backup = true
dry_run = false

[thresholds]
tenengrad_blur = 150.0
tenengrad_sharp = 350.0
composition_min = 1.2
highlight_clip_percent = 0.05
burst_interval_sec = 2.0

[tags]
blur = "red"
keeper = "green"
portfolio = "purple"
exposure_warn = "yellow"

[darktable]
cli_path = "darktable-cli"
config_dir = "~/.config/darktable"
label_case = "title"

[sensor]
iso_blur_above = 1600
region_center_fraction = 0.6
"""


@dataclass(frozen=True)
class Safety:
    preserve_existing: bool = True
    create_backup: bool = True
    dry_run: bool = False


@dataclass(frozen=True)
class Thresholds:
    tenengrad_blur: float = 150.0
    tenengrad_sharp: float = 350.0
    composition_min: float = 1.2
    highlight_clip_percent: float = 0.05
    burst_interval_sec: float = 2.0


@dataclass(frozen=True)
class Sensor:
    """Camera/processing behaviour (configurable, no camera-specific hardcoding)."""

    # Gaussian-blur noise suppression applied when the EXIF ISO rating exceeds
    # this value (Nikon D800E and similar high-ISO bodies benefit).
    iso_blur_above: int = 1600
    # Central percentage of the frame used to weight focus/composition
    # calculations, avoiding lens-edge falloff.
    region_center_fraction: float = 0.6


@dataclass(frozen=True)
class Tags:
    blur: str = "red"
    keeper: str = "green"
    portfolio: str = "purple"
    exposure_warn: str = "yellow"


@dataclass(frozen=True)
class Darktable:
    cli_path: str = "darktable-cli"
    config_dir: str = "~/.config/darktable"
    # How colour-label names are written to XMP: "title" -> Adobe "Green",
    # "lower" -> "green", "asis" -> exactly as configured.
    label_case: str = "title"


@dataclass(frozen=True)
class Config:
    safety: Safety
    thresholds: Thresholds
    tags: Tags
    darktable: Darktable
    sensor: Sensor = field(default_factory=Sensor)
    config_path: Optional[Path] = None


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME") or os.path.expanduser("~/.config")
    return Path(base) / CONFIG_DIR / CONFIG_FILE


def _section(data: Dict[str, Any], name: str) -> Dict[str, Any]:
    raw = data.get(name, {})
    return raw if isinstance(raw, dict) else {}


def _value(section: Dict[str, Any], key: str, default: Any) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def load_config(path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from ``path`` (or the default location).

    Returns a fully-populated Config, falling back to the built-in defaults
    for any missing key or an unreadable/corrupt file.
    """
    config_path = Path(path) if path else default_config_path()
    data: Dict[str, Any] = {}
    if config_path.exists():
        try:
            with open(config_path, "rb") as fh:
                data = tomllib.load(fh)
        except Exception as exc:  # noqa: BLE001 - corrupt file -> defaults
            logger.warning("Could not parse %s (%s); using defaults", config_path, exc)

    safety_raw = _section(data, "safety")
    thresholds_raw = _section(data, "thresholds")
    tags_raw = _section(data, "tags")
    darktable_raw = _section(data, "darktable")
    sensor_raw = _section(data, "sensor")

    safety = Safety(
        preserve_existing=_value(safety_raw, "preserve_existing", True),
        create_backup=_value(safety_raw, "create_backup", True),
        dry_run=_value(safety_raw, "dry_run", False),
    )
    thresholds = Thresholds(
        tenengrad_blur=_value(thresholds_raw, "tenengrad_blur", 150.0),
        tenengrad_sharp=_value(thresholds_raw, "tenengrad_sharp", 350.0),
        composition_min=_value(thresholds_raw, "composition_min", 1.2),
        highlight_clip_percent=_value(thresholds_raw, "highlight_clip_percent", 0.05),
        burst_interval_sec=_value(thresholds_raw, "burst_interval_sec", 2.0),
    )
    tags = Tags(
        blur=_value(tags_raw, "blur", "red"),
        keeper=_value(tags_raw, "keeper", "green"),
        portfolio=_value(tags_raw, "portfolio", "purple"),
        exposure_warn=_value(tags_raw, "exposure_warn", "yellow"),
    )
    darktable = Darktable(
        cli_path=_value(darktable_raw, "cli_path", "darktable-cli"),
        config_dir=_value(darktable_raw, "config_dir", "~/.config/darktable"),
        label_case=_value(darktable_raw, "label_case", "title"),
    )
    sensor = Sensor(
        iso_blur_above=_value(sensor_raw, "iso_blur_above", 1600),
        region_center_fraction=_value(sensor_raw, "region_center_fraction", 0.6),
    )
    return Config(
        safety=safety,
        thresholds=thresholds,
        tags=tags,
        darktable=darktable,
        sensor=sensor,
        config_path=config_path,
    )


def write_default_config(path: Optional[Union[str, Path]] = None) -> Path:
    """Write the default config template to disk and return its path."""
    target = Path(path) if path else default_config_path()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(DEFAULT_CONFIG_TEMPLATE, encoding="utf-8")
    return target