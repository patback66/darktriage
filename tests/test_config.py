"""Tests for darktriage.config."""

from __future__ import annotations

from darktriage.config import (
    Config,
    load_config,
    write_default_config,
)


def test_defaults_are_non_destructive(tmp_path):
    config = load_config(tmp_path / "missing.toml")
    assert config.safety.preserve_existing is True
    assert config.safety.create_backup is True
    assert config.safety.dry_run is False


def test_load_config_from_file(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text(
        "[safety]\n"
        "preserve_existing = false\n"
        "create_backup = false\n"
        "\n"
        "[thresholds]\n"
        "tenengrad_blur = 100.0\n"
    )
    config = load_config(path)
    assert config.safety.preserve_existing is False
    assert config.safety.create_backup is False
    assert config.thresholds.tenengrad_blur == 100.0
    # Missing keys fall back to defaults.
    assert config.thresholds.tenengrad_sharp == 350.0
    assert config.thresholds.composition_min == 1.2
    assert config.tags.blur == "red"
    assert config.darktable.cli_path == "darktable-cli"


def test_load_config_corrupt_file_falls_back(tmp_path):
    path = tmp_path / "config.toml"
    path.write_text("not valid toml [[[")
    config = load_config(path)
    assert config.safety.preserve_existing is True
    assert config.thresholds.tenengrad_sharp == 350.0


def test_write_default_config_roundtrip(tmp_path):
    path = write_default_config(tmp_path / "sub" / "config.toml")
    assert path.exists()
    config = load_config(path)
    assert config.safety.preserve_existing is True
    assert config.safety.create_backup is True
    assert config.tags.exposure_warn == "yellow"
    assert config.darktable.config_dir == "~/.config/darktable"
    assert isinstance(config, Config)