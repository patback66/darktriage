# DarkTriage (`darktriage`)

A deterministic, terminal-native (CLI/TUI) pre-import photo triage tool and batch post-processing engine built for RAW photographers using Darktable.

## Overview

DarkTriage replaces heavy, slow LLM vision models with fast, local computer vision heuristics (OpenCV) and EXIF metadata parser pipelines. It evaluates camera JPEG previews in memory, scores sharpness and composition, detects burst sequences, and writes sidecar metadata (`.xmp`) directly into your RAW folder—all before launching Darktable.

## Key Features

- **Micro-Sharpness Scoring:** Uses Tenengrad gradient magnitude (Sobel operators) on proxy images to identify crisp focus vs. motion blur.
- **Burst Sequence Grouping:** Automatically clusters shots taken within micro-second EXIF intervals; awards bonus ratings to the sharpest frame in each sequence.
- **Composition & Exposure Heuristics:** Evaluates luminance distribution against Rule-of-Thirds grid nodes and flags highlight clipping.
- **Native Darktable XMP Generator:** Appends 1–5 star ratings and color labels (Red=Blur, Yellow=Overexposed, Green=Keeper, Purple=Portfolio) via `exiftool`.
- **Interactive TUI Dashboard:** Built with `Textual` for terminal navigation, metric threshold tuning, and bulk sidecar commits.
- **Headless Batch Export Engine:** Direct invocation of `darktable-cli` to apply styles (`--style`) and render final JPEG/TIFF outputs headlessly.

## Installation

```bash
pip install -e .
```

Or with Poetry:

```bash
poetry install
```

This installs the `darktriage` console script.

## Usage

```bash
# Score a single image
darktriage score IMG_1234.ORF

# Triage an entire RAW folder, writing XMP sidecars
darktriage triage ./raw_images --dry-run

# Detect burst sequences within a folder
darktriage burst ./raw_images

# Launch the TUI dashboard
darktriage gui

# Headless batch export via darktable-cli
darktriage export ./raw_images --style "My Style" --output ./exports
```

## Module Layout

```
darktriage/
├── src/darktriage/
│   ├── __init__.py
│   ├── __main__.py
│   ├── cli.py                  # CLI argument parser (argparse/typer)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── exif_reader.py      # EXIF parsing & burst detection
│   │   ├── metrics.py          # OpenCV Tenengrad, composition & exposure algorithms
│   │   └── xmp_writer.py       # ExifTool sidecar writer wrapper
│   ├── tui/
│   │   ├── __init__.py
│   │   ├── app.py              # Textual TUI Application instance
│   │   └── widgets.py          # Custom TUI components (image list, metrics view)
│   └── export/
│       ├── __init__.py
│       └── darktable_cli.py    # Subprocess runner for darktable-cli batch rendering
├── tests/
│   ├── test_metrics.py
│   └── test_exif.py
├── .github/workflows/python-app.yml
├── docs/ARCHITECTURE.md
├── pyproject.toml
└── README.md
```

## Development

```bash
poetry install --with dev
poetry run pytest
```

## License

MIT
