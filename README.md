# DarkTriage (`darktriage`)

A deterministic, terminal-native (CLI + TUI) pre-import photo triage tool and headless batch-processing engine for RAW photographers using Darktable.

DarkTriage scores camera JPEG previews in memory with local computer-vision heuristics (OpenCV) plus EXIF metadata, then writes 1–5 star ratings and colour labels into Darktable-compatible `.xmp` sidecars — before you ever open Darktable. It is **non-destructive by default** and built to chew through large RAW+JPEG batches (e.g. 36 MP Nikon D800E) at high throughput.

## Features

- **Fast DCT-scaled JPEG decoding** — decodes previews at 1/2, 1/4 or 1/8 scale directly from JPEG DCT blocks (OpenCV reduced flags, PyTurboJPEG when available), ~4–6× faster than full decode + resize.
- **Micro-focus scoring** — Tenengrad (Sobel) energy on a central region-of-interest, with an ISO-driven noise pre-filter and a blur **early-exit** that skips work on soft frames.
- **Composition & exposure** — Rule-of-Thirds energy distribution, highlight clipping, plus a recommended EV offset written to `XMP-dc:Description`.
- **Burst detection** — clusters continuous-drive shots by sub-second capture time and awards the sharpest frame in each burst a bonus star.
- **Native Darktable XMP** — writes `XmpRating` / `XmpLabel` sidecars via `exiftool`; never touches RAW/JPEG binaries.
- **Subject auto-tagging** — local heuristics (Haar faces + sky/vegetation colour) tag `people`/`portrait`/`group`/`landscape`/`nature`/`night` into XMP keywords, no model download.
- **Non-destructive safety** — existing sidecars with a user rating or darktable history are preserved unless `--overwrite`; optional `.xmp.bak` backups.
- **Bounded-memory parallel engine** — `ProcessPoolExecutor` across all cores with `itertools.islice` chunking.
- **Optional embedded-preview scoring** — score the small JPEG embedded in each RAW instead of the full Fine JPEG (large speedups on big batches).
- **Interactive TUI** — Textual dashboard with live progress, metric breakdowns, and a bulk "Commit XMP sidecars" button.
- **Headless export** — `darktable-cli` batch renderer with chained styles and custom resolution.
- **Configurable** — a TOML config file controls thresholds, labels, safety and the darktable path.

## Requirements

- Python 3.10+
- [`exiftool`](https://exiftool.org/) — required to write XMP sidecars (`analyze`, TUI commit) and to extract embedded previews.
- `darktable-cli` — only required for `darktriage export`.

```bash
# macOS (Homebrew)
brew install exiftool
brew install --cask darktable

# Debian/Ubuntu
sudo apt install libimage-exiftool-perl darktable
```

## Installation

Homebrew Python is externally managed (PEP 668), so install into a virtual environment:

```bash
git clone https://github.com/patback66/darktriage.git
cd darktriage
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Or with Poetry (the project is Poetry-configured):

```bash
poetry install
poetry run darktriage --help
```

Either way you get the `darktriage` console script.

## Quick start

```bash
# Preview ratings for a folder without writing anything
darktriage analyze "/path/to/shoot" --dry-run

# Score the folder and write .xmp sidecars next to each RAW
darktriage analyze "/path/to/shoot"

# Interactive dashboard (browse metrics, then commit sidecars)
darktriage tui "/path/to/shoot"

# Headless render of the keepers through a darktable style
darktriage export "/path/to/shoot" --min-rating 4 --style "My Style"
```

See [Recommended workflow](#recommended-workflow) for the full first-run → cull → export loop.

## Recommended workflow

DarkTriage is a **pre-import → cull → export** loop: score previews, calibrate the rating bands to your camera, write non-destructive `.xmp` sidecars, then render the keepers with darktable.

| Step | Command | Writes |
| --- | --- | --- |
| Warm preview cache | `previews extract` | preview cache |
| Calibrate thresholds | `calibrate` | `config.toml` (with `--write`) |
| Review the mix | `analyze --dry-run` | nothing |
| Commit ratings | `analyze` | `.xmp` sidecars |
| Render keepers | `export` | JPEGs in `exported/` |

### 1. First run on a new folder (cold cache)

```bash
SHOOT="/path/to/shoot"

# a) Warm the embedded-preview cache (optional, makes the first analyze fast)
darktriage --threads 8 previews extract "$SHOOT"

# b) Calibrate the rating bands for this camera/source (do once per source/season)
darktriage --threads 8 calibrate "$SHOOT" --sample 300 --use-embedded-preview
#    review the distribution + suggestions, then persist them:
darktriage --threads 8 calibrate "$SHOOT" --sample 300 --use-embedded-preview --write

# c) Preview the ratings + rating/label histogram (writes nothing)
darktriage analyze "$SHOOT" --use-embedded-preview --dry-run

# d) Commit the sidecars (existing ratings/history are preserved)
darktriage analyze "$SHOOT" --use-embedded-preview
```

### 2. Every subsequent run (warm cache)

```bash
darktriage analyze "$SHOOT" --use-embedded-preview
# add --overwrite to recompute sidecars that already have ratings/history
```

### 3. Review in Darktable, then export the keepers

Open the folder in Darktable (it imports the star ratings and colour labels), cull as usual, then render the keepers headlessly:

```bash
darktriage export "$SHOOT" --min-rating 4 --style "Landscape" --style "Sharpen"
```

### TUI alternative

Prefer a dashboard? Same pipeline, interactive commit:

```bash
darktriage tui "$SHOOT" --use-embedded-preview
```

Browse the file list, check the selected frame's metric breakdown, confirm the summary rating/label histogram, then click **Commit XMP sidecars**.

### When to re-calibrate

- A different camera/sensor, season/lighting, or a big ISO-range change.
- Whenever the printed histogram looks off (too many 4★, too few 1★, no 5★) — adjust `--sharp-percentile` / `--blur-percentile` / `--composition-percentile` / `--clip-percentile` and re-run with `--write`.
- Always calibrate on the **same source** you analyse with (`--use-embedded-preview` vs Fine JPEG).

### Safety in the loop

- `analyze` writes **only** `.NEF.xmp` sidecars; RAW/JPEG binaries are never modified.
- Existing sidecars with a user rating or darktable history are **skipped** unless you pass `--overwrite` (and are backed up to `.xmp.bak` when `create_backup = true`).
- `--dry-run` (analyze) and `--dry-run` (export) plan everything without writing.

## Global options

These go **before** the subcommand:

| Option | Description |
| --- | --- |
| `--threads N` | Worker processes for scoring (default: CPU count). |
| `--config PATH` | Use an alternate `config.toml`. |
| `--debug` | Enable debug logging. |

```bash
darktriage --threads 8 analyze "/path/to/shoot"
```

## Commands

### `darktriage analyze <dir>`

Scores every `.NEF`/`.JPG` pair in `<dir>` and, unless `--dry-run`, writes a `<file>.NEF.xmp` sidecar for each RAW (rating + label + EV note).

| Option | Description |
| --- | --- |
| `--dry-run` | Print the ratings table without writing sidecars. |
| `--overwrite` | Allow modifying existing sidecars (otherwise files with a user rating or darktable history are preserved). |
| `--burst-threshold SEC` | Burst grouping window in seconds (default from config, `2.0`). |
| `--max-dim PX` | Proxy long-edge size for scoring (default `1920`). |
| `--use-embedded-preview` | Score the small JPEG embedded in each RAW instead of the full Fine JPEG (needs `exiftool`; falls back per-file, and previews are cached). |
| `--chunk-size N` | Files per parallel work chunk, bounds memory (default `256`). |
| `--preview-cache-dir PATH` | Override the preview cache (default: per-folder `$XDG_CACHE_HOME/darktriage/previews`, reused across runs). |
| `--autotag` | Detect subject tags and write them to `dc:Subject` (additive; existing keywords kept). See [Subject auto-tagging](#subject-auto-tagging). |

```bash
# Safe preview first, then commit
darktriage analyze "/path/to/shoot" --dry-run
darktriage analyze "/path/to/shoot"

# Force-recompute sidecars, 8 workers, faster preview path
darktriage --threads 8 analyze "/path/to/shoot" \
  --use-embedded-preview --overwrite
```

After the table it prints a one-line **rating/label histogram** (e.g. `1★ 15% · 3★ 60% · 4★ 24% · 5★ 1%`, plus label counts) so you can sanity-check the calibrated mix at a glance.

### `darktriage tui <dir>`

Launches the interactive Textual dashboard. Pairing/EXIF/scoring run in a background worker so the UI stays responsive; a progress bar shows scoring, the file list and metric breakdown populate when analysis finishes, and **Commit XMP sidecars** writes everything at once (respecting the non-destructive rules).

| Option | Description |
| --- | --- |
| `--burst-threshold SEC` | Burst window in seconds. |
| `--max-dim PX` | Proxy long-edge size (default `1920`). |
| `--use-embedded-preview` | Score embedded RAW previews instead of Fine JPEGs. |
| `--autotag` | Detect subject tags and include them when committing sidecars. |
| `--chunk-size N` | Files per parallel work chunk. |

Press `q` to quit. After analysis the dashboard shows the same one-line rating/label histogram, and the selected file's full metric breakdown.

### `darktriage export <dir>`

Renders RAWs whose `.xmp` rating is `>= --min-rating` through `darktable-cli`, writing JPEGs into an **isolated** output directory (default `<dir>/exported`) so source files are never overwritten.

| Option | Description |
| --- | --- |
| `--min-rating N` | Minimum star rating to export (default `4`). |
| `--style NAME` | darktable style to apply — **repeatable** for chained styles. |
| `--width PX` / `--height PX` | Explicit output resolution. |
| `--output DIR` | Output directory (default `<dir>/exported`). |
| `--dry-run` | Print the plan without invoking `darktable-cli`. |

```bash
darktriage export "/path/to/shoot" --min-rating 4 \
  --style "Landscape" --style "Sharpen" --width 2048

darktriage export "/path/to/shoot" --dry-run --min-rating 3
```

### `darktriage calibrate <dir>`

Derives rating thresholds from the **actual metric distribution** of a folder instead of the hard-coded defaults. It scores a sample with the blur early-exit disabled (so every image contributes sharpness, composition and clipping values), prints percentile tables, and suggests new `[thresholds]`.

| Option | Description |
| --- | --- |
| `--sample N` | Score at most N images (default `300`; `0` = all). |
| `--use-embedded-preview` | Calibrate against embedded previews (use this if you analyse with them). |
| `--preview-cache-dir PATH` | Override the preview cache. |
| `--max-dim PX` | Proxy long edge for scoring (default `1920`). |
| `--blur-percentile` | Sharpness percentile → `tenengrad_blur` (default `15`). |
| `--sharp-percentile` | Sharpness percentile → `tenengrad_sharp` (default `75`). |
| `--composition-percentile` | Composition percentile → `composition_min` (default `60`). |
| `--clip-percentile` | Clipping percentile → `highlight_clip_percent` (default `85`). |
| `--write [PATH]` | Write the suggested thresholds to a config (default: the standard config path), backing the old file up to `.bak`. |

```bash
# inspect the distribution + suggestions (no changes)
darktriage calibrate "/path/to/shoot" --sample 300

# calibrate for embedded-preview scoring and save to config.toml
darktriage calibrate "/path/to/shoot" --sample 500 --use-embedded-preview --write
```

> Calibrate on the **same source you analyse with** (Fine JPEG vs `--use-embedded-preview`): the embedded preview has a different resolution/compression, so its Tenengrad scale differs.

It also reports how many sampled frames scored **zero sharpness** (a sign of blank/failed preview extracts) so you can spot data problems before trusting the bands.

### `darktriage previews extract <dir>`

Warms the embedded-preview cache ahead of time so the first `analyze`/TUI run is already fast. Prints how many previews are available and where the cache lives.

```bash
darktriage previews extract "/path/to/shoot"
darktriage previews extract "/path/to/shoot" --preview-cache-dir /scratch/previews
```

Previews are cached under `${XDG_CACHE_HOME:-~/.cache}/darktriage/previews/<hash-of-folder>` and reused automatically by `--use-embedded-preview`.

## How scoring works

For each `.NEF`/`.JPG` pair, `analyze` runs this pipeline:

1. **Decode** the camera preview to grayscale at a reduced DCT scale (fast path), then **auto-rotate** it using the EXIF Orientation tag.
2. If **ISO > `sensor.iso_blur_above`** (default 1600), apply a 3×3 Gaussian noise pre-filter.
3. Compute **Tenengrad** (Sobel `Gx²+Gy²`) on the central `sensor.region_center_fraction` region (default 60%). If it is below `thresholds.tenengrad_blur`, the frame is flagged **soft/blur and exits early** (no composition/clipping work).
4. Otherwise compute **composition** (off-centre ÷ centre grid energy, clamped at 100 for stability) and **exposure** (mean luminance + clipped-highlight %), and derive a recommended **EV offset** `clamp(log2(128/mean), -2, +2)`.
5. **Burst grouping**: shots within `thresholds.burst_interval_sec` are clustered; for each **multi-frame** burst the sharpest frame is the **winner** and gets +1 star (lone frames are not "winners").

Ratings and labels are then derived from the configured thresholds:

| Condition | Rating | Label |
| --- | --- | --- |
| Multi-frame burst winner **and** `sharpness ≥ tenengrad_sharp` **and** `composition ≥ composition_min` | 5 | purple (portfolio) |
| `sharpness ≥ tenengrad_sharp` (crisp keeper) | 4 | green |
| In between (average) | 3 | — |
| `sharpness ≤ tenengrad_blur` (soft) | 1 | red |
| Highlight clipping `> highlight_clip_percent` | — | yellow (overrides the label) |

The EV offset is written to `XMP-dc:Description` for reference inside Darktable.

## Subject auto-tagging

`analyze --autotag` (and `tui --autotag`) derives subject tags from cheap, deterministic, local signals — no model downloads, no LLM:

- **Faces** (OpenCV's bundled Haar cascade) → `people`, `portrait` (one large face), `group` (2+ faces).
- **Scene colour/tonality** on a small colour proxy → `landscape` (sky present), `nature` (vegetation), `night` (low luminance).
- Faces take precedence; scene tags are only added when no faces are found.

Tags are written to the XMP `dc:subject` keyword list **additively** — existing user keywords are preserved and DarkTriage never removes them. They show up in darktable's tag list once it re-reads the sidecar.

```bash
darktriage analyze "$SHOOT" --use-embedded-preview --autotag --overwrite
```

Limits: object classes such as **cars** need a trained detector; these heuristics do not recognise vehicles (a pluggable OpenCV DNN hook is planned). Tagging is deliberately conservative — a tag is emitted only when a strong signal is present.

## Non-destructive safety

- **Sidecars only** — DarkTriage writes `<file>.NEF.xmp`; it never modifies `.NEF`/`.JPG` binaries.
- **Preserve existing edits** — a sidecar that already contains a user rating (>0) or `<darktable:history>` is left untouched unless you pass `--overwrite`.
- **Backups** — with `safety.create_backup = true`, an existing sidecar is copied to `<file>.NEF.xmp.bak` before it is rewritten.
- **Batched writes** — all sidecars for a run are written in a single `exiftool -@ argfile` pass.

## Configuration

DarkTriage reads `$XDG_CONFIG_HOME/darktriage/config.toml` (usually `~/.config/darktriage/config.toml`). Override the path with `--config PATH`. Missing keys fall back to these defaults:

```toml
[safety]
preserve_existing = true   # skip sidecars with user ratings / darktable history
create_backup = true       # write <file>.NEF.xmp.bak before overwriting
dry_run = false

[thresholds]
tenengrad_blur = 150.0     # <= this Tenengrad => soft/blur (1 star, red)
tenengrad_sharp = 350.0    # >= this Tenengrad => crisp keeper (4+ stars)
composition_min = 1.2      # min off-centre/centre energy ratio for the 5-star rule
highlight_clip_percent = 0.05   # fraction (0.05 = 5%) above which YELLOW is set
burst_interval_sec = 2.0   # continuous-drive burst window

[tags]
blur = "red"
keeper = "green"
portfolio = "purple"
exposure_warn = "yellow"

[darktable]
cli_path = "darktable-cli"
config_dir = "~/.config/darktable"
label_case = "title"           # write xmp:Label as Red/Yellow/Green/Purple

[sensor]
iso_blur_above = 1600          # apply 3x3 noise pre-filter above this ISO
region_center_fraction = 0.6   # central ROI used for focus/composition
```

Notes:
- Tenengrad thresholds depend on the proxy resolution (`--max-dim`) and image content; tune them against a folder of known keepers.
- `highlight_clip_percent` is a **fraction** (`0.05` = 5%); `composition_min` is an **energy ratio** (off-centre ÷ centre), not a percentage.
- Colour labels are written to XMP in **Adobe capitalisation** (`Red`/`Yellow`/`Green`/`Purple`) because darktable matches those names. Set `[darktable] label_case = "lower"` (or `"asis"`) if your darktable label names differ.
- `write_default_config()` (Python API) drops the template above to disk.

## Performance & benchmarks

`tests/benchmark_triage.py` measures decode throughput, scoring throughput, memory, and the end-to-end parallel engine.

**Synthetic batch** (self-contained, no fixtures):

```bash
python3 tests/benchmark_triage.py
python3 tests/benchmark_triage.py --images 8 --workers 4 --max-dim 1920
```

Useful flags: `--images`, `--workers`, `--max-dim`, `--frame-dim` (synthetic frame long edge), `--ref-samples` (cost of the reference full-decode baseline), `--keep` (keep the temp directory).

**Real camera folder** (skips generation, discovers your `.JPG`/`.NEF` pairs):

```bash
python3 tests/benchmark_triage.py \
  --dir "/Volumes/Koken Pass/Photos/Hokkaido 2026" \
  --images 40 --workers 8

# add --use-embedded-preview to benchmark scoring the RAW's embedded preview
python3 tests/benchmark_triage.py \
  --dir "/Volumes/Koken Pass/Photos/Hokkaido 2026" \
  --images 40 --workers 8 --use-embedded-preview

# also run phases 1-3 on the previews, matching the analyze path end-to-end
python3 tests/benchmark_triage.py \
  --dir "/Volumes/Koken Pass/Photos/Hokkaido 2026" \
  --images 40 --workers 8 --use-embedded-preview --bench-preview
```

The report covers four phases:

1. **Decode-only** — `prepare_proxy()` vs a reference full Pillow decode+resize (shows the DCT-scale speedup).
2. **Full scoring** — `score_image()` including the blur early-exit rate.
3. **Memory** — peak RSS and peak Python heap.
4. **Parallel engine** — whole-folder `analyze_directory()` end-to-end throughput.

Tips for reading the numbers:

- **Decode is entropy-bound.** JPEG reduced-scale decode barely changes cost with output size (1/2 ≈ 1/4 ≈ 1/8 in testing) because the entire compressed scan must be entropy-decoded. The dominant cost is the *file size* (a 36 MP Fine JPEG is ~24 MB), not the proxy resolution — lowering `--max-dim` won't make it much faster.
- **Embedded previews are the big lever.** `--use-embedded-preview` scores the ~1–3 MB JPEG most cameras embed in the RAW instead of the 24 MB Fine JPEG, cutting both I/O and decode. It requires `exiftool`, falls back per-file to the Fine JPEG when a RAW has no usable preview, and caches extracted previews under `$XDG_CACHE_HOME/darktriage/previews` so repeat runs skip extraction.
- **Parallel scaling** on a laptop is typically ~5× with 8 workers (performance + efficiency cores), not a clean 8×.

## Troubleshooting

- **`error: externally-managed-environment`** when running `pip install` — Homebrew Python is PEP 668 managed. Use a virtual environment (see [Installation](#installation)) or Poetry.
- **`ExifToolNotFoundError`** — `exiftool` is not on `PATH`; install it (`brew install exiftool` / `apt install libimage-exiftool-perl`).
- **`UnidentifiedImageError: .../._DSC1234.JPG`** — macOS AppleDouble sidecars. These are ignored automatically (names starting with `.`), so a folder copied to/from a Mac/exFAT volume is safe.
- **No speedup from `--use-embedded-preview`** — check the RAW actually contains a usable preview: `exiftool -PreviewImage -JpgFromRaw file.NEF | head`. On large folders the ExifTool extraction pass itself can dominate and offset the decode savings.
- **Ratings show in darktable but colour labels don't** — darktable matches Adobe capitalised names (`Red`/`Yellow`/`Green`/`Purple`). Ensure `[darktable] label_case = "title"`, re-run `analyze --overwrite` (existing sidecars are otherwise preserved), and confirm with `exiftool -Xmp:Label "<file>.NEF.xmp"`. If your darktable uses custom label names, set `label_case = "asis"` and put the exact names in `[tags]`.
- **darktable doesn't see newly written sidecars** — enable *Preferences → “look for updated XMP files on startup”* (or re-import into a fresh library), then restart darktable.

## Development

```bash
poetry install --with dev
poetry run pytest            # unit tests
poetry run python tests/benchmark_triage.py
```

## Module layout

```
src/darktriage/
├── __init__.py
├── __main__.py
├── cli.py                  # argparse CLI: analyze / tui / export
├── config.py               # TOML config loading + defaults
├── core/
│   ├── engine.py           # bounded parallel analysis engine (ProcessPool)
│   ├── calibration.py      # data-driven threshold calibration
│   ├── autotag.py          # subject auto-tagging (faces + scene heuristics)
│   ├── exif_reader.py      # EXIF parsing, pairing, burst detection
│   ├── metrics.py          # fast decode, Tenengrad, composition, exposure
│   ├── previews.py         # embedded RAW-preview extraction (ExifTool)
│   └── xmp_writer.py       # non-destructive XMP sidecar writer
├── tui/
│   ├── app.py              # Textual dashboard (background analysis)
│   └── widgets.py          # metric breakdown widget
└── export/
    └── darktable_cli.py    # headless darktable-cli batch renderer
tests/
├── benchmark_triage.py     # throughput / memory benchmark
├── test_config.py
├── test_engine.py
├── test_exif.py
├── test_export.py
├── test_metrics.py
├── test_previews.py
└── test_xmp_writer.py
```

## License

MIT



