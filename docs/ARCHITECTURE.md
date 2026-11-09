# Architecture

DarkTriage is organized as a deterministic, terminal-native pipeline: **scan → EXIF decode → metric scoring → rating + XMP sidecar → optional headless export**.

## Directory layout

```
src/darktriage/
├── cli.py                  # entry point + CLI dispatch
├── core/
│   ├── exif_reader.py      # EXIF decode + burst sequence grouping
│   ├── metrics.py          # OpenCV Tenengrad / composition / exposure
│   └── xmp_writer.py       # ExifTool wrapper (Rating + Label tags)
├── tui/
│   ├── app.py              # Textual application
│   └── widgets.py          # image list + metrics view widgets
└── export/
    └── darktable_cli.py    # darktable-cli batch renderer
```

## Dataflow

1. **Scanner** (`cli.py`) enumerates RAW/JPEG files in the target directory.
2. **EXIF reader** (`core/exif_reader.py`) decodes `DateTimeOriginal`, `OffsetTimeOriginal`, camera metadata, and any `UserComment` payloads; it groups files into burst sequences by tight timestamps.
3. **Proxy image scorer** (`core/metrics.py`) decodes JPEG preview data (via Pillow) to a NumPy RGB array and runs:
   - **Tenengrad sharpness** — squared Sobel gradient magnitude summed over the image.
   - **Rule-of-thirds composition** — luminance and local contrast sampled at the four grid nodes and nine cells.
   - **Exposure / highlight clipping** — mean luminance, highlight percentile, and clipping ratio.
4. **Rating engine** maps metrics to a 1–5 star rating plus a color label (Red/Yellow/Green/Purple).
5. **XMP writer** (`core/xmp_writer.py`) invokes `exiftool` to write `XmpRating` and `XmpLabel` into the same folder as the source file.
6. **Exporter** (`export/darktable_cli.py`) shells out to `darktable-cli` for headless batch rendering.

## Design notes

- All scoring is deterministic and order-independent: the same input bytes always produce the same scores.
- EXIF metadata, not pixel heuristics, defines the burst clock; this keeps frame grouping accurate even when two consecutive shots are near-identical in content.
- Metric thresholds (Tenengrad floor, highlight clip threshold, node-vs-edge contrast) are tunable via the TUI and passed to scoring helpers.
