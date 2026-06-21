# photo-extractor

Turn phone scans of physical photographs into clean, upright, color-normalized
digital copies — built for archiving old family prints and assembling a
wedding montage.

Point it at a folder of scans (JPEG, PNG, or iPhone HEIC) and it will, for each
photo:

1. **Detect & crop** the photograph out of the scan (floor / table / background
   removed) using closed-edge + adaptive-threshold contour detection.
2. **Perspective-correct** it into a flat rectangle (undoes camera tilt).
3. **Level residual tilt** (deskew) when the dominant edges clearly agree.
4. **Trim** any white matte / border.
5. **Tone-normalize** — gray-world white balance + contrast stretch + CLAHE — so
   a batch of mixed-era prints reads as one cohesive set.
6. **Auto-orient** to upright using face detection (YuNet).

Date-stamp removal is available as an experimental, opt-in step.

## Requirements

- macOS (the GUI's Tcl/Tk step is macOS-specific; the CLI works anywhere)
- [uv](https://docs.astral.sh/uv/) for Python dependency management
- For the **GUI only**: `brew install tcl-tk@8` (the uv Python ships a Tcl/Tk
  8.6 `_tkinter` but omits the 8.6 script library; the launcher wires it up
  automatically once tcl-tk@8 is installed)

## Quick start

```bash
git clone <this-repo> photo-extractor
cd photo-extractor
uv sync                       # install Python deps (opencv, pillow, pillow-heif)

# CLI — process a folder of scans
uv run photo_extractor.py raw/ -o out/

# GUI — double-click the launcher, or:
./run_gui.command
```

## CLI usage

```
uv run photo_extractor.py <inputs> -o <output> [options]
```

`<inputs>` is one or more image files or folders (folders are searched
recursively). Outputs are written as JPEG, named after each scan.

| Flag | Default | Effect |
|------|---------|--------|
| `-o, --output` | required | output directory |
| `--no-enhance` | on | skip tone normalization |
| `--no-crop-white` | on | skip white-matte trimming |
| `--no-orient` | on | skip face-based auto-orientation |
| `--no-deskew` | on | skip residual-tilt leveling |
| `--remove-dates` | **off** | attempt printed date-stamp removal (experimental — review outputs) |
| `-q, --quality` | 95 | JPEG quality (1–100) |

Examples:

```bash
uv run photo_extractor.py raw/ -o out/                 # sensible defaults
uv run photo_extractor.py raw/ -o out/ --remove-dates   # also try date cleanup
uv run photo_extractor.py scan.heic -o out/ --no-enhance
```

## GUI

`./run_gui.command` opens a desktop window (Tkinter):

- Pick input/output folders, toggle the pipeline options, set JPEG quality.
- **Process** runs the batch in the background with a live progress bar; the
  results gallery populates automatically.
- **Refresh gallery** reloads from the output folder; **Open output folder**
  reveals it in Finder.
- Double-click any thumbnail to open it full-size in your system viewer.

The GUI reuses the exact same pipeline as the CLI — no duplicated logic.

## Known limitations

- **Multi-photo scans** aren't reliably split. If two photos are in one frame,
  you'll usually get one combined image. Reshoot one-photo-per-frame for best
  results.
- **Auto-orientation** is face-based, so a single-subject photo with no clear
  face or text can stay sideways/upside-down. (OCR would fix the rest — the
  printed date stamps are a reliable text signal — but isn't bundled.)
- **Date-stamp removal** is opt-in and best-effort: classical CV can't always
  distinguish a date from other small corner detail, so it's off by default.
  Enable with `--remove-dates` and review the results.

## Diagnostics

`uv run python diagnose.py` processes `raw/` and writes a thumbnail of each
scan with the detected photo outline drawn in green (into `diag_out/`), handy
for sanity-checking detection on a new batch.

## Project layout

```
photo_extractor.py   # extraction + enhancement pipeline (library + CLI)
gui.py               # Tkinter batch + gallery GUI
run_gui.command      # macOS launcher (wires up tcl-tk@8 for the GUI)
diagnose.py          # detection visualization helper
make_test_scan.py    # generates a synthetic tilted scan for regression tests
pyproject.toml       # uv dependencies
```
