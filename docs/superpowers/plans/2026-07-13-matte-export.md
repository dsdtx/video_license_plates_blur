# Luma-Matte Export Mode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `--mode matte` to `blur_plates.py` that exports a white-on-black luma matte (ProRes `.mov`, or HEVC `.mp4` via `--matte-codec hevc`) instead of baking the redaction into the footage, so editors composite the blur in their own NLE.

**Architecture:** Reuse the existing decode → detect → track pipeline unchanged. At the redaction site, when mode is `matte`, render a synthetic black frame with white filled rects at each plate region instead of blurring the source. At the output stage, pipe those frames straight into a ProRes/HEVC encoder writing the final file, skipping the FFV1 lossless intermediate and the `mux_audio()` step.

**Tech Stack:** Python 3.12, OpenCV (`cv2`), NumPy, ffmpeg (`prores_videotoolbox`/`prores_ks`/`hevc_nvenc`/`libx265`), pytest.

---

## Environment Notes (this machine)

- Use the project venv: `./venv/bin/python` and `./venv/bin/pip`.
- This session's shell has a **stale PATH** that predates the ffmpeg PATH fix in
  `~/.zprofile`. For any command that runs ffmpeg/ffprobe (tests included), prefix:
  `export PATH="$HOME/Documents/ffmpeg:$PATH"` — e.g.
  `export PATH="$HOME/Documents/ffmpeg:$PATH"; ./venv/bin/python -m pytest ...`.
- Work happens on branch `feature/matte-export` (already created).

## Reference: existing code touchpoints

- Redaction helpers + dispatcher: `blur_plates.py:577-659` (`apply_blur`, `apply_solid_color`, `apply_image_overlay`, `apply_redaction`). Rect shape is `x1,y1,x2,y2 = rect[:4]`.
- ffmpeg command builders: `blur_plates.py:765-815` (`build_ffmpeg_extract`, `build_ffmpeg_encode_lossless`).
- Encoder availability probe (nested): `blur_plates.py:932-937` inside `mux_audio`.
- Main processing fn `blur_license_plates` signature: `blur_plates.py:1918-1921` (end of kwargs).
- Output banner: `blur_plates.py:1938`.
- Encode-command setup: `blur_plates.py:2018-2024`.
- Production render site: `blur_plates.py:2126-2132` (`elif plates: apply_redaction(...)`).
- Final mux call: `blur_plates.py:2171-2175`.
- `tmp_path` cleanup: `blur_plates.py:2195-2197` (removes it even if unused → no leak for matte).
- argparse `--mode`/`--color`/`--image`: `blur_plates.py:2275-2290`.
- `main()` → `blur_license_plates(...)` call: `blur_plates.py:2313-2348`.
- Config `[redact]`: `config.toml:107-124`.

---

## Task 1: Test scaffolding

**Files:**
- Modify: project venv (install pytest)
- Create: `tests/__init__.py`
- Create: `tests/conftest.py`

- [ ] **Step 1: Install pytest into the venv**

Run: `./venv/bin/pip install pytest`
Expected: `Successfully installed pytest-...`

- [ ] **Step 2: Create the tests package**

Create `tests/__init__.py` (empty file).

- [ ] **Step 3: Add a conftest that puts the repo root on sys.path**

Create `tests/conftest.py`:

```python
import os
import sys

# Make blur_plates importable when running pytest from the repo root.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
```

- [ ] **Step 4: Verify pytest collects nothing yet (sanity)**

Run: `./venv/bin/python -m pytest tests/ -q`
Expected: `no tests ran` (exit code 5) — confirms collection works with no errors.

- [ ] **Step 5: Commit**

```bash
git add tests/__init__.py tests/conftest.py
git commit -m "test: add pytest scaffolding for matte feature"
```

---

## Task 2: `render_matte_frame()`

**Files:**
- Modify: `blur_plates.py` (add function after `apply_image_overlay`, around line 641)
- Test: `tests/test_matte_render.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matte_render.py`:

```python
import numpy as np
from blur_plates import render_matte_frame


def test_hard_edges_white_inside_black_outside():
    m = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=0, feather=0)
    assert m.shape == (100, 100, 3)
    assert m.dtype == np.uint8
    # Clearly inside the box → white
    assert (m[50, 50] == [255, 255, 255]).all()
    # Clearly outside the box → black
    assert (m[10, 10] == [0, 0, 0]).all()


def test_padding_expands_white_region():
    m = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=5, feather=0)
    # 37 is inside the 5px-padded region (starts at x=35), still white
    assert (m[50, 37] == 255).all()
    # 30 is outside the padded region, still black
    assert (m[50, 30] == 0).all()


def test_empty_rects_is_all_black():
    m = render_matte_frame(64, 48, [], padding=8, feather=0)
    assert m.shape == (64, 48, 3)
    assert m.max() == 0


def test_feather_introduces_intermediate_values():
    hard = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=0, feather=0)
    soft = render_matte_frame(100, 100, [(40, 40, 60, 60)], padding=0, feather=5)
    # Hard matte is purely 0 or 255; feathered matte has grey edge pixels.
    assert set(np.unique(hard)).issubset({0, 255})
    assert ((soft > 0) & (soft < 255)).any()


def test_rect_with_extra_elements_ignored():
    # Plates may carry source tags beyond the first 4 coords; only coords matter.
    m = render_matte_frame(100, 100, [(40, 40, 60, 60, "sahi", 0.9)], padding=0)
    assert (m[50, 50] == 255).all()
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_matte_render.py -q`
Expected: FAIL — `ImportError: cannot import name 'render_matte_frame'`.

- [ ] **Step 3: Implement `render_matte_frame`**

In `blur_plates.py`, immediately after `apply_image_overlay` ends (line 640, before the blank line at 641), add:

```python
def render_matte_frame(height, width, rects, padding=8, feather=0):
    """
    Build a luma-matte frame: a black canvas with white filled rectangles at each
    (padded) plate region.  Used by redaction mode "matte" to export a matte for
    compositing the redaction in an external NLE instead of baking it into the
    footage.

    rects entries follow the same shape as the redaction helpers: the first four
    values are x1, y1, x2, y2 (any further elements — e.g. source tags — ignored).

    feather > 0 softens the whole matte with a Gaussian of that radius
    (kernel size 2*feather + 1), so the driven blur can fade at plate edges.
    """
    matte = np.zeros((height, width, 3), dtype=np.uint8)
    for rect in rects:
        x1, y1, x2, y2 = rect[:4]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(width, x2 + padding)
        y2 = min(height, y2 + padding)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(matte, (x1, y1), (x2, y2), (255, 255, 255), -1)
    if feather and feather > 0:
        k = int(feather) * 2 + 1
        matte = cv2.GaussianBlur(matte, (k, k), 0)
    return matte
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_matte_render.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add blur_plates.py tests/test_matte_render.py
git commit -m "feat: add render_matte_frame for luma-matte export"
```

---

## Task 3: `resolve_matte_output_path()`

**Files:**
- Modify: `blur_plates.py` (add helper near the other module-level helpers, after `render_matte_frame`)
- Test: `tests/test_matte_paths.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matte_paths.py`:

```python
from blur_plates import resolve_matte_output_path


def test_prores_forces_mov():
    assert resolve_matte_output_path("out.mp4", "prores") == "out.mov"


def test_prores_keeps_mov():
    assert resolve_matte_output_path("clip.mov", "prores") == "clip.mov"


def test_hevc_forces_mp4():
    assert resolve_matte_output_path("out.mkv", "hevc") == "out.mp4"


def test_hevc_keeps_mp4():
    assert resolve_matte_output_path("clip.mp4", "hevc") == "clip.mp4"


def test_preserves_directory_and_stem():
    assert resolve_matte_output_path("/a/b/matte.mxf", "prores") == "/a/b/matte.mov"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `./venv/bin/python -m pytest tests/test_matte_paths.py -q`
Expected: FAIL — `ImportError: cannot import name 'resolve_matte_output_path'`.

- [ ] **Step 3: Implement `resolve_matte_output_path`**

In `blur_plates.py`, directly after the `render_matte_frame` function you added in Task 2, add:

```python
def resolve_matte_output_path(output_path, codec):
    """
    Return an output path whose container extension matches the matte codec:
    'prores' → .mov, anything else ('hevc') → .mp4.  If the supplied path uses a
    different extension it is replaced (a single file at the corrected path) and a
    note is printed, so ProRes/HEVC never lands in a mismatched container.
    """
    ext = ".mov" if codec == "prores" else ".mp4"
    root, cur = os.path.splitext(output_path)
    if cur.lower() != ext:
        corrected = root + ext
        print(f"  Note: --matte-codec {codec} writes {ext}; "
              f"output path changed to {corrected}")
        return corrected
    return output_path
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `./venv/bin/python -m pytest tests/test_matte_paths.py -q`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add blur_plates.py tests/test_matte_paths.py
git commit -m "feat: add resolve_matte_output_path extension helper"
```

---

## Task 4: ffmpeg matte encoders

**Files:**
- Modify: `blur_plates.py` (add module-level `_ffmpeg_encoder_available`; add two builders after `build_ffmpeg_encode_lossless` at line 815; refactor nested `_nvenc_available` at line 932)
- Test: `tests/test_matte_encode.py`

Note: we **probe encoder availability up front** (not "encode then retry"), because the
frame stream is consumed once and cannot be replayed to a fallback encoder mid-run.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_matte_encode.py`:

```python
import shutil
import subprocess

import numpy as np
import pytest

from blur_plates import (
    build_ffmpeg_encode_prores,
    build_ffmpeg_encode_hevc_matte,
)


def test_prores_command_targets_mov_full_range():
    cmd = build_ffmpeg_encode_prores(1920, 1080, 30, "out.mov")
    assert cmd[0] == "ffmpeg"
    assert "pipe:0" in cmd
    assert "-color_range" in cmd and cmd[cmd.index("-color_range") + 1] == "pc"
    assert cmd[-1] == "out.mov"
    # Some ProRes encoder is selected (hardware or CPU).
    assert any(c in cmd for c in ("prores_videotoolbox", "prores_ks"))
    assert "yuv422p10le" in cmd


def test_hevc_command_targets_mp4_full_range():
    cmd = build_ffmpeg_encode_hevc_matte(1280, 720, 25, "out.mp4")
    assert cmd[0] == "ffmpeg"
    assert "pipe:0" in cmd
    assert "-color_range" in cmd and cmd[cmd.index("-color_range") + 1] == "pc"
    assert cmd[-1] == "out.mp4"
    assert any(c in cmd for c in ("hevc_nvenc", "libx265"))
    assert "yuv420p" in cmd


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_prores_encode_roundtrip(tmp_path):
    out = str(tmp_path / "m.mov")
    w, h, fps, n = 64, 48, 5, 3
    cmd = build_ffmpeg_encode_prores(w, h, fps, out)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame = np.zeros((h, w, 3), np.uint8)
    frame[10:20, 10:20] = 255
    for _ in range(n):
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", out],
        capture_output=True, text=True).stdout.strip()
    assert codec == "prores"
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `export PATH="$HOME/Documents/ffmpeg:$PATH"; ./venv/bin/python -m pytest tests/test_matte_encode.py -q`
Expected: FAIL — `ImportError: cannot import name 'build_ffmpeg_encode_prores'`.

- [ ] **Step 3: Add `_ffmpeg_encoder_available` and the two builders**

In `blur_plates.py`, immediately after `build_ffmpeg_encode_lossless` ends (line 815), add:

```python
def _ffmpeg_encoder_available(encoder: str) -> bool:
    """True if the named ffmpeg video encoder can be initialised on this machine."""
    r = subprocess.run(
        ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc",
         "-t", "0", "-c:v", encoder, "-f", "null", "-"],
        capture_output=True)
    return r.returncode == 0


def build_ffmpeg_encode_prores(width, height, fps, out_path):
    """
    Build ffmpeg command: raw BGR frames from stdin → ProRes 422 HQ (.mov),
    full-range luma.  Prefers the hardware VideoToolbox encoder (Apple platforms),
    falling back to the portable CPU prores_ks encoder.  Availability is probed
    up front because the frame stream cannot be replayed to a fallback mid-run.
    """
    if _ffmpeg_encoder_available("prores_videotoolbox"):
        codec_args = ["-c:v", "prores_videotoolbox", "-profile:v", "hq"]
    else:
        codec_args = ["-c:v", "prores_ks", "-profile:v", "3"]
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        *codec_args,
        "-pix_fmt", "yuv422p10le",
        "-color_range", "pc",
        out_path,
    ]


def build_ffmpeg_encode_hevc_matte(width, height, fps, out_path):
    """
    Build ffmpeg command: raw BGR frames from stdin → near-lossless HEVC (.mp4),
    full-range.  Performance escape hatch for PC/NVIDIA users where CPU ProRes is
    too slow: prefers GPU hevc_nvenc, falling back to CPU libx265.  A luma matte
    survives HEVC cleanly (signal is in luma; chroma is flat).
    """
    if _ffmpeg_encoder_available("hevc_nvenc"):
        codec_args = ["-c:v", "hevc_nvenc", "-rc", "vbr", "-cq", "12", "-preset", "p4"]
    else:
        codec_args = ["-c:v", "libx265", "-crf", "12", "-preset", "medium"]
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}", "-r", str(fps),
        "-i", "pipe:0",
        *codec_args,
        "-tag:v", "hvc1",
        "-pix_fmt", "yuv420p",
        "-color_range", "pc",
        out_path,
    ]
```

- [ ] **Step 4: Refactor the nested `_nvenc_available` to reuse the helper (DRY)**

In `blur_plates.py`, replace the nested function at lines 932-937:

```python
        def _nvenc_available():
            r = subprocess.run(
                ["ffmpeg", "-hide_banner", "-f", "lavfi", "-i", "nullsrc",
                 "-t", "0", "-c:v", "hevc_nvenc", "-f", "null", "-"],
                capture_output=True)
            return r.returncode == 0
```

with:

```python
        def _nvenc_available():
            return _ffmpeg_encoder_available("hevc_nvenc")
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `export PATH="$HOME/Documents/ffmpeg:$PATH"; ./venv/bin/python -m pytest tests/test_matte_encode.py -q`
Expected: PASS (3 passed) — on this Mac the ProRes roundtrip uses `prores_videotoolbox`.

- [ ] **Step 6: Run the full suite (regression check on the refactor)**

Run: `export PATH="$HOME/Documents/ffmpeg:$PATH"; ./venv/bin/python -m pytest tests/ -q`
Expected: PASS (all tests from Tasks 2-4).

- [ ] **Step 7: Commit**

```bash
git add blur_plates.py tests/test_matte_encode.py
git commit -m "feat: add ProRes/HEVC matte encoders and shared encoder probe"
```

---

## Task 5: Wire matte into `blur_license_plates`

**Files:**
- Modify: `blur_plates.py` (signature ~1918; output-path resolve near top ~1923; encode-cmd branch ~2019; render branch ~2126; skip mux ~2171)

No new unit test here (it's orchestration glue exercised by the end-to-end run in Task 9). Each step is a precise edit.

- [ ] **Step 1: Add the two new kwargs to the signature**

In `blur_plates.py`, at the end of the `blur_license_plates` kwargs (line 1920, the `redact_image_path: str = None,` line), change:

```python
    redact_mode: str = "blur",
    redact_color: tuple = (0, 0, 0),
    redact_image_path: str = None,
):
```

to:

```python
    redact_mode: str = "blur",
    redact_color: tuple = (0, 0, 0),
    redact_image_path: str = None,
    matte_feather: int = 0,
    matte_codec: str = "prores",
):
```

- [ ] **Step 2: Resolve the matte output path before the banner prints**

In `blur_plates.py`, right after the `tmp_dir` normalisation block (after line 1923, `tmp_dir = os.path.join(tempfile.gettempdir(), "plate-blur-tmp")`), add:

```python
    # Matte mode dictates the container; correct the extension before we print it.
    if redact_mode == "matte":
        output_path = resolve_matte_output_path(output_path, matte_codec)
```

- [ ] **Step 3: Select the encoder command for matte vs. normal**

In `blur_plates.py`, replace line 2019:

```python
        encode_cmd  = build_ffmpeg_encode_lossless(enc_width, enc_height, fps, tmp_path)
```

with:

```python
        if redact_mode == "matte":
            if matte_codec == "hevc":
                encode_cmd = build_ffmpeg_encode_hevc_matte(
                    enc_width, enc_height, fps, output_path)
            else:
                encode_cmd = build_ffmpeg_encode_prores(
                    enc_width, enc_height, fps, output_path)
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        else:
            encode_cmd = build_ffmpeg_encode_lossless(
                enc_width, enc_height, fps, tmp_path)
```

- [ ] **Step 4: Render the matte frame at the production render site**

In `blur_plates.py`, at the render site (line 2105, the `if debug_overlay:` that begins the render branch), insert a matte branch **before** it so it becomes the first condition. Change:

```python
                    if debug_overlay:
                        frame = draw_extended_overlay(
```

to:

```python
                    if redact_mode == "matte":
                        frame = render_matte_frame(
                            frame.shape[0], frame.shape[1], plates or [],
                            padding=blur_padding, feather=matte_feather)
                    elif debug_overlay:
                        frame = draw_extended_overlay(
```

Then change the following `elif debug:` and `elif plates:` stay as-is (they are now
later branches). Verify the chain reads: `if redact_mode == "matte": … elif debug_overlay: … elif debug: … elif plates: …`.

- [ ] **Step 5: Skip the mux for matte (output already written)**

In `blur_plates.py`, replace lines 2171-2175:

```python
        print("  Encoding final output (lossless HEVC + audio sync fix)...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mux_audio(tmp_path, input_path, output_path, start_time, end_time, fps,
                  total_frames=frame_num, preset=preset, tmp_dir=tmp_dir,
                  quality=quality)
```

with:

```python
        if redact_mode == "matte":
            print(f"  Matte written directly to {output_path} "
                  f"(codec: {matte_codec}, no audio, no HEVC mux).")
        else:
            print("  Encoding final output (lossless HEVC + audio sync fix)...")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            mux_audio(tmp_path, input_path, output_path, start_time, end_time, fps,
                      total_frames=frame_num, preset=preset, tmp_dir=tmp_dir,
                      quality=quality)
```

- [ ] **Step 6: Byte-compile to catch syntax/indentation errors**

Run: `./venv/bin/python -m py_compile blur_plates.py && echo OK`
Expected: `OK`

- [ ] **Step 7: Commit**

```bash
git add blur_plates.py
git commit -m "feat: wire matte mode through the processing/output pipeline"
```

---

## Task 6: CLI + main() wiring

**Files:**
- Modify: `blur_plates.py` (argparse ~2275-2290; main() call ~2345-2347)

- [ ] **Step 1: Add `matte` to `--mode` choices and add the two flags**

In `blur_plates.py`, replace the `--mode` argument (lines 2275-2280):

```python
    parser.add_argument("--mode", dest="mode",
                        default=red.get("mode", "blur"),
                        choices=["blur", "color", "image"],
                        help="Redaction style applied to detected plates "
                             "(default: blur). "
                             "color = solid fill, image = stretched overlay.")
```

with:

```python
    parser.add_argument("--mode", dest="mode",
                        default=red.get("mode", "blur"),
                        choices=["blur", "color", "image", "matte"],
                        help="Redaction style applied to detected plates "
                             "(default: blur). "
                             "color = solid fill, image = stretched overlay, "
                             "matte = export a white-on-black luma matte (no blur "
                             "baked in; footage is not re-encoded).")
    parser.add_argument("--matte-feather", dest="matte_feather", type=int,
                        default=int(red.get("matte_feather", 0)),
                        metavar="N",
                        help="For --mode matte: Gaussian edge-softening radius in "
                             "pixels (default: 0 = hard edges).")
    parser.add_argument("--matte-codec", dest="matte_codec",
                        default=red.get("matte_codec", "prores"),
                        choices=["prores", "hevc"],
                        help="For --mode matte: output codec. prores → ProRes 422 "
                             "HQ .mov (default). hevc → near-lossless HEVC .mp4 "
                             "(GPU-accelerated via NVENC where available).")
```

- [ ] **Step 2: Pass the new kwargs from `main()`**

In `blur_plates.py`, replace the tail of the `blur_license_plates(...)` call (lines 2345-2347):

```python
        redact_mode=args.mode,
        redact_color=redact_color,
        redact_image_path=args.image if args.mode == "image" else None,
    )
```

with:

```python
        redact_mode=args.mode,
        redact_color=redact_color,
        redact_image_path=args.image if args.mode == "image" else None,
        matte_feather=args.matte_feather,
        matte_codec=args.matte_codec,
    )
```

- [ ] **Step 3: Verify the CLI parses and shows the new flags**

Run: `./venv/bin/python blur_plates.py --help`
Expected: help text lists `--matte-feather` and `--matte-codec`, and `--mode` shows `{blur,color,image,matte}`.

- [ ] **Step 4: Commit**

```bash
git add blur_plates.py
git commit -m "feat: add --mode matte, --matte-feather, --matte-codec CLI flags"
```

---

## Task 7: config.toml defaults

**Files:**
- Modify: `config.toml` (`[redact]` section, lines 107-124)

- [ ] **Step 1: Add matte defaults to the `[redact]` section**

In `config.toml`, in the `[redact]` section, after the `mode = "blur"` line (line 112), add:

```toml
# For mode = "matte": edge-softening radius in pixels (0 = hard edges).
# Overridden on the CLI with --matte-feather.
matte_feather = 0

# For mode = "matte": output codec.  "prores" → ProRes 422 HQ .mov (best for
# NLE round-trip); "hevc" → near-lossless HEVC .mp4 (GPU-accelerated on PC).
# Overridden on the CLI with --matte-codec.
matte_codec = "prores"
```

- [ ] **Step 2: Verify config still loads and defaults flow through**

Run: `./venv/bin/python -c "import blur_plates; c = blur_plates.load_config(); print(c['redact'].get('matte_feather'), c['redact'].get('matte_codec'))"`
Expected: `0 prores`

- [ ] **Step 3: Commit**

```bash
git add config.toml
git commit -m "feat: add matte_feather and matte_codec defaults to config"
```

---

## Task 8: README documentation

**Files:**
- Modify: `README.md` (redaction modes table ~163-173; usage examples ~145-159; key-options table ~183-197)

- [ ] **Step 1: Add a matte row to the redaction-modes table**

In `README.md`, in the `### Redaction modes` table, add a row after the `image` row:

```markdown
| `matte` | Nothing — exports a **white-on-black luma matte** instead; footage is left untouched so you composite the blur/mosaic yourself in Premiere/DaVinci | `--matte-codec prores\|hevc`, `--matte-feather N` |
```

- [ ] **Step 2: Add usage examples**

In `README.md`, in the `### Single video` examples block (before the closing fence around line 159), add:

```bash
# Export a luma matte (ProRes .mov) — original footage stays untouched
python blur_plates.py input.mp4 matte.mov --mode matte

# Soft-edged matte + HEVC .mp4 (GPU-accelerated on NVIDIA PCs)
python blur_plates.py input.mp4 matte.mp4 --mode matte --matte-codec hevc --matte-feather 8
```

- [ ] **Step 3: Add the flags to the Key options table**

In `README.md`, in the `### Key options` table, add after the `--image` row:

```markdown
| `--matte-codec` | `prores` | For `--mode matte`: `prores` (.mov) or `hevc` (.mp4) |
| `--matte-feather` | `0` | For `--mode matte`: edge-softening radius in px |
```

- [ ] **Step 4: Update the `--mode` row in the Key options table**

In `README.md`, change the existing `--mode` row:

```markdown
| `--mode` | `blur` | Redaction style: `blur`, `color`, or `image` |
```

to:

```markdown
| `--mode` | `blur` | Redaction style: `blur`, `color`, `image`, or `matte` |
```

- [ ] **Step 5: Commit**

```bash
git add README.md
git commit -m "docs: document --mode matte and matte options in README"
```

---

## Task 9: End-to-end verification

**Files:**
- None modified (verification only; produces temp artifacts under `/tmp`)

- [ ] **Step 1: Create a short synthetic test clip**

Run:
```bash
export PATH="$HOME/Documents/ffmpeg:$PATH"
ffmpeg -y -f lavfi -i testsrc=size=640x360:rate=15:duration=2 \
  -pix_fmt yuv420p -c:v libx264 /tmp/matte_src.mp4
ffprobe -v error -show_entries stream=width,height,codec_name -of default=nw=1 /tmp/matte_src.mp4
```
Expected: a 640x360 h264 clip is written.

- [ ] **Step 2: Run matte mode (ProRes) end-to-end**

Run:
```bash
export PATH="$HOME/Documents/ffmpeg:$PATH"
./venv/bin/python blur_plates.py /tmp/matte_src.mp4 /tmp/matte_out.mov --mode matte
```
Expected: run completes; prints "Matte written directly to /tmp/matte_out.mov (codec: prores, …)". (Detections may be zero on synthetic content — that is fine; the matte will be all-black, which is a valid matte and confirms the pipeline.)

- [ ] **Step 3: Verify the ProRes output**

Run:
```bash
export PATH="$HOME/Documents/ffmpeg:$PATH"
ffprobe -v error -select_streams v:0 \
  -show_entries stream=codec_name,width,height,r_frame_rate,color_range \
  -of default=nw=1 /tmp/matte_out.mov
```
Expected: `codec_name=prores`, `width=640`, `height=360`, frame rate `15/1`, `color_range=pc`.

- [ ] **Step 4: Run matte mode (HEVC) and verify**

Run:
```bash
export PATH="$HOME/Documents/ffmpeg:$PATH"
./venv/bin/python blur_plates.py /tmp/matte_src.mp4 /tmp/matte_out.mp4 --mode matte --matte-codec hevc
ffprobe -v error -select_streams v:0 -show_entries stream=codec_name -of csv=p=0 /tmp/matte_out.mp4
```
Expected: encode completes and ffprobe reports `hevc`.

- [ ] **Step 5: Confirm extension auto-correction**

Run:
```bash
export PATH="$HOME/Documents/ffmpeg:$PATH"
./venv/bin/python blur_plates.py /tmp/matte_src.mp4 /tmp/wrongext.mkv --mode matte 2>&1 | grep -i "output path changed"
ls -la /tmp/wrongext.mov
```
Expected: a "output path changed to /tmp/wrongext.mov" note is printed and `/tmp/wrongext.mov` exists.

- [ ] **Step 6: Run the full test suite one final time**

Run: `export PATH="$HOME/Documents/ffmpeg:$PATH"; ./venv/bin/python -m pytest tests/ -q`
Expected: all pass.

- [ ] **Step 7: Clean up temp artifacts**

Run: `rm -f /tmp/matte_src.mp4 /tmp/matte_out.mov /tmp/matte_out.mp4 /tmp/wrongext.mov`

---

## Post-plan: PR preparation (separate step, after implementation)

Not a task in this plan, but the agreed follow-up: remove the local-only render
helpers from version control (`docs/superpowers/specs/_render_md.py`, `*.html`) or add
them to `.gitignore`, then open a PR from `feature/matte-export` explaining the what and
why (keeps master footage pristine; matte composited in-NLE; ProRes default with an
HEVC/NVENC escape hatch for PC). Confirm push access to `dsdtx/video_license_plates_blur`
(fork if needed). **Do not push until the user confirms.**
```
