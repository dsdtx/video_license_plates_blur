# Video License Plate Blur

Automatically detects and blurs license plates in dashcam, action-cam and motorsport
footage. Built on YOLOv11 + SAHI sliced inference, with a custom-trained model
specifically optimized for **motorbike riding** — extreme lean angles, motion blur,
tunnels, rain, and small distant plates.

Output is **lossless** (FFV1 intermediate → HEVC CRF 0) with the **original audio
preserved**.

---

## What's included

| Component | Purpose |
|---|---|
| `blur_plates.py` | Production: detect + track + blur plates in a video |
| `batch_blur.py` | Run the production blur on every video in a folder |
| The included **`license-plate-finetune-v1m.pt`** | Fine-tuned model — drop-in replacement for the HuggingFace baseline, much better on motorsport / dashcam footage |
| Fine-tuning pipeline (see below) | Specialize the model further on your own footage if v1m doesn't quite hit your edge cases |

The fine-tuned weights were trained on real motorbike footage: cornering, tunnels,
rain, autobahn, low-light, and heavy motion blur. On a held-out 72-frame validation
set, v1m scores **mAP@50 = 0.75** and **mAP@50-95 = 0.57** — a meaningful jump from
the stock weights (0.58 / 0.36 respectively).

---

## AI Setup Prompt

New to the project or setting it up on a fresh machine? Copy and paste the prompt
below into any AI assistant (Claude, ChatGPT, etc.) and it will guide you through
the entire setup interactively:

---

> I want to set up and use this tool that automatically detects and blurs license
> plates in dashcam and action-cam videos.
>
> Before doing anything else, silently run the necessary checks to figure out my
> environment yourself — detect my OS, check if Python is installed and what
> version, check if ffmpeg is installed and on PATH, and check if I have an NVIDIA
> GPU available. Do not ask me for any of this information.
>
> Then take me through the full setup in this order, giving me one step at a time
> and waiting for me to confirm before moving on:
>
> 1. **Clone the repo** to a sensible location based on my OS.
> 2. **Install ffmpeg** if it is not already installed, using the best method for
>    my OS (winget, brew, apt, etc.) and verify it is on PATH.
> 3. **Create a Python virtual environment** inside the cloned folder, activate
>    it, and install all dependencies — use CUDA-enabled PyTorch if I have an
>    NVIDIA GPU, CPU-only otherwise. Detect this automatically.
> 4. **Download the fine-tuned model weights** `license-plate-finetune-v1m.pt`
>    from the repo's latest GitHub Release into the project folder, if it's
>    not already there.
> 5. **Run a quick debug test** on any video file I have so I can see detections
>    before any real blurring happens.
> 6. **Explain `config.toml`** in plain language — only the settings I am likely
>    to actually change.
>
> Keep instructions short and copy-pasteable. If something fails, diagnose it and
> fix it before moving on.

---

## Requirements

- Python 3.11+
- [ffmpeg](https://ffmpeg.org/download.html) (must be on `PATH`)
- NVIDIA GPU recommended (CPU works but is slow)

---

## Installation

```bash
# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS / Linux

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install ultralytics sahi opencv-python tqdm
```

> For CPU-only, replace the torch line with: `pip install torch torchvision`

### Model weights

The fine-tuned **`license-plate-finetune-v1m.pt`** is published as a GitHub
Release asset (the `.pt` files are not committed to the repo).

Download it into the project folder — `blur_plates.py` looks for it by exact
filename in the same directory:

```bash
# From the project root
curl -L -o license-plate-finetune-v1m.pt \
  https://github.com/dsdtx/video_license_plates_blur/releases/latest/download/license-plate-finetune-v1m.pt
```

If you ever want the baseline HuggingFace model for comparison, it's at
[morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).

---

## Configuration

All defaults live in `config.toml` — edit this file to tune thresholds, blur
strength, SAHI settings, and encoding options without touching the code.

Key settings:

```toml
[detection]
plate_conf = 0.15              # confidence threshold (full frame)
plate_conf_in_vehicle = 0.02   # lower threshold inside vehicle bounding boxes

[blur]
strength = 61                  # Gaussian kernel size (odd number)
```

---

## Usage

### Single video

```bash
python blur_plates.py input.mp4 output.mp4
```

```bash
# Clip a time range
python blur_plates.py input.mp4 output.mp4 --start 2:30 --end 3:00

# Motorbike plates only
python blur_plates.py input.mp4 output.mp4 --vehicles motorbike

# Camera mounted behind your own plate (always cover a fixed region)
python blur_plates.py input.mp4 output.mp4 --own-plate 1700,900,2200,1100

# Debug mode — draws detection boxes instead of blurring (blue=vehicle, green=plate)
python blur_plates.py input.mp4 debug.mp4 --debug

# DEBUG DATA — extended in-frame overlay with source tags & trajectory trails
python blur_plates.py input.mp4 debug.mp4 --debug-overlay

# DEBUG DATA — add a side-panel HUD with frame/track/timing telemetry
python blur_plates.py input.mp4 debug.mp4 --debug-overlay --debug-hud

# Replace plates with a solid colour (R,G,B) instead of blurring
python blur_plates.py input.mp4 output.mp4 --mode color --color 0,0,0

# Stamp a custom image (logo / sticker / portrait) onto every plate
python blur_plates.py input.mp4 output.mp4 --mode image --image my_sticker.png
```

### Redaction modes

| `--mode` | What gets drawn over each plate | Extra flag |
|---|---|---|
| `blur` (default) | Strong Gaussian blur | `--blur N` for kernel size |
| `color` | Solid colour fill | `--color R,G,B` (0-255 each) |
| `image` | A PNG/JPG stretched to fill the plate. PNG alpha is honoured. | `--image PATH` |

The selected mode applies to **every plate** in the output, including the
`--own-plate` fixed region — so the result has a consistent look.

Defaults for these flags can also be set in `config.toml` under the `[redact]`
section so you don't have to pass them every time.

### Batch — entire folder

```bash
python batch_blur.py /path/to/folder --outdir /path/to/output --vehicles motorbike
```

### Key options

| Flag | Default | Description |
|------|---------|-------------|
| `--vehicles` | `all` | Filter by vehicle type: `all`, `motorbike`, `car`, `bus`, `truck` |
| `--plate-conf` | `0.15` | Plate confidence threshold (full frame) |
| `--plate-conf-in-vehicle` | `0.02` | Plate confidence inside vehicle boxes |
| `--blur` | `61` | Gaussian blur kernel size |
| `--conf` | `0.30` | Vehicle detector confidence |
| `--start` / `--end` | — | Process a time range (`MM:SS` or `HH:MM:SS`) |
| `--own-plate` | — | Fixed region to always blur (`x1,y1,x2,y2`) |
| `--debug` | off | Overlay detection boxes instead of blurring |
| `--debug-overlay` | off | DEBUG DATA mode (A): rich in-frame overlay with source tags (SAHI/crop+/pred), trajectory trails, ghost tracks |
| `--debug-hud` | off | DEBUG DATA mode (B): brand-styled side panel with frame#, counts, track list, per-stage timings (output gets +320 px wider) |
| `--mode` | `blur` | Redaction style: `blur`, `color`, or `image` |
| `--color` | `0,0,0` | Solid fill colour for `--mode color` (R,G,B) |
| `--image` | — | Path to overlay image for `--mode image` (PNG with alpha supported) |

---

## Specializing the model on your own footage

The shipped v1m model is great on motorsport / dashcam material, but every camera
and every riding style has its own quirks (different plate styles, unusual mounts,
camera angles, weather, etc.). If you notice the model missing or over-detecting
on **your specific** footage, the repo includes a complete fine-tuning pipeline
to specialize it further.

A full extract → review → augment → train cycle on ~500 of your own frames
typically takes a few hours and produces a model meaningfully better on **your**
edge cases.

### Workflow

```
your videos ─┐
             ▼
extract_training_frames.py  ──▶  auto-labelled frames + dataset.yaml
             ▼
review_annotations.py       ──▶  you fix wrong / missed boxes (mouse + keyboard)
             ▼
augment_training_data.py    ──▶  5× variants per frame: motion blur, lean, perspective
             ▼
review_annotations.py       ──▶  optional second pass on augmented frames
             ▼
finetune.py                 ──▶  two-phase fine-tune (frozen backbone → full)
```

### 1. Extract training frames

Smart sampling — uncertain detections (conf 0.05–0.25) are kept aggressively,
confident ones and "no plate" frames are sub-sampled — so the dataset is dense
in the frames that carry the most learning signal.

```bash
# All videos in a folder → ~2000 frames @ 1 fps, PNG (lossless)
python extract_training_frames.py /path/to/videos/ --outdir training_data/ \
       --fps 1 --max-frames 2000 --format png --no-review
```

Output structure:
```
training_data/
  images/train/  images/val/   ← 85 / 15 split
  labels/train/  labels/val/   ← YOLO .txt files (auto-generated)
  dataset.yaml
```

### 2. Review the auto-labels

```bash
python review_annotations.py --dataset training_data/
```

| Control | Action |
|---|---|
| **Left-drag** | Draw a new bounding box (auto-saves on release) |
| **Right-click** on a box | Delete it (auto-saves) |
| `k` | Keep frame as-is, mark reviewed, advance |
| `d` | Empty the label file (no plates in this frame) |
| `s` | Skip without marking, advance |
| `←` / `→` | Navigate |
| `q` | Quit (progress saved to `review_progress.json`) |

Useful filters:
- `--augmented-only` — only show augmented variants
- `--source-only`   — only show source (non-augmented) frames
- `--claude-edited` — only show frames listed in `claude_review_log.json`

### 3. Augment the training set

Boosts the dataset 5× with **motion blur, corner-lean rotation, perspective warp,
shear, and HSV jitter** — the conditions stock detectors struggle with. Bounding
boxes are transformed in lock-step (corners warped, axis-aligned bbox of the
warped corners becomes the new label).

```bash
# Default: 5 variants per source frame; motion-blur + lean enabled
python augment_training_data.py --dataset training_data/

# Heavier corner-lean for motorsport / track riding
python augment_training_data.py --dataset training_data/ \
       --variants 6 --lean-prob 0.5 --max-lean 40

# Wipe previous augmented copies first
python augment_training_data.py --dataset training_data/ --clean
```

Lean rotations use cover-zoom (scale = `|cos|+|sin|·aspect`) so the rotated image
fully fills the frame — no streaky `BORDER_REPLICATE` edges the model could
latch onto as spurious features.

### 4. (Optional) Cycle: fix sources flagged via augmented review

When you fix a bad box on an augmented frame, the **source** frame almost
certainly has the same underlying label problem (plus its other 4 augmented
copies inherited it). If this matters for your accuracy:

1. Note which augmented stems you fixed (e.g. `IMG_xxx_aug2`).
2. Re-review the corresponding **source** frames with
   `python review_annotations.py --dataset training_data/ --source-only` and
   correct the underlying label.
3. Delete the source's stale augmented copies
   (`rm training_data/images/train/<source>_aug*.png` and the matching
   `.txt` files) and re-run `augment_training_data.py` to regenerate.

In practice, training tolerates a few percent of label noise just fine — you
usually don't need this loop.

### 5. Fine-tune

Two phases: **(1)** backbone frozen, head trains fast; **(2)** full unfreeze at
lower LR. Starts from the shipped fine-tuned weights so convergence is hours,
not days.

```bash
python finetune.py --data training_data/dataset.yaml --name my_run
```

| Flag | Default | What it does |
|---|---|---|
| `--epochs` | 60 | Total epochs (Phase 1 + Phase 2) |
| `--freeze-epochs` | 15 | Frozen-backbone epochs at the start |
| `--freeze` | 10 | How many backbone layers to freeze |
| `--batch` | 16 | Batch size |
| `--imgsz` | 640 | Training image size |
| `--lr` | 0.001 | Initial LR (Phase 2 uses 1/10 of this) |
| `--name` | `exp` | Run name → `runs/finetune/<name>/` |

Best model lands at `runs/finetune/<name>/weights/best.pt`.

### 6. Verify the improvement

The simplest sanity check: run `blur_plates.py --debug` on a short clip with
each model in turn (rename `license-plate-finetune-v1m.pt` between runs) and
eyeball the green plate boxes. For a stricter side-by-side, the validation
mAP printed at the end of training is also a good comparison point — the
v1m model in this repo scores **mAP@50 ≈ 0.75 / mAP@50-95 ≈ 0.57** on its
held-out set.

### 7. Promote your new model into production

```bash
# Back up the shipped model first
cp license-plate-finetune-v1m.pt license-plate-finetune-v1m-shipped.pt

# Drop in your own
cp runs/finetune/my_run/weights/best.pt license-plate-finetune-v1m.pt
```

`blur_plates.py` reads from `license-plate-finetune-v1m.pt` by name — no code
changes needed.

---

## License & credits

Detection architecture: YOLOv11 / Ultralytics. Sliced inference: SAHI.
Baseline weights: [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection).
