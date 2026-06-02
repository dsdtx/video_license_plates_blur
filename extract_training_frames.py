#!/usr/bin/env python3
"""
Extract training frames from dashcam/action-cam videos and auto-annotate
them with the existing plate detection model.  Produces a YOLO-format
dataset ready for correction (review_annotations.py) and fine-tuning
(finetune.py).

How auto-annotation works
─────────────────────────
The existing license-plate-finetune-v1m.pt model is run on every sampled
frame.  Its detections are saved as YOLO .txt label files — this gives you
a starting point so that annotation means *correcting mistakes*, not drawing
every box from scratch.  On typical dashcam footage 70-90 % of boxes need
no editing at all.

Smart sampling strategy
───────────────────────
Frames are sorted into buckets and kept at different rates so the dataset is
dense in the frames that carry the most learning signal:

  uncertain    conf 0.05 – 0.25  → always keep   (model unsure → top priority)
  confident    conf ≥ 0.25        → keep 1-in-4   (already working, add diversity)
  vehicle / no plate detected     → keep 1-in-8   (potential false negatives)
  nothing detected                → keep 1-in-30  (true-negative variety)

Output layout
─────────────
  <outdir>/
    images/train/   images/val/   — JPEG frames  (85 % / 15 % split)
    labels/train/   labels/val/   — YOLO .txt files from the model (edit these!)
    review/                        — overlay PNGs for quick spot-check
    dataset.yaml                   — ready for `python finetune.py`

Usage
─────
    # Single video, 1 frame/s
    python extract_training_frames.py VID.mp4

    # Entire folder of videos
    python extract_training_frames.py /path/to/videos/ --outdir dataset/ --fps 1

    # Short clip
    python extract_training_frames.py VID.mp4 --start 1:00 --end 3:00

    # Skip review images (faster, saves disk)
    python extract_training_frames.py /videos/ --no-review

Then correct with:
    python review_annotations.py --dataset <outdir>

Then fine-tune with:
    python finetune.py --data <outdir>/dataset.yaml
"""

import argparse
import random
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from blur_plates import (
    load_models,
    detect_plates,
    get_video_info,
    time_to_seconds,
    VEHICLE_CLASSES,
    _dbg_box,
    _DBG_PLATE_COLOR,
    _DBG_VEHICLE_COLOR,
    _DBG_PREDICT_COLOR,
)

VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".mts", ".m4v", ".mpg", ".mpeg"}

# ── Sampling thresholds ────────────────────────────────────────────────────────
CONF_UNCERTAIN_LOW  = 0.05   # minimum conf to write a box to the label file
CONF_UNCERTAIN_HIGH = 0.25   # above this = "confident" detection
CONFIDENT_KEEP_1IN  = 4
VEHICLE_KEEP_1IN    = 8
NOTHING_KEEP_1IN    = 30
VAL_FRACTION        = 0.15


# ── Frame extraction via ffmpeg pipe ──────────────────────────────────────────

def _iter_frames(video_path: Path, sample_fps: float, start_sec, end_sec):
    """Yield (raw_frame_idx, timestamp, bgr_frame, w, h, fps).

    Pushes the temporal subsampling into ffmpeg via the `fps=` filter so we
    only decode the frames we actually want. Without this, ffmpeg decodes the
    full video (e.g. 30 fps × 10 min = 18,000 frames) and we throw most away —
    extremely wasteful for sample_fps=1.
    """
    info = get_video_info(str(video_path))
    w, h, fps = info["width"], info["height"], info["fps"]
    frame_size = w * h * 3

    cmd = ["ffmpeg", "-y"]
    if start_sec is not None:
        cmd += ["-ss", f"{start_sec:.6f}"]
    cmd += ["-i", str(video_path)]
    if end_sec is not None:
        cmd += ["-t", f"{end_sec - (start_sec or 0.0):.6f}"]
    cmd += ["-f", "rawvideo", "-pix_fmt", "bgr24",
            "-vf", f"fps={sample_fps},scale=trunc(iw/2)*2:trunc(ih/2)*2",
            "pipe:1"]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    raw_idx = 0
    try:
        while True:
            raw = proc.stdout.read(frame_size)
            if len(raw) < frame_size:
                break
            # Every emitted frame is a sample now — no skip step needed.
            if True:
                ts    = (start_sec or 0.0) + raw_idx / sample_fps
                frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
                yield raw_idx, ts, frame, w, h, fps
            raw_idx += 1
    finally:
        proc.stdout.close()
        proc.wait()


# ── YOLO label helpers ─────────────────────────────────────────────────────────

def _to_yolo(x1, y1, x2, y2, img_w, img_h):
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    bw = (x2 - x1)       / img_w
    bh = (y2 - y1)       / img_h
    # Clamp — SAHI can produce tiny out-of-bounds edge coords
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))
    return cx, cy, bw, bh


# ── Review overlay ────────────────────────────────────────────────────────────

def _review_image(frame, plate_rects, vehicles):
    """Draw all detections onto a copy of the frame for human review."""
    vis = frame.copy()
    for (cls, x1, y1, x2, y2, conf) in vehicles:
        _dbg_box(vis, x1, y1, x2, y2, _DBG_VEHICLE_COLOR,
                 f"{VEHICLE_CLASSES[cls]} {conf:.2f}", thickness=2)
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf = rect[4] if len(rect) > 4 else 0.0
        uncertain = conf < CONF_UNCERTAIN_HIGH
        color = _DBG_PREDICT_COLOR if uncertain else _DBG_PLATE_COLOR
        label = f"plate {conf:.2f}" + ("  LOW" if uncertain else "")
        _dbg_box(vis, x1, y1, x2, y2, color, label, thickness=2)
    return vis


# ── Sampling decision ──────────────────────────────────────────────────────────

def _should_keep(plate_rects, has_vehicles, counters):
    """Return True if this frame should be added to the dataset."""
    uncertain = any(
        CONF_UNCERTAIN_LOW <= (r[4] if len(r) > 4 else 0) < CONF_UNCERTAIN_HIGH
        for r in plate_rects
    )
    if uncertain:
        return True

    confident = any((r[4] if len(r) > 4 else 0) >= CONF_UNCERTAIN_HIGH
                    for r in plate_rects)
    if confident:
        counters["confident"] = counters.get("confident", 0) + 1
        return counters["confident"] % CONFIDENT_KEEP_1IN == 0

    if has_vehicles:
        counters["vehicle"] = counters.get("vehicle", 0) + 1
        return counters["vehicle"] % VEHICLE_KEEP_1IN == 0

    counters["nothing"] = counters.get("nothing", 0) + 1
    return counters["nothing"] % NOTHING_KEEP_1IN == 0


# ── Per-video processing ───────────────────────────────────────────────────────

def _is_dead_frame(frame):
    """Reject frames with near-zero information content (all-black, all-white,
    cameras-off, etc.).  A frame is 'dead' when its per-channel std-dev is
    below a small threshold — there's nothing for the model to learn from."""
    return frame.std() < 5.0


def _process_video(video_path, outdir, vehicle_model, plate_model, device,
                   sample_fps, start_sec, end_sec, write_review, budget, rng,
                   image_format="png"):
    """Returns list of (stem, split) for every frame written."""
    vid_stem    = video_path.stem
    counters    = {}
    collected   = []
    n_dead      = 0

    if image_format == "png":
        ext, write_params = ".png", []   # PNG is lossless; default compression
    else:
        ext, write_params = ".jpg", [cv2.IMWRITE_JPEG_QUALITY, 95]

    for raw_idx, ts, frame, w, h, fps in _iter_frames(
            video_path, sample_fps, start_sec, end_sec):

        if len(collected) >= budget:
            break

        # Quality filter — skip dead frames (black, all-white, sensor off)
        if _is_dead_frame(frame):
            n_dead += 1
            continue

        plate_rects, vehicles = detect_plates(
            frame, vehicle_model, plate_model, device=device,
            vehicle_conf=0.30,
            plate_conf=CONF_UNCERTAIN_LOW,
            plate_conf_in_vehicle=CONF_UNCERTAIN_LOW,
            sahi_slice_size=640, sahi_overlap=0.2,
            sharpen=True, sharpen_amount=1.5, sharpen_sigma=1.0,
            vehicle_crop_scale=2.0,
        )

        if not _should_keep(plate_rects, bool(vehicles), counters):
            continue

        split = "val" if rng.random() < VAL_FRACTION else "train"
        stem  = f"{vid_stem}_f{raw_idx:07d}"

        # Image (PNG lossless by default — preserves small plate detail)
        img_path = outdir / "images" / split / f"{stem}{ext}"
        cv2.imwrite(str(img_path), frame, write_params)

        # YOLO label  (all detections above minimum threshold)
        lbl_path = outdir / "labels" / split / f"{stem}.txt"
        lines = []
        for rect in plate_rects:
            x1, y1, x2, y2 = rect[:4]
            conf = rect[4] if len(rect) > 4 else 0.0
            if conf >= CONF_UNCERTAIN_LOW:
                cx, cy, bw, bh = _to_yolo(x1, y1, x2, y2, w, h)
                lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        lbl_path.write_text("\n".join(lines))

        # Review overlay (JPEG is fine — these are just for visual checking)
        if write_review:
            rev  = _review_image(frame, plate_rects, vehicles)
            rev_path = outdir / "review" / f"{stem}.jpg"
            cv2.imwrite(str(rev_path), rev, [cv2.IMWRITE_JPEG_QUALITY, 82])

        collected.append((stem, split))

    if n_dead:
        tqdm.write(f"    (skipped {n_dead} dead frame(s))")
    return collected


# ── dataset.yaml ───────────────────────────────────────────────────────────────

def _write_yaml(outdir: Path):
    yaml = f"""\
# Auto-generated by extract_training_frames.py
# Review/correct labels with:   python review_annotations.py --dataset {outdir}
# Fine-tune with:               python finetune.py --data {outdir.resolve()}/dataset.yaml

path:  {outdir.resolve()}
train: images/train
val:   images/val

nc: 1
names:
  - license_plate
"""
    (outdir / "dataset.yaml").write_text(yaml)


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input",
                    help="Video file or folder of videos")
    ap.add_argument("--outdir",     default="training_data",
                    help="Output dataset directory (default: training_data/)")
    ap.add_argument("--fps",        type=float, default=1.0,
                    help="Frames per second to sample from each video (default: 1.0)")
    ap.add_argument("--start",      default=None, help="Start time MM:SS")
    ap.add_argument("--end",        default=None, help="End time MM:SS")
    ap.add_argument("--max-frames", type=int, default=5000,
                    help="Total frame budget across all videos (default: 5000)")
    ap.add_argument("--no-review",  action="store_true",
                    help="Skip writing review overlay images")
    ap.add_argument("--format",     choices=["png", "jpg"], default="png",
                    help="Image format for training frames (default: png, lossless).  "
                         "Use jpg for ~5x smaller files at minor quality cost.")
    ap.add_argument("--seed",       type=int, default=42)
    args = ap.parse_args()

    inp    = Path(args.input)
    outdir = Path(args.outdir)
    rng    = random.Random(args.seed)

    # Locate videos
    if inp.is_dir():
        videos = sorted(p for p in inp.iterdir()
                        if p.suffix.lower() in VIDEO_EXTENSIONS
                        and "_blurred" not in p.stem)
        if not videos:
            sys.exit(f"No video files found in {inp}")
    elif inp.is_file():
        videos = [inp]
    else:
        sys.exit(f"Input not found: {inp}")

    # Create dirs
    for split in ("train", "val"):
        (outdir / "images" / split).mkdir(parents=True, exist_ok=True)
        (outdir / "labels" / split).mkdir(parents=True, exist_ok=True)
    if not args.no_review:
        (outdir / "review").mkdir(parents=True, exist_ok=True)

    start_sec = time_to_seconds(args.start) if args.start else None
    end_sec   = time_to_seconds(args.end)   if args.end   else None

    print(f"\nLoading models...")
    vehicle_model, plate_model, device = load_models(plate_conf=CONF_UNCERTAIN_LOW)
    print(f"Models ready. Processing {len(videos)} video(s).\n")

    all_frames: list = []
    budget = args.max_frames

    for video in tqdm(videos, unit="video", desc="Videos"):
        if budget <= 0:
            tqdm.write("  Frame budget reached — stopping.")
            break
        tqdm.write(f"\n  {video.name}")
        collected = _process_video(
            video, outdir, vehicle_model, plate_model, device,
            sample_fps=args.fps,
            start_sec=start_sec, end_sec=end_sec,
            write_review=not args.no_review,
            budget=budget, rng=rng,
            image_format=args.format,
        )
        budget -= len(collected)
        all_frames.extend(collected)
        n_t = sum(1 for _, s in collected if s == "train")
        n_v = sum(1 for _, s in collected if s == "val")
        tqdm.write(f"    → {len(collected)} frames  (train={n_t}, val={n_v})")

    _write_yaml(outdir)

    n_train = sum(1 for _, s in all_frames if s == "train")
    n_val   = sum(1 for _, s in all_frames if s == "val")

    print(f"\n{'═'*58}")
    print(f"  Dataset ready   {len(all_frames)} frames  (train={n_train}, val={n_val})")
    print(f"  Output: {outdir.resolve()}/")
    print(f"{'═'*58}")
    print(f"""
Step 2 — Correct auto-annotations
    python review_annotations.py --dataset {outdir}
    Keys: [k] keep  [d] delete label  [← →] prev/next  [q] quit

Step 3 — Fine-tune
    python finetune.py --data {outdir}/dataset.yaml
""")


if __name__ == "__main__":
    main()
