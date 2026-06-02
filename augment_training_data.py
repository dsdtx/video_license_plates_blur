#!/usr/bin/env python3
"""
Generate motion-blur and angled augmentation copies of the training set.

For each (image, label) pair in <dataset>/images/train/ + labels/train/,
this writes N new variants with one or more transformations applied:

  • motion blur         — random direction + strength (simulates camera/subject motion)
  • rotation            — random small angle  (-rot..+rot deg)
  • perspective warp    — random per-corner inward shift (3D-style tilt)
  • shear               — random horizontal shear

Bounding boxes are transformed in lock-step:
  - each box's 4 corners are run through the same transform
  - the axis-aligned bounding box of the warped corners becomes the new box
  - boxes that land entirely outside the frame are dropped

Validation split is left untouched so it remains a clean held-out test set.

Output
──────
  <dataset>/images/train/<stem>_aug<N>.jpg
  <dataset>/labels/train/<stem>_aug<N>.txt

Usage
─────
  # 4 variants per source frame (default)
  python augment_training_data.py --dataset training_data

  # More aggressive — 6 variants, stronger blur
  python augment_training_data.py --dataset training_data --variants 6 --max-blur 30

  # Remove all existing augmented files first
  python augment_training_data.py --dataset training_data --clean
"""

import argparse
import math
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

AUG_TAG = "_aug"   # appended to stems so augmented files are recognisable


# ── Bounding-box helpers ──────────────────────────────────────────────────────

def yolo_to_xyxy(line: str, w: int, h: int):
    """Parse one YOLO line. Returns (cls, x1, y1, x2, y2) in pixel coords or None."""
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    cls = int(parts[0])
    cx, cy, bw, bh = (float(p) for p in parts[1:5])
    x1 = (cx - bw / 2) * w
    y1 = (cy - bh / 2) * h
    x2 = (cx + bw / 2) * w
    y2 = (cy + bh / 2) * h
    return cls, x1, y1, x2, y2


def xyxy_to_yolo(cls, x1, y1, x2, y2, w, h):
    cx = ((x1 + x2) / 2) / w
    cy = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    cx = max(0.0, min(1.0, cx))
    cy = max(0.0, min(1.0, cy))
    bw = max(0.001, min(1.0, bw))
    bh = max(0.001, min(1.0, bh))
    return f"{cls} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def warp_points(pts: np.ndarray, M: np.ndarray, perspective: bool) -> np.ndarray:
    """Apply 3x3 (perspective) or 2x3 (affine) matrix to Nx2 points."""
    if perspective:
        homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)  # Nx3
        out = homog @ M.T                                              # Nx3
        out = out[:, :2] / out[:, 2:3]
        return out
    else:
        homog = np.concatenate([pts, np.ones((len(pts), 1))], axis=1)  # Nx3
        return homog @ M.T                                             # Nx2


def transform_boxes(boxes, M, perspective, out_w, out_h, min_area=24):
    """
    Apply M to every box, return list of (cls, x1, y1, x2, y2) that survive clipping.

    The 4 corners of each box are warped, then the axis-aligned bounding box of
    the warped corners is taken — this is the standard way to keep YOLO labels
    correct under rotation/perspective.
    """
    new_boxes = []
    for (cls, x1, y1, x2, y2) in boxes:
        corners = np.array([[x1, y1], [x2, y1], [x2, y2], [x1, y2]], dtype=np.float32)
        warped  = warp_points(corners, M, perspective)
        nx1, ny1 = warped[:, 0].min(), warped[:, 1].min()
        nx2, ny2 = warped[:, 0].max(), warped[:, 1].max()

        # Clip to image bounds
        nx1 = max(0.0, min(out_w, nx1))
        nx2 = max(0.0, min(out_w, nx2))
        ny1 = max(0.0, min(out_h, ny1))
        ny2 = max(0.0, min(out_h, ny2))

        if (nx2 - nx1) * (ny2 - ny1) < min_area:    # mostly off-screen → drop
            continue
        new_boxes.append((cls, nx1, ny1, nx2, ny2))
    return new_boxes


# ── Image transforms ──────────────────────────────────────────────────────────

def build_geometric_transform(w, h, rng,
                              max_rotate=12.0,
                              max_perspective=0.06,
                              max_shear=0.04):
    """
    Return (M, is_perspective) — a 3x3 perspective or 2x3 affine matrix that
    combines random rotation + perspective + shear within configured limits.

    The matrix is built around the image centre so the warp looks natural.
    """
    use_persp = rng.random() < 0.6        # 60 % of geometric augs are perspective

    if use_persp:
        # Random per-corner inward shift (positive = corner moves inward)
        m = max_perspective
        src = np.float32([[0,0], [w,0], [0,h], [w,h]])
        dst = np.float32([
            [w * rng.uniform(0,  m),       h * rng.uniform(0,  m)],
            [w * (1 - rng.uniform(0, m)),  h * rng.uniform(0,  m)],
            [w * rng.uniform(0,  m),       h * (1 - rng.uniform(0, m))],
            [w * (1 - rng.uniform(0, m)),  h * (1 - rng.uniform(0, m))],
        ])
        M = cv2.getPerspectiveTransform(src, dst)
        return M, True

    # Otherwise — rotation + small shear (affine)
    angle  = rng.uniform(-max_rotate, max_rotate)
    a      = math.radians(abs(angle))
    aspect = w / h if w >= h else h / w
    cover  = (abs(math.cos(a)) + abs(math.sin(a)) * aspect) * 1.02   # zoom to hide borders
    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, cover)
    sx = rng.uniform(-max_shear, max_shear)
    sy = rng.uniform(-max_shear, max_shear)
    S = np.array([[1, sx, 0], [sy, 1, 0]], dtype=np.float32)
    # Compose: apply rotation first, then shear  →  M = S @ R (as 3x3)
    R3 = np.vstack([R, [0, 0, 1]])
    S3 = np.vstack([S, [0, 0, 1]])
    M  = (S3 @ R3)[:2]
    return M, False


def apply_geometric(img, M, perspective):
    """
    Apply M with BORDER_REFLECT_101 fill so any uncovered region is filled
    with reflected (visually plausible) image content rather than streaky
    REPLICATE smears that the model can spuriously latch onto.
    The lean transform's cover-zoom usually leaves no uncovered region at all;
    perspective warps can still leave thin triangular gaps where reflection
    looks much more natural than replication.
    """
    h, w = img.shape[:2]
    if perspective:
        return cv2.warpPerspective(img, M, (w, h),
                                   borderMode=cv2.BORDER_REFLECT_101)
    return cv2.warpAffine(img, M, (w, h),
                          borderMode=cv2.BORDER_REFLECT_101)


def motion_blur_kernel(size: int, angle_deg: float):
    """Directional 1-D blur kernel of length `size` at angle `angle_deg`."""
    k = np.zeros((size, size), dtype=np.float32)
    k[size // 2, :] = 1.0
    R = cv2.getRotationMatrix2D((size / 2, size / 2), angle_deg, 1.0)
    k = cv2.warpAffine(k, R, (size, size))
    s = k.sum()
    return k / s if s > 0 else k


def apply_motion_blur(img, size: int, angle_deg: float):
    return cv2.filter2D(img, -1, motion_blur_kernel(size, angle_deg))


# ── Main augmentation loop ────────────────────────────────────────────────────

def build_lean_transform(w, h, rng, max_lean):
    """
    Pure rotation transform for simulating a leaning motorbike at corner angles.
    Returns (M, False) — 2x3 affine, never perspective.
    Skewed toward larger angles than the everyday rotation so we actually get
    cornering-grade lean rather than tiny camera tilts.

    The rotation includes a zoom factor sized to cover the whole frame after
    rotation, so no border streaks appear at the corners — the rotated image
    fills the frame fully.  Geometrically:  scale = |sin| + |cos| × aspect.
    """
    mag  = rng.uniform(0.4, 1.0) * max_lean
    sign = 1 if rng.random() < 0.5 else -1
    angle = sign * mag

    # Cover scale — how much we have to zoom in so the rotated frame leaves
    # no empty triangles at the corners. cos+|sin|*aspect for the wide side.
    a = math.radians(abs(angle))
    aspect = w / h if w >= h else h / w
    cover  = abs(math.cos(a)) + abs(math.sin(a)) * aspect
    cover *= 1.02                                 # tiny extra margin

    R = cv2.getRotationMatrix2D((w / 2, h / 2), angle, cover)
    return R, False


def augment_one(img, boxes, rng,
                motion_prob, max_blur,
                geometric_prob, max_rotate, max_perspective, max_shear,
                lean_prob=0.0, max_lean=30.0,
                hsv_prob=0.5):
    """
    Apply a random combination of augmentations to one image+boxes pair.
    Returns (new_img, new_boxes).

    Two mutually exclusive geometric paths:
      lean_prob       → strong rotation only (corner lean simulation)
      geometric_prob  → mixed rotation/perspective/shear (everyday tilt)
    Lean is checked first so when both probabilities fire we get the lean
    variant rather than diluting it with small rotations.
    """
    h, w = img.shape[:2]

    # Lean simulation (strong rotation only)
    did_geom = False
    if rng.random() < lean_prob:
        M, is_persp = build_lean_transform(w, h, rng, max_lean=max_lean)
        img    = apply_geometric(img, M, is_persp)
        boxes  = transform_boxes(boxes, M, is_persp, w, h)
        did_geom = True

    # Otherwise: everyday geometric (rotation/perspective/shear)
    if not did_geom and rng.random() < geometric_prob:
        M, is_persp = build_geometric_transform(
            w, h, rng,
            max_rotate=max_rotate,
            max_perspective=max_perspective,
            max_shear=max_shear,
        )
        img    = apply_geometric(img, M, is_persp)
        boxes  = transform_boxes(boxes, M, is_persp, w, h)

    # HSV jitter (light)
    if rng.random() < hsv_prob:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.int16)
        hsv[..., 1] = np.clip(hsv[..., 1] + rng.randint(-25, 25), 0, 255)
        hsv[..., 2] = np.clip(hsv[..., 2] + rng.randint(-30, 30), 0, 255)
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    # Motion blur (after geometric so the blur direction stays plausible)
    if rng.random() < motion_prob:
        k_size = rng.randrange(7, max_blur + 1, 2)   # odd kernel sizes 7..max
        angle  = rng.uniform(-90, 90)
        img    = apply_motion_blur(img, k_size, angle)

    return img, boxes


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",        required=True,
                    help="Dataset dir (with images/train + labels/train)")
    ap.add_argument("--variants",       type=int,   default=4,
                    help="How many augmented copies per source frame (default: 4)")
    ap.add_argument("--motion-prob",    type=float, default=0.7,
                    help="Probability of motion blur per variant (default: 0.7)")
    ap.add_argument("--max-blur",       type=int,   default=25,
                    help="Max motion-blur kernel size in px (default: 25)")
    ap.add_argument("--geometric-prob", type=float, default=0.9,
                    help="Probability of rotation/perspective/shear per variant (default: 0.9)")
    ap.add_argument("--max-rotate",     type=float, default=12.0,
                    help="Max rotation degrees (default: 12)")
    ap.add_argument("--max-perspective", type=float, default=0.06,
                    help="Max perspective corner shift as fraction (default: 0.06)")
    ap.add_argument("--max-shear",      type=float, default=0.04,
                    help="Max shear factor (default: 0.04)")
    ap.add_argument("--lean-prob",      type=float, default=0.35,
                    help="Probability of strong lean rotation per variant — "
                         "simulates motorbike cornering (default: 0.35)")
    ap.add_argument("--max-lean",       type=float, default=30.0,
                    help="Max lean angle in degrees (default: 30; "
                         "40-45 for full motorsport cornering)")
    ap.add_argument("--seed",           type=int,   default=42,
                    help="Random seed (default: 42)")
    ap.add_argument("--clean",          action="store_true",
                    help="Remove existing augmented files before generating new ones")
    ap.add_argument("--include-empty",  action="store_true",
                    help="Also augment frames whose label file is empty (no plates)")
    args = ap.parse_args()

    dataset = Path(args.dataset)
    img_dir = dataset / "images" / "train"
    lbl_dir = dataset / "labels" / "train"
    if not img_dir.is_dir() or not lbl_dir.is_dir():
        sys.exit(f"Dataset structure not found in {dataset}/")

    rng = random.Random(args.seed)

    # ── Optional clean of previous augmentation runs ──────────────────────────
    if args.clean:
        removed = 0
        for f in (list(img_dir.glob(f"*{AUG_TAG}*.jpg"))
                  + list(img_dir.glob(f"*{AUG_TAG}*.png"))
                  + list(lbl_dir.glob(f"*{AUG_TAG}*.txt"))):
            f.unlink()
            removed += 1
        print(f"Removed {removed} previous augmented files.")

    # ── Find source frames (skip already-augmented) ───────────────────────────
    sources = sorted(p for p in (list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")))
                     if AUG_TAG not in p.stem)
    print(f"Found {len(sources)} source frames in {img_dir}")
    print(f"Generating {args.variants} variant(s) per frame → "
          f"~{len(sources) * args.variants} new files\n")

    n_written = 0
    n_dropped = 0          # variants where ALL boxes got clipped out

    for src in tqdm(sources, unit="frame", desc="Augmenting"):
        lbl = lbl_dir / f"{src.stem}.txt"
        if not lbl.exists():
            continue

        label_text = lbl.read_text().strip()
        has_boxes  = bool(label_text)
        if not has_boxes and not args.include_empty:
            continue

        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]

        # Parse YOLO labels → pixel boxes
        boxes = []
        for line in label_text.splitlines():
            b = yolo_to_xyxy(line, w, h)
            if b is not None:
                boxes.append(b)

        for vi in range(args.variants):
            new_img, new_boxes = augment_one(
                img.copy(), list(boxes), rng,
                motion_prob     = args.motion_prob,
                max_blur        = args.max_blur,
                geometric_prob  = args.geometric_prob,
                max_rotate      = args.max_rotate,
                max_perspective = args.max_perspective,
                max_shear       = args.max_shear,
                lean_prob       = args.lean_prob,
                max_lean        = args.max_lean,
            )

            # If we started with boxes and all of them clipped out, skip this variant
            if has_boxes and not new_boxes:
                n_dropped += 1
                continue

            stem = f"{src.stem}{AUG_TAG}{vi}"
            # Match the source format so a PNG source produces a PNG augmented copy
            out_img = img_dir / f"{stem}{src.suffix}"
            out_lbl = lbl_dir / f"{stem}.txt"

            if src.suffix.lower() == ".png":
                cv2.imwrite(str(out_img), new_img)
            else:
                cv2.imwrite(str(out_img), new_img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            out_lbl.write_text(
                "\n".join(xyxy_to_yolo(c, *b, w=w, h=h) for (c, *b) in new_boxes)
            )
            n_written += 1

    print(f"\nDone — wrote {n_written} augmented samples.")
    if n_dropped:
        print(f"Dropped {n_dropped} variants whose boxes all clipped off-screen.")
    print(f"\nNext: python finetune.py --data {dataset}/dataset.yaml")


if __name__ == "__main__":
    main()
