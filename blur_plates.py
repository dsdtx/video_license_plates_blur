#!/usr/bin/env python3
"""
License Plate Blurring Tool
============================
Detects license plates using YOLOv8 vehicle detection + OpenCV Haar cascades,
then blurs them. Output is lossless (FFV1 intermediate → HEVC CRF 0).

Usage:
    python blur_plates.py <input> <output> [--start MM:SS] [--end MM:SS] [--blur N] [--conf F]
    python blur_plates.py <input> <output> --own-plate x1,y1,x2,y2   # fixed region for camera-mounted plates

Example:
    python blur_plates.py final.mov output_blurred.mov --start 2:48 --end 2:51
    python blur_plates.py final.mov output.mov --own-plate 1700,900,2200,1100
"""

import cv2
import math
import numpy as np
import subprocess
import json
import os
import sys
import argparse
import tempfile
import threading
import time
import tomllib
from datetime import datetime
from tqdm import tqdm
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction


_CONFIG_PATH = os.path.join(os.path.dirname(__file__), "config.toml")


def load_config() -> dict:
    if not os.path.exists(_CONFIG_PATH):
        sys.exit(f"ERROR: config file not found: {_CONFIG_PATH}")
    with open(_CONFIG_PATH, "rb") as f:
        return tomllib.load(f)


# ─── YOLO vehicle class IDs (COCO dataset) ───────────────────────────────────
VEHICLE_CLASSES = {2: "car", 3: "motorcycle", 5: "bus", 7: "truck"}

VEHICLE_FILTER_MAP = {
    "all":       {2, 3, 5, 7},
    "motorbike": {3},
    "car":       {2},
    "bus":       {5},
    "truck":     {7},
}


def time_to_seconds(time_str: str) -> float:
    """Convert MM:SS or HH:MM:SS to float seconds."""
    parts = time_str.split(":")
    if len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    return float(time_str)


def parse_region(s: str):
    """Parse 'x1,y1,x2,y2' string into a tuple of ints."""
    parts = [int(v.strip()) for v in s.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected x1,y1,x2,y2 but got: {s!r}")
    return tuple(parts)


def get_video_info(video_path: str) -> dict:
    """Return width, height, fps, codec via ffprobe."""
    result = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", video_path],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(result.stdout)["streams"]
    vs = next(s for s in streams if s["codec_type"] == "video")
    num, den = map(int, vs["r_frame_rate"].split("/"))
    duration = float(vs.get("duration") or
                     json.loads(subprocess.run(
                         ["ffprobe", "-v", "quiet", "-print_format", "json",
                          "-show_format", video_path],
                         capture_output=True, text=True).stdout)["format"]["duration"])

    stored_w = int(vs["width"])
    stored_h = int(vs["height"])

    # Detect rotation metadata — ffmpeg auto-rotates output by default.
    # For 90°/270° clips (e.g. iPhone, GoPro portrait) the display dimensions
    # are swapped vs the stored stream dimensions.
    rotate = int(vs.get("tags", {}).get("rotate", 0))
    for sd in vs.get("side_data_list", []):          # newer ffmpeg uses display matrix
        if sd.get("side_data_type") == "Display Matrix":
            rotate = -int(sd.get("rotation", 0))
            break
    if abs(rotate) in (90, 270):
        stored_w, stored_h = stored_h, stored_w

    # Apply the same even-rounding the scale filter uses, so frame_size is exact.
    width  = (stored_w // 2) * 2
    height = (stored_h // 2) * 2

    return {
        "width":    width,
        "height":   height,
        "fps":      num / den,
        "codec":    vs["codec_name"],
        "pix_fmt":  vs.get("pix_fmt", "yuv420p"),
        "duration": duration,
    }


PLATE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "license-plate-finetune-v1m.pt")
DETECT_WIDTH = 1280  # both models run at this width; coords scaled back to full-res


def load_models(plate_conf: float = 0.07):
    """Load YOLOv8 vehicle detector + SAHI-wrapped license plate detector.

    plate_conf is set to the lowest threshold that will be used so SAHI doesn't
    discard low-confidence detections before context-aware filtering can run.
    """
    import torch
    from ultralytics import YOLO

    device = "cuda" if torch.cuda.is_available() else "cpu"
    gpu_name = torch.cuda.get_device_name(0) if device == "cuda" else "none"
    print(f"  Device : {device.upper()}" + (f"  ({gpu_name})" if device == "cuda" else " (install CUDA PyTorch for GPU acceleration)"))

    vehicle_model = YOLO("yolov8n.pt")
    vehicle_model.to(device)

    plate_model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics",
        model_path=PLATE_MODEL_PATH,
        confidence_threshold=plate_conf,
        device=device,
    )

    return vehicle_model, plate_model, device


def _overlaps(a, b):
    """Return True if rectangle a overlaps rectangle b."""
    return a[0] < b[2] and a[2] > b[0] and a[1] < b[3] and a[3] > b[1]


def _unsharp_mask(img: np.ndarray, amount: float = 1.5, sigma: float = 1.0) -> np.ndarray:
    """
    Sharpen *img* via unsharp masking.

    A Gaussian-blurred copy is subtracted from the original and the
    difference is added back at *amount* × strength.  Works entirely in
    uint8 space through OpenCV's addWeighted so there is no float cast.

    amount : 0.5 = subtle,  1.5 = strong,  3.0 = very aggressive
    sigma  : Gaussian radius in pixels (1.0–2.0 is typical)
    """
    blurred = cv2.GaussianBlur(img, (0, 0), sigmaX=sigma)
    # img*(1+amount) - blurred*amount  ≡ img + amount*(img - blurred)
    return cv2.addWeighted(img, 1.0 + amount, blurred, -amount, 0)


def detect_plates(frame, vehicle_model, plate_model, device="cpu", vehicle_conf=0.3,
                  vehicle_filter="all", plate_conf=0.15, plate_conf_in_vehicle=0.07,
                  sahi_slice_size=640, sahi_overlap=0.2, detect_scale=1.0,
                  sharpen=False, sharpen_amount=1.5, sharpen_sigma=1.0,
                  vehicle_crop_scale=1.0, collect_rejected=False):
    """
    Returns (plate_rects, all_vehicles, rejected) where:
      plate_rects  — list of (x1, y1, x2, y2, conf) regions to blur
      all_vehicles — list of (cls_id, x1, y1, x2, y2, conf) for every detected vehicle

    Dual-confidence strategy:
      - Plates inside a vehicle bounding box use plate_conf_in_vehicle (lower).
      - Plates outside any vehicle box use plate_conf (stricter).
      This lets you catch blurry / angled plates on motorbikes without flooding
      the full frame with false positives.

    detect_scale:
      Fraction of the original resolution sent to the plate model (default 1.0).
      0.5 processes 4K footage at 2K for ~5× faster detection on a compute-bound
      GPU; blur is always applied at the original full resolution.

    sharpen / sharpen_amount / sharpen_sigma:
      Apply unsharp masking to the detection frame before SAHI tiling.
      Helps with lens blur, mild motion blur, or heavily compressed footage.

    vehicle_crop_scale:
      When > 1.0, each detected vehicle bounding box is extracted from the
      original full-resolution frame, upscaled by this factor with Lanczos
      resampling, and fed to the plate model directly.  Plates that were
      30 px wide become 60–90 px wide, dramatically improving confidence on
      distant or small vehicles.  Results are merged with the SAHI pass.
      1.0 = disabled (default).  2.0 is recommended when enabling.

    collect_rejected:
      When True, below-threshold detections are collected into `rejected`
      (tagged source 'rejected') instead of being silently dropped, so the
      debug overlay can draw them.  Adds no inference cost — the model already
      runs at the lowest threshold; this only changes a `continue` into an
      append.  Defaults False so production output is byte-for-byte unchanged.
    """
    h, w = frame.shape[:2]
    scale = DETECT_WIDTH / w
    small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)), interpolation=cv2.INTER_LINEAR)

    # ── Step 1: vehicle detection — collect all vehicles ──────────────────────
    inv = 1.0 / scale
    all_vehicles = []   # (cls_id, x1, y1, x2, y2, conf)
    v_results = vehicle_model(small, conf=vehicle_conf, verbose=False)
    for r in v_results:
        if r.boxes is None:
            continue
        for box in r.boxes:
            cls = int(box.cls[0])
            if cls not in VEHICLE_CLASSES:
                continue
            vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            all_vehicles.append((cls, int(vx1*inv), int(vy1*inv), int(vx2*inv), int(vy2*inv), conf))

    # Boxes for the active filter (used for plate overlap check)
    filter_classes = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
    filter_boxes = [(x1, y1, x2, y2) for (cls, x1, y1, x2, y2, _) in all_vehicles
                    if cls in filter_classes]

    # ── Step 2: SAHI sliced plate detection ───────────────────────────────────
    # Optionally downsample the frame for faster detection (detect_scale < 1.0).
    # Coordinates are scaled back to full resolution after detection so the blur
    # is always applied at the original quality.
    if detect_scale < 1.0:
        det_w = max(1, int(w * detect_scale))
        det_h = max(1, int(h * detect_scale))
        frame_for_det = cv2.resize(frame, (det_w, det_h), interpolation=cv2.INTER_LINEAR)
        coord_scale   = 1.0 / detect_scale   # multiply detected coords by this
        # Vehicle filter boxes also need to be in detection-space
        filter_boxes_det = [(int(x1 * detect_scale), int(y1 * detect_scale),
                             int(x2 * detect_scale), int(y2 * detect_scale))
                            for (x1, y1, x2, y2) in filter_boxes]
    else:
        frame_for_det    = frame
        coord_scale      = 1.0
        filter_boxes_det = filter_boxes

    # Optional: sharpen the detection frame to recover edge contrast on blurry footage
    if sharpen:
        frame_for_det = _unsharp_mask(frame_for_det, sharpen_amount, sharpen_sigma)

    # Model threshold is pre-set to min(plate_conf, plate_conf_in_vehicle) in
    # load_models() so no valid detection is thrown away before we can filter.
    plate_model.confidence_threshold = min(plate_conf, plate_conf_in_vehicle)
    frame_rgb = cv2.cvtColor(frame_for_det, cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        frame_rgb, plate_model,
        slice_height=sahi_slice_size, slice_width=sahi_slice_size,
        overlap_height_ratio=sahi_overlap, overlap_width_ratio=sahi_overlap,
        verbose=0,
    )

    # ── Step 3: context-aware confidence filtering ────────────────────────────
    plate_rects = []
    rejected    = []   # below-threshold detections (only populated if collect_rejected)
    for det in result.object_prediction_list:
        x1, y1 = int(det.bbox.minx), int(det.bbox.miny)
        x2, y2 = int(det.bbox.maxx), int(det.bbox.maxy)
        if x2 <= x1 or y2 <= y1:
            continue
        conf       = det.score.value
        plate      = (x1, y1, x2, y2)
        in_vehicle = any(_overlaps(plate, vb) for vb in filter_boxes_det)

        if vehicle_filter != "all" and not in_vehicle:
            continue  # vehicle filter requires vehicle overlap

        required = plate_conf_in_vehicle if in_vehicle else plate_conf
        if conf < required:
            if collect_rejected:
                rejected.append((
                    int(x1 * coord_scale), int(y1 * coord_scale),
                    int(x2 * coord_scale), int(y2 * coord_scale),
                    conf, "rejected",
                ))
            continue

        # Scale coords back to full-resolution space.
        # Tuple format: (x1, y1, x2, y2, conf, source) where source is one of
        # 'sahi' | 'crop' | 'pred' | 'own'.  Older code reading rect[:5] is
        # unaffected by the extra trailing field.
        plate_rects.append((
            int(x1 * coord_scale), int(y1 * coord_scale),
            int(x2 * coord_scale), int(y2 * coord_scale),
            conf, "sahi",
        ))

    # ── Step 3b: per-vehicle crop upscale pass ────────────────────────────────
    # For each detected vehicle, extract its bounding box from the original
    # full-res frame, upscale by vehicle_crop_scale, run the plate model on the
    # larger crop, then map detections back to frame coordinates.
    # This is highly effective for distant/small plates that SAHI misses because
    # they are too few pixels wide for the model to score confidently.
    if vehicle_crop_scale > 1.0:
        _CROP_PAD = 12   # extra pixels around each vehicle bbox (original-frame px)
        inv_crop  = 1.0 / vehicle_crop_scale
        for (cls, vx1, vy1, vx2, vy2, _vc) in all_vehicles:
            if cls not in filter_classes:
                continue
            # Crop coords in original-frame space, clamped to frame edges
            cx1 = max(0, vx1 - _CROP_PAD)
            cy1 = max(0, vy1 - _CROP_PAD)
            cx2 = min(w, vx2 + _CROP_PAD)
            cy2 = min(h, vy2 + _CROP_PAD)
            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0:
                continue

            # Upscale with Lanczos for sharpest result
            up_w = max(1, int((cx2 - cx1) * vehicle_crop_scale))
            up_h = max(1, int((cy2 - cy1) * vehicle_crop_scale))
            crop_up = cv2.resize(crop, (up_w, up_h), interpolation=cv2.INTER_LANCZOS4)

            # Apply sharpening to the upscaled crop if enabled
            if sharpen:
                crop_up = _unsharp_mask(crop_up, sharpen_amount, sharpen_sigma)

            # Run plate model directly on the upscaled crop (no SAHI — crop is
            # already the right scale; one model call per vehicle)
            crop_rgb     = cv2.cvtColor(crop_up, cv2.COLOR_BGR2RGB)
            min_conf_thr = min(plate_conf, plate_conf_in_vehicle)
            crop_results = plate_model.model(
                crop_rgb, conf=min_conf_thr, verbose=False
            )

            for r in crop_results:
                if r.boxes is None:
                    continue
                for box in r.boxes:
                    c = float(box.conf[0])
                    px1, py1, px2, py2 = map(int, box.xyxy[0].tolist())
                    # Map from upscaled-crop space → original-frame space
                    ox1 = cx1 + int(px1 * inv_crop)
                    oy1 = cy1 + int(py1 * inv_crop)
                    ox2 = cx1 + int(px2 * inv_crop)
                    oy2 = cy1 + int(py2 * inv_crop)
                    if ox2 <= ox1 or oy2 <= oy1:
                        continue
                    if c < plate_conf_in_vehicle:   # all crop plates are "in vehicle"
                        if collect_rejected:
                            rejected.append((ox1, oy1, ox2, oy2, c, "rejected"))
                        continue
                    plate_rects.append((ox1, oy1, ox2, oy2, c, "crop"))

    # ── Step 4: dedup ─────────────────────────────────────────────────────────
    return merge_overlapping(plate_rects), all_vehicles, rejected


# ─── Batch / GPU-parallel detection ──────────────────────────────────────────

def auto_batch_size(width: int, height: int, sahi_slice_size: int = 640,
                    sahi_overlap: float = 0.2) -> int:
    """
    Calculate how many frames to batch based on free GPU memory after models
    have been loaded.  Returns 1 when CUDA is not available (CPU mode).

    Memory estimate per frame:
        tiles_per_frame × tile_bytes × 4  (activation headroom, float16)
    """
    try:
        import torch
        if not torch.cuda.is_available():
            return 1
        free_bytes, _ = torch.cuda.mem_get_info()
        usable = max(0, free_bytes - 3 * 1024 ** 3)   # keep 3 GB headroom

        stride = int(sahi_slice_size * (1 - sahi_overlap))
        tiles_per_frame = (
            math.ceil(width  / stride) *
            math.ceil(height / stride)
        )
        # float16 tile tensor × 4 for intermediate activations
        bytes_per_frame = tiles_per_frame * sahi_slice_size * sahi_slice_size * 3 * 2 * 4

        batch = max(1, int(usable / bytes_per_frame))
        return min(batch, 64)          # practical cap: 64 frames at once
    except Exception:
        return 1


def detect_plates_batched(frames, vehicle_model, plate_model, device,
                          vehicle_conf=0.3, vehicle_filter="all",
                          plate_conf=0.45, plate_conf_in_vehicle=0.10,
                          sahi_slice_size=640, sahi_overlap=0.2):
    """
    GPU-efficient alternative to calling detect_plates() per frame.

    Strategy:
      1. One batched vehicle-detection call for all N frames.
      2. All SAHI tiles from all N frames are pooled and sent to the plate
         model in a single GPU inference call — far fewer kernel launches.
      3. Tile coordinates are mapped back to full-frame space per-frame.
      4. The same context-aware confidence filtering and NMS used in
         detect_plates() is applied, so results are equivalent.

    Falls back to single-frame detect_plates() when len(frames) == 1.
    """
    if len(frames) == 1:
        r, v, _ = detect_plates(
            frames[0], vehicle_model, plate_model, device,
            vehicle_conf=vehicle_conf, vehicle_filter=vehicle_filter,
            plate_conf=plate_conf, plate_conf_in_vehicle=plate_conf_in_vehicle,
            sahi_slice_size=sahi_slice_size, sahi_overlap=sahi_overlap,
        )
        return [(r, v)]

    h, w = frames[0].shape[:2]
    scale  = DETECT_WIDTH / w
    inv    = 1.0 / scale
    filter_classes = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
    min_conf = min(plate_conf, plate_conf_in_vehicle)

    # ── 1. Batch vehicle detection (one GPU call for all N frames) ─────────────
    smalls = [cv2.resize(f, (DETECT_WIDTH, int(h * scale)),
                         interpolation=cv2.INTER_LINEAR) for f in frames]
    batch_v = vehicle_model(smalls, conf=vehicle_conf, verbose=False)

    per_frame_vehicles = []
    for r in batch_v:
        vehicles = []
        if r.boxes is not None:
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in VEHICLE_CLASSES:
                    continue
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                conf_v = float(box.conf[0])
                vehicles.append((cls, int(vx1 * inv), int(vy1 * inv),
                                  int(vx2 * inv), int(vy2 * inv), conf_v))
        per_frame_vehicles.append(vehicles)

    # ── 2. Pool SAHI tiles from all frames into one list ──────────────────────
    stride     = int(sahi_slice_size * (1 - sahi_overlap))
    all_tiles  = []   # flat list of tile ndarrays (RGB, padded to sahi_slice_size²)
    tile_meta  = []   # (frame_idx, origin_x, origin_y, actual_w, actual_h)

    for fi, frame in enumerate(frames):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        fh, fw = rgb.shape[:2]
        y = 0
        while True:
            y2 = min(y + sahi_slice_size, fh)
            x = 0
            while True:
                x2 = min(x + sahi_slice_size, fw)
                actual_w, actual_h = x2 - x, y2 - y
                tile = rgb[y:y2, x:x2]
                if actual_h < sahi_slice_size or actual_w < sahi_slice_size:
                    pad = np.zeros((sahi_slice_size, sahi_slice_size, 3), dtype=np.uint8)
                    pad[:actual_h, :actual_w] = tile
                    tile = pad
                all_tiles.append(tile)
                tile_meta.append((fi, x, y, actual_w, actual_h))
                if x2 >= fw:
                    break
                x += stride
            if y2 >= fh:
                break
            y += stride

    # ── 3. Single batched plate inference on all tiles ────────────────────────
    tile_results = plate_model.model(
        all_tiles, conf=min_conf, verbose=False, imgsz=sahi_slice_size,
    )

    # ── 4. Map tile detections back to full-frame coordinates ─────────────────
    per_frame_raw = [[] for _ in frames]
    for tile_r, (fi, ox, oy, tw, th) in zip(tile_results, tile_meta):
        if tile_r.boxes is None:
            continue
        for box in tile_r.boxes:
            bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
            # clamp to the unpadded tile area
            bx1, bx2 = min(bx1, tw), min(bx2, tw)
            by1, by2 = min(by1, th), min(by2, th)
            if bx2 <= bx1 or by2 <= by1:
                continue
            per_frame_raw[fi].append(
                (ox + bx1, oy + by1, ox + bx2, oy + by2, float(box.conf[0]))
            )

    # ── 5. Context-aware filtering + NMS, per frame ───────────────────────────
    results = []
    for fi, all_vehicles in enumerate(per_frame_vehicles):
        filter_boxes = [
            (x1, y1, x2, y2)
            for (cls, x1, y1, x2, y2, _) in all_vehicles
            if cls in filter_classes
        ]
        plate_rects = []
        for (x1, y1, x2, y2, conf_val) in per_frame_raw[fi]:
            plate     = (x1, y1, x2, y2)
            in_vehicle = any(_overlaps(plate, vb) for vb in filter_boxes)
            if vehicle_filter != "all" and not in_vehicle:
                continue
            required = plate_conf_in_vehicle if in_vehicle else plate_conf
            if conf_val < required:
                continue
            plate_rects.append((x1, y1, x2, y2, conf_val, "sahi"))
        results.append((merge_overlapping(plate_rects), all_vehicles))

    return results


def suppress_duplicate_plates(plate_rects, vehicle_boxes):
    """Keep only the highest-confidence plate inside each vehicle box.

    SAHI tiles overlap, so the same physical plate can produce 2-3 slightly
    offset detections that survive IoU-NMS. This enforces the physical
    constraint that only one plate face is visible per vehicle at a time.
    Plates not inside any vehicle box are passed through unchanged.

    Returns (kept, suppressed) so callers can visualise the dropped detections.
    """
    if not vehicle_boxes or not plate_rects:
        return plate_rects, []
    claimed    = [False] * len(plate_rects)
    kept       = []
    suppressed = []
    for vb in vehicle_boxes:
        inside = [(i, p) for i, p in enumerate(plate_rects)
                  if not claimed[i] and _overlaps(p[:4], vb)]
        if inside:
            best_i, best_p = max(inside, key=lambda ip: ip[1][4] if len(ip[1]) > 4 else 1.0)
            kept.append(best_p)
            for i, p in inside:
                claimed[i] = True
                if i != best_i:
                    suppressed.append(p)
    kept.extend(p for i, p in enumerate(plate_rects) if not claimed[i])
    return kept, suppressed


def merge_overlapping(rects, iou_thresh=0.3):
    """Merge highly overlapping rectangles (greedy NMS)."""
    if not rects:
        return []
    rects = sorted(rects, key=lambda r: (r[2] - r[0]) * (r[3] - r[1]), reverse=True)
    kept = []
    suppressed = [False] * len(rects)
    for i, r in enumerate(rects):
        if suppressed[i]:
            continue
        kept.append(r)
        for j in range(i + 1, len(rects)):
            if suppressed[j]:
                continue
            if iou(r, rects[j]) > iou_thresh:
                suppressed[j] = True
    return kept


def iou(a, b):
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def apply_blur(frame, rects, blur_strength=61, padding=8):
    """Apply strong Gaussian blur to each rectangle region."""
    h, w = frame.shape[:2]
    k = blur_strength | 1  # ensure odd
    for rect in rects:
        x1, y1, x2, y2 = rect[:4]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        roi = frame[y1:y2, x1:x2]
        if roi.size == 0:
            continue
        blurred = cv2.GaussianBlur(roi, (k, k), 0)
        blurred = cv2.GaussianBlur(blurred, (k, k), 0)
        frame[y1:y2, x1:x2] = blurred
    return frame


def apply_solid_color(frame, rects, color=(0, 0, 0), padding=8):
    """Fill each rectangle region with a solid BGR colour."""
    h, w = frame.shape[:2]
    for rect in rects:
        x1, y1, x2, y2 = rect[:4]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        if x2 > x1 and y2 > y1:
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, -1)
    return frame


def apply_image_overlay(frame, rects, overlay_img, padding=8):
    """
    Paste *overlay_img* (BGR ndarray, already loaded) onto each rectangle
    region, stretched to fill the padded rect exactly.

    overlay_img may include an alpha channel (4 channels).  If alpha is
    present it is used for compositing so the corners of e.g. a circular
    logo blend with the underlying frame; otherwise the overlay covers
    the rect opaquely.
    """
    h, w = frame.shape[:2]
    has_alpha = overlay_img.ndim == 3 and overlay_img.shape[2] == 4
    for rect in rects:
        x1, y1, x2, y2 = rect[:4]
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(w, x2 + padding)
        y2 = min(h, y2 + padding)
        rw, rh = x2 - x1, y2 - y1
        if rw <= 0 or rh <= 0:
            continue
        resized = cv2.resize(overlay_img, (rw, rh), interpolation=cv2.INTER_LINEAR)
        if has_alpha:
            bgr = resized[:, :, :3].astype(np.float32)
            alpha = (resized[:, :, 3:4].astype(np.float32)) / 255.0
            roi = frame[y1:y2, x1:x2].astype(np.float32)
            blended = bgr * alpha + roi * (1.0 - alpha)
            frame[y1:y2, x1:x2] = blended.astype(np.uint8)
        else:
            frame[y1:y2, x1:x2] = resized
    return frame


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


def apply_redaction(frame, rects, mode="blur",
                    blur_strength=61, color=(0, 0, 0),
                    overlay_img=None, padding=8):
    """
    Dispatcher for the three redaction modes.

    mode = "blur"  → Gaussian blur (apply_blur)
    mode = "color" → solid colour fill (apply_solid_color)
    mode = "image" → stretched overlay image (apply_image_overlay)

    Falls back to blur if mode == "image" but no overlay_img is provided.
    """
    if mode == "color":
        return apply_solid_color(frame, rects, color=color, padding=padding)
    if mode == "image" and overlay_img is not None:
        return apply_image_overlay(frame, rects, overlay_img, padding=padding)
    return apply_blur(frame, rects, blur_strength=blur_strength, padding=padding)


def load_overlay_image(path: str):
    """Load an overlay image (PNG/JPG); preserves alpha channel when present."""
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise FileNotFoundError(f"Could not read overlay image: {path}")
    return img


def parse_color(s: str):
    """Parse 'R,G,B' (0-255) into a BGR tuple for OpenCV."""
    parts = [int(v.strip()) for v in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected R,G,B but got: {s!r}")
    if any(p < 0 or p > 255 for p in parts):
        raise ValueError(f"Color components must be 0-255: {s!r}")
    r, g, b = parts
    return (b, g, r)   # OpenCV uses BGR


def _format_duration(seconds: float) -> str:
    """Human-readable duration: '12.3s', '5m23s', or '1h12m05s'."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    if seconds < 3600:
        m, s = divmod(int(round(seconds)), 60)
        return f"{m}m{s:02d}s"
    h, rem = divmod(int(round(seconds)), 3600)
    m, s   = divmod(rem, 60)
    return f"{h}h{m:02d}m{s:02d}s"


def _hardware_label(device: str) -> str:
    """Short string describing the compute device used (GPU name + VRAM, or CPU)."""
    if device == "cuda":
        try:
            import torch
            name = torch.cuda.get_device_name(0)
            total_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            return f"GPU - {name} ({total_gb:.1f} GB VRAM)"
        except Exception:
            return "GPU (CUDA)"
    # CPU fallback
    try:
        import platform
        cpu = platform.processor() or platform.machine() or "CPU"
        n   = os.cpu_count() or "?"
        return f"CPU - {cpu} ({n} threads)"
    except Exception:
        return "CPU"


def _print_run_summary(*, elapsed_total, elapsed_process, device, info,
                       input_path, output_path, frame_num, total_plates,
                       redact_mode, redact_color, redact_image_path,
                       vehicle_filter):
    """Print a compact one-page summary at the end of a video run."""
    bar = "═" * 60

    in_name  = os.path.basename(input_path)
    out_name = os.path.basename(output_path)
    res      = f"{info['width']}x{info['height']}"
    in_fps   = info["fps"]
    in_codec = info.get("codec", "?")
    in_dur   = info.get("duration", 0.0)
    out_size = os.path.getsize(output_path) / 1024 / 1024 if os.path.exists(output_path) else 0.0

    proc_fps   = frame_num / elapsed_process if elapsed_process > 0 else 0.0
    speed_x    = (frame_num / in_fps) / elapsed_total if in_fps and elapsed_total > 0 else 0.0
    plates_pf  = total_plates / frame_num if frame_num else 0.0

    mode_desc = redact_mode
    if redact_mode == "color":
        b, g, r = redact_color
        mode_desc = f"color (R={r} G={g} B={b})"
    elif redact_mode == "image":
        mode_desc = f"image ({os.path.basename(redact_image_path)})" if redact_image_path else "image"

    print(f"\n{bar}")
    print(f"  Run summary")
    print(f"{bar}")
    print(f"  Hardware       :  {_hardware_label(device)}")
    print(f"  Input          :  {in_name}")
    print(f"                    {res} @ {in_fps:.2f} fps  |  {in_codec}  |  {_format_duration(in_dur)}")
    print(f"  Output         :  {out_name}")
    print(f"                    {out_size:.1f} MB  |  HEVC (lossless intermediate → visually lossless mux)")
    print(f"  Redaction      :  {mode_desc}")
    if vehicle_filter and vehicle_filter != "all":
        print(f"  Vehicle filter :  {vehicle_filter} only")
    print(f"  Frames         :  {frame_num:,} processed")
    print(f"  Plates         :  {total_plates:,} redacted   ({plates_pf:.2f} per frame)")
    print(f"  Throughput     :  {proc_fps:.1f} fps processing  |  {speed_x:.2f}x realtime")
    print(f"  Total time     :  {_format_duration(elapsed_total)}")
    print(f"{bar}\n")


_CUVID_DECODERS = {
    "h264": "h264_cuvid",
    "hevc": "hevc_cuvid",
    "av1":  "av1_cuvid",
    "vp9":  "vp9_cuvid",
}


def build_ffmpeg_extract(input_path, start_sec, end_sec, codec=None):
    """Build ffmpeg command that pipes raw BGR frames to stdout.

    Uses NVIDIA CUVID hardware decode when the input codec is supported,
    falling back to software decode silently.
    """
    cmd = ["ffmpeg", "-y"]

    hw_decoder = _CUVID_DECODERS.get(codec or "")
    if hw_decoder:
        # CUVID decoder must come before -i; seek with -ss after -i for accuracy
        cmd += ["-c:v", hw_decoder]
        cmd += ["-i", input_path]
        if start_sec is not None:
            cmd += ["-ss", f"{start_sec:.6f}"]
    else:
        if start_sec is not None:
            cmd += ["-ss", f"{start_sec:.6f}"]
        cmd += ["-i", input_path]

    if end_sec is not None:
        duration = end_sec - (start_sec or 0.0)
        cmd += ["-t", f"{duration:.6f}"]

    # CUVID decoders output nv12 and sometimes emit wrong color-range metadata,
    # causing orange-cast corruption when scale converts nv12→bgr24 directly.
    # Inserting format=yuv420p forces a clean software nv12→yuv420p step first.
    vf = ("format=yuv420p," if hw_decoder else "") + "scale=trunc(iw/2)*2:trunc(ih/2)*2"
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", vf,
        "pipe:1",
    ]
    return cmd


def build_ffmpeg_encode_lossless(width, height, fps, out_path):
    """Build ffmpeg command that reads raw BGR frames from stdin → FFV1 lossless."""
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-s", f"{width}x{height}",
        "-r", str(fps),
        "-i", "pipe:0",
        "-c:v", "ffv1",
        "-level", "3",
        "-threads", "0",
        out_path,
    ]


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


def _ffmpeg_with_progress(cmd, total_frames, desc="  Encoding"):
    """Run an ffmpeg command and show a tqdm frame progress bar. Returns (returncode, stderr)."""
    # Insert -progress pipe:1 -nostats right after 'ffmpeg'
    cmd = [cmd[0], "-progress", "pipe:1", "-nostats"] + cmd[1:]

    stderr_lines = []

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    def _drain_stderr():
        for line in proc.stderr:
            stderr_lines.append(line)

    t = threading.Thread(target=_drain_stderr, daemon=True)
    t.start()

    bar_fmt = "{l_bar}{bar}| {n_fmt}/{total_fmt} frames [{elapsed}<{remaining}, {rate_fmt}]"
    with tqdm(total=total_frames, unit="frame", desc=desc,
              dynamic_ncols=True, bar_format=bar_fmt) as pbar:
        current = 0
        for line in proc.stdout:
            if line.startswith("frame="):
                try:
                    new = int(line.split("=", 1)[1])
                    if new > current:
                        pbar.update(new - current)
                        current = new
                except ValueError:
                    pass

    t.join()
    proc.wait()
    return proc.returncode, "".join(stderr_lines)


def _source_color_args(source_path):
    """
    Probe the source video for color metadata (primaries, transfer, space,
    range) and return ffmpeg flags that preserve it on the output.

    Without this, the raw-BGR pipe in the middle of the pipeline strips all
    colorimetry — leaving the final encoded HEVC with color_primaries=unknown
    etc.  Different players then guess different colorspaces, which is exactly
    the colour-shift artefact users see on iPhone footage (BT.709 source ←→
    BT.601 default-guess).
    """
    try:
        out = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries",
             "stream=color_primaries,color_transfer,color_space,color_range",
             "-of", "default=nw=1:nk=0", source_path],
            capture_output=True, text=True, check=True
        ).stdout
    except subprocess.CalledProcessError:
        return []
    # Parse key=value pairs (order from ffprobe is alphabetical, not request order).
    fields = {}
    for line in out.splitlines():
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        fields[k.strip()] = v.strip()

    args = []
    primaries = fields.get("color_primaries", "")
    transfer  = fields.get("color_transfer",  "")
    space     = fields.get("color_space",     "")
    rng       = fields.get("color_range",     "")
    if primaries and primaries != "unknown":
        args += ["-color_primaries", primaries]
    if transfer and transfer != "unknown":
        args += ["-color_trc", transfer]
    if space and space != "unknown":
        args += ["-colorspace", space]
    if rng and rng != "unknown":
        # ffmpeg's -color_range only accepts 'tv'/'pc' (or 1/2), not 'mpeg'/'jpeg'
        # — both refer to the same thing; pass the ffprobe label through as-is.
        args += ["-color_range", rng]
    return args


def mux_audio(video_only_path, original_path, output_path,
              start_sec, end_sec, fps, total_frames=None,
              preset="medium", tmp_dir="D:/pip-tmp", quality=18):
    """
    Combine processed video with original audio.
    Audio is extracted to a temp file first (resets timestamps to 0)
    so it stays in sync with the processed video regardless of clip position.
    Final encode: HEVC CRF 0 (lossless) video + original audio.
    """
    os.makedirs(tmp_dir, exist_ok=True)
    # ── Extract audio segment to temp file (timestamps start at 0) ───────────
    with tempfile.NamedTemporaryFile(suffix=".mov", delete=False, dir=tmp_dir) as tmp_audio:
        tmp_audio_path = tmp_audio.name

    try:
        audio_cmd = ["ffmpeg", "-y"]
        if start_sec is not None:
            audio_cmd += ["-ss", f"{start_sec:.6f}"]
        audio_cmd += ["-i", original_path]
        if end_sec is not None:
            duration = end_sec - (start_sec or 0.0)
            audio_cmd += ["-t", f"{duration:.6f}"]
        audio_cmd += [
            "-vn",                    # no video
            "-c:a", "copy",
            "-reset_timestamps", "1", # force PTS to start at 0
            tmp_audio_path,
        ]
        subprocess.run(audio_cmd, capture_output=True, check=True)

        # ── Mux video (FFV1, t=0) + extracted audio (t=0) → final output ─────
        # Use hevc_nvenc (GPU) if available, fall back to libx265 (CPU)
        def _nvenc_available():
            return _ffmpeg_encoder_available("hevc_nvenc")

        # Capture source colorimetry so we can re-tag the output with the same
        # BT.709 (or whatever the source had) values.  The raw-BGR pipe strips
        # this metadata mid-pipeline and players have to guess otherwise.
        color_args = _source_color_args(original_path)

        if _nvenc_available():
            video_codec_args = [
                "-c:v", "hevc_nvenc",
                # Visually-lossless constant quality.  QP 0 used to produce
                # ~400 Mbps output that many players refused to open or froze
                # mid-playback; cq 18 is indistinguishable to the eye and
                # ~10× smaller.
                "-rc", "vbr",
                "-cq", str(quality),
                "-preset", "p4",    # p1=fastest … p7=slowest (GPU-side)
                "-bf", "0",         # match iPhone source (no B-frames)
                "-tag:v", "hvc1",   # QuickTime / macOS compatible
                "-pix_fmt", "yuv420p",
                *color_args,
            ]
            enc_label = "  Encoding (HEVC NVENC GPU + audio)"
        else:
            video_codec_args = [
                "-c:v", "libx265",
                "-crf", str(quality),   # visually lossless, ~10× smaller than crf 0
                "-preset", preset,
                "-tag:v", "hvc1",
                "-pix_fmt", "yuv420p",
                *color_args,
            ]
            enc_label = "  Encoding (HEVC CPU + audio)"

        mux_cmd = [
            "ffmpeg", "-y",
            "-i", video_only_path,   # processed video, PTS 0..N
            "-i", tmp_audio_path,    # audio, PTS 0..N
            "-map", "0:v:0",
            "-map", "1:a:0",
            *video_codec_args,
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        rc, stderr = _ffmpeg_with_progress(mux_cmd, total_frames, enc_label)
        if rc != 0:
            # Retry without audio
            print("  Warning: audio mux failed, retrying video-only...")
            print(stderr[-400:])
            mux_cmd_no_audio = [
                "ffmpeg", "-y",
                "-i", video_only_path,
                *video_codec_args,
                "-movflags", "+faststart",
                output_path,
            ]
            rc2, _ = _ffmpeg_with_progress(mux_cmd_no_audio, total_frames, "  Encoding (video-only)")
            if rc2 != 0:
                raise subprocess.CalledProcessError(rc2, mux_cmd_no_audio)

    finally:
        if os.path.exists(tmp_audio_path):
            os.remove(tmp_audio_path)


def estimate_frame_count(info, start_sec, end_sec):
    """Estimate total frames for the progress bar."""
    total_duration = info["duration"]
    clip_start = start_sec or 0.0
    clip_end   = end_sec   or total_duration
    return max(1, int((clip_end - clip_start) * info["fps"]))


# ─── Temporal tracker ────────────────────────────────────────────────────────

class PlateHistory:
    """
    Rolling history of confirmed plate detections for one tracked vehicle.
    Provides velocity-based position prediction for gap frames.
    """

    def __init__(self, max_history: int = 15):
        self._data: list = []          # (frame_idx, cx, cy, w, h, conf)
        self.max_history = max_history
        self.miss_count  = 0           # consecutive frames without a detection
        self._lk_pts: np.ndarray  = None   # shape (N,1,2) float32, full-frame px
        self._lk_gray: np.ndarray = None   # grayscale frame where _lk_pts were last set

    @property
    def has_history(self) -> bool:
        return len(self._data) > 0

    def record(self, frame_idx: int, x1, y1, x2, y2, conf: float):
        cx = (x1 + x2) * 0.5
        cy = (y1 + y2) * 0.5
        w  = float(x2 - x1)
        h  = float(y2 - y1)
        self._data.append((frame_idx, cx, cy, w, h, conf))
        if len(self._data) > self.max_history:
            self._data.pop(0)
        self.miss_count = 0

    def predict_rect(self, frame_idx: int, max_expand: int = 20):
        """
        Return (x1, y1, x2, y2) predicted at frame_idx.

        Velocity is computed as an exponentially-weighted average of
        per-frame deltas so recent movement dominates.  The box grows by
        2 px per missed frame to account for growing positional uncertainty,
        capped at max_expand pixels per side so a large gap_frames setting
        doesn't balloon the blur region across the frame.
        Returns None if no history exists.
        """
        if not self._data:
            return None

        _, cx_last, cy_last, w_last, h_last, _ = self._data[-1]
        f_last = self._data[-1][0]

        if len(self._data) >= 2:
            vxs, vys, weights = [], [], []
            for i in range(1, len(self._data)):
                f0, cx0, cy0 = self._data[i-1][0], self._data[i-1][1], self._data[i-1][2]
                f1, cx1, cy1 = self._data[i][0],   self._data[i][1],   self._data[i][2]
                dt = f1 - f0
                if dt > 0:
                    vxs.append((cx1 - cx0) / dt)
                    vys.append((cy1 - cy0) / dt)
                    weights.append(2.0 ** i)        # exponential: recent frames dominate
            if vxs:
                tw = sum(weights)
                vx = sum(v * w for v, w in zip(vxs, weights)) / tw
                vy = sum(v * w for v, w in zip(vys, weights)) / tw
                dt = frame_idx - f_last
                cx_last = cx_last + vx * dt
                cy_last = cy_last + vy * dt

        # Grow 2 px per missed frame, but never more than max_expand px per side
        expand = min(self.miss_count * 2, max_expand)
        x1 = int(cx_last - w_last * 0.5 - expand)
        y1 = int(cy_last - h_last * 0.5 - expand)
        x2 = int(cx_last + w_last * 0.5 + expand)
        y2 = int(cy_last + h_last * 0.5 + expand)
        return (x1, y1, x2, y2)

    def refresh_lk(self, gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> None:
        """
        (Re-)initialise Lucas-Kanade tracking from a freshly-confirmed plate bbox.

        Extracts corner features from the plate crop and stores them alongside
        the grayscale frame so predict_rect_lk() can propagate them forward.
        Falls back silently when the crop has too few trackable corners.
        """
        crop = gray[y1:y2, x1:x2]
        if crop.size == 0:
            return
        pts = cv2.goodFeaturesToTrack(
            crop, maxCorners=16, qualityLevel=0.1, minDistance=3, blockSize=5,
        )
        if pts is None or len(pts) < 3:
            return
        pts[:, 0, 0] += x1   # offset crop-space → full-frame space
        pts[:, 0, 1] += y1
        self._lk_pts  = pts
        self._lk_gray = gray.copy()

    def predict_rect_lk(
        self, gray: np.ndarray, max_expand: int = 20
    ) -> "tuple[int,int,int,int] | None":
        """
        Propagate stored corner points to *gray* via Lucas-Kanade sparse optical
        flow, then derive a bounding box from where they landed.

        Updates the stored points and frame so chained gap-fill calls each build
        on the latest tracked position.  Returns None when too few points survive
        (caller should fall back to velocity prediction).
        """
        if self._lk_pts is None or self._lk_gray is None or not self._data:
            return None

        new_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self._lk_gray, gray, self._lk_pts, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 20, 0.03),
        )
        if new_pts is None or status is None:
            return None

        good = new_pts[status.ravel() == 1].reshape(-1, 2)   # (M, 2)
        if len(good) < 3:
            return None

        _, _, _, w_last, h_last, _ = self._data[-1]
        cx = float(np.mean(good[:, 0]))
        cy = float(np.mean(good[:, 1]))

        expand = min(self.miss_count * 2, max_expand)
        x1 = int(cx - w_last * 0.5 - expand)
        y1 = int(cy - h_last * 0.5 - expand)
        x2 = int(cx + w_last * 0.5 + expand)
        y2 = int(cy + h_last * 0.5 + expand)

        # Update for the next gap frame — LK expects (N,1,2)
        self._lk_pts  = good.reshape(-1, 1, 2).astype(np.float32)
        self._lk_gray = gray.copy()
        return (x1, y1, x2, y2)


class VehicleTrack:
    """One tracked vehicle with its associated plate history."""
    _id_counter = 0

    def __init__(self, box: tuple, frame_idx: int, history_frames: int = 15):
        VehicleTrack._id_counter += 1
        self.id          = VehicleTrack._id_counter
        self.box         = box           # (x1, y1, x2, y2) full-res
        self.last_frame  = frame_idx
        self.miss_count  = 0
        self.frames_seen = 1             # frames vehicle was successfully detected
        self.plate       = PlateHistory(max_history=history_frames)

    def update_box(self, box: tuple, frame_idx: int):
        self.box         = box
        self.last_frame  = frame_idx
        self.miss_count  = 0
        self.frames_seen += 1

    def mark_missed(self):
        self.miss_count += 1


class _VehicleDetections:
    """
    Minimal wrapper that presents vehicle bounding boxes in the format
    BYTETracker.update() expects (supports boolean-mask and integer indexing).
    """
    def __init__(self, xyxy, confs, clss):
        self.xyxy = np.asarray(xyxy, dtype=np.float32).reshape(-1, 4)
        self.conf = np.asarray(confs, dtype=np.float32).reshape(-1)
        self.cls  = np.asarray(clss,  dtype=np.float32).reshape(-1)
        if len(self.xyxy):
            x1, y1, x2, y2 = self.xyxy[:,0], self.xyxy[:,1], self.xyxy[:,2], self.xyxy[:,3]
            self.xywh = np.stack([(x1+x2)/2, (y1+y2)/2, x2-x1, y2-y1], axis=1)
        else:
            self.xywh = np.zeros((0, 4), dtype=np.float32)

    def __len__(self):
        return len(self.conf)

    def __getitem__(self, idx):
        return _VehicleDetections(self.xyxy[idx], self.conf[idx], self.cls[idx])


class SceneTracker:
    """
    Frame-level coordinator using BYTETracker for vehicle association.

    Replaces the original greedy-IoU matcher with ByteTrack, which provides:
      - Kalman-filter position prediction  → survives fast camera / subject motion
      - Two-stage matching                 → recovers vehicles that briefly disappear
      - Stable track IDs across gaps       → plate history survives detection drops

    PlateHistory velocity-based gap-fill works on top: when the vehicle detector
    drops a track temporarily, the plate's last known position + velocity is
    extrapolated for up to max_gap_frames frames.
    """

    def __init__(self, max_gap_frames: int = 8, history_frames: int = 15,
                 min_vehicle_conf: float = 0.60, vehicle_iou_thresh: float = 0.30,
                 standalone_min_ar: float = 1.2, standalone_max_ar: float = 6.0,
                 predict_expand_max: int = 20):
        from types import SimpleNamespace
        from ultralytics.trackers import BYTETracker

        self.max_gap_frames      = max_gap_frames
        self.history_frames      = history_frames
        self.standalone_min_ar   = standalone_min_ar
        self.standalone_max_ar   = standalone_max_ar
        self.predict_expand_max  = predict_expand_max
        self.frame_idx           = 0

        # BYTETracker hyperparameters tuned for dashcam / action-cam footage:
        #   track_high_thresh — stage-1: detections above this are matched first
        #   track_low_thresh  — stage-2: weaker detections used to recover lost tracks
        #   new_track_thresh  — minimum conf to start a brand-new track
        #   match_thresh      — max IoU *distance* (= 1 − IoU) for a valid match
        #   track_buffer      — frames BYTETracker holds a lost track before discarding
        bt_args = SimpleNamespace(
            track_high_thresh = min_vehicle_conf,
            track_low_thresh  = max(0.05, min_vehicle_conf * 0.25),
            new_track_thresh  = min_vehicle_conf,
            match_thresh      = 1.0 - vehicle_iou_thresh,    # IoU 0.3 → distance 0.7
            track_buffer      = max(max_gap_frames * 2, 30), # keep lost tracks long enough
            fuse_score        = True,
        )
        self._byte       = BYTETracker(bt_args)
        self._track_dict: dict = {}   # track_id (int) → VehicleTrack

    # ── public API ────────────────────────────────────────────────────────────

    @property
    def tracks(self) -> list:
        """List of currently active VehicleTrack objects (for debug overlay)."""
        return list(self._track_dict.values())

    def update(self, all_vehicles: list, plate_rects: list,
               frame: np.ndarray = None) -> list:
        """
        all_vehicles : [(cls, x1, y1, x2, y2, conf), ...]
        plate_rects  : [(x1, y1, x2, y2, conf), ...]  — current frame detections
        frame        : optional BGR frame; enables LK optical-flow gap-fill when given

        Returns effective plate list: detected plates + gap-filled predicted plates.
        Predicted entries carry conf = -1.0 so the debug overlay can label them.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None

        # ── 1. Feed vehicle detections to BYTETracker ─────────────────────────
        if all_vehicles:
            xyxy  = np.array([[x1, y1, x2, y2] for (_, x1, y1, x2, y2, _) in all_vehicles],
                             dtype=np.float32)
            confs = np.array([c   for (*_, c)          in all_vehicles], dtype=np.float32)
            clss  = np.array([cls for (cls, *_)         in all_vehicles], dtype=np.float32)
            det   = _VehicleDetections(xyxy, confs, clss)
        else:
            det   = _VehicleDetections(np.zeros((0, 4)), [], [])

        # BYTETracker returns active tracks: rows of [x1,y1,x2,y2, id,conf,cls,idx]
        active = self._byte.update(det)

        # ── 2. Sync our VehicleTrack dict with BYTETracker's output ───────────
        active_ids = set()
        for row in active:
            x1, y1, x2, y2 = int(row[0]), int(row[1]), int(row[2]), int(row[3])
            tid = int(row[4])
            active_ids.add(tid)

            if tid not in self._track_dict:
                vt    = VehicleTrack((x1, y1, x2, y2), self.frame_idx, self.history_frames)
                vt.id = tid   # use BYTETracker's stable ID instead of the counter
                self._track_dict[tid] = vt
            else:
                self._track_dict[tid].update_box((x1, y1, x2, y2), self.frame_idx)

        # Tracks not returned this frame are missed; drop after gap-fill window
        for tid in list(self._track_dict):
            if tid not in active_ids:
                self._track_dict[tid].mark_missed()
                if self._track_dict[tid].miss_count > self.max_gap_frames:
                    del self._track_dict[tid]

        # ── 3. AR-filter standalone plates ────────────────────────────────────
        # Plates overlapping any current or last-known vehicle zone pass through.
        # Standalone plates (outside all vehicle boxes) must have a realistic AR.
        vehicle_zones = [(v[1], v[2], v[3], v[4]) for v in all_vehicles] + \
                        [t.box for t in self._track_dict.values()]
        filtered = []
        for p in plate_rects:
            if any(_overlaps(p[:4], z) for z in vehicle_zones):
                filtered.append(p)
            else:
                x1, y1, x2, y2 = p[:4]
                h = y2 - y1
                if h > 0 and self.standalone_min_ar <= (x2 - x1) / h <= self.standalone_max_ar:
                    filtered.append(p)
        plate_rects = filtered

        # ── 4. Plate recording + gap-fill (no suppression) ───────────────────────
        #
        # We blur EVERY distinct candidate that survives the confidence and AR
        # filters.  "One plate per vehicle" suppression has been removed because:
        #
        #   • Missing a real plate (privacy failure) is always worse than blurring
        #     an extra region (cosmetic issue).
        #   • A persistent false positive that keeps winning by raw confidence
        #     would otherwise permanently shadow the real plate.
        #
        # merge_overlapping() in detect_plates() still collapses near-identical
        # detections of the *same* plate produced by overlapping SAHI tiles, so
        # we don't blur the same region multiple times.
        #
        # History-aware trajectory recording (for gap-fill only):
        #   Even though we blur all candidates, the PlateHistory still needs a
        #   single position to track for velocity-based gap-fill prediction.
        #   We pick the candidate most consistent with the established trajectory:
        #   - No history yet            → highest confidence
        #   - ≥ GATE_HISTORY confirmed  → prefer overlap with predicted zone;
        #                                 fall back to highest confidence if none
        #                                 overlap (fast vehicle, bad prediction)
        _GATE_HISTORY = 3

        effective = []
        claimed   = set()   # indices of plate_rects already added to effective

        for track in self._track_dict.values():
            indexed = [(i, p) for i, p in enumerate(plate_rects)
                       if i not in claimed and _overlaps(p[:4], track.box)]

            if indexed:
                # Add ALL candidates — blur every distinct detection in this box
                for i, p in indexed:
                    effective.append(p)
                    claimed.add(i)

                # Pick history-consistent winner for trajectory recording only
                if track.plate.has_history and len(track.plate._data) >= _GATE_HISTORY:
                    predicted = track.plate.predict_rect(self.frame_idx,
                                                         self.predict_expand_max)
                    if predicted is not None:
                        gated = [(i, p) for i, p in indexed if _overlaps(p[:4], predicted)]
                        candidates = gated if gated else indexed
                    else:
                        candidates = indexed
                else:
                    candidates = indexed

                _, best = max(candidates, key=lambda ip: ip[1][4] if len(ip[1]) > 4 else 1.0)
                track.plate.record(self.frame_idx, *best[:4],
                                   best[4] if len(best) > 4 else 1.0)
                if gray is not None:
                    track.plate.refresh_lk(gray, *best[:4])
            else:
                # No detection overlaps this vehicle — advance the miss counter
                track.plate.miss_count += 1
                if 1 <= track.plate.miss_count <= self.max_gap_frames \
                        and track.plate.has_history:
                    # LK optical flow first (pixel-level); velocity prediction as fallback
                    predicted = None
                    if gray is not None:
                        predicted = track.plate.predict_rect_lk(
                            gray, self.predict_expand_max
                        )
                    if predicted is None:
                        predicted = track.plate.predict_rect(
                            self.frame_idx, self.predict_expand_max
                        )
                    if predicted is not None:
                        # conf=-1 marks gap-fill; source 'pred' lets the overlay
                        # render these as dashed yellow boxes instead of solid green
                        effective.append((*predicted, -1.0, "pred"))

        # Standalone plates not claimed by any vehicle track pass through as-is
        for i, p in enumerate(plate_rects):
            if i not in claimed:
                effective.append(p)

        self.frame_idx += 1
        return effective, []   # suppressed is always empty — nothing is discarded


# ─── Debug overlay colours ───────────────────────────────────────────────────

_DBG_VEHICLE_COLOR = (255, 100,   0)   # blue
_DBG_GHOST_COLOR      = (180,  80,  80)   # dim teal — tracked vehicle, detector missed
_DBG_SUPPRESSED_COLOR = (120, 120, 120)   # grey    — duplicate plate, suppressed
_DBG_PLATE_COLOR      = (  0, 220,   0)   # green  — raw model detection
_DBG_PREDICT_COLOR = (  0, 220, 220)   # yellow — tracker-predicted (gap fill)
_DBG_BLUR_COLOR    = (  0,   0, 220)   # red    — padded blur region
_DBG_OWN_COLOR     = (  0, 140, 255)   # orange — own plate fixed region
_DBG_FONT          = cv2.FONT_HERSHEY_SIMPLEX


def _dbg_box(img, x1, y1, x2, y2, color, label, thickness=3):
    h_img, w_img = img.shape[:2]
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    fs = max(0.6, (x2 - x1) / 500)
    (tw, th), _ = cv2.getTextSize(label, _DBG_FONT, fs, 2)
    pad = 6
    # Flip label below the box when it would clip above the frame top
    if y1 - th - pad * 2 >= 0:
        bg_y1, bg_y2, ty = y1 - th - pad * 2, y1, y1 - pad
    else:
        bg_y1, bg_y2, ty = y2, y2 + th + pad * 2, y2 + th + pad
    # Clamp label right edge to frame width
    lx = min(x1, w_img - tw - pad * 2)
    cv2.rectangle(img, (lx, bg_y1), (lx + tw + pad * 2, bg_y2), color, -1)
    cv2.putText(img, label, (lx + pad, ty), _DBG_FONT, fs, (255, 255, 255), 2, cv2.LINE_AA)


def draw_debug_overlay(frame, plate_rects, all_vehicles, own_plate_region=None,
                       blur_padding=8, blur_strength=61, tracker=None,
                       suppressed_plates=None,
                       redact_mode="blur", redact_color=(0, 0, 0),
                       overlay_img=None):
    """
    Returns a debug frame that shows exactly what the production output will look like:
      - Redaction (blur / solid colour / image overlay) is applied to all detected
        regions, matching production output exactly
      - Blue box     = vehicle detection  (label: class conf | #id Nf detected)
      - Teal box     = tracked vehicle whose detector dropped this frame (label: gap N/M)
      - Green box    = raw plate detection boundary
      - Yellow box   = tracker-predicted plate (gap fill)
      - Red box      = padded redaction region (what was actually erased)
      - Orange box   = own-plate fixed region
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ── Step 1: apply the real redaction so the frame looks like production ──
    all_rects = list(plate_rects)
    if own_plate_region:
        all_rects.append(own_plate_region)
    if all_rects:
        vis = apply_redaction(vis, all_rects, mode=redact_mode,
                              blur_strength=blur_strength,
                              color=redact_color,
                              overlay_img=overlay_img,
                              padding=blur_padding)

    # ── Step 2: build a lookup of track data keyed by closest vehicle box ─────
    # Maps each track to its detected vehicle (if any) so we can annotate labels.
    track_by_vehicle = {}   # index into all_vehicles → VehicleTrack
    ghost_tracks     = []   # tracks whose vehicle wasn't detected this frame
    if tracker is not None:
        for track in tracker.tracks:
            if track.miss_count == 0:
                # Find the all_vehicles entry closest to this track's box
                best_vi, best_iou = None, 0.0
                for vi, (_, vx1, vy1, vx2, vy2, _) in enumerate(all_vehicles):
                    score = iou(track.box, (vx1, vy1, vx2, vy2))
                    if score > best_iou:
                        best_iou, best_vi = score, vi
                if best_vi is not None and best_iou > 0.1:
                    track_by_vehicle[best_vi] = track
            else:
                ghost_tracks.append(track)

    # ── Step 3: detected vehicle boxes (blue) ────────────────────────────────
    for vi, (cls, x1, y1, x2, y2, conf) in enumerate(all_vehicles):
        track = track_by_vehicle.get(vi)
        if track:
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f} | #{track.id} {track.frames_seen}f"
        else:
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
        _dbg_box(vis, x1, y1, x2, y2, _DBG_VEHICLE_COLOR, label, thickness=3)

    # ── Step 4: ghost vehicle boxes — tracked but not detected this frame ─────
    for track in ghost_tracks:
        x1, y1, x2, y2 = track.box
        label = f"#{track.id} gap {track.miss_count}/{tracker.max_gap_frames} | {track.frames_seen}f"
        _dbg_box(vis, x1, y1, x2, y2, _DBG_GHOST_COLOR, label, thickness=2)

    # ── Step 3: plate boxes — green (detected), yellow (predicted), red (blur) ─
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf = rect[4] if len(rect) > 4 else None

        if conf is not None and conf < 0:      # tracker gap fill (SAHI missed this frame)
            color = _DBG_PREDICT_COLOR
            label = "gap fill"
        else:
            color = _DBG_PLATE_COLOR
            label = f"plate {conf:.2f}" if conf is not None else "plate"

        # Colour-coded boundary
        _dbg_box(vis, x1, y1, x2, y2, color, label, thickness=2)

        # Red: padded region that was blurred
        px1 = max(0, x1 - blur_padding)
        py1 = max(0, y1 - blur_padding)
        px2 = min(w, x2 + blur_padding)
        py2 = min(h, y2 + blur_padding)
        cv2.rectangle(vis, (px1, py1), (px2, py2), _DBG_BLUR_COLOR, 4)

    # ── Step 4: own-plate fixed region (orange) ───────────────────────────────
    if own_plate_region:
        ox1, oy1, ox2, oy2 = own_plate_region
        _dbg_box(vis, ox1, oy1, ox2, oy2, _DBG_OWN_COLOR, "own plate", thickness=3)

    # ── Step 5: suppressed duplicates (grey, thin) ────────────────────────────
    for rect in (suppressed_plates or []):
        x1, y1, x2, y2 = rect[:4]
        conf  = rect[4] if len(rect) > 4 else None
        label = f"dup {conf:.2f}" if conf is not None else "dup"
        _dbg_box(vis, x1, y1, x2, y2, _DBG_SUPPRESSED_COLOR, label, thickness=1)

    return vis


# ─── DEBUG DATA mode (A: extended overlay, B: side HUD panel) ────────────────

# Brand palette (BGR for OpenCV).  Kept in sync with the dsdt.x assets.
_DD_VOID_BG   = (15,  15,  18)
_DD_HUD_BG    = (18,  18,  24)
_DD_HUD_LINE  = (40,  40,  50)
_DD_WHITE     = (240, 240, 240)
_DD_DIM       = (140, 140, 150)
_DD_RED       = (26,  0,   226)    # Signal Red
_DD_BLUE      = (255, 102, 0)      # Data Blue
_DD_GREEN     = (0,   230, 0)      # detection box
_DD_YELLOW    = (0,   220, 220)    # predicted (gap-fill)
_DD_ORANGE    = (0,   140, 255)    # own-plate
_DD_TRAIL     = (255, 200, 80)     # cyan trajectory tail
_DD_GHOST     = (90,  90,  100)    # rejected / dropped

# Source-to-colour map for plate boxes
_DD_SOURCE_COLOR = {
    "sahi": _DD_GREEN,
    "crop": _DD_GREEN,
    "pred": _DD_YELLOW,
    "own":  _DD_ORANGE,
}

_DD_FONT_DIR = os.path.join(os.path.expanduser("~"),
                             ".local/share/fonts/dsdtx")
_DD_F_DISP  = os.path.join(_DD_FONT_DIR, "OrbitronVar.ttf")
_DD_F_BODY  = os.path.join(_DD_FONT_DIR, "Rajdhani-SemiBold.ttf")
_DD_F_MONO  = os.path.join(_DD_FONT_DIR, "JetBrainsMono-Regular.ttf")
_DD_F_MONOB = os.path.join(_DD_FONT_DIR, "JetBrainsMono-Bold.ttf")
_DD_HUD_W   = 320


def _dd_have_pil_fonts() -> bool:
    """All brand fonts present? Falls back to OpenCV Hershey if missing."""
    return all(os.path.exists(p) for p in
               (_DD_F_DISP, _DD_F_BODY, _DD_F_MONO, _DD_F_MONOB))


def _dd_tag(img, x, y, text, bg_bgr, fg_bgr=_DD_WHITE, fs=0.5, pad=4):
    """Solid pill-style tag for plate / vehicle labels."""
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_DUPLEX, fs, 1)
    h_img, w_img = img.shape[:2]
    # Draw above the box if there's room, otherwise below
    if y - th - pad * 2 >= 0:
        bg_y1, bg_y2, ty = y - th - pad * 2, y, y - pad
    else:
        bg_y1, bg_y2, ty = y, y + th + pad * 2, y + th + pad
    lx = min(x, w_img - tw - pad * 2)
    cv2.rectangle(img, (lx, bg_y1), (lx + tw + pad * 2, bg_y2), bg_bgr, -1)
    cv2.putText(img, text, (lx + pad, ty), cv2.FONT_HERSHEY_DUPLEX, fs,
                fg_bgr, 1, cv2.LINE_AA)


def _dd_dashed_rect(img, x1, y1, x2, y2, color, thickness=2, dash=8, gap=4):
    """Dashed rectangle for predicted (gap-fill) plates."""
    step = dash + gap
    for ix in range(x1, x2, step):
        cv2.line(img, (ix, y1), (min(ix+dash, x2), y1), color, thickness)
        cv2.line(img, (ix, y2), (min(ix+dash, x2), y2), color, thickness)
    for iy in range(y1, y2, step):
        cv2.line(img, (x1, iy), (x1, min(iy+dash, y2)), color, thickness)
        cv2.line(img, (x2, iy), (x2, min(iy+dash, y2)), color, thickness)


def draw_extended_overlay(frame, plate_rects, all_vehicles, tracker=None,
                          blur_padding=8, blur_strength=61,
                          redact_mode="blur", redact_color=(0, 0, 0),
                          overlay_img=None, rejected_plates=None):
    """
    DEBUG DATA - mode A.  Returns a frame with the actual redaction applied
    plus rich annotations: source-tagged plate boxes, vehicle boxes with
    track IDs, ghost tracks, plate trajectory trails, and a top HUD strip
    summarising what's in this frame.
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ── Apply real redaction so the user sees production output ───────────
    if plate_rects:
        vis = apply_redaction(vis, plate_rects,
                              mode=redact_mode,
                              blur_strength=blur_strength,
                              color=redact_color,
                              overlay_img=overlay_img,
                              padding=blur_padding)

    # ── Rejected detections (grey, dashed) — below confidence threshold ───
    # Drawn first so accepted/predicted boxes render on top of them.
    for rect in (rejected_plates or []):
        x1, y1, x2, y2 = rect[:4]
        conf = rect[4] if len(rect) > 4 else None
        _dd_dashed_rect(vis, x1, y1, x2, y2, _DD_GHOST, thickness=1)
        label = f"REJ {conf:.2f}" if conf is not None else "REJ"
        _dd_tag(vis, x1, y1, label, _DD_GHOST, fs=0.4)

    # ── Vehicle boxes (blue) with track-aware rich tag ────────────────────
    track_by_vidx = {}
    ghost_tracks  = []
    if tracker is not None:
        for tr in tracker.tracks:
            if tr.miss_count == 0:
                # Find closest current vehicle box for labelling
                best_vi, best_iou = None, 0.0
                for vi, (_, vx1, vy1, vx2, vy2, _) in enumerate(all_vehicles):
                    score = iou(tr.box, (vx1, vy1, vx2, vy2))
                    if score > best_iou:
                        best_iou, best_vi = score, vi
                if best_vi is not None and best_iou > 0.1:
                    track_by_vidx[best_vi] = tr
            else:
                ghost_tracks.append(tr)

    for vi, (cls, x1, y1, x2, y2, conf) in enumerate(all_vehicles):
        tr = track_by_vidx.get(vi)
        if tr:
            label = (f"{VEHICLE_CLASSES[cls]} {conf:.2f} | "
                     f"#TRK{tr.id} age:{tr.frames_seen}f miss:{tr.miss_count}")
        else:
            label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), _DD_BLUE, 2)
        _dd_tag(vis, x1, y1, label, _DD_BLUE)

    # ── Ghost tracks (detector missed this frame, tracker still holds) ────
    for tr in ghost_tracks:
        x1, y1, x2, y2 = tr.box
        label = f"GHOST #{tr.id} gap {tr.miss_count}/{tracker.max_gap_frames}"
        cv2.rectangle(vis, (x1, y1), (x2, y2), _DD_GHOST, 1)
        _dd_tag(vis, x1, y1, label, _DD_GHOST, fs=0.45)

    # ── Plate trajectory trails — fading cyan line per track ──────────────
    if tracker is not None:
        for tr in tracker.tracks:
            if not tr.plate.has_history:
                continue
            # Use the centres of the recorded plate positions
            pts = [(int(d[1]), int(d[2])) for d in tr.plate._data[-15:]]
            for i in range(len(pts) - 1):
                alpha = 0.25 + 0.75 * (i / max(1, len(pts) - 1))
                c = tuple(int(v * alpha) for v in _DD_TRAIL)
                cv2.line(vis, pts[i], pts[i+1], c, 2, cv2.LINE_AA)

    # ── Plate boxes with source tag ───────────────────────────────────────
    src_counts = {"sahi": 0, "crop": 0, "pred": 0, "own": 0}
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf   = rect[4] if len(rect) > 4 else None
        source = rect[5] if len(rect) > 5 else "sahi"
        src_counts[source] = src_counts.get(source, 0) + 1
        color = _DD_SOURCE_COLOR.get(source, _DD_GREEN)

        if source == "pred":
            _dd_dashed_rect(vis, x1, y1, x2, y2, color, thickness=2)
            _dd_tag(vis, x1, y2 + 20, "PRED  gap-fill", color,
                    fg_bgr=(0, 0, 0), fs=0.45)
        else:
            cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
            if conf is not None and conf >= 0:
                _dd_tag(vis, x1, y1, f"{source.upper()} {conf:.2f}", color,
                        fg_bgr=(0, 0, 0))
            else:
                _dd_tag(vis, x1, y1, source.upper(), color, fg_bgr=(0, 0, 0))

    # ── Top status strip ──────────────────────────────────────────────────
    strip_h = 38
    cv2.rectangle(vis, (0, 0), (w, strip_h), _DD_VOID_BG, -1)
    cv2.rectangle(vis, (0, strip_h - 1), (w, strip_h), _DD_BLUE, 1)
    strip_text = (f"DEBUG DATA   VEH {len(all_vehicles)}   "
                  f"PLT {len(plate_rects)} "
                  f"(SAHI {src_counts['sahi']}, crop+ {src_counts['crop']}, "
                  f"pred {src_counts['pred']}, own {src_counts['own']})   "
                  f"REJ {len(rejected_plates or [])}")
    cv2.putText(vis, strip_text, (12, 26),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, _DD_WHITE, 1, cv2.LINE_AA)

    return vis


def _dd_font(path, size, variation=None):
    """Load PIL font; honours OpenType variation axes (e.g. Orbitron Bold)."""
    from PIL import ImageFont
    fnt = ImageFont.truetype(path, size)
    if variation is not None:
        try:
            fnt.set_variation_by_name(variation)
        except OSError:
            pass
    return fnt


def draw_hud_panel(frame_h, telemetry):
    """
    DEBUG DATA - mode B.  Renders a 320×frame_h BGR side-panel with:
      ULTRA-style header, FRAME / VEHICLES / PLATES / TRACKS sections and
      a TIMINGS (ms) block.  Falls back to a Hershey-rendered panel if the
      brand TTFs aren't installed locally.
    """
    if not _dd_have_pil_fonts():
        return _draw_hud_panel_fallback(frame_h, telemetry)

    from PIL import Image, ImageDraw

    hud = np.full((frame_h, _DD_HUD_W, 3), _DD_HUD_BG, dtype=np.uint8)
    # Left-edge accent bar (Data Blue)
    cv2.rectangle(hud, (0, 0), (3, frame_h), _DD_BLUE, -1)

    pil = Image.fromarray(cv2.cvtColor(hud, cv2.COLOR_BGR2RGB)).convert("RGBA")
    over = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(over)

    fnt_display = _dd_font(_DD_F_DISP,  22, "Bold")
    fnt_body    = _dd_font(_DD_F_BODY,  20)
    fnt_mono    = _dd_font(_DD_F_MONO,  17)
    fnt_mono_b  = _dd_font(_DD_F_MONOB, 22)
    fnt_small   = _dd_font(_DD_F_DISP,  14, "Bold")

    # Cursor (vertical) and helpers — use a mutable container so nested fns
    # can advance the cursor without leaning on `nonlocal`.
    x0   = 18
    yp   = [18]

    # PIL uses RGB so colours need a flip from our BGR brand constants.
    def _rgb(bgr): return (bgr[2], bgr[1], bgr[0])
    BLUE_RGB  = _rgb(_DD_BLUE)  + (255,)
    RED_RGB   = _rgb(_DD_RED)   + (255,)
    WHITE_RGB = _rgb(_DD_WHITE) + (255,)
    DIM_RGB   = _rgb(_DD_DIM)   + (255,)
    GREEN_RGB = _rgb(_DD_GREEN) + (255,)

    def section(title, color):
        draw.text((x0, yp[0]), title, font=fnt_body, fill=color)
        yp[0] += 24

    def kv(label, value, value_color=WHITE_RGB):
        draw.text((x0, yp[0]),       label, font=fnt_mono,   fill=DIM_RGB)
        draw.text((x0 + 120, yp[0]), value, font=fnt_mono_b, fill=value_color)
        yp[0] += 22

    def bar(label, val_ms, max_ms, color):
        draw.text((x0,         yp[0]),     label,           font=fnt_mono,   fill=DIM_RGB)
        draw.text((x0 + 230,   yp[0] - 1), f"{val_ms:5.0f}", font=fnt_mono_b, fill=WHITE_RGB)
        bx, by, bw_, bh_ = x0 + 70, yp[0] + 5, 150, 10
        draw.rectangle([(bx, by), (bx + bw_, by + bh_)], fill=(40, 40, 50, 255))
        fill_w = int(bw_ * min(1.0, val_ms / max_ms))
        draw.rectangle([(bx, by), (bx + fill_w, by + bh_)], fill=color)
        yp[0] += 22

    # ── Header ─────────────────────────────────────────────────────────────
    draw.text((x0, yp[0]), "DEBUG  DATA", font=fnt_display, fill=WHITE_RGB)
    yp[0] += 32
    draw.rectangle([(x0, yp[0]), (_DD_HUD_W - 18, yp[0] + 2)], fill=BLUE_RGB)
    yp[0] += 18

    # ── FRAME ─────────────────────────────────────────────────────────────
    section("FRAME", BLUE_RGB)
    frame_idx    = telemetry.get("frame_num", 0)
    total_frames = telemetry.get("total_frames", 0)
    fps_target   = telemetry.get("fps_target", 30.0)
    timings      = telemetry.get("timings", {})
    ts_sec       = frame_idx / fps_target if fps_target else 0
    ts_min, ts_s = divmod(ts_sec, 60)
    total_ms     = sum(timings.values()) if timings else 0
    inst_fps     = (1000.0 / total_ms) if total_ms > 0 else 0
    kv("idx",  f"{frame_idx:05d} / {total_frames}")
    kv("time", f"{int(ts_min):02d}:{ts_s:06.3f}")
    kv("fps",  f"{inst_fps:5.1f}",
       value_color=GREEN_RGB if inst_fps >= fps_target * 0.5 else RED_RGB)
    yp[0] += 8

    # ── VEHICLES ──────────────────────────────────────────────────────────
    vehicles = telemetry.get("vehicles", [])
    tracks   = telemetry.get("tracks", [])
    section("VEHICLES", BLUE_RGB)
    kv("found",   f"{len(vehicles)}")
    kv("tracked", f"{len(tracks)}")
    yp[0] += 8

    # ── PLATES ────────────────────────────────────────────────────────────
    plates = telemetry.get("plates", [])
    by_src = {"sahi": 0, "crop": 0, "pred": 0, "own": 0}
    for r in plates:
        s = r[5] if len(r) > 5 else "sahi"
        by_src[s] = by_src.get(s, 0) + 1
    section("PLATES", BLUE_RGB)
    kv("SAHI",  f"{by_src['sahi']}",
       value_color=WHITE_RGB if by_src["sahi"] else DIM_RGB)
    kv("crop+", f"{by_src['crop']}",
       value_color=WHITE_RGB if by_src["crop"] else DIM_RGB)
    kv("pred",  f"{by_src['pred']}",
       value_color=WHITE_RGB if by_src["pred"] else DIM_RGB)
    kv("own",   f"{by_src['own']}",
       value_color=WHITE_RGB if by_src["own"] else DIM_RGB)
    yp[0] += 8

    # ── TRACKS list (up to 6 rows) ────────────────────────────────────────
    section("TRACKS", RED_RGB)
    if not tracks:
        draw.text((x0, yp[0]), "—  no active tracks",
                  font=fnt_mono, fill=DIM_RGB)
        yp[0] += 24
    else:
        for tr in tracks[:6]:
            draw.text((x0,        yp[0]), f"#{tr.id}",
                      font=fnt_mono_b, fill=WHITE_RGB)
            draw.text((x0 + 50,   yp[0]),
                      f"age {tr.frames_seen}f",
                      font=fnt_mono, fill=(180, 180, 200, 255))
            draw.text((x0 + 165,  yp[0]),
                      f"miss {tr.miss_count}",
                      font=fnt_mono, fill=(180, 180, 200, 255))
            yp[0] += 22
        if len(tracks) > 6:
            draw.text((x0, yp[0]), f"+ {len(tracks) - 6} more...",
                      font=fnt_mono, fill=DIM_RGB)
            yp[0] += 22
    yp[0] += 4

    # ── TIMINGS (ms) ──────────────────────────────────────────────────────
    section("TIMINGS (ms)", RED_RGB)
    detect_ms = timings.get("detect", 0)
    track_ms  = timings.get("track",  0)
    render_ms = timings.get("render", 0)
    max_ms    = max(60.0, detect_ms * 1.1)
    bar("detect", detect_ms, max_ms, RED_RGB)
    bar("track",  track_ms,  max_ms, RED_RGB)
    bar("render", render_ms, max_ms, RED_RGB)
    yp[0] += 4
    draw.rectangle([(x0, yp[0]), (_DD_HUD_W - 18, yp[0] + 1)],
                   fill=(70, 70, 80, 255))
    yp[0] += 8
    bar("TOTAL",  total_ms, max(80.0, total_ms * 1.1), BLUE_RGB)

    # ── Footer ────────────────────────────────────────────────────────────
    draw.text((x0, frame_h - 30), "DSDT.X / AI VISION",
              font=fnt_small, fill=DIM_RGB)

    pil.alpha_composite(over)
    return cv2.cvtColor(np.array(pil.convert("RGB")), cv2.COLOR_RGB2BGR)


def _draw_hud_panel_fallback(frame_h, telemetry):
    """OpenCV-only HUD when brand TTFs aren't available."""
    hud = np.full((frame_h, _DD_HUD_W, 3), _DD_HUD_BG, dtype=np.uint8)
    cv2.rectangle(hud, (0, 0), (3, frame_h), _DD_BLUE, -1)
    y = 30
    cv2.putText(hud, "DEBUG DATA", (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.9,
                _DD_WHITE, 2, cv2.LINE_AA)
    y += 26
    cv2.line(hud, (18, y), (_DD_HUD_W - 18, y), _DD_BLUE, 2); y += 18

    timings = telemetry.get("timings", {})
    total_ms = sum(timings.values()) if timings else 0
    for line in (
        f"frame  {telemetry.get('frame_num', 0):>5d}/{telemetry.get('total_frames', 0)}",
        f"VEH    {len(telemetry.get('vehicles', []))}",
        f"PLT    {len(telemetry.get('plates', []))}",
        f"TRK    {len(telemetry.get('tracks', []))}",
        "",
        f"detect {timings.get('detect', 0):6.1f} ms",
        f"track  {timings.get('track',  0):6.1f} ms",
        f"render {timings.get('render', 0):6.1f} ms",
        f"TOTAL  {total_ms:6.1f} ms",
    ):
        cv2.putText(hud, line, (18, y), cv2.FONT_HERSHEY_DUPLEX, 0.55,
                    _DD_WHITE, 1, cv2.LINE_AA)
        y += 24
    return hud


def blur_license_plates(
    input_path: str,
    output_path: str,
    start_time: float = None,
    end_time: float = None,
    blur_strength: int = 61,
    blur_padding: int = 8,
    vehicle_conf: float = 0.3,
    plate_conf: float = 0.15,
    plate_conf_in_vehicle: float = 0.07,
    sahi_slice_size: int = 640,
    sahi_overlap: float = 0.2,
    own_plate_region: tuple = None,
    vehicle_filter: str = "all",
    preset: str = "medium",
    quality: int = 18,            # final HEVC quality (CRF / CQ) — see config.toml
    tmp_dir: str = "auto",
    debug: bool = False,
    debug_overlay: bool = False,   # extended in-frame overlay (DEBUG DATA mode)
    debug_hud: bool = False,       # side HUD panel (DEBUG DATA mode)
    tracking_enabled: bool = True,
    max_gap_frames: int = 8,
    history_frames: int = 15,
    min_vehicle_conf: float = 0.60,
    standalone_min_ar: float = 1.2,
    standalone_max_ar: float = 6.0,
    predict_expand_max: int = 20,
    detect_scale: float = 1.0,
    sharpen: bool = False,
    sharpen_amount: float = 1.5,
    sharpen_sigma: float = 1.0,
    vehicle_crop_scale: float = 1.0,
    redact_mode: str = "blur",
    redact_color: tuple = (0, 0, 0),
    redact_image_path: str = None,
    matte_feather: int = 0,
    matte_codec: str = "prores",
):
    if tmp_dir == "auto":
        tmp_dir = os.path.join(tempfile.gettempdir(), "plate-blur-tmp")

    # Matte mode dictates the container; correct the extension before we print it.
    if redact_mode == "matte":
        output_path = resolve_matte_output_path(output_path, matte_codec)
    # Capture wall-clock start so we can report total + processing-only time
    # at the end. Uses a different name from the `start_time` parameter (which
    # is the video trim start, not a timestamp).
    _run_start_ts = time.perf_counter()

    # Load the overlay image (if any) once, up front
    overlay_img = None
    if redact_mode == "image" and redact_image_path:
        overlay_img = load_overlay_image(redact_image_path)

    print(f"\n{'='*60}")
    print(f"  License Plate Redaction Tool")
    print(f"{'='*60}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    if start_time is not None:
        print(f"  Clip  : {start_time:.1f}s to {end_time:.1f}s")
    if vehicle_filter != "all":
        print(f"  Filter: {vehicle_filter} plates only")
    print(f"  Plate conf : {plate_conf} (global)  |  {plate_conf_in_vehicle} (inside vehicle boxes)")
    if own_plate_region:
        print(f"  Own plate region (always redacted): {own_plate_region}")
    print(f"  Mode  : {redact_mode}", end="")
    if redact_mode == "color":
        print(f"  (BGR {redact_color})")
    elif redact_mode == "image" and overlay_img is not None:
        print(f"  (overlay {redact_image_path})")
    else:
        print()
    if detect_scale < 1.0:
        print(f"  Detect : {detect_scale:.2f}× scale  (detection at {detect_scale*100:.0f}% res, blur at full res)")
    if sharpen:
        print(f"  Sharpen: enabled  (amount={sharpen_amount}, sigma={sharpen_sigma})")
    if vehicle_crop_scale > 1.0:
        print(f"  Crop upscale: {vehicle_crop_scale:.1f}×  (per-vehicle plate pass on full-res crops)")
    if debug:
        print(f"  Mode  : DEBUG (blur applied + detection overlay)")
    if tracking_enabled:
        print(f"  Track : enabled  (gap={max_gap_frames} frames, history={history_frames}, "
              f"expand_max={predict_expand_max}px)")
    print(f"{'='*60}\n")

    info = get_video_info(input_path)
    width, height, fps = info["width"], info["height"], info["fps"]
    print(f"  Video : {width}x{height} @ {fps:.3f}fps  [{info['codec']}]")

    print("  Loading models...")
    vehicle_model, plate_model, device = load_models(
        plate_conf=min(plate_conf, plate_conf_in_vehicle)
    )
    print(f"  Models ready  |  vehicle detector + license plate detector\n")

    tracker = SceneTracker(
        max_gap_frames=max_gap_frames,
        history_frames=history_frames,
        min_vehicle_conf=min_vehicle_conf,
        standalone_min_ar=standalone_min_ar,
        standalone_max_ar=standalone_max_ar,
        predict_expand_max=predict_expand_max,
    ) if tracking_enabled else None

    frame_size     = width * height * 3
    total_frames   = estimate_frame_count(info, start_time, end_time)

    # When the HUD side-panel is enabled, output frames are wider than input.
    # The encoder is told this widened size; ffmpeg's `-s WxH` reads the raw
    # buffer at the new dimensions so the side panel survives encoding.
    enc_width  = width + _DD_HUD_W if debug_hud else width
    enc_height = height

    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False, dir=tmp_dir) as tmp:
        tmp_path = tmp.name

    try:
        # CUVID pre-check: probe one decoded frame to detect unsupported codec
        # profiles (e.g. iPhone 10-bit HEVC, some MOV variants).  CUVID failure
        # is silent — it produces 0 bytes — which would otherwise create an
        # empty intermediate and an unplayable output file.
        video_codec = info.get("codec")
        if video_codec in _CUVID_DECODERS:
            _probe_end = (start_time or 0.0) + 1.0 / fps
            _probe = subprocess.Popen(
                build_ffmpeg_extract(input_path, start_time, _probe_end, codec=video_codec),
                stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
            )
            _probe_ok = len(_probe.stdout.read(frame_size)) >= frame_size
            _probe.stdout.close()
            _probe.wait()
            if not _probe_ok:
                print(f"  Warning: {_CUVID_DECODERS[video_codec]} decode failed for this file "
                      f"— falling back to software decode")
                video_codec = None   # None → build_ffmpeg_extract uses software path

        extract_cmd = build_ffmpeg_extract(input_path, start_time, end_time, codec=video_codec)
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

        extract_proc = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
        encode_proc  = subprocess.Popen(encode_cmd,  stdin=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)

        frame_num    = 0
        total_plates = 0
        _process_start_ts = time.perf_counter()

        with tqdm(total=total_frames, unit="frame", dynamic_ncols=True,
                  bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} frames "
                             "[{elapsed}<{remaining}, {rate_fmt}] {postfix}") as pbar:
            try:
                while True:
                    raw = extract_proc.stdout.read(frame_size)
                    if len(raw) < frame_size:
                        break

                    frame = np.frombuffer(raw, dtype=np.uint8).reshape(
                        (height, width, 3)).copy()

                    # Per-frame timing dict for the debug HUD's TIMINGS bars.
                    # Each stage is wrapped with perf_counter() deltas in ms.
                    timings = {}
                    _t0 = time.perf_counter()

                    suppressed_plates = []   # reset each frame; populated by tracker or dedup
                    plates, vehicles, rejected_plates = detect_plates(
                        frame, vehicle_model, plate_model,
                        vehicle_conf=vehicle_conf,
                        vehicle_filter=vehicle_filter,
                        plate_conf=plate_conf,
                        plate_conf_in_vehicle=plate_conf_in_vehicle,
                        sahi_slice_size=sahi_slice_size,
                        sahi_overlap=sahi_overlap,
                        detect_scale=detect_scale,
                        sharpen=sharpen,
                        sharpen_amount=sharpen_amount,
                        sharpen_sigma=sharpen_sigma,
                        vehicle_crop_scale=vehicle_crop_scale,
                        collect_rejected=debug_overlay,
                    )

                    timings["detect"] = (time.perf_counter() - _t0) * 1000
                    _t1 = time.perf_counter()

                    filter_classes  = VEHICLE_FILTER_MAP.get(vehicle_filter, set(VEHICLE_CLASSES))
                    vehicle_boxes   = [(v[1], v[2], v[3], v[4]) for v in vehicles
                                       if v[0] in filter_classes]

                    if tracker is not None:
                        # Tracker owns dedup: it uses plate history to gate out
                        # spatially inconsistent false positives before recording.
                        # suppress_duplicate_plates is intentionally skipped here
                        # so the tracker sees all candidates, not just the
                        # highest-confidence one chosen without temporal context.
                        plates, suppressed_plates = tracker.update(vehicles, plates, frame)
                    else:
                        # Tracking disabled: AR filter only — no suppression
                        vehicle_zones = vehicle_boxes
                        ar_filtered = []
                        for p in plates:
                            if any(_overlaps(p[:4], z) for z in vehicle_zones):
                                ar_filtered.append(p)
                            else:
                                x1, y1, x2, y2 = p[:4]
                                h = y2 - y1
                                if h > 0 and standalone_min_ar <= (x2-x1)/h <= standalone_max_ar:
                                    ar_filtered.append(p)
                        plates = ar_filtered

                    # Always include own-plate region
                    timings["track"] = (time.perf_counter() - _t1) * 1000
                    _t2 = time.perf_counter()

                    if own_plate_region:
                        # Tag own-plate with conf 1.0 and source 'own' so the
                        # debug overlay can label / colour it distinctly.
                        own_with_source = (*own_plate_region, 1.0, "own")
                        plates = list(plates) + [own_with_source]

                    if plates:
                        total_plates += len(plates)

                    if redact_mode == "matte":
                        frame = render_matte_frame(
                            frame.shape[0], frame.shape[1], plates or [],
                            padding=blur_padding, feather=matte_feather)
                    elif debug_overlay:
                        frame = draw_extended_overlay(
                            frame, plates, vehicles,
                            tracker=tracker,
                            blur_padding=blur_padding,
                            blur_strength=blur_strength,
                            redact_mode=redact_mode,
                            redact_color=redact_color,
                            overlay_img=overlay_img,
                            rejected_plates=rejected_plates,
                        )
                    elif debug:
                        frame = draw_debug_overlay(frame, plates, vehicles,
                                                   own_plate_region=own_plate_region,
                                                   blur_padding=blur_padding,
                                                   blur_strength=blur_strength,
                                                   tracker=tracker,
                                                   suppressed_plates=suppressed_plates,
                                                   redact_mode=redact_mode,
                                                   redact_color=redact_color,
                                                   overlay_img=overlay_img)
                    elif plates:
                        frame = apply_redaction(frame, plates,
                                                mode=redact_mode,
                                                blur_strength=blur_strength,
                                                color=redact_color,
                                                overlay_img=overlay_img,
                                                padding=blur_padding)

                    timings["render"] = (time.perf_counter() - _t2) * 1000

                    # HUD side panel (DEBUG DATA mode B) — appended to the right
                    if debug_hud:
                        hud = draw_hud_panel(
                            frame_h    = frame.shape[0],
                            telemetry  = {
                                "frame_num"   : frame_num,
                                "total_frames": total_frames,
                                "fps_target"  : fps,
                                "vehicles"    : vehicles,
                                "plates"      : plates,
                                "tracks"      : tracker.tracks if tracker else [],
                                "timings"     : timings,
                            },
                        )
                        frame = np.concatenate([frame, hud], axis=1)

                    encode_proc.stdin.write(frame.tobytes())
                    frame_num += 1

                    pbar.update(1)
                    pbar.set_postfix(plates=total_plates, refresh=False)

            finally:
                extract_proc.stdout.close()
                extract_proc.wait()
                encode_proc.stdin.close()
                encode_proc.wait()

        # Wall-clock processing time = detection + draw loop, before muxing.
        _elapsed_process = time.perf_counter() - _process_start_ts

        action = "annotated" if debug else "redacted"
        print(f"\n  Processed : {frame_num} frames")
        print(f"  Detections: {total_plates} plate regions {action}")

        if redact_mode == "matte":
            print(f"  Matte written directly to {output_path} "
                  f"(codec: {matte_codec}, no audio, no HEVC mux).")
        else:
            print("  Encoding final output (lossless HEVC + audio sync fix)...")
            os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
            mux_audio(tmp_path, input_path, output_path, start_time, end_time, fps,
                      total_frames=frame_num, preset=preset, tmp_dir=tmp_dir,
                      quality=quality)

        # End-of-run summary: hardware, timing, throughput, mode, settings.
        _elapsed_total = time.perf_counter() - _run_start_ts
        _print_run_summary(
            elapsed_total   = _elapsed_total,
            elapsed_process = _elapsed_process,
            device          = device,
            info            = info,
            input_path      = input_path,
            output_path     = output_path,
            frame_num       = frame_num,
            total_plates    = total_plates,
            redact_mode     = redact_mode,
            redact_color    = redact_color,
            redact_image_path = redact_image_path,
            vehicle_filter  = vehicle_filter,
        )
        return frame_num, total_plates

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    cfg = load_config()
    det  = cfg["detection"]
    sahi = cfg["sahi"]
    blr  = cfg["blur"]
    out  = cfg["output"]
    trk  = cfg.get("tracking", {})
    pre  = cfg.get("preprocessing", {})
    red  = cfg.get("redact", {})

    parser = argparse.ArgumentParser(
        description="Blur license plates in video with zero quality loss.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full video
  python blur_plates.py final.mov output.mov

  # Clip from 2:48 to 2:51
  python blur_plates.py final.mov sample.mov --start 2:48 --end 2:51

  # Camera mounted behind own plate (Insta360 on motorbike etc.)
  python blur_plates.py final.mov output.mov --own-plate 1700,900,2200,1100

  # Motorbike-only with lower in-vehicle plate threshold
  python blur_plates.py final.mov output.mov --vehicles motorbike --plate-conf-in-vehicle 0.05

  # Debug mode — see what gets detected without blurring
  python blur_plates.py final.mov debug.mov --start 0:10 --end 0:20 --debug

  # Replace plates with a solid colour instead of blurring
  python blur_plates.py final.mov output.mov --mode color --color 255,0,0

  # Overlay a custom image (logo / sticker / portrait) onto every plate
  python blur_plates.py final.mov output.mov --mode image --image my_sticker.png
        """,
    )
    parser.add_argument("input",  help="Input video path")
    parser.add_argument("output", help="Output video path")
    parser.add_argument("--start", help="Start time MM:SS or HH:MM:SS", default=None)
    parser.add_argument("--end",   help="End time   MM:SS or HH:MM:SS", default=None)
    parser.add_argument("--blur",  type=int,   default=blr["strength"],
                        help=f"Blur kernel size, odd (default: {blr['strength']})")
    parser.add_argument("--conf",  type=float, default=det["vehicle_conf"],
                        help=f"Vehicle detection confidence (default: {det['vehicle_conf']})")
    parser.add_argument("--plate-conf", dest="plate_conf", type=float,
                        default=det["plate_conf"],
                        help=f"Plate confidence — full frame (default: {det['plate_conf']})")
    parser.add_argument("--plate-conf-in-vehicle", dest="plate_conf_in_vehicle", type=float,
                        default=det["plate_conf_in_vehicle"],
                        help=f"Plate confidence inside vehicle boxes (default: {det['plate_conf_in_vehicle']})")
    parser.add_argument("--own-plate", dest="own_plate", default=None,
                        metavar="x1,y1,x2,y2",
                        help="Fixed region to always blur (e.g. camera behind own plate)")
    parser.add_argument("--vehicles", dest="vehicles", default="all",
                        choices=["all", "motorbike", "car", "bus", "truck"],
                        help="Only blur plates on the specified vehicle type (default: all)")
    parser.add_argument("--debug", action="store_true",
                        help="Write detection overlay video instead of blurring "
                             "(blue=vehicles, green=plate regions, orange=own plate)")
    parser.add_argument("--debug-overlay", dest="debug_overlay",
                        action="store_true",
                        help="DEBUG DATA mode (A): rich in-frame overlay with source "
                             "tags (SAHI/crop+/pred), trajectory trails and ghost "
                             "boxes for tracked vehicles whose detector missed.")
    parser.add_argument("--debug-hud", dest="debug_hud", action="store_true",
                        help="DEBUG DATA mode (B): add a brand-styled side panel "
                             "with frame#, counts, track list and per-stage timings. "
                             "Output video gets wider by ~320 px.")
    parser.add_argument("--detect-scale", dest="detect_scale", type=float,
                        default=float(det.get("detect_scale", 1.0)),
                        help="Fraction of resolution used for detection (default: "
                             f"{det.get('detect_scale', 1.0)}).  "
                             "0.5 = ~5× faster on 4K, minimal accuracy loss at "
                             "typical dashcam distances. Blur always at full res.")
    parser.add_argument("--mode", dest="mode",
                        default=red.get("mode", "blur"),
                        choices=["blur", "color", "image"],
                        help="Redaction style applied to detected plates "
                             "(default: blur). "
                             "color = solid fill, image = stretched overlay.")
    parser.add_argument("--color", dest="color",
                        default=red.get("color", "0,0,0"),
                        metavar="R,G,B",
                        help="Solid fill colour when --mode color (default: 0,0,0 = black). "
                             "Values 0-255.")
    parser.add_argument("--image", dest="image",
                        default=red.get("image", None),
                        metavar="PATH",
                        help="Overlay image when --mode image. PNG with alpha is supported. "
                             "The image is stretched to fill each plate rectangle.")

    args = parser.parse_args()

    start_sec = time_to_seconds(args.start) if args.start else None
    end_sec   = time_to_seconds(args.end)   if args.end   else None

    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        print("Error: --end must be after --start")
        sys.exit(1)

    own_plate = parse_region(args.own_plate) if args.own_plate else None

    # Validate redaction-mode prerequisites
    redact_color = parse_color(args.color) if args.mode == "color" else (0, 0, 0)
    if args.mode == "image":
        if not args.image:
            print("Error: --mode image requires --image PATH")
            sys.exit(1)
        if not os.path.exists(args.image):
            print(f"Error: overlay image not found: {args.image}")
            sys.exit(1)

    blur_license_plates(
        input_path=args.input,
        output_path=args.output,
        start_time=start_sec,
        end_time=end_sec,
        blur_strength=args.blur,
        blur_padding=blr["padding"],
        vehicle_conf=args.conf,
        plate_conf=args.plate_conf,
        plate_conf_in_vehicle=args.plate_conf_in_vehicle,
        sahi_slice_size=sahi["slice_size"],
        sahi_overlap=sahi["overlap"],
        own_plate_region=own_plate,
        vehicle_filter=args.vehicles,
        preset=out["preset"],
        quality=int(out.get("quality", 18)),
        tmp_dir=out["tmp_dir"],
        debug=args.debug,
        debug_overlay=args.debug_overlay,
        debug_hud=args.debug_hud,
        tracking_enabled=bool(trk.get("enabled", True)),
        max_gap_frames=int(trk.get("max_gap_frames", 8)),
        history_frames=int(trk.get("history_frames", 15)),
        min_vehicle_conf=float(trk.get("min_vehicle_conf", 0.60)),
        predict_expand_max=int(trk.get("predict_expand_max", 20)),
        standalone_min_ar=float(det.get("standalone_min_ar", 1.2)),
        standalone_max_ar=float(det.get("standalone_max_ar", 6.0)),
        detect_scale=args.detect_scale,
        sharpen=bool(pre.get("sharpen", False)),
        sharpen_amount=float(pre.get("sharpen_amount", 1.5)),
        sharpen_sigma=float(pre.get("sharpen_sigma", 1.0)),
        vehicle_crop_scale=float(pre.get("vehicle_crop_scale", 1.0)),
        redact_mode=args.mode,
        redact_color=redact_color,
        redact_image_path=args.image if args.mode == "image" else None,
    )


if __name__ == "__main__":
    main()
