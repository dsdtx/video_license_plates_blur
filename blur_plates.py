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
import numpy as np
import subprocess
import json
import os
import sys
import argparse
import tempfile
import threading
import tomllib
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


PLATE_MODEL_PATH = os.path.join(os.path.dirname(__file__), "license-plate-finetune-v1n.pt")
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


def detect_plates(frame, vehicle_model, plate_model, device="cpu", vehicle_conf=0.3,
                  vehicle_filter="all", plate_conf=0.15, plate_conf_in_vehicle=0.07,
                  sahi_slice_size=640, sahi_overlap=0.2):
    """
    Returns (plate_rects, all_vehicles) where:
      plate_rects  — list of (x1, y1, x2, y2, conf) regions to blur
      all_vehicles — list of (cls_id, x1, y1, x2, y2, conf) for every detected vehicle

    Dual-confidence strategy:
      - Plates inside a vehicle bounding box use plate_conf_in_vehicle (lower).
      - Plates outside any vehicle box use plate_conf (stricter).
      This lets you catch blurry / angled plates on motorbikes without flooding
      the full frame with false positives.
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
    # Model threshold is pre-set to min(plate_conf, plate_conf_in_vehicle) in
    # load_models() so no valid detection is thrown away before we can filter.
    plate_model.confidence_threshold = min(plate_conf, plate_conf_in_vehicle)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = get_sliced_prediction(
        frame_rgb, plate_model,
        slice_height=sahi_slice_size, slice_width=sahi_slice_size,
        overlap_height_ratio=sahi_overlap, overlap_width_ratio=sahi_overlap,
        verbose=0,
    )

    # ── Step 3: context-aware confidence filtering ────────────────────────────
    plate_rects = []
    for det in result.object_prediction_list:
        x1, y1 = int(det.bbox.minx), int(det.bbox.miny)
        x2, y2 = int(det.bbox.maxx), int(det.bbox.maxy)
        if x2 <= x1 or y2 <= y1:
            continue
        conf       = det.score.value
        plate      = (x1, y1, x2, y2)
        in_vehicle = any(_overlaps(plate, vb) for vb in filter_boxes)

        if vehicle_filter != "all" and not in_vehicle:
            continue  # vehicle filter requires vehicle overlap

        required = plate_conf_in_vehicle if in_vehicle else plate_conf
        if conf < required:
            continue

        plate_rects.append((x1, y1, x2, y2, conf))

    # ── Step 4: dedup ─────────────────────────────────────────────────────────
    return merge_overlapping(plate_rects), all_vehicles


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


def build_ffmpeg_extract(input_path, start_sec, end_sec):
    """Build ffmpeg command that pipes raw BGR frames to stdout."""
    cmd = ["ffmpeg", "-y"]
    if start_sec is not None:
        cmd += ["-ss", f"{start_sec:.6f}"]
    cmd += ["-i", input_path]
    if end_sec is not None:
        duration = end_sec - (start_sec or 0.0)
        cmd += ["-t", f"{duration:.6f}"]
    cmd += [
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",  # ensure even dimensions
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


def mux_audio(video_only_path, original_path, output_path,
              start_sec, end_sec, fps, total_frames=None,
              preset="medium", tmp_dir="D:/pip-tmp"):
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
        mux_cmd = [
            "ffmpeg", "-y",
            "-i", video_only_path,   # processed video, PTS 0..N
            "-i", tmp_audio_path,    # audio, PTS 0..N
            "-map", "0:v:0",
            "-map", "1:a:0",
            "-c:v", "libx265",
            "-crf", "0",             # lossless HEVC
            "-preset", preset,
            "-tag:v", "hvc1",        # QuickTime / macOS compatible
            "-pix_fmt", "yuv420p",
            "-c:a", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        rc, stderr = _ffmpeg_with_progress(mux_cmd, total_frames, "  Encoding (HEVC + audio)")
        if rc != 0:
            # Retry without audio
            print("  Warning: audio mux failed, retrying video-only...")
            print(stderr[-400:])
            mux_cmd_no_audio = [
                "ffmpeg", "-y",
                "-i", video_only_path,
                "-c:v", "libx265", "-crf", "0", "-preset", preset,
                "-tag:v", "hvc1", "-pix_fmt", "yuv420p",
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


_DBG_VEHICLE_COLOR = (255, 100,   0)   # blue
_DBG_PLATE_COLOR   = (  0, 220,   0)   # green  — raw model detection
_DBG_BLUR_COLOR    = (  0,   0, 220)   # red    — padded blur region
_DBG_OWN_COLOR     = (  0, 140, 255)   # orange — own plate fixed region
_DBG_FONT          = cv2.FONT_HERSHEY_SIMPLEX


def _dbg_box(img, x1, y1, x2, y2, color, label, thickness=3):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    fs = max(0.6, (x2 - x1) / 500)
    (tw, th), _ = cv2.getTextSize(label, _DBG_FONT, fs, 2)
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4), _DBG_FONT, fs, (255, 255, 255), 2)


def draw_debug_overlay(frame, plate_rects, all_vehicles, own_plate_region=None,
                       blur_padding=8, blur_strength=61):
    """
    Returns a debug frame that shows exactly what the production output will look like:
      - Blur is applied to all detected regions (identical to production)
      - Green box  = raw model detection boundary
      - Red box    = padded blur region (what was actually erased)
      - Blue box   = vehicle detection
      - Orange box = own-plate fixed region
    """
    h, w = frame.shape[:2]
    vis = frame.copy()

    # ── Step 1: apply the real blur so the frame looks like production output ──
    all_rects = list(plate_rects)
    if own_plate_region:
        all_rects.append(own_plate_region)
    if all_rects:
        vis = apply_blur(vis, all_rects, blur_strength=blur_strength, padding=blur_padding)

    # ── Step 2: vehicle boxes (blue) ──────────────────────────────────────────
    for (cls, x1, y1, x2, y2, conf) in all_vehicles:
        _dbg_box(vis, x1, y1, x2, y2, _DBG_VEHICLE_COLOR,
                 f"{VEHICLE_CLASSES[cls]} {conf:.2f}", thickness=3)

    # ── Step 3: plate boxes — green (raw detection) + red (blur region) ───────
    for rect in plate_rects:
        x1, y1, x2, y2 = rect[:4]
        conf  = rect[4] if len(rect) > 4 else None
        label = f"plate {conf:.2f}" if conf is not None else "plate"

        # Green: raw model detection boundary
        _dbg_box(vis, x1, y1, x2, y2, _DBG_PLATE_COLOR, label, thickness=2)

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

    return vis


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
    tmp_dir: str = "D:/pip-tmp",
    debug: bool = False,
):
    print(f"\n{'='*60}")
    print(f"  License Plate Blurring Tool")
    print(f"{'='*60}")
    print(f"  Input : {input_path}")
    print(f"  Output: {output_path}")
    if start_time is not None:
        print(f"  Clip  : {start_time:.1f}s to {end_time:.1f}s")
    if vehicle_filter != "all":
        print(f"  Filter: {vehicle_filter} plates only")
    print(f"  Plate conf : {plate_conf} (global)  |  {plate_conf_in_vehicle} (inside vehicle boxes)")
    if own_plate_region:
        print(f"  Own plate region (always blurred): {own_plate_region}")
    if debug:
        print(f"  Mode  : DEBUG (boxes drawn, no blurring)")
    print(f"{'='*60}\n")

    info = get_video_info(input_path)
    width, height, fps = info["width"], info["height"], info["fps"]
    print(f"  Video : {width}x{height} @ {fps:.3f}fps  [{info['codec']}]")

    print("  Loading models...")
    vehicle_model, plate_model, device = load_models(
        plate_conf=min(plate_conf, plate_conf_in_vehicle)
    )
    print(f"  Models ready  |  vehicle detector + license plate detector\n")

    frame_size     = width * height * 3
    total_frames   = estimate_frame_count(info, start_time, end_time)

    os.makedirs(tmp_dir, exist_ok=True)
    with tempfile.NamedTemporaryFile(suffix=".mkv", delete=False, dir=tmp_dir) as tmp:
        tmp_path = tmp.name

    try:
        extract_cmd = build_ffmpeg_extract(input_path, start_time, end_time)
        encode_cmd  = build_ffmpeg_encode_lossless(width, height, fps, tmp_path)

        extract_proc = subprocess.Popen(extract_cmd, stdout=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)
        encode_proc  = subprocess.Popen(encode_cmd,  stdin=subprocess.PIPE,
                                        stderr=subprocess.DEVNULL)

        frame_num    = 0
        total_plates = 0

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

                    plates, vehicles = detect_plates(
                        frame, vehicle_model, plate_model,
                        vehicle_conf=vehicle_conf,
                        vehicle_filter=vehicle_filter,
                        plate_conf=plate_conf,
                        plate_conf_in_vehicle=plate_conf_in_vehicle,
                        sahi_slice_size=sahi_slice_size,
                        sahi_overlap=sahi_overlap,
                    )

                    # Always include own-plate region
                    if own_plate_region:
                        plates = list(plates) + [own_plate_region]

                    if plates:
                        total_plates += len(plates)

                    if debug:
                        frame = draw_debug_overlay(frame, plates, vehicles,
                                                   own_plate_region=own_plate_region,
                                                   blur_padding=blur_padding,
                                                   blur_strength=blur_strength)
                    elif plates:
                        frame = apply_blur(frame, plates, blur_strength=blur_strength,
                                           padding=blur_padding)

                    encode_proc.stdin.write(frame.tobytes())
                    frame_num += 1

                    pbar.update(1)
                    pbar.set_postfix(plates=total_plates, refresh=False)

            finally:
                extract_proc.stdout.close()
                extract_proc.wait()
                encode_proc.stdin.close()
                encode_proc.wait()

        action = "annotated" if debug else "blurred"
        print(f"\n  Processed : {frame_num} frames")
        print(f"  Detections: {total_plates} plate regions {action}")

        print("  Encoding final output (lossless HEVC + audio sync fix)...")
        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        mux_audio(tmp_path, input_path, output_path, start_time, end_time, fps,
                  total_frames=frame_num, preset=preset, tmp_dir=tmp_dir)

        size_mb = os.path.getsize(output_path) / 1024 / 1024
        print(f"\n  Done!  Output: {output_path}  ({size_mb:.1f} MB)\n")
        return frame_num, total_plates

    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def main():
    cfg = load_config()
    det = cfg["detection"]
    sahi = cfg["sahi"]
    blr = cfg["blur"]
    out = cfg["output"]

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

    args = parser.parse_args()

    start_sec = time_to_seconds(args.start) if args.start else None
    end_sec   = time_to_seconds(args.end)   if args.end   else None

    if start_sec is not None and end_sec is not None and end_sec <= start_sec:
        print("Error: --end must be after --start")
        sys.exit(1)

    own_plate = parse_region(args.own_plate) if args.own_plate else None

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
        tmp_dir=out["tmp_dir"],
        debug=args.debug,
    )


if __name__ == "__main__":
    main()
