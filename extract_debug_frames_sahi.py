"""
Extract annotated debug frames using SAHI sliced inference for plate detection.
Usage: python extract_debug_frames_sahi.py --start 1:59 --end 2:00
"""
import cv2, subprocess, numpy as np, os, sys, json
sys.path.insert(0, os.path.dirname(__file__))
from blur_plates import load_models, DETECT_WIDTH, VEHICLE_CLASSES, time_to_seconds
from extract_debug_frames import draw_box
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction
import argparse

VEHICLE_COLOR = (255, 80, 0)
PLATE_COLOR   = (0, 220, 0)
FONT          = cv2.FONT_HERSHEY_SIMPLEX


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",      default="../final.mov")
    parser.add_argument("--start",      default="1:59")
    parser.add_argument("--end",        default="2:00")
    parser.add_argument("--outdir",     default="debug_frames_sahi")
    parser.add_argument("--conf",       type=float, default=0.15)
    parser.add_argument("--slice-size", type=int,   default=640)
    parser.add_argument("--overlap",    type=float, default=0.2)
    parser.add_argument("--clahe",      action="store_true", help="Apply CLAHE contrast enhancement before detection")
    args = parser.parse_args()

    start_sec = time_to_seconds(args.start)
    end_sec   = time_to_seconds(args.end)
    os.makedirs(args.outdir, exist_ok=True)

    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json", "-show_streams", args.input],
        capture_output=True, text=True, check=True,
    )
    vs = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    num, den = map(int, vs["r_frame_rate"].split("/"))
    fps = num / den
    frame_size = w * h * 3
    duration   = end_sec - start_sec

    # ── load models ──────────────────────────────────────────────────────────
    vehicle_model, _, device = load_models()

    plate_model_sahi = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=os.path.join(os.path.dirname(__file__), 'license-plate-finetune-v1n.pt'),
        confidence_threshold=args.conf,
        device=device,
    )
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)) if args.clahe else None
    print(f"Processing {args.start}–{args.end}  ({duration:.1f}s @ {fps:.0f}fps)  "
          f"slice={args.slice_size}px  overlap={args.overlap}"
          + ("  CLAHE=on" if args.clahe else ""))

    # ── extract frames ───────────────────────────────────────────────────────
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.6f}", "-i", args.input,
           "-t", f"{duration:.6f}", "-f", "rawvideo", "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_idx = 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        vis   = frame.copy()

        # ── vehicle detection (downscaled) ────────────────────────────────
        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)))
        inv   = 1.0 / scale
        for r in vehicle_model(small, conf=0.3, verbose=False):
            if r.boxes is None: continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in VEHICLE_CLASSES: continue
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                draw_box(vis, int(vx1*inv), int(vy1*inv), int(vx2*inv), int(vy2*inv),
                         VEHICLE_COLOR, f"{VEHICLE_CLASSES[cls]} {float(box.conf[0]):.2f}")

        # ── SAHI plate detection ──────────────────────────────────────────
        detect_frame = frame
        if clahe is not None:
            lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            lab[:, :, 0] = clahe.apply(lab[:, :, 0])
            detect_frame = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        frame_rgb = cv2.cvtColor(detect_frame, cv2.COLOR_BGR2RGB)
        result = get_sliced_prediction(
            frame_rgb, plate_model_sahi,
            slice_height=args.slice_size, slice_width=args.slice_size,
            overlap_height_ratio=args.overlap, overlap_width_ratio=args.overlap,
            verbose=0,
        )
        for det in result.object_prediction_list:
            x1, y1 = int(det.bbox.minx), int(det.bbox.miny)
            x2, y2 = int(det.bbox.maxx), int(det.bbox.maxy)
            if x2 <= x1 or y2 <= y1:
                continue
            conf = det.score.value
            draw_box(vis, x1, y1, x2, y2, PLATE_COLOR, f"plate {conf:.2f}")

        # ── timestamp ─────────────────────────────────────────────────────
        ts = start_sec + frame_idx / fps
        m, s = int(ts) // 60, ts % 60
        cv2.putText(vis, f"{m:02d}:{s:05.2f}  frame {frame_idx}", (30, 70), FONT, 2, (0,0,0), 6)
        cv2.putText(vis, f"{m:02d}:{s:05.2f}  frame {frame_idx}", (30, 70), FONT, 2, (255,255,255), 3)

        out_path = os.path.join(args.outdir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, vis)
        print(f"  {out_path}  ({len(result.object_prediction_list)} plates)")
        frame_idx += 1

    proc.stdout.close()
    proc.wait()
    print(f"\nDone — {frame_idx} frames saved to {args.outdir}/")


if __name__ == "__main__":
    main()
