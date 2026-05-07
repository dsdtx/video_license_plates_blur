"""
Extract annotated debug frames showing vehicle + plate detections.
Usage: python extract_debug_frames.py --start 1:51 --end 1:52
"""
import cv2, subprocess, numpy as np, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from blur_plates import load_models, DETECT_WIDTH, VEHICLE_CLASSES, time_to_seconds
import argparse

VEHICLE_COLOR = (255, 80, 0)    # blue
PLATE_COLOR   = (0, 220, 0)     # green
FONT          = cv2.FONT_HERSHEY_SIMPLEX

def draw_box(img, x1, y1, x2, y2, color, label, thickness=4):
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    fs = max(0.7, (x2 - x1) / 400)
    tw, th = cv2.getTextSize(label, FONT, fs, 2)[0]
    cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 6, y1), color, -1)
    cv2.putText(img, label, (x1 + 3, y1 - 4), FONT, fs, (255, 255, 255), 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="../final.mov")
    parser.add_argument("--start",  default="1:51")
    parser.add_argument("--end",    default="1:52")
    parser.add_argument("--outdir", default="debug_frames")
    parser.add_argument("--conf",   type=float, default=0.15)
    args = parser.parse_args()

    start_sec = time_to_seconds(args.start)
    end_sec   = time_to_seconds(args.end)
    os.makedirs(args.outdir, exist_ok=True)

    # ── probe video ──────────────────────────────────────────────────────────
    import json
    probe = subprocess.run(
        ["ffprobe", "-v", "quiet", "-print_format", "json",
         "-show_streams", args.input],
        capture_output=True, text=True, check=True,
    )
    vs = next(s for s in json.loads(probe.stdout)["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    num, den = map(int, vs["r_frame_rate"].split("/"))
    fps = num / den

    frame_size = w * h * 3
    duration   = end_sec - start_sec

    # ── load models ──────────────────────────────────────────────────────────
    vehicle_model, plate_model, device = load_models()
    print(f"Processing {args.start}–{args.end}  ({duration:.1f}s @ {fps:.0f}fps)")

    # ── extract raw frames ───────────────────────────────────────────────────
    cmd = ["ffmpeg", "-y", "-ss", f"{start_sec:.6f}", "-i", args.input,
           "-t", f"{duration:.6f}", "-f", "rawvideo",
           "-pix_fmt", "bgr24", "pipe:1"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

    frame_idx = 0
    while True:
        raw = proc.stdout.read(frame_size)
        if len(raw) < frame_size:
            break

        frame = np.frombuffer(raw, dtype=np.uint8).reshape((h, w, 3)).copy()
        vis   = frame.copy()

        scale = DETECT_WIDTH / w
        small = cv2.resize(frame, (DETECT_WIDTH, int(h * scale)))
        inv   = 1.0 / scale

        # ── vehicle detection ─────────────────────────────────────────────
        v_results = vehicle_model(small, conf=0.3, verbose=False)
        vehicle_boxes = []
        for r in v_results:
            if r.boxes is None: continue
            for box in r.boxes:
                cls = int(box.cls[0])
                if cls not in VEHICLE_CLASSES: continue
                vx1, vy1, vx2, vy2 = map(int, box.xyxy[0].tolist())
                fx1, fy1 = int(vx1*inv), int(vy1*inv)
                fx2, fy2 = int(vx2*inv), int(vy2*inv)
                conf = float(box.conf[0])
                label = f"{VEHICLE_CLASSES[cls]} {conf:.2f}"
                draw_box(vis, fx1, fy1, fx2, fy2, VEHICLE_COLOR, label)
                vehicle_boxes.append((cls, fx1, fy1, fx2, fy2))

        # ── plate detection: full frame ───────────────────────────────────
        p_results = plate_model(frame, conf=args.conf, verbose=False, imgsz=1280)
        for r in p_results:
            if r.boxes is None: continue
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                draw_box(vis, x1, y1, x2, y2, PLATE_COLOR, f"plate {conf:.2f}")

        # ── plate detection: vehicle crops ────────────────────────────────
        for (cls, fvx1, fvy1, fvx2, fvy2) in vehicle_boxes:
            top_frac = 0.0 if cls == 3 else 0.45
            mid_y = fvy1 + int((fvy2 - fvy1) * top_frac)
            cx1 = max(0, fvx1 - 15);  cy1 = max(0, mid_y)
            cx2 = min(w, fvx2 + 15);  cy2 = min(h, fvy2 + 20)
            if cx2 <= cx1 or cy2 <= cy1: continue
            crop = frame[cy1:cy2, cx1:cx2]
            pr2 = plate_model(crop, conf=args.conf, verbose=False, imgsz=640)
            for r in pr2:
                if r.boxes is None: continue
                for box in r.boxes:
                    bx1, by1, bx2, by2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    draw_box(vis, cx1+bx1, cy1+by1, cx1+bx2, cy1+by2,
                             PLATE_COLOR, f"plate(crop) {conf:.2f}", thickness=3)

        # ── timestamp overlay ─────────────────────────────────────────────
        ts = start_sec + frame_idx / fps
        m, s = int(ts) // 60, ts % 60
        cv2.putText(vis, f"{m:02d}:{s:05.2f}  frame {frame_idx}", (30, 70),
                    FONT, 2, (0, 0, 0), 6)
        cv2.putText(vis, f"{m:02d}:{s:05.2f}  frame {frame_idx}", (30, 70),
                    FONT, 2, (255, 255, 255), 3)

        out_path = os.path.join(args.outdir, f"frame_{frame_idx:04d}.png")
        cv2.imwrite(out_path, vis)
        print(f"  {out_path}")
        frame_idx += 1

    proc.stdout.close()
    proc.wait()
    print(f"\nDone — {frame_idx} frames saved to {args.outdir}/")

if __name__ == "__main__":
    main()
