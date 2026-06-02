#!/usr/bin/env python3
"""
Interactive annotation review and correction tool.

Opens each frame + its auto-generated bounding boxes in an OpenCV window.
You correct mistakes with the keyboard — no external tools needed.

Controls
────────
  Mouse left-drag   draw a new bounding box (auto-saves on release)
  Mouse right-click delete the box under the cursor (auto-saves)
  ← / →             previous / next frame
  k                 mark frame as KEPT and advance (label unchanged)
  d                 delete ALL boxes (label becomes empty — "no plates here")
  s                 skip without marking, just advance
  q                 quit and save progress

Edits are persisted to the label file as soon as the mouse button is released.
Progress (kept / deleted / edited) is saved in <dataset>/review_progress.json
so you can stop and resume at any time.

Usage
─────
    # Normal full review
    python review_annotations.py --dataset training_data/

    # Spot-check Claude's auto-edits (frames listed in claude_review_log.json)
    python review_annotations.py --dataset training_data_v2/ --claude-edited

    # Only show frames that have at least one detected box
    python review_annotations.py --dataset training_data/ --only-uncertain
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from blur_plates import _DBG_PLATE_COLOR, _DBG_VEHICLE_COLOR, _DBG_BLUR_COLOR

# ── Colours used in the editor ────────────────────────────────────────────────
COL_BOX_SAVED   = _DBG_PLATE_COLOR       # green  — saved box
COL_BOX_NEW     = (0, 180, 255)          # orange — box being drawn
COL_BOX_DELETE  = (30, 30, 200)          # red    — selected for deletion
COL_BG_KEPT     = (0, 60, 0)            # dark green banner
COL_BG_DELETED  = (0, 0, 80)            # dark red banner
COL_BG_UNSEEN   = (50, 50, 50)          # dark grey banner
FONT            = cv2.FONT_HERSHEY_SIMPLEX
WIN             = "Annotation Review  [k]eep [d]elete [e]dit [<>] nav [q]uit"

DISPLAY_MAX_W = 1600    # max window width (downscale large frames for display only)
DISPLAY_MAX_H = 900


# ── YOLO I/O ──────────────────────────────────────────────────────────────────

def load_yolo(lbl_path: Path, img_w, img_h):
    """Return list of (x1, y1, x2, y2) pixel boxes from a YOLO .txt file."""
    if not lbl_path.exists():
        return []
    boxes = []
    for line in lbl_path.read_text().splitlines():
        parts = line.strip().split()
        if len(parts) < 5:
            continue
        _, cx, cy, bw, bh = float(parts[0]), float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        x1 = int((cx - bw / 2) * img_w)
        y1 = int((cy - bh / 2) * img_h)
        x2 = int((cx + bw / 2) * img_w)
        y2 = int((cy + bh / 2) * img_h)
        boxes.append((x1, y1, x2, y2))
    return boxes


def save_yolo(lbl_path: Path, boxes, img_w, img_h):
    """Write YOLO .txt from list of (x1, y1, x2, y2) pixel boxes."""
    lines = []
    for (x1, y1, x2, y2) in boxes:
        cx = (x1 + x2) / 2.0 / img_w
        cy = (y1 + y2) / 2.0 / img_h
        bw = (x2 - x1)       / img_w
        bh = (y2 - y1)       / img_h
        cx = max(0.0, min(1.0, cx))
        cy = max(0.0, min(1.0, cy))
        bw = max(0.001, min(1.0, bw))
        bh = max(0.001, min(1.0, bh))
        lines.append(f"0 {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    lbl_path.write_text("\n".join(lines))


# ── Display helpers ────────────────────────────────────────────────────────────

def _fit(img):
    """Downscale image to fit display, return (display_img, scale_factor)."""
    h, w = img.shape[:2]
    scale = min(DISPLAY_MAX_W / w, DISPLAY_MAX_H / h, 1.0)
    if scale < 1.0:
        nw, nh = int(w * scale), int(h * scale)
        return cv2.resize(img, (nw, nh), interpolation=cv2.INTER_AREA), scale
    return img.copy(), 1.0


def _draw_boxes(vis, boxes, highlight_idx=None):
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        color = COL_BOX_DELETE if i == highlight_idx else COL_BOX_SAVED
        cv2.rectangle(vis, (x1, y1), (x2, y2), color, 2)
        cv2.putText(vis, f"[{i}]", (x1 + 4, y1 + 22),
                    FONT, 0.65, color, 2, cv2.LINE_AA)


def _banner(vis, text, bg_color, text_color=(255, 255, 255)):
    h, w = vis.shape[:2]
    bh   = 38
    cv2.rectangle(vis, (0, 0), (w, bh), bg_color, -1)
    cv2.putText(vis, text, (8, bh - 10), FONT, 0.65, text_color, 2, cv2.LINE_AA)


# ── Interactive box editor ─────────────────────────────────────────────────────

class _BoxEditor:
    """Lets the user draw new boxes and right-click-delete existing ones."""

    def __init__(self, frame_orig, boxes_orig):
        self._orig  = frame_orig
        self._boxes = list(boxes_orig)   # working copy
        self._drag_start = None
        self._cur_pt     = None
        self._done       = False
        self._saved      = False

    def _redraw(self):
        disp, _ = _fit(self._orig)
        h, w = self._orig.shape[:2]
        dh, dw = disp.shape[:2]
        sx, sy = dw / w, dh / h
        vis = disp.copy()
        for (x1, y1, x2, y2) in self._boxes:
            px1, py1 = int(x1 * sx), int(y1 * sy)
            px2, py2 = int(x2 * sx), int(y2 * sy)
            cv2.rectangle(vis, (px1, py1), (px2, py2), COL_BOX_SAVED, 2)
        if self._drag_start and self._cur_pt:
            cv2.rectangle(vis, self._drag_start, self._cur_pt, COL_BOX_NEW, 2)
        _banner(vis,
                "EDIT: left-drag=draw box  right-click=delete  Enter=save  Esc=cancel",
                (0, 0, 100))
        cv2.imshow(WIN, vis)

    def _mouse(self, event, x, y, flags, _param):
        h, w = self._orig.shape[:2]
        dh, dw = _fit(self._orig)[0].shape[:2]
        sx, sy = w / dw, h / dh   # display→original scale

        ox, oy = int(x * sx), int(y * sy)   # original-frame coordinates

        if event == cv2.EVENT_LBUTTONDOWN:
            self._drag_start = (x, y)
            self._cur_pt     = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and self._drag_start:
            self._cur_pt = (x, y)
            self._redraw()
        elif event == cv2.EVENT_LBUTTONUP and self._drag_start:
            x0d, y0d = self._drag_start
            # Convert both corners to original-frame space
            ax1, ay1 = int(min(x0d, x) * sx), int(min(y0d, y) * sy)
            ax2, ay2 = int(max(x0d, x) * sx), int(max(y0d, y) * sy)
            if ax2 - ax1 > 8 and ay2 - ay1 > 4:
                self._boxes.append((ax1, ay1, ax2, ay2))
            self._drag_start = None
            self._cur_pt     = None
            self._redraw()
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Priority 1: click is INSIDE a box → delete the smallest one
            #             (most specific when boxes overlap).
            # Priority 2: click is within BORDER_TOL px of a box edge → delete
            #             the closest one.
            # Working in display-space throughout so the tolerance feels the
            # same regardless of the original frame resolution.
            BORDER_TOL = 18   # pixels in display space

            def _disp(bx1, by1, bx2, by2):
                """Original-frame box → display-space box."""
                return (int(bx1 / sx), int(by1 / sy),
                        int(bx2 / sx), int(by2 / sy))

            def _dist_to_box(px, py, dx1, dy1, dx2, dy2):
                """Distance from (px,py) to the nearest point on the box.
                Returns 0 if the point is inside the box."""
                if dx1 <= px <= dx2 and dy1 <= py <= dy2:
                    return 0.0
                cx = max(dx1, min(dx2, px))
                cy = max(dy1, min(dy2, py))
                return ((px - cx) ** 2 + (py - cy) ** 2) ** 0.5

            inside = []   # (area_in_display_px, original_index)
            near   = []   # (dist_px, original_index)

            for i, (bx1, by1, bx2, by2) in enumerate(self._boxes):
                dx1, dy1, dx2, dy2 = _disp(bx1, by1, bx2, by2)
                d = _dist_to_box(x, y, dx1, dy1, dx2, dy2)
                if d == 0.0:
                    area = (dx2 - dx1) * (dy2 - dy1)
                    inside.append((area, i))
                elif d <= BORDER_TOL:
                    near.append((d, i))

            if inside:
                _, best_i = min(inside)          # smallest containing box
                self._boxes.pop(best_i)
                self._redraw()
            elif near:
                _, best_i = min(near)            # closest border
                self._boxes.pop(best_i)
                self._redraw()

    def run(self):
        # Render the frame first so Qt fully initialises the window handle,
        # then attach the mouse callback — doing it the other way around
        # produces a "NULL window handler" error in the Qt backend.
        self._redraw()
        cv2.waitKey(1)   # pump the Qt event loop once to flush the imshow
        cv2.setMouseCallback(WIN, self._mouse)
        while True:
            key = cv2.waitKey(30) & 0xFF
            if key == 13:   # Enter — save
                self._saved = True
                break
            elif key == 27:  # Esc — cancel
                break
        cv2.setMouseCallback(WIN, lambda *a: None)   # detach
        return self._saved, self._boxes


# ── Progress persistence ───────────────────────────────────────────────────────

def load_progress(path: Path):
    if path.exists():
        return json.loads(path.read_text())
    return {}


def save_progress(path: Path, progress: dict):
    path.write_text(json.dumps(progress, indent=2))


# ── Main review loop ───────────────────────────────────────────────────────────

class _ReviewState:
    """Per-frame mutable state shared with the mouse callback."""
    def __init__(self):
        self.boxes        = []          # in original-frame coords
        self.dirty        = False       # set when boxes change → caller saves
        self.drag_start   = None        # display-space start of current drag
        self.cur_pt       = None        # display-space current mouse pos
        self.scale        = (1.0, 1.0)  # display→original (sx, sy)
        self.need_redraw  = False


def review(dataset: Path, split: str, only_uncertain: bool, claude_edited_only: bool,
           augmented_only: bool = False, source_only: bool = False):
    progress_file = dataset / "review_progress.json"
    progress      = load_progress(progress_file)

    # claude_review_log.json lists which frames Claude has edited
    claude_log_file = dataset / "claude_review_log.json"
    claude_edited   = set()
    if claude_log_file.exists():
        claude_edited = set(json.loads(claude_log_file.read_text()).keys())

    # Collect frames from requested split(s)
    splits = [split] if split != "all" else ["train", "val"]
    entries = []
    for sp in splits:
        img_dir = dataset / "images" / sp
        lbl_dir = dataset / "labels" / sp
        if not img_dir.exists():
            continue
        for img_path in sorted(list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))):
            stem     = img_path.stem
            lbl_path = lbl_dir / f"{stem}.txt"
            is_augmented = "_aug" in stem
            if augmented_only and not is_augmented:
                continue
            if source_only and is_augmented:
                continue
            if claude_edited_only and stem not in claude_edited:
                continue
            if only_uncertain:
                if lbl_path.exists() and lbl_path.read_text().strip():
                    entries.append((img_path, lbl_path, stem))
            else:
                entries.append((img_path, lbl_path, stem))

    if not entries:
        print("No frames found.")
        return

    # Sort: claude-edited frames first (so you see Claude's work without hunting),
    # then unreviewed, then already-reviewed.
    def _sort_key(e):
        stem = e[2]
        rank = 0 if stem in claude_edited else (1 if stem not in progress else 2)
        return (rank, stem)
    entries.sort(key=_sort_key)

    print(f"  {len(entries)} frames  |  {len(progress)} already reviewed"
          f"  |  {len(claude_edited)} claude-edited")
    print("  Controls:")
    print("    mouse left-drag  = draw new box      mouse right-click = delete box under cursor")
    print("    [k]eep & next    [d]elete all boxes  [s]kip            [< >] navigate   [q]uit\n")

    # WINDOW_GUI_NORMAL strips Qt's extra toolbar/status; combined with our
    # right-click handler this also suppresses the context menu in most builds.
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
    cv2.resizeWindow(WIN, DISPLAY_MAX_W, DISPLAY_MAX_H + 50)

    state = _ReviewState()

    def _on_mouse(event, x, y, flags, _param):
        sx, sy = state.scale            # display→original
        if event == cv2.EVENT_LBUTTONDOWN:
            state.drag_start = (x, y)
            state.cur_pt     = (x, y)
        elif event == cv2.EVENT_MOUSEMOVE and state.drag_start:
            state.cur_pt = (x, y)
            state.need_redraw = True
        elif event == cv2.EVENT_LBUTTONUP and state.drag_start:
            x0, y0 = state.drag_start
            ax1, ay1 = int(min(x0, x) * sx), int(min(y0, y) * sy)
            ax2, ay2 = int(max(x0, x) * sx), int(max(y0, y) * sy)
            if ax2 - ax1 > 8 and ay2 - ay1 > 4:
                state.boxes.append((ax1, ay1, ax2, ay2))
                state.dirty = True
            state.drag_start = None
            state.cur_pt     = None
            state.need_redraw = True
        elif event == cv2.EVENT_RBUTTONDOWN:
            # Delete the box the cursor is INSIDE (smallest containing one wins).
            # If not inside any box, delete the nearest within BORDER_TOL display px.
            BORDER_TOL = 18
            inside, near = [], []
            for i, (bx1, by1, bx2, by2) in enumerate(state.boxes):
                dx1, dy1 = int(bx1 / sx), int(by1 / sy)
                dx2, dy2 = int(bx2 / sx), int(by2 / sy)
                if dx1 <= x <= dx2 and dy1 <= y <= dy2:
                    inside.append(((dx2 - dx1) * (dy2 - dy1), i))
                else:
                    cx = max(dx1, min(dx2, x)); cy = max(dy1, min(dy2, y))
                    d = ((cx - x) ** 2 + (cy - y) ** 2) ** 0.5
                    if d <= BORDER_TOL:
                        near.append((d, i))
            target = None
            if inside:
                target = min(inside)[1]      # smallest box
            elif near:
                target = min(near)[1]        # closest
            if target is not None:
                state.boxes.pop(target)
                state.dirty = True
                state.need_redraw = True

    # First imshow must happen before setMouseCallback to ensure Qt has
    # a real window handle (otherwise we get NULL handler errors).
    cv2.imshow(WIN, np.zeros((100, 200, 3), dtype=np.uint8))
    cv2.waitKey(1)
    cv2.setMouseCallback(WIN, _on_mouse)

    def _render(img_orig, boxes, header_text, header_color):
        disp, _ = _fit(img_orig)
        dh, dw  = disp.shape[:2]
        sx_d = img_orig.shape[1] / dw     # original-pixel per display-pixel
        sy_d = img_orig.shape[0] / dh
        # Boxes stored in original-image space; convert to display
        for (bx1, by1, bx2, by2) in boxes:
            cv2.rectangle(disp,
                          (int(bx1 / sx_d), int(by1 / sy_d)),
                          (int(bx2 / sx_d), int(by2 / sy_d)),
                          COL_BOX_SAVED, 2)
        # In-progress drag rectangle
        if state.drag_start and state.cur_pt:
            cv2.rectangle(disp, state.drag_start, state.cur_pt, COL_BOX_NEW, 2)
        _banner(disp, header_text, header_color)
        return disp, sx_d, sy_d

    idx = 0
    while True:
        img_path, lbl_path, stem = entries[idx]
        img = cv2.imread(str(img_path))
        if img is None:
            idx = (idx + 1) % len(entries)
            continue

        h, w = img.shape[:2]
        state.boxes = load_yolo(lbl_path, w, h)
        state.dirty = False

        status = progress.get(stem, "")
        if stem in claude_edited:
            status = "claude-edited" if not status else f"claude-edited + {status}"

        bg_col = (COL_BG_KEPT    if "kept"    in status else
                  COL_BG_DELETED if "deleted" in status else
                  COL_BG_UNSEEN)

        # Inner loop: stay on this frame while user draws/deletes boxes; advance on keystroke.
        advance = 0    # -1 prev, 0 stay, +1 next, "quit" to quit
        while True:
            tick = " ✓" if (status and "claude" not in status) else (" *" if "claude" in status else "")
            tag = f"  [{status}]" if status else ""
            header = f"[{idx+1}/{len(entries)}]  {stem}{tick}{tag}  |  {len(state.boxes)} box"

            disp, sxd, syd = _render(img, state.boxes, header, bg_col)
            state.scale = (sxd, syd)
            cv2.imshow(WIN, disp)
            state.need_redraw = False

            # Short waitKey so mouse-move drag rectangle is redrawn live
            key = cv2.waitKey(15) & 0xFF
            if key == 255:
                # No key — check if we just need to redraw (drag in progress)
                continue

            if key in (ord('q'), 27):
                advance = "quit"; break
            elif key in (81, 2):
                advance = -1; break
            elif key in (83, 3, ord('s')):
                advance = +1; break
            elif key == ord('k'):
                progress[stem] = "kept"
                save_progress(progress_file, progress)
                advance = +1; break
            elif key == ord('d'):
                state.boxes = []
                state.dirty = True
                progress[stem] = "deleted"
                save_progress(progress_file, progress)
                advance = +1; break

        # If user drew/deleted boxes, persist label and mark edited
        if state.dirty:
            save_yolo(lbl_path, state.boxes, w, h)
            if progress.get(stem) not in ("deleted",):
                progress[stem] = "edited"
                save_progress(progress_file, progress)

        if advance == "quit":
            break
        idx = (idx + advance) % len(entries)

    cv2.destroyAllWindows()

    n_kept    = sum(1 for v in progress.values() if v == "kept")
    n_deleted = sum(1 for v in progress.values() if v == "deleted")
    n_edited  = sum(1 for v in progress.values() if v == "edited")
    n_total   = len(entries)

    print(f"\nReview complete  ({len(progress)}/{n_total} frames reviewed)")
    print(f"  kept={n_kept}  deleted-label={n_deleted}  edited={n_edited}")
    print(f"\nNext: python finetune.py --data {dataset}/dataset.yaml\n")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset",        required=True,
                    help="Dataset directory produced by extract_training_frames.py")
    ap.add_argument("--split",          default="all",
                    choices=["all", "train", "val"],
                    help="Which split to review (default: all)")
    ap.add_argument("--only-uncertain", action="store_true",
                    help="Only show frames that have at least one detected box")
    ap.add_argument("--claude-edited",   action="store_true",
                    help="Only show frames listed in claude_review_log.json — "
                         "useful for spot-checking automated edits")
    ap.add_argument("--augmented-only",  action="store_true",
                    help="Only show augmented frames (with '_aug' in stem)")
    ap.add_argument("--source-only",     action="store_true",
                    help="Only show source (non-augmented) frames")
    args = ap.parse_args()

    if args.augmented_only and args.source_only:
        sys.exit("--augmented-only and --source-only are mutually exclusive")

    dataset = Path(args.dataset)
    if not dataset.is_dir():
        sys.exit(f"Dataset directory not found: {dataset}")

    review(dataset, args.split, args.only_uncertain, args.claude_edited,
           augmented_only=args.augmented_only, source_only=args.source_only)


if __name__ == "__main__":
    main()
