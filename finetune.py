#!/usr/bin/env python3
"""
Fine-tune the license plate detection model on newly annotated data.

Starts from the existing license-plate-finetune-v1m.pt weights so training
converges quickly (hours, not days) — the backbone already knows what plates
look like; we're just teaching it the gaps in YOUR footage.

Strategy
────────
  • First N epochs: backbone frozen  → only the detection head trains.
    Fast convergence, no catastrophic forgetting.
  • Remaining epochs: full model unfrozen, lower LR  → fine adaptation.
  (Controlled by --freeze and --freeze-epochs.)

Output
──────
  runs/finetune/<name>/weights/best.pt   ← use this as the new model
  runs/finetune/<name>/weights/last.pt
  runs/finetune/<name>/results.csv       ← training curves

Usage
─────
    # Minimal — use all defaults
    python finetune.py --data training_data/dataset.yaml

    # More epochs, larger batch
    python finetune.py --data training_data/dataset.yaml --epochs 100 --batch 32

    # Start from a different base model
    python finetune.py --data training_data/dataset.yaml --model license-plate-finetune-v1n.pt

After training
──────────────
    # Quick sanity check on a clip
    python blur_plates.py video.mp4 test_out.mp4 --debug

    # If happy, copy the new model over the old one
    cp runs/finetune/exp/weights/best.pt license-plate-finetune-v1m.pt
"""

import argparse
import shutil
import sys
from pathlib import Path

# ── Defaults ──────────────────────────────────────────────────────────────────
_HERE         = Path(__file__).parent
DEFAULT_MODEL = str(_HERE / "license-plate-finetune-v1m.pt")
DEFAULT_OUT   = str(_HERE / "runs" / "finetune")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data",          required=True,
                    help="Path to dataset.yaml")
    ap.add_argument("--model",         default=DEFAULT_MODEL,
                    help=f"Starting weights (default: license-plate-finetune-v1m.pt)")
    ap.add_argument("--epochs",        type=int,   default=60,
                    help="Total training epochs (default: 60)")
    ap.add_argument("--batch",         type=int,   default=16,
                    help="Batch size — use -1 for auto (default: 16)")
    ap.add_argument("--imgsz",         type=int,   default=640,
                    help="Training image size (default: 640)")
    ap.add_argument("--lr",            type=float, default=0.001,
                    help="Initial learning rate (default: 0.001)")
    ap.add_argument("--freeze",        type=int,   default=10,
                    help="Number of backbone layers to freeze (default: 10; 0=train all)")
    ap.add_argument("--freeze-epochs", type=int,   default=15,
                    help="Epochs to train with backbone frozen before unfreezing (default: 15)")
    ap.add_argument("--workers",       type=int,   default=4,
                    help="DataLoader worker processes (default: 4)")
    ap.add_argument("--project",       default=DEFAULT_OUT,
                    help=f"Run output directory (default: {DEFAULT_OUT})")
    ap.add_argument("--name",          default="exp",
                    help="Run name — output goes to <project>/<name>/ (default: exp)")
    ap.add_argument("--device",        default=None,
                    help="Device override: 'cpu', '0', '0,1', etc. (default: auto)")
    args = ap.parse_args()

    from ultralytics import YOLO

    data_path  = Path(args.data)
    model_path = Path(args.model)

    if not data_path.exists():
        sys.exit(f"dataset.yaml not found: {data_path}")
    if not model_path.exists():
        sys.exit(f"Model file not found: {model_path}")

    print(f"\n{'═'*60}")
    print(f"  License Plate Fine-Tuning")
    print(f"{'═'*60}")
    print(f"  Base model   : {model_path}")
    print(f"  Dataset      : {data_path}")
    print(f"  Epochs       : {args.epochs}  "
          f"(backbone frozen for first {args.freeze_epochs})")
    print(f"  Batch        : {args.batch}")
    print(f"  Image size   : {args.imgsz}")
    print(f"  LR           : {args.lr}")
    print(f"  Frozen layers: {args.freeze}")
    print(f"{'═'*60}\n")

    model = YOLO(str(model_path))

    # ── Phase 1: train with backbone frozen ───────────────────────────────────
    if args.freeze > 0 and args.freeze_epochs > 0:
        print(f"Phase 1 — frozen backbone ({args.freeze_epochs} epochs)...")
        model.train(
            data          = str(data_path),
            epochs        = args.freeze_epochs,
            imgsz         = args.imgsz,
            batch         = args.batch,
            lr0           = args.lr,
            lrf           = 0.1,
            momentum      = 0.937,
            weight_decay  = 0.0005,
            warmup_epochs = 3,
            warmup_momentum = 0.8,
            box           = 7.5,
            cls           = 0.5,
            dfl           = 1.5,
            hsv_h         = 0.015,
            hsv_s         = 0.7,
            hsv_v         = 0.4,
            degrees       = 5.0,
            translate     = 0.1,
            scale         = 0.5,
            mosaic        = 1.0,
            close_mosaic  = 5,
            freeze        = args.freeze,
            workers       = args.workers,
            device        = args.device,
            project       = args.project,
            name          = args.name,
            save          = True,
            save_period   = 10,
            val           = True,
            plots         = True,
            verbose       = True,
            exist_ok      = True,
        )
        print(f"\nPhase 1 complete — unfreezing backbone...\n")

    # ── Phase 2: full model training (or all epochs if freeze=0) ─────────────
    remaining = args.epochs - (args.freeze_epochs if args.freeze > 0 else 0)
    if remaining > 0:
        # Ultralytics clears the model's internal overrides after train() so
        # calling .train() a second time on the same object raises KeyError.
        # Reload from the Phase 1 best checkpoint to start Phase 2 fresh.
        if args.freeze > 0 and args.freeze_epochs > 0:
            phase1_best = Path(args.project) / args.name / "weights" / "best.pt"
            print(f"  Reloading from Phase 1 best: {phase1_best}")
            model = YOLO(str(phase1_best))

        print(f"Phase 2 — full model ({remaining} epochs)...")
        model.train(
            data          = str(data_path),
            epochs        = remaining,
            imgsz         = args.imgsz,
            batch         = args.batch,
            lr0           = args.lr * 0.1,   # lower LR for fine adaptation
            lrf           = 0.01,
            momentum      = 0.937,
            weight_decay  = 0.0005,
            warmup_epochs = 1,
            warmup_momentum = 0.8,
            box           = 7.5,
            cls           = 0.5,
            dfl           = 1.5,
            hsv_h         = 0.015,
            hsv_s         = 0.7,
            hsv_v         = 0.4,
            degrees       = 5.0,
            translate     = 0.1,
            scale         = 0.5,
            mosaic        = 1.0,
            close_mosaic  = 10,
            freeze        = None,            # unfreeze everything
            workers       = args.workers,
            device        = args.device,
            project       = args.project,
            name          = args.name,
            save          = True,
            save_period   = 10,
            val           = True,
            plots         = True,
            verbose       = True,
            exist_ok      = True,
        )

    # ── Summary ───────────────────────────────────────────────────────────────
    best = Path(args.project) / args.name / "weights" / "best.pt"
    last = Path(args.project) / args.name / "weights" / "last.pt"

    print(f"\n{'═'*60}")
    print(f"  Training complete!")
    print(f"  Best model : {best}")
    print(f"  Last model : {last}")
    print(f"\nTest on a short clip first:")
    print(f"  python blur_plates.py video.mp4 debug.mp4 --debug")
    print(f"\nIf happy, replace the production model:")
    print(f"  cp {best} {model_path}")
    print(f"{'═'*60}\n")


if __name__ == "__main__":
    main()
