"""
Batch license plate blurring — processes all videos in a folder sequentially.

Usage:
    python batch_blur.py /path/to/folder
    python batch_blur.py /path/to/folder --outdir /path/to/output
    python batch_blur.py /path/to/folder --own-plate 100,200,300,400 --conf 0.15

Any extra arguments are forwarded directly to blur_plates.py for every video.
"""
import argparse, os, subprocess, sys
from pathlib import Path
from tqdm import tqdm

VIDEO_EXTENSIONS = {".mov", ".mp4", ".mkv", ".avi", ".mts", ".m4v", ".mpg", ".mpeg"}
PYTHON = sys.executable  # same venv that runs this script


def main():
    parser = argparse.ArgumentParser(
        description="Batch-blur license plates in all videos inside a folder."
    )
    parser.add_argument("folder", help="Folder containing input videos")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output folder (default: same folder as input, files get _blurred suffix)",
    )
    parser.add_argument(
        "--ext",
        default=None,
        help="Comma-separated extensions to process, e.g. mp4,mov (default: all common video types)",
    )

    # Capture unknown args and pass them straight to blur_plates.py
    args, extra = parser.parse_known_args()

    folder = Path(args.folder).resolve()
    if not folder.is_dir():
        sys.exit(f"ERROR: '{folder}' is not a directory.")

    allowed_exts = (
        {f".{e.lstrip('.')}" for e in args.ext.split(",")} if args.ext else VIDEO_EXTENSIONS
    )

    videos = sorted(p for p in folder.iterdir() if p.suffix.lower() in allowed_exts)
    if not videos:
        sys.exit(f"No video files found in '{folder}'.")

    outdir = Path(args.outdir).resolve() if args.outdir else None
    if outdir:
        outdir.mkdir(parents=True, exist_ok=True)

    blur_script = Path(__file__).parent / "blur_plates.py"
    total = len(videos)

    print(f"Found {total} video(s) in '{folder}'")
    print(f"Output: {'same folder (_blurred suffix)' if outdir is None else outdir}")
    if extra:
        print(f"Extra args passed to blur_plates.py: {' '.join(extra)}")
    print()

    # Ensure ffmpeg/ffprobe are findable in child processes
    env = os.environ.copy()
    ffmpeg_bin = r"C:\ffmpeg\bin"
    if ffmpeg_bin not in env.get("PATH", ""):
        env["PATH"] = ffmpeg_bin + os.pathsep + env.get("PATH", "")

    failed = []
    bar = tqdm(videos, unit="video", dynamic_ncols=True,
               bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} videos [{elapsed}<{remaining}] {postfix}")

    for video in bar:
        if outdir:
            out_file = outdir / (video.stem + "_blurred" + video.suffix)
        else:
            out_file = video.parent / (video.stem + "_blurred" + video.suffix)

        bar.set_postfix_str(video.name, refresh=True)
        tqdm.write(f"\n{'─'*60}")
        tqdm.write(f"  {video.name}  →  {out_file.name}")
        tqdm.write(f"{'─'*60}")

        cmd = [PYTHON, str(blur_script), str(video), str(out_file)] + extra
        result = subprocess.run(cmd, env=env)

        if result.returncode != 0:
            tqdm.write(f"  !! FAILED (exit code {result.returncode})")
            failed.append(video.name)
        else:
            tqdm.write(f"  ✓ Done")

    bar.set_postfix_str("complete", refresh=True)
    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"Finished: {total - len(failed)}/{total} succeeded.")
    if failed:
        tqdm.write("Failed:")
        for f in failed:
            tqdm.write(f"  - {f}")


if __name__ == "__main__":
    main()
