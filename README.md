# Video License Plate Blur

Automatically detects and blurs license plates in dashcam / action-cam footage using YOLOv11 + SAHI sliced inference. Output is lossless (FFV1 intermediate → HEVC CRF 0) with original audio preserved.

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

### Download model weights

Download from [morsetechlab/yolov11-license-plate-detection](https://huggingface.co/morsetechlab/yolov11-license-plate-detection) and place in this folder:

| File | Size | Notes |
|------|------|-------|
| `license-plate-finetune-v1n.pt` | 5.5 MB | Fast, lower accuracy |
| `license-plate-finetune-v1m.pt` | 40.5 MB | Balanced (recommended) |

---

## Configuration

All defaults live in `config.toml` — edit this file to tune thresholds, blur strength, SAHI settings, and encoding options without touching the code.

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

# Camera mounted behind your own plate (always blur a fixed region)
python blur_plates.py input.mp4 output.mp4 --own-plate 1700,900,2200,1100

# Debug mode — draws detection boxes instead of blurring (blue=vehicle, green=plate)
python blur_plates.py input.mp4 debug.mp4 --debug
```

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
