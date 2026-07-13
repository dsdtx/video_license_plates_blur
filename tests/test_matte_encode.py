import shutil
import subprocess

import numpy as np
import pytest

from blur_plates import (
    build_ffmpeg_encode_prores,
    build_ffmpeg_encode_hevc_matte,
)


def test_prores_command_targets_mov_full_range():
    cmd = build_ffmpeg_encode_prores(1920, 1080, 30, "out.mov")
    assert cmd[0] == "ffmpeg"
    assert "pipe:0" in cmd
    assert "-color_range" in cmd and cmd[cmd.index("-color_range") + 1] == "pc"
    assert cmd[-1] == "out.mov"
    # Some ProRes encoder is selected (hardware or CPU).
    assert any(c in cmd for c in ("prores_videotoolbox", "prores_ks"))
    assert "yuv422p10le" in cmd


def test_hevc_command_targets_mp4_full_range():
    cmd = build_ffmpeg_encode_hevc_matte(1280, 720, 25, "out.mp4")
    assert cmd[0] == "ffmpeg"
    assert "pipe:0" in cmd
    assert "-color_range" in cmd and cmd[cmd.index("-color_range") + 1] == "pc"
    assert cmd[-1] == "out.mp4"
    assert any(c in cmd for c in ("hevc_nvenc", "libx265"))
    assert "yuv420p" in cmd


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not on PATH")
def test_prores_encode_roundtrip(tmp_path):
    out = str(tmp_path / "m.mov")
    w, h, fps, n = 64, 48, 5, 3
    cmd = build_ffmpeg_encode_prores(w, h, fps, out)
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.DEVNULL)
    frame = np.zeros((h, w, 3), np.uint8)
    frame[10:20, 10:20] = 255
    for _ in range(n):
        proc.stdin.write(frame.tobytes())
    proc.stdin.close()
    assert proc.wait() == 0
    codec = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=codec_name", "-of", "csv=p=0", out],
        capture_output=True, text=True).stdout.strip()
    assert codec == "prores"
