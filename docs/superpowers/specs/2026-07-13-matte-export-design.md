# Design: Luma-Matte Export Mode (`--mode matte`)

**Date:** 2026-07-13
**Status:** Approved, pending implementation
**Component:** `blur_plates.py` (and `batch_blur.py` by inheritance)

## Problem

The current pipeline bakes the redaction (blur / colour / image) directly into a
full re-encode of **every** frame. For a high-fidelity editorial workflow this has
two costs:

1. **Quality loss on the master.** The decode path pipes frames as 8-bit `bgr24`
   and the output is HEVC `yuv420p` (`blur_plates.py:795`, `:956`/`:966`). Feeding a
   high-fidelity master (e.g. a 4K ProRes HQ intermediate) through this transcodes the
   **entire frame** to 8-bit 4:2:0 HEVC — the untouched parts of the picture are not
   preserved, and the result no longer ingests natively into a ProRes-based
   Premiere/DaVinci edit.
2. **No way to composite the blur yourself.** There is no output that lets an editor
   keep the original footage pristine and apply the redaction in their own NLE/grade.

## Goal

Add a mode that detects and tracks plates exactly as today, but instead of altering
the footage, **exports a luma matte** (white plates on black) as a ProRes `.mov`. The
editor keeps the original ProRes untouched and drives a blur/mosaic through the matte
in Premiere (Track Matte Key) or DaVinci (external matte / Matte node). This gives:

- **Zero quality loss** on the actual footage (it is never re-encoded).
- **Native ProRes** ingest, matched frame-for-frame to the source.
- **Faster** runs than a blur pass (see Non-Goals / performance note).

## Decisions (locked during brainstorming)

| Decision | Choice | Rationale |
|---|---|---|
| Matte type | **Luma matte** — white plates on black, ProRes 422 | Most universal; works as a Premiere Track Matte Key (luma) and a DaVinci external matte everywhere. |
| Edge treatment | **Hard edges by default**, optional `--matte-feather N` | Editor controls softness in the NLE; feather flag available when baking is convenient. |
| Encoder | **`prores_videotoolbox`, profile HQ** when available (Apple platforms), auto-fallback to CPU `prores_ks` | Hardware ProRes via VideoToolbox is fast and low-power on Apple Silicon; `prores_ks` is the portable CPU path everywhere else. HQ keeps hard matte edges clean. |
| Colour range | **Full range (`-color_range pc`)** | Ensures white encodes as true 255 (not limited-range 235) for a clean key. Source colorimetry is intentionally *not* copied for the matte. |
| Scope | Matte mode outputs **only** the matte | No blur baked in, no audio, no sidecar, no alpha/4444. |

## CLI Surface

- Extend `--mode` choices from `{blur, color, image}` to `{blur, color, image, matte}`.
- New flag `--matte-feather N` (int, default `0`): Gaussian falloff width in pixels
  applied to the matte after boxes are drawn. `0` = hard edges.
- New flag `--matte-codec {prores, hevc}` (default `prores`): choice of matte encoder.
  - `prores` → ProRes 422 HQ in a `.mov` (see Encode Path). Best quality/edge fidelity;
    hardware-accelerated only on Apple platforms, CPU (`prores_ks`) everywhere else.
  - `hevc` → near-lossless HEVC in an `.mp4`, GPU-accelerated via `hevc_nvenc` where
    available (NVIDIA), CPU `libx265` fallback. This is the **performance escape hatch**
    for PC/NVIDIA users where CPU ProRes is too slow. A luma matte survives HEVC well:
    the signal lives entirely in luma, chroma is flat/neutral so 4:2:0 subsampling is
    effectively lossless there, and only white/black edge crispness matters — which a
    near-lossless CRF preserves cleanly.
- All existing detection/tracking flags are reused unchanged and honoured by the
  matte: `--vehicles`, `--plate-conf`, `--plate-conf-in-vehicle`, `--conf`,
  `--own-plate`, `--start`/`--end`, padding, SAHI settings, and tracking gap-fill.
  The matte covers the exact same regions a blur run would have redacted.
- Output extension: the container is dictated by `--matte-codec` (`prores` → `.mov`,
  `hevc` → `.mp4`). If the user-supplied output path does not end in the expected
  extension, print a warning and **replace the extension** (writing a single file at
  the corrected path). Rationale: silently writing a codec into a mismatched container
  confuses NLEs.
- Defaults may also be set in `config.toml` under the existing `[redact]` section
  (`mode`, plus new `matte_feather` and `matte_codec`), consistent with how
  `color`/`image` defaults work.

## Matte Rendering

For each output frame:

1. Start from a black frame: `np.zeros((H, W, 3), np.uint8)`.
2. For every final plate region for that frame — detections, predicted/gap-filled
   boxes from the tracker, and the `--own-plate` fixed region — draw a filled white
   rectangle: `cv2.rectangle(frame, (x1,y1), (x2,y2), (255,255,255), -1)`, using the
   **same padded coordinates** the redactor computes. This is a drop-in substitution
   at the point where blur/colour/image is currently applied.
3. If `--matte-feather N > 0`, apply `cv2.GaussianBlur` with a kernel derived from `N`
   (odd kernel size, e.g. `k = 2*N+1`) to the whole matte.

Because R=G=B for every pixel, the result is a clean luma matte with neutral chroma.

## Encode Path

Matte frames are synthetic, so this mode **bypasses the FFV1 lossless intermediate
and the `mux_audio()` step entirely** — raw BGR frames pipe straight into a single
encode. The encode is selected by `--matte-codec`; both paths write full-range
(`-color_range pc`) so white encodes as true 255 for a clean key, and neither writes
audio.

### `--matte-codec prores` (default)

New helper `build_ffmpeg_encode_prores(width, height, fps, out_path)`.
Prefer the hardware encoder when `prores_videotoolbox` is present in the ffmpeg build
(probed the same way `mux_audio()` probes for `hevc_nvenc`); otherwise use `prores_ks`.

Primary (hardware, Apple platforms):
```
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i pipe:0 \
  -c:v prores_videotoolbox -profile:v hq -pix_fmt yuv422p10le \
  -color_range pc out.mov
```

Fallback (CPU, if the hardware encode returns non-zero):
```
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i pipe:0 \
  -c:v prores_ks -profile:v 3 -pix_fmt yuv422p10le \
  -color_range pc out.mov
```

### `--matte-codec hevc` (performance escape hatch)

New helper `build_ffmpeg_encode_hevc_matte(width, height, fps, out_path)`. Prefer
`hevc_nvenc` (GPU, NVIDIA) when available — same probe as `mux_audio()` — otherwise
`libx265` (CPU). Encoded near-lossless to keep matte edges crisp.

Primary (GPU, NVIDIA):
```
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i pipe:0 \
  -c:v hevc_nvenc -rc vbr -cq 12 -preset p4 -tag:v hvc1 \
  -pix_fmt yuv420p -color_range pc out.mp4
```

Fallback (CPU):
```
ffmpeg -y -f rawvideo -pix_fmt bgr24 -s WxH -r FPS -i pipe:0 \
  -c:v libx265 -crf 12 -preset medium -tag:v hvc1 \
  -pix_fmt yuv420p -color_range pc out.mp4
```

Both paths mirror the existing `hevc_nvenc → libx265` fallback pattern in
`mux_audio()`. Progress reporting reuses `_ffmpeg_with_progress()`.

The matte is frame-accurate to the source: same resolution, same fps, and it honours
`--start`/`--end` trim so it aligns on the timeline. No audio is written (a matte
needs none).

**Verified caveat (ProRes range tag):** `prores_videotoolbox` does not write a
`color_range` atom, so ffprobe reports `color_range=tv` on the ProRes output even with
`-color_range pc`. The *sample values* are nonetheless full-range — a pure-white input
frame decodes back to 255 — so the matte keys cleanly (white = fully on, black = fully
off). The tag is cosmetic. The HEVC path tags `color_range=pc` correctly.

## Pipeline Integration

- One branch at the redaction site: when `mode == "matte"`, render the matte frame
  instead of blurring/colour/image.
- One branch at the output/finalise stage: when `mode == "matte"`, route to the encoder
  selected by `--matte-codec` (`build_ffmpeg_encode_prores()` or
  `build_ffmpeg_encode_hevc_matte()`) and pipe frames directly, skipping the FFV1
  intermediate and `mux_audio()`. All upstream decode/detect/track code is untouched.

## Testing

- **Unit — renderer:** given a set of boxes with `feather=0`, assert white pixels
  exactly inside the padded boxes and black elsewhere; with `feather>0`, assert a soft
  falloff at the edges (intermediate grey values present).
- **Smoke — encode:** pipe 2–3 synthetic frames through each codec path and assert
  `ffprobe` reports the expected codec (`prores` for `.mov`, `hevc` for `.mp4`) with the
  expected dimensions/fps. Exercise the CPU fallback branch (`prores_ks` / `libx265`) as
  well, since CI typically lacks the hardware/GPU encoders.
- **Manual — end-to-end:** run `--mode matte` on a short clip for each codec; confirm the
  output opens, reports the right codec, and the white regions align with the plates.
  Verify in an NLE that the matte keys cleanly (white = fully affected).

## Non-Goals / Deferred

- **JSON sidecar (Approach C):** decoupling detection from matte rendering so feather /
  padding can be re-rendered without re-detecting. Deferred until there's a real need
  to iterate on the matte without re-running the (slow) SAHI detection.
- **Alpha / ProRes 4444 matte:** not needed for a luma-matte workflow.
- **Baked-in blur alongside the matte:** out of scope; matte mode outputs only the matte.

## Platform Notes

**ProRes has no GPU encoder on PC.** Hardware ProRes *encode* exists only on Apple
Silicon (the dedicated media/ProRes engine, used via `prores_videotoolbox`). On
Windows/Linux there is no GPU path:

- **NVENC / AMD VCN / Intel QuickSync** are fixed-function encoders limited to H.264,
  HEVC, and AV1 — ProRes is not in their silicon.
- **CUDA** is general-purpose compute and *could* in principle run a ProRes encoder,
  but no usable one exists (ffmpeg ships only the CPU `prores_ks` / `prores_aw`). The
  gap is largely non-technical: ProRes is Apple-proprietary and certified ProRes
  encoders are tightly controlled (hence sanctioned implementations from Apple,
  Blackmagic/Resolve on Windows, and capture-card ASICs — but nothing droppable into
  ffmpeg).

**Consequence:** on PC, `--matte-codec prores` runs on the CPU via `prores_ks`. For a
flat luma matte that is cheap, so most users won't notice. Users who *do* need GPU
acceleration (e.g. long 4K jobs on an NVIDIA box) can switch to `--matte-codec hevc`,
which uses `hevc_nvenc` — accepting a near-lossless HEVC/`.mp4` matte instead of ProRes.
Because a luma matte carries its signal in luma with neutral chroma, this trade is
visually negligible for keying.

## Performance Note

Because the matte skips the lossless FFV1 intermediate and encodes near-empty frames
(hardware ProRes where available, else CPU `prores_ks`; or `hevc_nvenc`/`libx265` for
`--matte-codec hevc`), a matte run is expected to be **faster** than an equivalent blur
run — the dominant cost remains SAHI plate detection, which is unchanged.
