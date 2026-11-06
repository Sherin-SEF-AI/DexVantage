# Policies: what a robot-learning policy takes from each camera

`dexvantage/core/policies.py`, the policy library `dexvantage/resources/library/policy/*.json`, the project's
`policies` section (`core/project.py` `Policy`), the Policies page `dexvantage/ui/pages/policies.py`, the validation
check `dexvantage/validation/checks_policies.py`.

A project names the policies its data is for (spec §5). Each is a **policy library entry** plus the **cameras it
takes** plus the **user's own figures** over the entry's. The token, latency and action-chunk studies (§7.12, §7.13)
read a policy through one call:

```python
from dexvantage.core import policies
fields = policies.resolve(project, policy_uid)   # {field: (value, source text)}, overrides applied
geom = policies.input_geometry(project, policy_uid, (width, height))   # where camera pixels land in the input
```

## The fields

Every field of an entry is `{"value": …, "source": <index into the entry's sources> | null, "quote": "…"}`; a
figure no source states is `null` and shows as "—", *not stated by the source* (§1.1.1, §1.1.6). Nothing fills a gap.

| Field | Meaning |
|---|---|
| `input_resolution` | `[w, h]` px of each image the model takes |
| `crop_rule` | how a camera image becomes the input (below) |
| `crop_side_fraction` | `center_crop_resize`: the fraction of each side the centred crop keeps |
| `pre_resize` | `center_crop`: `[w, h]` the image is resized to before the crop |
| `patch_px` | ViT patch edge in input pixels (`null` for a CNN encoder) |
| `tokens_per_image` | visual tokens per image (after any pixel shuffle) |
| `wrist_input_resolution`, `wrist_tokens_per_image` | where wrist images are sized differently (Octo) |
| `n_cameras` | camera images per step |
| `action_chunk` | H, actions predicted per inference |
| `execute_steps` | actions executed before the next inference |
| `temporal_ensemble_m` | m in the weights exp(−m·i); `false` when the source says it uses none |
| `inference_ms` | one inference, with `hardware` naming what it was measured on |
| `control_hz` | the rate actions are sent; `null` where a source gives several robots' rates and none is the policy's own |

`resolve()` returns the source as text: the source's title, URL and date read, and what it says. A user's figure
(an override, entered on the Policies page) carries its own source, always prefixed "measured by the user: …";
`check_value()` normalises it (a size as whole pixels, a count as an integer) and refuses what cannot be a figure.

## The library

Entries were written on 2026-11-06 from the model card, the paper and the official code; each source is listed with
its URL, title, date read and the fields it gave. Figures in the code (configuration defaults, preprocessing
constants) are cited to a commit.

| Entry | Input | Crop rule | Patch | Tokens | H | Executed | m | Inference | Control |
|---|---|---|---|---|---|---|---|---|---|
| OpenVLA 7B | 224 × 224 | resize | 14 | — | 1 | 1 | — | 166.7 ms (6 Hz, RTX 4090) | — (5 and 15 Hz) |
| π0 | 224 × 224 | resize_pad | 14 | 256 | 50 | 25 | not used | 73 ms (RTX 4090) | 50 Hz (up to) |
| π0-FAST | 224 × 224 | resize_pad | 14 | 256 | 32 (default) | — | — | 750 ms (GPU not stated) | — (5–50 Hz) |
| SmolVLA base | 512 × 512 | resize_pad_top_left | 16 | 64 | 50 | 50 | — | — | 30 Hz |
| ACT | 640 × 480 | none | — (CNN) | 300 | 100 | 1 | 0.01 | 10 ms (RTX 2080 Ti) | 50 Hz |
| Diffusion Policy (CNN, real) | 288 × 216 | center_crop of 320 × 240 | — (CNN) | — | 16 | 8 | — | 100 ms (3080) | 10 Hz |
| Octo-Base | 256 × 256 (wrist 128 × 128) | center_crop_resize, 0.9487 | 16 | 256 (wrist 64) | 4 | — | — | 76.9 ms (13 it/s, 4090) | — (5–15 Hz) |
| GR00T N1 2B | 224 × 224 | center_crop_resize, 0.95 | — | 64 | 16 | — | — | 63.9 ms (L40, bf16) | 120 Hz (action module) |

What the sources say about crops and resizing:

- **OpenVLA**: `"image_resize_strategy": "resize-naive"`, no letterbox (checkpoint config). A fine-tune with image
  augmentation expects "the center 90% crop" at test time (README): override `crop_rule` with `center_crop_resize`
  and `crop_side_fraction` 0.949 (√0.9) for such a model.
- **π0, π0-FAST** (openpi): `resize_with_pad` keeps the aspect ratio and pads with black, split evenly
  (`divmod(h − resized_h, 2)`). Training adds a 95 % random crop and ±5° rotation to non-wrist images; not at inference.
- **SmolVLA** (LeRobot): the same fit, but padded on the **left and top** ("the smolvla/xvla convention"), so a 16:9
  image sits at the bottom of the square.
- **ACT**: images go in as captured (`get_image` only reorders the axes); the paper's rig streams 480 × 640.
- **Diffusion Policy**: video "down-sampled to 320x240", random crop in training, "a static center crop with the same
  size" (288 × 216) at inference. How a 16:9 stream becomes 320 × 240 is not stated: DexVantage stretches it, an
  assumption.
- **Octo**: the primary image's random crop is replaced at inference by its average, a centred crop of √0.9 of each
  side and a resize to 256 × 256; the wrist image is only resized to 128 × 128.
- **GR00T N1** (Isaac-GR00T `n1-release`): `VideoCrop(scale=0.95)` (centred in eval) then `VideoResize` to 224 × 224,
  not keeping the aspect ratio.

The token counts agree with the patch grids where both are cited from different sources: π0 (224 / 14)² = 256 from
the PaliGemma paper; Octo (256 / 16)² = 256 and (128 / 16)² = 64; SmolVLA (512 / 16)² / 4² = 64 with SmolVLM2's
pixel-shuffle `scale_factor` 4 (test `test_token_counts_agree_with_the_patch_grid`).

DexVantage bundles these figures only, never the models or weights; the weights' licences differ (OpenVLA may carry
the Llama 2 licence, GR00T N1 is non-commercial) and each entry's note says so.

## The crop geometry

`crop_geometry(rule, (W, H), (w, h), crop_side_fraction, pre_resize)` gives the camera rectangle the input shows
(`src_rect`), where it lands in the input (`dst_rect`; the rest is padding) and the scales `sx`, `sy` (input pixels
per camera pixel). Coordinates are pixel edges: x from 0 at the image's left edge to W at its right; an OpenCV pixel
centre u is at x = u + 0.5. The integer rounding is each source's own:

- `resize`, `none`: src = the whole image, dst = the whole input, sx = w / W, sy = h / H.
- `resize_pad`, `resize_pad_top_left`: r = max(W / w, H / h); content int(W / r) × int(H / r); offset
  ((w − rw) // 2, (h − rh) // 2) or (w − rw, h − rh); sx = rw / W, sy = rh / H.
- `center_crop`: the image stretched to pre_resize (W′, H′) if given; left = int(round((W′ − w) / 2.0)), top likewise
  (torchvision `center_crop`); sx = W′ / W.
- `center_crop_resize`: crop int(W·f) × int(H·f) at int(round((W − cw) / 2.0)), then stretched to w × h.

A camera point (x, y) maps to `dst0 + (x − src0)·s`; `to_source` is the inverse. Example, π0 on a 1280 × 720
stream: the whole image becomes 224 × 126 at y = 49 to 175, 43.8 % of the input is padding, 0.175 input pixels per
camera pixel (one 14 px patch covers 80 camera pixels).

## Validation

Check `policies.crop_geometry` (model `vla`): each crop rule's pipeline is written out again in the check from its
source code (openpi `resize_with_pad`, LeRobot's left-top variant, torchvision `center_crop`, GR00T `VideoCrop`) as
array operations, with OpenCV's `cv2.resize` (bilinear, centre-aligned) doing the resampling. It runs on an image
whose channels are each camera pixel's own x and y (a linear ramp, which bilinear resampling reproduces exactly away
from the border) and a channel of ones (zero where padded). For every input pixel, the camera point the pipeline
shows is compared with `CropGeometry.to_source`, and the padded pixels with `dst_rect`.

- Cases: 16, covering the six rules, landscape and portrait streams, odd sizes (847 × 479, 333 × 517, 853 × 481) and
  the library's inputs (224, 256, 512, 288 × 216); 1 420 672 input pixels compared (those whose camera point lies
  between the outer pixel centres, where bilinear resampling does not clamp).
- Tolerance: 10⁻⁶ px. Why: both sides are exact in real arithmetic; the bound leaves room for double-precision
  rounding only, so any off-by-one in a size, an offset or a rounding direction (a whole pixel) fails.
- Measured: worst 2.5 × 10⁻¹³ px; 0 padding mismatches. The test also shows the check fails (padding mismatches,
  errors above 1 px) when the geometry pads on the wrong side.

The library figures are checked against the sources by pinned values in `tests/test_policies.py`, each with the
quote it came from in the entry.

## Known limits

Entries hold what the sources publish, mostly one configuration: a fine-tune can change the input size, the crop and
the chunk length, so enter your own as overrides. Inference times are for the hardware each source names, and
π0-FAST's 750 ms names none; measure on your host. `control_hz` is null where a source lists several robots' rates.
The crop geometry is the inference-time crop: training augmentation (random crops, rotation, colour jitter) is not
modelled, nor are the interpolation filters (bilinear, bicubic, Lanczos, antialiasing), only where each camera pixel
lands. Diffusion Policy's stretch to 320 × 240 is an assumption. Per-camera-role inputs (Octo's wrist image) are
listed, not yet applied per camera. A policy's camera count is compared with the cameras assigned to it; which slot
(base or wrist) each camera fills is not modelled.
