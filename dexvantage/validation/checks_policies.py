"""Checks of the policy input model (core/policies.py): where each camera pixel lands in a policy's input.

The reference is independent of core.policies.crop_geometry: each crop rule's image pipeline is written out again here
as the array operations of the code it comes from (openpi image_tools.resize_with_pad: resize to int(W / ratio) ×
int(H / ratio), then divmod padding; LeRobot vla_utils.resize_with_pad: padding on the left and top; torchvision
CenterCrop: int(round((H − h) / 2)); GR00T N1 VideoCrop: int(H · scale)), with OpenCV's cv2.resize doing the
resampling, and run on an image whose channels are the camera pixel's own x and y (a linear ramp, which bilinear
resampling reproduces exactly away from the border) plus a channel of ones (zero where padded). The pipeline's output
says which camera point each input pixel shows; the model's to_source() must give the same point, and its content
rectangle must cover exactly the pixels the pipeline did not pad.
"""
from __future__ import annotations

import numpy as np

from .suite import Outcome, check

# (rule, camera size, input size, crop_side_fraction, pre_resize): every rule, landscape and portrait streams, sizes
# with odd remainders, the library's own inputs (224, 256, 512, 288 × 216)
CASES = (
    ('resize', (640, 480), (224, 224), None, None),
    ('resize', (1280, 720), (224, 224), None, None),
    ('none', (640, 480), (640, 480), None, None),
    ('resize_pad', (640, 480), (224, 224), None, None),
    ('resize_pad', (1280, 720), (224, 224), None, None),
    ('resize_pad', (480, 640), (224, 224), None, None),
    ('resize_pad', (847, 479), (224, 224), None, None),
    ('resize_pad_top_left', (640, 480), (512, 512), None, None),
    ('resize_pad_top_left', (1280, 800), (512, 512), None, None),
    ('resize_pad_top_left', (333, 517), (512, 512), None, None),
    ('center_crop', (1280, 720), (288, 216), None, (320, 240)),
    ('center_crop', (640, 480), (288, 216), None, (320, 240)),
    ('center_crop', (321, 241), (288, 216), None, None),
    ('center_crop_resize', (640, 480), (224, 224), 0.95, None),
    ('center_crop_resize', (1280, 720), (224, 224), 0.95, None),
    ('center_crop_resize', (853, 481), (256, 256), 0.9487, None),
)


def _ramp(W, H):
    x = np.broadcast_to(np.arange(W, dtype=np.float64) + 0.5, (H, W))
    y = np.broadcast_to((np.arange(H, dtype=np.float64) + 0.5)[:, None], (H, W))
    return np.dstack([x, y, np.ones((H, W))])


def _pipeline(rule, src, inp, frac, pre):
    """The rule's image pipeline written as array operations (see the module docstring), with cv2.resize."""
    import cv2
    W, H = src
    w, h = inp
    img = _ramp(W, H)

    def resize(a, size):
        return cv2.resize(a, size, interpolation=cv2.INTER_LINEAR)

    if rule in ('resize', 'none'):
        return img if (W, H) == (w, h) else resize(img, (w, h))
    if rule in ('resize_pad', 'resize_pad_top_left'):
        if (W, H) == (w, h):
            return img
        ratio = max(W / w, H / h)
        rh, rw = int(H / ratio), int(W / ratio)
        r = resize(img, (rw, rh))
        if rule == 'resize_pad':                       # openpi: divmod(height - resized_height, 2), centred
            ph0, rem_h = divmod(h - rh, 2)
            pw0, rem_w = divmod(w - rw, 2)
            pads = ((ph0, ph0 + rem_h), (pw0, pw0 + rem_w), (0, 0))
        else:                                          # LeRobot: F.pad(img, (pad_width, 0, pad_height, 0))
            pads = ((max(0, h - rh), 0), (max(0, w - rw), 0), (0, 0))
        return np.pad(r, pads, constant_values=0.0)
    if rule == 'center_crop':
        a = resize(img, tuple(pre)) if pre is not None else img
        Hp, Wp = a.shape[:2]
        top, left = int(round((Hp - h) / 2.0)), int(round((Wp - w) / 2.0))   # torchvision CenterCrop
        return a[top:top + h, left:left + w]
    ch, cw = int(H * frac), int(W * frac)                                  # GR00T VideoCrop size
    top, left = int(round((H - ch) / 2.0)), int(round((W - cw) / 2.0))
    return resize(img[top:top + ch, left:left + cw], (w, h))


def compare(rule, src, inp, frac=None, pre=None):
    """(worst point error in camera pixels, pixels whose padding differs, pixels compared)."""
    from ..core.policies import crop_geometry
    out = _pipeline(rule, src, inp, frac, pre)
    g = crop_geometry(rule, src, inp, frac, pre)
    w, h = inp
    assert out.shape[:2] == (h, w), (rule, out.shape)
    u = np.arange(w) + 0.5
    v = np.arange(h) + 0.5
    U, V = np.meshgrid(u, v)
    x0, y0, x1, y1 = g.dst_rect
    inside = (U > x0) & (U < x1) & (V > y0) & (V < y1)
    content = out[..., 2] > 0.5
    pad_diff = int(np.count_nonzero(inside != content))
    xs, ys = g.to_source(U, V)
    # bilinear resampling clamps at the image border: compare where the point lies between the outer pixel centres
    W, H = src
    ok = content & (xs >= 0.5) & (xs <= W - 0.5) & (ys >= 0.5) & (ys <= H - 0.5)
    err = np.hypot(out[..., 0] - xs, out[..., 1] - ys)[ok]
    return (float(err.max()) if err.size else 0.0), pad_diff, int(ok.sum())


@check('policies.crop_geometry', 'vla', 'Policy input crop and resize geometry',
       'worst distance between the camera point an input pixel shows by the model (CropGeometry.to_source) and by '
       'the rule\'s pipeline, over every input pixel of 16 cases covering the six crop rules; padded pixels must '
       'agree exactly', 'each rule\'s pipeline written out again from its source code (openpi, LeRobot, '
       'torchvision CenterCrop, GR00T N1 VideoCrop) with OpenCV cv2.resize resampling coordinate ramps', 1e-6, 'px',
       refs=('openpi', 'lerobot', 'torchvision_centercrop', 'isaac_gr00t', 'opencv_resize'))
def _crop_geometry(ctx):
    worst, pads, n, rows = 0.0, 0, 0, []
    for rule, src, inp, frac, pre in CASES:
        e, p, k = compare(rule, src, inp, frac, pre)
        worst, pads, n = max(worst, e), pads + p, n + k
        rows.append({'rule': rule, 'camera': list(src), 'input': list(inp), 'error_px': e, 'padding_mismatch': p})
    return Outcome(worst, {'cases': rows, 'pixels_compared': n, 'padding_mismatches': pads}, ok=pads == 0,
                   note='' if pads == 0 else f'{pads} input pixels padded by one and not the other')
