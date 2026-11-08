"""Copy the sample tours' screenshots from out/screens (scripts/screens.py) into docs/media as downscaled PNGs, each
under 300 KB, for docs/guide/samples.md.

    QT_QPA_PLATFORM=offscreen python scripts/screens.py --only samples   # make or refresh the screenshots first
    python scripts/tour_media.py [--src out/screens]

Each image is shrunk with area averaging to at most 1120 px wide; one that is still over the limit is shrunk further
in steps of 10 %, then stored with a 256-colour adaptive palette (Pillow, median cut) if that is what it takes. The
names are tour-<sample>-<view>.png. tests/test_docs_reference.py checks that every image the tours name exists and is
under the limit.
"""
from __future__ import annotations

import argparse
import io
import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
LIMIT = 300 * 1024
WIDTH = 1120
SHOTS = {                                        # docs/media name: out/screens name
    'tour-so100-main.png': 'sample-so100-main.png',
    'tour-so100-teleop.png': 'p458-so100-teleop-page.png',
    'tour-ur5e-main.png': 'sample-ur5e-main.png',
    'tour-ur5e-grasp.png': 'phase3-ur5e-grasp-page.png',
    'tour-dexhand-matrix.png': 'phase3-dexhand-matrix-page.png',
    'tour-aloha-main.png': 'sample-aloha-main.png',
    'tour-aloha-left-wrist.png': 'sample-aloha-camview-cam-left-wrist.png',
    'tour-panda-bimanual-main.png': 'sample-panda-bimanual-main.png',
    'tour-panda-bimanual-teleop.png': 'sample-panda-bimanual-teleop-page.png',
    'tour-g1-main.png': 'sample-g1-main.png',
    'tour-g1-head.png': 'sample-g1-camview-head-d435-unitree-urdf-d435-link.png',
}


def _png(img) -> bytes:
    import cv2
    ok, buf = cv2.imencode('.png', img, [cv2.IMWRITE_PNG_COMPRESSION, 9])
    if not ok:
        raise RuntimeError('OpenCV could not encode the image as PNG.')
    return buf.tobytes()


def shrink(path: str) -> bytes:
    """The PNG bytes of the image at path, downscaled to fit WIDTH and LIMIT."""
    import cv2
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise RuntimeError(f'{os.path.basename(path)} is not an image OpenCV can read.')
    h, w = img.shape[:2]
    scale = min(1.0, WIDTH / w)
    while True:
        im = img if scale >= 1.0 else cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))),
                                                 interpolation=cv2.INTER_AREA)
        data = _png(im)
        if len(data) <= LIMIT:
            return data
        try:
            from PIL import Image
            pil = Image.fromarray(im[:, :, ::-1]).quantize(colors=256, method=Image.Quantize.MEDIANCUT)
            out = io.BytesIO()
            pil.save(out, format='PNG', optimize=True)
            if out.tell() <= LIMIT:
                return out.getvalue()
        except ImportError:
            pass
        scale *= 0.9


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Copy the sample tours\' screenshots into docs/media.')
    ap.add_argument('--src', default=os.path.join(ROOT, 'out', 'screens'), help='the screenshots folder')
    args = ap.parse_args(argv)
    dst = os.path.join(ROOT, 'docs', 'media')
    os.makedirs(dst, exist_ok=True)
    missing = [s for s in SHOTS.values() if not os.path.isfile(os.path.join(args.src, s))]
    if missing:
        print(f'tour_media: {", ".join(missing)} not found in {args.src}: run scripts/screens.py --only samples '
              'first.', file=sys.stderr)
        return 2
    for name, src in SHOTS.items():
        data = shrink(os.path.join(args.src, src))
        with open(os.path.join(dst, name), 'wb') as fh:
            fh.write(data)
        print(f'{name}: {len(data) // 1024} KB')
    return 0


if __name__ == '__main__':
    sys.exit(main())
