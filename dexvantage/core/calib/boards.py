"""Calibration boards: ChArUco (OpenCV's layout) and AprilGrid (Kalibr's layout), their corner coordinates, and the
board itself as an image or a vector PDF at true scale for printing.

Ported from Vantage core/targets.py (board_points) and the target records of core/observability.py (the Kalibr
AprilGrid spacing ratio), with the floor-target placement and the vehicle frame removed: a board here is a rigid,
flat object in its own frame, mounted in the world or on a robot link by the hand-eye problem (handeye.py).

ChArUco (cv2.aruco.CharucoBoard, OpenCV 4.7 and later): squares_x × squares_y squares of side square_m, ArUco
markers of side marker_m in the white squares, from a named predefined dictionary. Board frame (OpenCV's): origin at
the pattern's top-left outer corner as printed, x along the columns (right), y down the rows, z into the board
(x × y), so a camera facing the printed side sees the board's z axis pointing away from it. The calibration points
are the (squares_x − 1)(squares_y − 1) inner chessboard corners, id = row · (squares_x − 1) + column, the order of
getChessboardCorners. `legacy=True` is the layout of boards made before OpenCV 4.6 (setLegacyPattern: with an even
number of rows the top-left square is white).

AprilGrid (Kalibr, ethz-asl/kalibr, read 2026-11-06: the "Calibration targets" wiki page,
aslam_cv/aslam_cameras_april/src/GridCalibrationTargetAprilgrid.cpp and its header, and
aslam_offline_calibration/kalibr/python/kalibr_create_target_pdf): tag_cols × tag_rows AprilTags of the 36h11
family, tag_m the size of a tag "edge to edge" (the outer edge of its black border), the gap between tags
spacing · tag_m (Kalibr's tagSpacing, "ratio of space between tags to tagSize"; 0.3 is the wiki's example and the
creator's command-line default). Each tag is 6 × 6 data bits inside a black border two bits wide (Kalibr's
blackTagBorder default 2 and the creator's borderBits 2), so a bit is tag_m / 10. Tag ids run row by row from the
bottom-left, columns fastest (id = cols · row + col). The creator draws a black square of side spacing · tag_m at each
tag corner ("symmetric corners"), drawn here when corner_squares is True. Board frame (Kalibr's): origin at the
bottom-left corner of tag 0, x right, y up as printed, z = x × y out of the printed face (towards a camera that sees
it). The points are the four corners of every tag, numbered as Kalibr numbers them: a (2 · tag_rows) × (2 · tag_cols)
grid, index = R · 2 · tag_cols + C, at x = (C // 2)(1 + spacing) tag_m + (C % 2) tag_m and likewise y from R.

Detection (detect.py) uses OpenCV's DICT_APRILTAG_36h11 with markerBorderBits = 2 for an AprilGrid. Kalibr's
detector needs at least 4 tags in a view (minTagsForValidObs default 4); APRILGRID_MIN_TAGS keeps that default.

Printing: save_board writes a PNG with its resolution in the file (pHYs) or a one-page PDF drawn in vectors at
exactly the board's size; print at 100 % ("actual size", no fit-to-page) and measure a square or a tag with a ruler,
then enter the measured size (the board records are what the solvers use).

Known limits: the tags are drawn as OpenCV's generateImageMarker draws the dictionary's codes (upright, bit row 0 at
the top). Whether a board printed by Kalibr's own kalibr_create_target_pdf (which turns its code matrix by 180°
before drawing, and uses the AprilTags C++ library's 36h11 codes) decodes with the same corner order in OpenCV was
not tested here; print the board from DexVantage, or check one detection's corner order against Kalibr's before
mixing tools. Boards are taken as perfectly flat and printed true to size.
"""
from __future__ import annotations

import math
import os
import struct
import zlib
from dataclasses import asdict, dataclass

import numpy as np

DEFAULT_CHARUCO_DICT = 'DICT_5X5_100'     # a default (any predefined ArUco dictionary with enough markers works)
APRILGRID_SPACING = 0.3                   # Kalibr's tagSpacing in its wiki example and creator default (see above)
APRILGRID_BORDER_BITS = 2                 # Kalibr's blackTagBorder default
APRILGRID_MIN_TAGS = 4                    # Kalibr's minTagsForValidObs default
APRILGRID_DICT = 'DICT_APRILTAG_36h11'
DEFAULT_DPI = 300                         # a default for PNG output
INCH_M = 0.0254

_CV_BOARDS: dict = {}


class BoardError(ValueError):
    """A board definition that cannot be built; the message says why in plain words."""


def dictionary_names() -> tuple:
    """The predefined ArUco dictionaries OpenCV offers (names as cv2.aruco spells them)."""
    import cv2
    return tuple(sorted(n for n in dir(cv2.aruco) if n.startswith('DICT_') and n == _canon(n)))


def _canon(name: str) -> str:
    n = str(name).strip()
    if not n.upper().startswith('DICT_'):
        n = 'DICT_' + n
    # OpenCV spells the AprilTag ones both DICT_APRILTAG_36h11 and DICT_APRILTAG_36H11
    if n.upper().startswith('DICT_APRILTAG_'):
        return 'DICT_APRILTAG_' + n[len('DICT_APRILTAG_'):].lower()
    if n.upper().startswith('DICT_ARUCO_MIP_'):
        return 'DICT_ARUCO_MIP_' + n[len('DICT_ARUCO_MIP_'):].lower()
    return n.upper()


def get_dictionary(name: str):
    import cv2
    n = _canon(name)
    if not hasattr(cv2.aruco, n):
        raise BoardError(f'OpenCV has no ArUco dictionary named {name!r}; use one of: '
                         f'{", ".join(dictionary_names())}')
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, n))


def _positive(name, v, unit='m'):
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise BoardError(f'the {name} must be a number of {unit}, got {v!r}') from None
    if not math.isfinite(f) or f <= 0:
        raise BoardError(f'the {name} must be a positive number of {unit}, got {v!r}')
    return f


def _count(name, v, lo):
    try:
        i = int(v)
    except (TypeError, ValueError):
        raise BoardError(f'the {name} must be a whole number, got {v!r}') from None
    if i != v or i < lo:
        raise BoardError(f'the {name} must be a whole number of at least {lo}, got {v!r}')
    return i


def _cell_rects(img: np.ndarray, x0: float, y0: float, cell: float) -> list:
    """Black cells of a one-pixel-per-cell marker image as page rectangles, runs along each row merged."""
    out = []
    for i, row in enumerate(np.asarray(img) < 128):
        j = 0
        n = len(row)
        while j < n:
            if row[j]:
                k = j
                while k < n and row[k]:
                    k += 1
                out.append((x0 + j * cell, y0 + i * cell, x0 + k * cell, y0 + (i + 1) * cell))
                j = k
            else:
                j += 1
    return out


# =========================================================================== ChArUco
@dataclass(frozen=True)
class CharucoBoard:
    """A ChArUco board (see the module docstring for the layout and the frame). Sizes in metres."""
    squares_x: int
    squares_y: int
    square_m: float
    marker_m: float
    dictionary: str = DEFAULT_CHARUCO_DICT
    legacy: bool = False

    kind = 'charuco'

    def __post_init__(self):
        s = object.__setattr__
        s(self, 'squares_x', _count('number of squares across', self.squares_x, 3))
        s(self, 'squares_y', _count('number of squares down', self.squares_y, 3))
        s(self, 'square_m', _positive('square size', self.square_m))
        s(self, 'marker_m', _positive('marker size', self.marker_m))
        if self.marker_m >= self.square_m:
            raise BoardError(f'the marker ({1000 * self.marker_m:g} mm) must be smaller than the square '
                             f'({1000 * self.square_m:g} mm) it sits in')
        s(self, 'dictionary', _canon(self.dictionary))
        s(self, 'legacy', bool(self.legacy))
        d = get_dictionary(self.dictionary)
        need = (self.squares_x * self.squares_y) // 2
        if need > d.bytesList.shape[0]:
            raise BoardError(f'a {self.squares_x} × {self.squares_y} ChArUco board needs {need} markers; '
                             f'{self.dictionary} has {d.bytesList.shape[0]}')

    # ------------------------------------------------------------------ OpenCV
    def cv(self):
        """The cv2.aruco.CharucoBoard (built once per definition)."""
        b = _CV_BOARDS.get(self)
        if b is None:
            import cv2
            b = cv2.aruco.CharucoBoard((self.squares_x, self.squares_y), self.square_m, self.marker_m,
                                       get_dictionary(self.dictionary))
            if self.legacy:
                b.setLegacyPattern(True)
            _CV_BOARDS[self] = b
        return b

    # ------------------------------------------------------------------ geometry
    @property
    def n_points(self) -> int:
        return (self.squares_x - 1) * (self.squares_y - 1)

    def points(self) -> np.ndarray:
        """(N, 3) board-frame coordinates (m) of the inner chessboard corners, indexed by ChArUco corner id."""
        return np.asarray(self.cv().getChessboardCorners(), float).reshape(-1, 3)

    @property
    def size_m(self) -> tuple:
        """(width, height) of the printed pattern, m."""
        return (self.squares_x * self.square_m, self.squares_y * self.square_m)

    def page_origin(self) -> np.ndarray:
        """Page coordinates are board coordinates (x right, y down from the pattern's top-left)."""
        return np.zeros(2)

    def board_to_page(self, xy) -> np.ndarray:
        return np.asarray(xy, float)[..., :2].copy()

    def page_to_board(self, xy) -> np.ndarray:
        return np.asarray(xy, float).copy()

    def page_extent(self) -> tuple:
        """(x0, y0, x1, y1) of everything drawn, page metres."""
        w, h = self.size_m
        return (0.0, 0.0, w, h)

    def page_rects(self) -> list:
        """Black rectangles (x0, y0, x1, y1) in page metres: the chessboard's black squares and every marker's
        black cells."""
        b = self.cv()
        sq = self.square_m
        obj = np.asarray(b.getObjPoints(), float)            # (M, 4, 3) marker corners tl, tr, br, bl
        ids = np.asarray(b.getIds()).ravel()
        d = get_dictionary(self.dictionary)
        bits = int(d.markerSize)
        host = set()
        out = []
        for m, mid in zip(obj, ids):
            x0, y0 = m[:, 0].min(), m[:, 1].min()
            host.add((int(math.floor(x0 / sq)), int(math.floor(y0 / sq))))
            import cv2
            img = cv2.aruco.generateImageMarker(d, int(mid), bits + 2, borderBits=1)
            out += _cell_rects(img, x0, y0, self.marker_m / (bits + 2))
        for r in range(self.squares_y):
            for c in range(self.squares_x):
                if (c, r) not in host:
                    out.append((c * sq, r * sq, (c + 1) * sq, (r + 1) * sq))
        return out

    def describe(self) -> str:
        return (f'ChArUco {self.squares_x} × {self.squares_y} squares, square {1000 * self.square_m:g} mm, marker '
                f'{1000 * self.marker_m:g} mm, {self.dictionary}' + (' (legacy layout)' if self.legacy else ''))

    def to_dict(self) -> dict:
        return {'kind': self.kind, **asdict(self)}


# =========================================================================== AprilGrid
@dataclass(frozen=True)
class AprilGrid:
    """A Kalibr AprilGrid (see the module docstring for the layout and the frame). Sizes in metres."""
    tag_cols: int
    tag_rows: int
    tag_m: float
    spacing: float = APRILGRID_SPACING
    first_id: int = 0
    corner_squares: bool = True

    kind = 'aprilgrid'

    def __post_init__(self):
        s = object.__setattr__
        s(self, 'tag_cols', _count('number of tag columns', self.tag_cols, 1))
        s(self, 'tag_rows', _count('number of tag rows', self.tag_rows, 1))
        s(self, 'tag_m', _positive('tag size', self.tag_m))
        s(self, 'spacing', _positive('tag spacing ratio', self.spacing, 'tag sizes'))
        s(self, 'first_id', _count('first tag id', self.first_id, 0))
        s(self, 'corner_squares', bool(self.corner_squares))
        n = get_dictionary(APRILGRID_DICT).bytesList.shape[0]
        if self.first_id + self.tag_cols * self.tag_rows > n:
            raise BoardError(f'a {self.tag_cols} × {self.tag_rows} AprilGrid from id {self.first_id} needs ids up to '
                             f'{self.first_id + self.tag_cols * self.tag_rows - 1}; the 36h11 family has {n} tags')

    @property
    def gap_m(self) -> float:
        return self.spacing * self.tag_m

    @property
    def pitch_m(self) -> float:
        return (1.0 + self.spacing) * self.tag_m

    @property
    def n_tags(self) -> int:
        return self.tag_cols * self.tag_rows

    @property
    def n_points(self) -> int:
        return 4 * self.n_tags

    def points(self) -> np.ndarray:
        """(N, 3) board-frame coordinates (m) of every tag corner, in Kalibr's point order."""
        C, R = np.meshgrid(np.arange(2 * self.tag_cols), np.arange(2 * self.tag_rows))
        x = (C // 2) * self.pitch_m + (C % 2) * self.tag_m
        y = (R // 2) * self.pitch_m + (R % 2) * self.tag_m
        return np.stack([x.ravel(), y.ravel(), np.zeros(x.size)], 1).astype(float)

    def tag_point_ids(self, tag_id: int) -> np.ndarray:
        """Point indices of a tag's corners in OpenCV's marker corner order (top-left, top-right, bottom-right,
        bottom-left of the upright tag): Kalibr's corners 3, 2, 1, 0."""
        t = int(tag_id) - self.first_id
        if not 0 <= t < self.n_tags:
            raise KeyError(f'tag {tag_id} is not on this AprilGrid')
        cols2 = 2 * self.tag_cols
        base = (t // self.tag_cols) * cols2 * 2 + (t % self.tag_cols) * 2
        return np.array([base + cols2, base + cols2 + 1, base + 1, base], int)

    @property
    def size_m(self) -> tuple:
        """(width, height) of the tag grid itself (outer black edges), m."""
        return (self.tag_cols * self.tag_m + (self.tag_cols - 1) * self.gap_m,
                self.tag_rows * self.tag_m + (self.tag_rows - 1) * self.gap_m)

    def _pad(self) -> float:
        return self.gap_m if self.corner_squares else 0.0

    def page_extent(self) -> tuple:
        w, h = self.size_m
        p = self._pad()
        return (0.0, 0.0, w + 2 * p, h + 2 * p)

    def board_to_page(self, xy) -> np.ndarray:
        """Board (x right, y up from tag 0's bottom-left corner) to page (x right, y down from the top-left)."""
        xy = np.asarray(xy, float)[..., :2]
        p = self._pad()
        h = self.size_m[1]
        return np.stack([xy[..., 0] + p, (h + p) - xy[..., 1]], -1)

    def page_to_board(self, xy) -> np.ndarray:
        xy = np.asarray(xy, float)
        p = self._pad()
        h = self.size_m[1]
        return np.stack([xy[..., 0] - p, (h + p) - xy[..., 1]], -1)

    def page_rects(self) -> list:
        import cv2
        d = get_dictionary(APRILGRID_DICT)
        bits = int(d.markerSize) + 2 * APRILGRID_BORDER_BITS
        cell = self.tag_m / bits
        g = self.gap_m
        out = []
        for r in range(self.tag_rows):
            for c in range(self.tag_cols):
                tid = self.first_id + r * self.tag_cols + c
                x0, y0 = c * self.pitch_m, r * self.pitch_m              # bottom-left, board frame
                px, py = self.board_to_page([x0, y0 + self.tag_m])       # top-left, page
                img = cv2.aruco.generateImageMarker(d, tid, bits, borderBits=APRILGRID_BORDER_BITS)
                out += _cell_rects(img, float(px), float(py), cell)
                if self.corner_squares:
                    for cx, cy in ((x0 - g, y0 - g), (x0 + self.tag_m, y0 - g), (x0 - g, y0 + self.tag_m),
                                   (x0 + self.tag_m, y0 + self.tag_m)):
                        a = self.board_to_page([cx, cy + g])
                        out.append((float(a[0]), float(a[1]), float(a[0]) + g, float(a[1]) + g))
        return sorted(set(out))

    def describe(self) -> str:
        return (f'AprilGrid {self.tag_cols} × {self.tag_rows} tags (36h11 from id {self.first_id}), tag '
                f'{1000 * self.tag_m:g} mm, spacing {self.spacing:g}' + ('' if self.corner_squares else
                                                                          ', no corner squares'))

    def to_dict(self) -> dict:
        return {'kind': self.kind, **asdict(self)}


Board = CharucoBoard | AprilGrid


def board_from_dict(d: dict) -> Board:
    """A board from to_dict's output; BoardError says what is wrong."""
    if not isinstance(d, dict):
        raise BoardError('a board must be a dictionary of its figures')
    d = dict(d)
    kind = d.pop('kind', None)
    cls = {'charuco': CharucoBoard, 'aprilgrid': AprilGrid}.get(kind)
    if cls is None:
        raise BoardError(f'unknown board kind {kind!r}: use "charuco" or "aprilgrid"')
    try:
        return cls(**d)
    except TypeError as e:
        raise BoardError(f'the {kind} board record is not complete or has unknown fields ({e})') from None


def board_points(board: Board) -> np.ndarray:
    """(N, 3) calibration point coordinates (m) in the board frame (the module docstring gives each frame)."""
    return board.points()


# =========================================================================== raster and print
def _cover(lo: float, hi: float, n: int) -> tuple[int, np.ndarray]:
    """First pixel index and the fraction of each pixel [i, i + 1) that [lo, hi) covers."""
    i0 = max(int(math.floor(lo)), 0)
    i1 = min(int(math.ceil(hi)), n)
    if i1 <= i0:
        return i0, np.zeros(0)
    idx = np.arange(i0, i1, dtype=float)
    return i0, np.clip(np.minimum(idx + 1, hi) - np.maximum(idx, lo), 0.0, 1.0)


def rasterise(board: Board, px_per_m: float, margin_m: float = 0.0, white: int = 255,
              black: int = 0) -> np.ndarray:
    """The board as a grey uint8 image (exact area coverage of every black rectangle, so edges are anti-aliased),
    px_per_m pixels per metre, a white margin of margin_m around what is drawn. Pixel (i, j) covers page
    [j, j + 1) / px_per_m − margin_m (OpenCV pixel centres at integers + ½ of a page pixel)."""
    s = _positive('resolution', px_per_m, 'pixels per metre')
    m = max(float(margin_m), 0.0)
    x0, y0, x1, y1 = board.page_extent()
    W = int(math.ceil((x1 - x0 + 2 * m) * s - 1e-9))
    H = int(math.ceil((y1 - y0 + 2 * m) * s - 1e-9))
    if W * H > 400_000_000:
        raise BoardError(f'a {W} × {H} pixel image is too large; lower the resolution')
    ink = np.zeros((H, W), np.float32)
    for a, b, c, d in board.page_rects():
        j0, cx = _cover((a - x0 + m) * s, (c - x0 + m) * s, W)
        i0, cy = _cover((b - y0 + m) * s, (d - y0 + m) * s, H)
        if len(cx) and len(cy):
            ink[i0:i0 + len(cy), j0:j0 + len(cx)] += np.outer(cy, cx).astype(np.float32)
    ink = np.clip(ink, 0.0, 1.0)
    return np.round(white + (black - white) * ink).astype(np.uint8)


def render_board(board: Board, dpi: float = DEFAULT_DPI, margin_m: float | None = None) -> np.ndarray:
    """The board for printing as a grey uint8 image at dpi dots per inch, with a white margin (default: one square
    of a ChArUco board, one tag of an AprilGrid, as Kalibr's wiki asks: "a white border around the calibration
    target of min. the size of one grid element")."""
    return rasterise(board, _positive('resolution', dpi, 'dots per inch') / INCH_M, default_margin(board)
                     if margin_m is None else margin_m)


def default_margin(board: Board) -> float:
    return board.square_m if isinstance(board, CharucoBoard) else board.tag_m


def _png(img: np.ndarray, px_per_m: float) -> bytes:
    h, w = img.shape

    def chunk(t: bytes, data: bytes) -> bytes:
        return struct.pack('>I', len(data)) + t + data + struct.pack('>I', zlib.crc32(t + data) & 0xFFFFFFFF)
    raw = b''.join(b'\x00' + img[i].tobytes() for i in range(h))
    ppm = int(round(px_per_m))
    return (b'\x89PNG\r\n\x1a\n' + chunk(b'IHDR', struct.pack('>IIBBBBB', w, h, 8, 0, 0, 0, 0)) +
            chunk(b'pHYs', struct.pack('>IIB', ppm, ppm, 1)) + chunk(b'IDAT', zlib.compress(raw, 9)) +
            chunk(b'IEND', b''))


def _pdf(board: Board, margin_m: float, caption: str) -> bytes:
    pt = 72.0 / INCH_M                                   # PDF points per metre
    x0, y0, x1, y1 = board.page_extent()
    W, H = (x1 - x0 + 2 * margin_m) * pt, (y1 - y0 + 2 * margin_m) * pt
    ops = ['0 g']
    for a, b, c, d in board.page_rects():
        ops.append(f'{(a - x0 + margin_m) * pt:.4f} {H - (d - y0 + margin_m) * pt:.4f} {(c - a) * pt:.4f} '
                   f'{(d - b) * pt:.4f} re f')
    size = min(8.0, 0.4 * margin_m * pt)
    if caption and size >= 4.0:
        txt = caption.encode('ascii', 'replace').decode('ascii').replace('\\', '\\\\').replace('(', '\\(') \
            .replace(')', '\\)')
        ops.append(f'BT /F1 {size:.2f} Tf {0.5 * margin_m * pt:.2f} {0.35 * margin_m * pt:.2f} Td ({txt}) Tj ET')
    content = '\n'.join(ops).encode('ascii')
    objs = [b'<< /Type /Catalog /Pages 2 0 R >>',
            b'<< /Type /Pages /Kids [3 0 R] /Count 1 >>',
            (f'<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {W:.4f} {H:.4f}] /Contents 4 0 R '
             f'/Resources << /Font << /F1 5 0 R >> >> >>').encode('ascii'),
            b'<< /Length ' + str(len(content)).encode() + b' >>\nstream\n' + content + b'\nendstream',
            b'<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>']
    out = bytearray(b'%PDF-1.4\n')
    offs = []
    for i, o in enumerate(objs, 1):
        offs.append(len(out))
        out += f'{i} 0 obj\n'.encode() + o + b'\nendobj\n'
    xref = len(out)
    out += f'xref\n0 {len(objs) + 1}\n0000000000 65535 f \n'.encode()
    for o in offs:
        out += f'{o:010d} 00000 n \n'.encode()
    out += f'trailer\n<< /Size {len(objs) + 1} /Root 1 0 R >>\nstartxref\n{xref}\n%%EOF\n'.encode()
    return bytes(out)


def print_caption(board: Board) -> str:
    return (board.describe().replace('×', 'x') + '. Print at 100 % (actual size, no fit to page); measure a '
            + ('square' if isinstance(board, CharucoBoard) else 'tag') + ' and enter the measured size.')


def save_board(board: Board, path: str, dpi: float = DEFAULT_DPI, margin_m: float | None = None) -> str:
    """Write the board for printing at true scale: '.png' (grey, its resolution stored in the file) or '.pdf' (one
    page of exactly the board's size plus the margin, drawn in vectors, with a caption in the margin). Returns
    the path. Raises BoardError for another extension."""
    m = default_margin(board) if margin_m is None else max(float(margin_m), 0.0)
    ext = os.path.splitext(str(path))[1].lower()
    if ext == '.png':
        s = _positive('resolution', dpi, 'dots per inch') / INCH_M
        data = _png(rasterise(board, s, m), s)
    elif ext == '.pdf':
        data = _pdf(board, m, print_caption(board))
    else:
        raise BoardError(f'save the board as .png or .pdf, not {ext or "a file without an extension"}')
    with open(path, 'wb') as f:
        f.write(data)
    return str(path)
