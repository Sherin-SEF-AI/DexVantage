"""Plücker ray maps at a policy's input resolution (spec §7.12): one ray per input pixel, through the camera's real
lens model and the policy's crop and resize, in the robot base frame, for cross-embodiment models that condition on
camera geometry.

Ported from Vantage core/plucker.py: the per-pixel ray field (origin o, unit direction d unprojected through the exact
lens, the moment m = o × d, channel order 'dm' or 'md') and the export of one file per camera plus a manifest. What
changed: the grid is the policy's input image, not a stride grid over the sensor (each input pixel centre is mapped
back to the sensor through core/policies.py CropGeometry, then unprojected); pixel centres follow OpenCV (DexVantage's
convention: integer centres in the sensor image); the frame is the robot base (or the world, or the camera), not the
ISO 8855 vehicle frame; the ground-plane range, incidence and ego mask of the automotive version are left out; the
files are .npy and .safetensors (written here, see below) with a provenance JSON beside each.

    input_rays(camera, geom) -> (u (w,), v (h,), rays (h, w, 3) optical, valid (h, w), content (h, w))
    ray_field(project, sensor, imager=None, policy_uid=None, geom=None, q=None, frame='base', robot_uid=None)
        -> RayField
    export_plucker(project, out_dir, policy_uid, q=None, frame='base', robot_uid=None, dtype='float32', order='dm',
                   formats=('npy', 'safetensors'), cameras=None) -> [paths]
    write_safetensors(path, tensors, metadata=None);  read_safetensors(path) -> (tensors, metadata)

The mapping. Input pixel (row j, column i) has its centre at input edge coordinates (i + ½, j + ½) (core/policies.py
uses pixel-edge coordinates: x runs from 0 at the left edge to w at the right edge). CropGeometry.to_source maps it to
sensor edge coordinates (x, y), the sensor pixel coordinate is (u, v) = (x − ½, y − ½) (OpenCV: pixel centres at
integers), and CameraModel.unproject gives the unit ray in the optical frame (z forward, x right, y down). Every rule
is separable (u depends on the column only, v on the row only), so u and v are stored as 1-D arrays. A pixel is valid
when it shows the camera image (inside the crop's content rectangle, not padding), its source point lies inside the
sensor image (u in [−0.5, W − 0.5], v likewise) and the lens has a ray there (within the valid angle).

Plücker coordinates: d = the unit direction in the chosen frame, m = o × d with o the optical centre in that frame
(metres), so m · d = 0 and m is the same for every point o + t d of the line. Channels [d_x, d_y, d_z, m_x, m_y, m_z]
for order 'dm' (default), [m, d] for 'md'. Invalid pixels hold zeros in all six channels and False in 'valid'.

Frames: 'base' (default) is the base frame of robot_uid (default the project's primary robot, whose base is the world
origin, so 'base' and 'world' agree unless another robot is named; another robot's base pose from Robot.base);
'world' is the project's world (the primary robot's base, REP-103: x forward, y left, z up); 'camera' is the
optical frame (o = 0, so the moments are zero: the directions alone, a ray-direction embedding).

safetensors. The format is written here (the package is optional): 8 bytes, the header length N as an unsigned
little-endian 64-bit integer; N bytes of UTF-8 JSON, starting with '{', padded with spaces to a multiple of 8 bytes
(as the reference writer does), mapping each tensor name to {"dtype", "shape", "data_offsets": [begin, end]} relative
to the start of the byte buffer, plus "__metadata__" (string to string); then the byte buffer, every tensor
little-endian and row-major, back to back with no holes [safetensors README "Format" and safetensors/src/tensor.rs,
https://github.com/huggingface/safetensors, read 2026-11-06]. Tensors are written in name order.

Known limits: a pixel's ray is the ray of its centre (no averaging over the pixel's footprint, which an
antialiasing resize would do); the crop and resize are the inference-time geometry of core/policies.py (training
augmentation is not modelled); the camera pose is the one at q (a wrist camera moves with the arm: export at the pose
the policy sees, or once per pose); rays are not clipped by the scene (no ego or self-occlusion mask: the occlusion
studies treat that).
"""
from __future__ import annotations

import json
import os
import re
import struct
from dataclasses import dataclass

import numpy as np

from .jobs import StudyError

SCHEMA = 'dexvantage.plucker/1'
FRAMES = ('base', 'world', 'camera')
CONVENTION = ('Plücker line coordinates of each input pixel ray: d = unit direction, m = o × d (moment about the '
              'frame origin, metres); channels [d_x, d_y, d_z, m_x, m_y, m_z] for order "dm", [m, d] for "md". Input '
              'pixel (row j, column i) is the ray of sensor pixel (u[i], v[j]) (OpenCV: pixel centres at integers), '
              'the point its centre (i + 0.5, j + 0.5) maps to through the policy\'s crop and resize. Invalid pixels '
              '(padding, outside the sensor image or the lens\'s valid angle) are zero with valid = false.')
FRAME_TEXT = {
    'base': 'the robot\'s base frame (REP-103: x forward, y left, z up, metres)',
    'world': 'the project\'s world frame (the primary robot\'s base, REP-103: x forward, y left, z up, metres)',
    'camera': 'the camera optical frame (OpenCV: x right, y down, z forward); the origin is the optical centre, so '
              'the moments are zero',
}
# numpy dtype -> safetensors dtype name (safetensors/src/tensor.rs Dtype)
ST_DTYPES = {'bool': 'BOOL', 'uint8': 'U8', 'int8': 'I8', 'int16': 'I16', 'uint16': 'U16', 'float16': 'F16',
             'int32': 'I32', 'uint32': 'U32', 'float32': 'F32', 'float64': 'F64', 'int64': 'I64', 'uint64': 'U64'}
ST_NUMPY = {v: k for k, v in ST_DTYPES.items()}


class PluckerError(StudyError):
    """A ray map that cannot be made; the message says why in plain words."""


# ====================================================================== rays through the crop
def input_rays(camera, geom):
    """Rays of every input pixel of a crop geometry (core/policies.py CropGeometry) through camera (an
    optics.CameraModel at the geometry's source size): (u (w,), v (h,) sensor pixel coordinates (OpenCV), rays
    (h, w, 3) unit optical-frame directions (NaN where there is none), valid (h, w), content (h, w): not padding)."""
    W, H = (int(x) for x in geom.src_size)
    if (camera.width, camera.height) != (W, H):
        raise PluckerError(f'The crop geometry is for a {W} × {H} image but the camera is {camera.width} × '
                           f'{camera.height}.')
    w, h = (int(x) for x in geom.input_size)
    xi = np.arange(w, dtype=np.float64) + 0.5
    yi = np.arange(h, dtype=np.float64) + 0.5
    xs, _ = geom.to_source(xi, np.zeros_like(xi))
    _, ys = geom.to_source(np.zeros_like(yi), yi)
    u, v = xs - 0.5, ys - 0.5
    x0, y0, x1, y1 = geom.dst_rect
    cx = (xi > x0) & (xi < x1)
    cy = (yi > y0) & (yi < y1)
    content = cy[:, None] & cx[None, :]
    e = 1e-9
    in_img = ((v >= -0.5 - e) & (v <= H - 0.5 + e))[:, None] & ((u >= -0.5 - e) & (u <= W - 0.5 + e))[None, :]
    U, V = np.meshgrid(u, v)
    rays, ok = camera.unproject(np.stack([U, V], axis=-1), with_ok=True)
    valid = content & in_img & ok
    rays = np.where(valid[..., None], rays, np.nan)
    return u, v, rays, valid, content


@dataclass
class RayField:
    """One camera's rays at a policy's input resolution. Arrays are (h, w[, 3]); row j is v, column i is u."""
    sensor_uid: str
    name: str
    imager: str
    frame: str
    frame_robot: str              # the robot whose base 'base' means ('' otherwise)
    width: int                    # input pixels
    height: int
    u: np.ndarray                 # (w,) sensor u of each column (px, OpenCV centres)
    v: np.ndarray                 # (h,)
    origin: np.ndarray            # (3,) o in the frame
    rotation: np.ndarray          # (3, 3) optical -> frame: d_frame = rotation @ d_optical
    d: np.ndarray                 # (h, w, 3) unit directions (zero where not valid)
    m: np.ndarray                 # (h, w, 3) moments o × d (zero where not valid)
    valid: np.ndarray             # (h, w)
    content: np.ndarray           # (h, w) not padding
    camera: object                # optics.CameraModel at the sensor stream
    geometry: object              # policies.CropGeometry

    def plucker(self, order: str = 'dm') -> np.ndarray:
        """(h, w, 6): [d, m] ('dm') or [m, d] ('md')."""
        if order not in ('dm', 'md'):
            raise PluckerError(f'The channel order must be "dm" or "md", not {order!r}.')
        return np.concatenate([self.d, self.m] if order == 'dm' else [self.m, self.d], axis=-1)


def frame_transform(project, frame: str = 'base', robot_uid: str | None = None) -> tuple[np.ndarray, str]:
    """(4 × 4 world -> frame, robot uid the frame belongs to or ''). 'camera' is handled by the caller."""
    if frame not in FRAMES:
        raise PluckerError(f'The frame must be one of {", ".join(FRAMES)}, not {frame!r}.')
    if frame != 'base':
        return np.eye(4), ''
    if robot_uid:
        rob = project.robot(robot_uid)
        if rob is None:
            raise PluckerError('The robot named for the base frame is not in the project.')
    else:
        rob = project.primary_robot               # the primary (else the first): its base is the world origin
        if rob is None:
            return np.eye(4), ''                  # no robot: the world (the caller says so)
    return np.linalg.inv(rob.base.matrix()), rob.uid


def pick_imager(sensor, mode=None):
    """The imager a policy sees from a sensor: its RGB imager when it has one, else its first (as the Camera view)."""
    from . import library as LB
    ims = LB.sensor_imagers(sensor, mode)
    if not ims:
        return None
    return next((i for i in ims if i.role == 'rgb'), ims[0])


def sensor_stream(sensor):
    m = getattr(sensor, 'stream', None)
    return (int(m[0]), int(m[1])) if m and len(m) == 2 else None


def ray_field(project, sensor, imager=None, policy_uid: str | None = None, geom=None, q: dict | None = None,
              frame: str = 'base', robot_uid: str | None = None) -> RayField:
    """The rays of sensor's imager (default pick_imager) at the policy's input resolution (geom, else
    policies.input_geometry for policy_uid), at joint state q ({robot uid: q}; None = home), in frame."""
    from . import camrender as CR
    from . import policies as PO
    from .frames import quat_to_matrix
    im = imager or pick_imager(sensor, sensor_stream(sensor))
    if im is None:
        raise PluckerError(f'{sensor.name} has no imager to make rays for.')
    cam = im.camera
    if geom is None:
        if not policy_uid:
            raise PluckerError('Name the policy whose input the rays are for.')
        geom = PO.input_geometry(project, policy_uid, (cam.width, cam.height))
    u, v, rays, valid, content = input_rays(cam, geom)
    if frame == 'camera':
        R, o, who = np.eye(3), np.zeros(3), ''
    else:
        pose = CR.optical_pose(project, sensor, im, q)
        T_fw, who = frame_transform(project, frame, robot_uid)
        R = T_fw[:3, :3] @ quat_to_matrix(pose.quaternion)
        o = T_fw[:3, :3] @ np.asarray(pose.position, float) + T_fw[:3, 3]
    d = np.where(valid[..., None], rays, 0.0) @ R.T
    nrm = np.linalg.norm(d, axis=-1, keepdims=True)
    d = np.where(valid[..., None], d / np.maximum(nrm, 1e-300), 0.0)
    m = np.where(valid[..., None], np.cross(np.broadcast_to(o, d.shape), d), 0.0)
    return RayField(sensor.uid, sensor.name, im.name, frame, who, int(geom.input_size[0]), int(geom.input_size[1]),
                    u, v, o, R, d, m, valid, content, cam, geom)


# ====================================================================== safetensors
def write_safetensors(path: str, tensors: dict, metadata: dict | None = None) -> str:
    """Write tensors {name: numpy array} as a .safetensors file (module docstring); metadata: str -> str."""
    header: dict = {}
    if metadata:
        header['__metadata__'] = {str(k): str(v) for k, v in metadata.items()}
    bufs, off = [], 0
    for name in sorted(tensors):
        if name == '__metadata__':
            raise PluckerError('"__metadata__" is reserved in a safetensors file.')
        a = np.asarray(tensors[name])
        key = a.dtype.name
        if key not in ST_DTYPES:
            raise PluckerError(f'A {key} array cannot be written to safetensors here.')
        b = np.ascontiguousarray(a, dtype=a.dtype.newbyteorder('<')).tobytes(order='C')
        header[str(name)] = {'dtype': ST_DTYPES[key], 'shape': [int(s) for s in a.shape],
                             'data_offsets': [off, off + len(b)]}
        bufs.append(b)
        off += len(b)
    hb = json.dumps(header, separators=(',', ':'), ensure_ascii=False).encode('utf-8')
    hb += b' ' * ((-len(hb)) % 8)
    tmp = path + '.part'
    with open(tmp, 'wb') as fh:
        fh.write(struct.pack('<Q', len(hb)))
        fh.write(hb)
        for b in bufs:
            fh.write(b)
    os.replace(tmp, path)
    return path


def read_safetensors(path: str) -> tuple[dict, dict]:
    """({name: numpy array}, metadata) of a .safetensors file; PluckerError when it is not a valid one."""
    with open(path, 'rb') as fh:
        data = fh.read()
    if len(data) < 8:
        raise PluckerError(f'{os.path.basename(path)} is too short to be a safetensors file.')
    (n,) = struct.unpack('<Q', data[:8])
    if 8 + n > len(data):
        raise PluckerError(f'{os.path.basename(path)}: the header length runs past the end of the file.')
    raw = data[8:8 + n]
    if not raw.startswith(b'{'):
        raise PluckerError(f'{os.path.basename(path)}: the header does not start with "{{".')
    try:
        header = json.loads(raw.decode('utf-8'))
    except (UnicodeDecodeError, ValueError):
        raise PluckerError(f'{os.path.basename(path)}: the header is not JSON.') from None
    meta = header.pop('__metadata__', {}) or {}
    buf = memoryview(data)[8 + n:]
    out, spans = {}, []
    for name, t in header.items():
        dt = ST_NUMPY.get(t.get('dtype'))
        if dt is None:
            raise PluckerError(f'{name}: dtype {t.get("dtype")!r} is not read here.')
        b, e = (int(x) for x in t['data_offsets'])
        shape = tuple(int(s) for s in t['shape'])
        nd = np.dtype(dt).newbyteorder('<')
        if not 0 <= b <= e <= len(buf) or (e - b) != int(np.prod(shape, dtype=np.int64)) * nd.itemsize:
            raise PluckerError(f'{name}: its data offsets do not match its shape.')
        out[name] = np.frombuffer(buf[b:e], dtype=nd).reshape(shape).astype(np.dtype(dt), copy=True)
        spans.append((b, e))
    spans.sort()
    pos = 0
    for b, e in spans:
        if b != pos:
            raise PluckerError(f'{os.path.basename(path)}: the byte buffer has a hole or an overlap.')
        pos = e
    if pos != len(buf):
        raise PluckerError(f'{os.path.basename(path)}: the byte buffer has bytes no tensor names.')
    return out, dict(meta)


# ====================================================================== the export
def _tag(text: str) -> str:
    t = re.sub(r'[^A-Za-z0-9_-]+', '_', str(text)).strip('_')
    return t[:48] or 'camera'


def field_record(f: RayField, order: str = 'dm') -> dict:
    """What the provenance JSON says about one ray field (JSON-able)."""
    W, H = f.camera.width, f.camera.height
    g = f.geometry
    x0, y0, x1, y1 = g.src_rect
    return {'sensor_uid': f.sensor_uid, 'camera': f.name, 'imager': f.imager, 'frame': f.frame,
            'frame_robot': f.frame_robot, 'shape': [f.height, f.width, 6], 'order': order,
            'origin_m': [float(x) for x in f.origin],
            'rotation_optical_to_frame': [[float(x) for x in r] for r in f.rotation],
            'valid_share': float(f.valid.mean()), 'padding_share': float(1.0 - f.content.mean()),
            'sensor_size': [W, H], 'intrinsics': f.camera.to_dict(),
            'crop': dict(g.to_dict(), crop_box_sensor_px={'u0': x0 - 0.5, 'v0': y0 - 0.5, 'u1': x1 - 0.5,
                                                          'v1': y1 - 0.5, 'convention': 'OpenCV pixel centres; '
                                                          'the box runs from edge to edge'})}


def export_plucker(project, out_dir: str, policy_uid: str, q: dict | None = None, frame: str = 'base',
                   robot_uid: str | None = None, dtype: str = 'float32', order: str = 'dm',
                   formats=('npy', 'safetensors'), cameras=None) -> list:
    """Write the ray map of every camera the policy takes (cameras: sensor uids to restrict to): per camera
    plucker_<policy>_<camera>.npy ((h, w, 6) in dtype) and/or .safetensors (tensors 'plucker' (h, w, 6), 'valid'
    (h, w) bool, 'u' (w,), 'v' (h,) float32, 'origin' (3,), 'rotation' (3, 3) float64), and a
    plucker_<policy>_<camera>.json beside each with the convention, the lens, the crop and the provenance. Returns
    the paths written."""
    from . import policies as PO
    from . import provenance as PV
    if dtype not in ('float16', 'float32'):
        raise PluckerError('The data type must be float16 or float32.')
    if order not in ('dm', 'md'):
        raise PluckerError(f'The channel order must be "dm" or "md", not {order!r}.')
    formats = tuple(formats)
    if not formats or any(f not in ('npy', 'safetensors') for f in formats):
        raise PluckerError('The formats are npy and safetensors.')
    pol = PO.policy(project, policy_uid)
    if pol is None:
        raise PluckerError('That policy is not in the project.')
    sensors = [s for s in PO.policy_cameras(project, pol) if s.enabled and (cameras is None or s.uid in cameras)]
    if not sensors:
        raise PluckerError(f'{pol.name} takes no camera that is in use: assign its cameras on the Policies page.')
    try:
        os.makedirs(out_dir, exist_ok=True)
    except OSError as e:
        raise PluckerError(f'Could not make the folder {os.path.basename(out_dir)}: {e.strerror or e}.') from None
    paths, used = [], set()
    for s in sensors:
        f = ray_field(project, s, policy_uid=pol.uid, q=q, frame=frame, robot_uid=robot_uid)
        stem = f'plucker_{_tag(pol.name)}_{_tag(s.name)}'
        k = 2
        while stem in used:
            stem = f'plucker_{_tag(pol.name)}_{_tag(s.name)}_{k}'
            k += 1
        used.add(stem)
        pk = f.plucker(order).astype(dtype)
        rec = field_record(f, order)
        files = {}
        try:
            if 'npy' in formats:
                p = os.path.join(out_dir, stem + '.npy')
                np.save(p, pk)
                files['npy'] = os.path.basename(p)
                paths.append(p)
            if 'safetensors' in formats:
                p = os.path.join(out_dir, stem + '.safetensors')
                meta = {'schema': SCHEMA, 'frame': frame, 'order': order, 'convention': CONVENTION,
                        'camera': s.name, 'policy': pol.name}
                write_safetensors(p, {'plucker': pk, 'valid': f.valid, 'u': f.u.astype(np.float32),
                                      'v': f.v.astype(np.float32), 'origin': f.origin.astype(np.float64),
                                      'rotation': f.rotation.astype(np.float64)}, meta)
                files['safetensors'] = os.path.basename(p)
                paths.append(p)
            doc = {'schema': SCHEMA, 'policy': {'uid': pol.uid, 'name': pol.name, 'library_id': pol.library_id},
                   'frame_text': FRAME_TEXT[frame] + ('' if frame != 'base' or rec['frame_robot'] else
                                                      ' (the project has no robot: the world frame is used)'),
                   'convention': CONVENTION, 'dtype': dtype, 'files': files,
                   'q': {k2: [float(x) for x in np.asarray(v2, float).reshape(-1)] for k2, v2 in (q or {}).items()},
                   'q_rule': 'given' if q else 'each robot at its home pose', 'field': rec}
            p = os.path.join(out_dir, stem + '.json')
            doc = PV.stamp(doc, project, output={'option': 'plucker', 'name': os.path.basename(p), 'in_folder': True})
            with open(p, 'w', encoding='utf-8') as fh:
                json.dump(doc, fh, indent=1, ensure_ascii=False)
                fh.write('\n')
            paths.append(p)
        except OSError as e:
            raise PluckerError(f'Could not write the ray map for {s.name}: {e.strerror or e}.') from None
    return paths
