"""The auditor's reports (spec §8.2 output, §9.4 FiftyOne): JSON with provenance, a standalone HTML page, and a
FiftyOne dataset of the audit's per-episode and per-frame scores.

    write_json(result, project, path) -> path              the audit result with its provenance block
    html(result, project) -> str                           a page that needs nothing else (inline SVG, light or dark)
    write_html(result, project, path) -> path
    export_fiftyone(result, dataset_path, out_dir, project=None, media='clips', progress=None, cancel=None) -> dict

FiftyOne. FiftyOne's importer for its own on-disk format, fiftyone.utils.data.importers.FiftyOneDatasetImporter, reads
a folder with metadata.json and samples.json; when metadata.json has no "version" key it hands the folder to
LegacyFiftyOneDatasetImporter, which this export writes (read from the FiftyOne source on 2026-11-07,
https://github.com/voxel51/fiftyone, branch develop, fiftyone/utils/data/importers.py: LegacyFiftyOneDatasetImporter
setup() and __next__(), _is_legacy_format_data(); fiftyone/core/sample.py Sample.from_dict; the format is described in
the user guide, https://docs.voxel51.com/user_guide/export_datasets.html, "FiftyOneDataset"):

    out_dir/metadata.json       {"name", "media_type": "video", "info": {...}}      (no "version": the legacy layout)
    out_dir/samples.json        {"samples": [sample, ...]}
    out_dir/frames/<n>.json     {"frames": {"<frame number>": {field: value}}}   (frame numbers from 1)
    out_dir/data/<clip>.mp4     the media (media='clips'), or a path to the recorded video (media='reference')

A sample is one camera of one episode: {"filepath" (relative to out_dir, which the importer prepends), "tags",
"_media_type": "video", "frames": "frames/<n>.json", and the audit's fields}. Sample tags: the episode's verdict
('audit:pass', 'audit:fix', 'audit:drop', 'audit:recalibrate') and every fault kind found on that camera ('drop',
'duplicate', 'desync', 'drift', 'jitter', 'usb_reset', 'exposure_change', 'blur', 'gripper_hidden', 'mount_slip').
Sample fields: episode_index, episode_id, camera, sensor, dataset_format, verdict, desync_offset_ms,
desync_sigma_ms, drift_ppm, jitter_ms, fps_measured, dropped_frames, gap_frames, duplicate_frames, blur_frames,
gripper_hidden_frames, exposure_steps, mount_shift_deg, action_lag_rows. Frame fields (per decoded frame, frame k as
frame number k + 1): t_s, change_dn, sharpness_ratio, gain_log_ratio, gripper_ncc, gap_before_s, duplicate, blur,
exposure_changed, gripper_hidden. Load it with
fiftyone.Dataset.from_dir(out_dir, dataset_type=fiftyone.types.FiftyOneDataset).

The fiftyone package is optional and not a dependency: the files are written without it, and nothing here imports it.
When it is installed, the validation check imports the folder with FiftyOne's own importer.

Clips (media='clips', default): each camera's frames of each episode re-encoded as H.264 MP4 (PyAV, libx264, yuv420p,
crf 18) at the frames' own timestamps, so frame k of the clip is frame k of the audit whatever the dataset's packing
(a LeRobot v3.0 video file holds many episodes). media='reference': filepath points at the recorded file itself (the
path the user chose; only for formats with one video per episode, LeRobot v2.1).

Known limits: the export holds the scores, not the images' content (no detections or masks); clips are re-encoded
(a second lossy step, crf 18); a dataset whose cameras are not videos (RLDS images, bag topics) is exported as clips.
"""
from __future__ import annotations

import html as _html
import json
import math
import os

import numpy as np

from ..datasets.episode import format_title
from ..jobs import StudyError

FIFTYONE_SOURCE = 'https://github.com/voxel51/fiftyone (develop, fiftyone/utils/data/importers.py, read 2026-11-07)'
FRAME_FIELDS = ('t', 'change_dn', 'sharpness_ratio', 'gain_log_ratio', 'gripper_ncc', 'gap_before_s', 'duplicate',
                'blur', 'exposure_changed', 'gripper_hidden')
CLIP_CRF = 18


# ====================================================================== JSON
def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, np.ndarray):
        return _clean(o.tolist())
    if isinstance(o, (np.bool_, bool)):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, (float, np.floating)):
        f = float(o)
        return f if math.isfinite(f) else None
    return o


def stamped(result: dict, project) -> dict:
    from .. import provenance as PV
    doc = {k: v for k, v in _clean(result).items() if k not in ('cached', 'revision')}
    return PV.stamp(doc, project)


def write_json(result: dict, project, path: str) -> str:
    doc = stamped(result, project)
    with open(path, 'w', encoding='utf-8') as fh:
        json.dump(doc, fh, indent=1, ensure_ascii=False)
        fh.write('\n')
    return path


# ====================================================================== HTML
def _f(x, nd=1, unit=''):
    if x is None or (isinstance(x, float) and not math.isfinite(x)):
        return '—'
    if isinstance(x, (list, tuple)):
        return '–'.join(_f(v, nd) for v in x) + (f' {unit}' if unit else '')
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, int):
        return f'{x:,}' + (f' {unit}' if unit else '')
    return f'{x:,.{nd}f}' + (f' {unit}' if unit else '')


def slip_svg(mounts: dict, width: int = 720, height: int = 240) -> str:
    """The mount estimates over the campaign (rotation, deg, each camera's largest axis), with σ bars."""
    from ...validation.report import CSS_VARS, SERIES
    series = []
    for k, (key, m) in enumerate(mounts.items()):
        S = m.get('episodes') or []
        if len(S) < 2:
            continue
        R = np.array([[np.nan if v is None else v for v in s['rotation_deg']] for s in S], float)
        if not np.isfinite(R).any():
            continue
        ax = int(np.nanargmax(np.nanmax(np.abs(R), axis=0)))
        sg = np.array([np.nan if (s.get('sigma_deg') or [None] * 3)[ax] is None else s['sigma_deg'][ax] for s in S])
        series.append((key, m, [s['t'] for s in S], R[:, ax], sg, ax, k))
    if not series:
        return '<p class="meta">No mount estimates to plot.</p>'
    L, Rm, T, B = 56, 16, 16, 34
    xs = np.concatenate([np.asarray(s[2], float) for s in series])
    ys = np.concatenate([np.asarray(s[3], float) + np.nan_to_num(s[4]) for s in series] +
                        [np.asarray(s[3], float) - np.nan_to_num(s[4]) for s in series])
    x0, x1 = float(np.nanmin(xs)), float(np.nanmax(xs)) or 1.0
    y0, y1 = float(np.nanmin(ys)), float(np.nanmax(ys))
    if y1 - y0 < 0.2:
        y0, y1 = y0 - 0.1, y1 + 0.1
    pad = 0.08 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad

    def X(v):
        return L + (width - L - Rm) * (v - x0) / max(x1 - x0, 1e-9)

    def Y(v):
        return T + (height - T - B) * (1 - (v - y0) / max(y1 - y0, 1e-9))
    parts = [f'<line x1="{L}" y1="{Y(0):.1f}" x2="{width - Rm}" y2="{Y(0):.1f}" class="g"/>']
    for v in np.linspace(y0, y1, 5):
        parts.append(f'<text x="{L - 6}" y="{Y(v) + 4:.1f}" class="t" text-anchor="end">{v:.2f}</text>')
    for v in sorted(set(np.round(xs, 3))):
        parts.append(f'<text x="{X(v):.1f}" y="{height - 14}" class="t" text-anchor="middle">{v:g}</text>')
    legend = []
    for key, m, t, y, sg, ax, k in series:
        col = SERIES[k % len(SERIES)][0]
        pts = ' '.join(f'{X(a):.1f},{Y(b):.1f}' for a, b in zip(t, y) if np.isfinite(b))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="var(--s{k % len(SERIES) + 1})" '
                     'stroke-width="1.5"/>')
        for a, b, s_ in zip(t, y, sg):
            if not np.isfinite(b):
                continue
            if np.isfinite(s_) and s_ > 0:
                parts.append(f'<line x1="{X(a):.1f}" y1="{Y(b - s_):.1f}" x2="{X(a):.1f}" y2="{Y(b + s_):.1f}" '
                             f'stroke="var(--s{k % len(SERIES) + 1})"/>')
            parts.append(f'<circle cx="{X(a):.1f}" cy="{Y(b):.1f}" r="3" fill="var(--s{k % len(SERIES) + 1})"><title>'
                         f'{_html.escape(key)} at {a:g}: {b:+.3f}° ± {s_:.3f}</title></circle>')
        legend.append(f'<span style="color:{col}">■</span> {_html.escape(key)} ({("roll", "pitch", "yaw")[ax]}, '
                      f'{_html.escape(m.get("kind") or "")})')
    return (f'<figure><svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="Mount over the campaign">'
            f'<style>{CSS_VARS}.g{{stroke:var(--axis)}}.t{{font:10px ui-monospace,monospace;fill:var(--muted)}}'
            f'</style>{"".join(parts)}</svg><figcaption class="meta">{" · ".join(legend)} — rotation of each '
            'camera\'s mount (°, its largest axis, body frame) against the first estimate, ±1 σ.</figcaption></figure>')


def html(result: dict, project=None, title: str = 'Dataset audit') -> str:
    from ...validation.report import PAGE_CSS
    from .. import provenance as PV
    s = result.get('summary') or {}
    ds = result.get('dataset') or {}
    rows = []
    for d in result.get('requirements') or []:
        rows.append(f'<tr><td>{_html.escape(d["title"])}</td><td>{_f(d.get("limit"), 2)} {d["unit"]}'
                    f'{" (default)" if d.get("default") else ""}</td><td>{d["checked"] - d["failed"]} of {d["checked"]}'
                    f'</td><td><span class="badge {"pass" if d["pass"] else "fail"}">'
                    f'{"pass" if d["pass"] else "fail"}</span></td><td>{_f(d.get("worst"), 2)}</td><td>'
                    f'{_html.escape(", ".join(str(x) for x in d["episodes_failed"][:20]))}</td></tr>')
    eps = []
    for e in result.get('episodes') or []:
        cams = []
        for key, c in e['cameras'].items():
            sy = c['sync']
            cams.append(f'{_html.escape(key.replace("observation.images.", ""))}: offset '
                        f'{_f(sy.get("offset_ms"))} ± {_f(sy.get("sigma_ms"))} ms')
        eps.append(f'<tr><td>{e["index"]}</td><td>{_f(e["duration_s"], 2)} s</td><td><span class="badge '
                   f'{"pass" if e["verdict"] == "pass" else "fail"}">{_html.escape(e["verdict"])}</span></td><td>'
                   f'{_html.escape(", ".join(e["flags"]) or "—")}</td><td>{"<br>".join(cams)}</td></tr>')
    prov = PV.block(project, result) if project is not None else {}
    ptxt = _html.escape(json.dumps(PV.scrub(prov), indent=1, ensure_ascii=False))
    tiles = (f'<div class="tiles"><div class="tile"><b>{s.get("pass", 0)} / {s.get("episodes", 0)}</b><span>episodes '
             f'meet every requirement</span></div><div class="tile"><b>{len(s.get("drop_or_rerecord") or [])}</b>'
             f'<span>to drop or re-record</span></div><div class="tile"><b>{len(s.get("fix") or [])}</b><span>fixable '
             f'by re-stamping</span></div><div class="tile"><b>{len(s.get("recalibrate") or [])}</b><span>need new '
             'extrinsics</span></div></div>')
    return ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
            f'content="width=device-width,initial-scale=1"><title>{_html.escape(title)}</title><style>{PAGE_CSS}'
            f'</style></head><body><main><h1>{_html.escape(title)}</h1><p class="meta">'
            f'{_html.escape(ds.get("name", ""))}'
            f' · {_html.escape(format_title(ds.get("format", ""), ds.get("version", "")))} · '
            f'{ds.get("episodes_audited", 0)} of {ds.get("episodes_total", 0)} episodes audited · '
            f'{_f(result.get("seconds"), 1)} s</p><p><b>{_html.escape(s.get("word", ""))}</b>: '
            f'{_html.escape(s.get("advice", ""))}</p>{tiles}<h2>Requirements</h2><p>Every limit is a DexVantage '
            'default the user can change, not a standard.</p><div class="wrap"><table><tr><th>Requirement</th><th>Limit'
            '</th><th>Checks passed</th><th>Verdict</th><th>Worst</th><th>Episodes failing</th></tr>'
            f'{"".join(rows)}</table></div><h2>Mount over the campaign</h2>{slip_svg(result.get("mounts") or {})}'
            f'<h2>Episodes</h2><div class="wrap"><table><tr><th>Episode</th><th>Length</th><th>Verdict</th><th>Faults'
            f'</th><th>Camera offsets</th></tr>{"".join(eps)}</table></div><h2>Provenance</h2><details><summary>'
            f'The inputs, code and environment of this report</summary><pre><code>{ptxt}</code></pre></details>'
            '</main></body></html>')


def write_html(result: dict, project, path: str) -> str:
    with open(path, 'w', encoding='utf-8') as fh:
        fh.write(html(result, project))
    return path


# ====================================================================== FiftyOne
def _tags(e: dict, key: str, cam: dict) -> list:
    tags = [f'audit:{e["verdict"]}']
    from . import episode as AE
    for r in e['requirements']:
        if not r['pass'] and r['camera'] == key:
            tags.append(AE.FLAG[r['id']])
    return sorted(set(tags))


def _sample_fields(e: dict, key: str, cam: dict, ds_format: str) -> dict:
    tm, sy = cam['timing'], cam['sync']
    shift = next((r['value'] for r in e['requirements'] if r['id'] == 'max_mount_shift_deg' and r['camera'] == key),
                 None)
    return _clean({
        'episode_index': e['index'], 'episode_id': e['id'], 'camera': key, 'sensor': cam.get('sensor'),
        'dataset_format': ds_format, 'verdict': e['verdict'], 'desync_offset_ms': sy.get('offset_ms'),
        'desync_sigma_ms': sy.get('sigma_ms'), 'drift_ppm': sy.get('drift_ppm'), 'jitter_ms': tm.get('jitter_ms'),
        'fps_measured': tm.get('fps_measured'), 'dropped_frames': tm.get('dropped_frames', 0),
        'gap_frames': tm.get('gap_frames', 0), 'duplicate_frames': len(cam['duplicates']['frames']),
        'blur_frames': len(cam['blur']['frames']), 'gripper_hidden_frames': len(cam['gripper'].get('hidden') or []),
        'exposure_steps': len(cam['exposure']['steps']), 'mount_shift_deg': shift,
        'action_lag_rows': (e.get('action') or {}).get('lag_rows')})


def _write_clip(stream, path: str, cancel=None) -> int:
    try:
        import av
    except ImportError:
        raise StudyError('Writing FiftyOne clips needs the Python package av (install the lerobot extra: pip install '
                         '"dexvantage[lerobot]"), or export with media="reference".') from None
    from fractions import Fraction
    ts = np.asarray(stream.timestamps, float)
    n = 0
    tb = Fraction(1, 90000)
    with av.open(path, 'w') as out:
        st = None
        last = -1
        for i, img in stream.frames():
            if cancel is not None and cancel():
                break
            a = np.ascontiguousarray(np.asarray(img, np.uint8)[..., :3])
            h, w = a.shape[:2]
            if st is None:
                st = out.add_stream('libx264', rate=int(round(stream.fps or 30)))
                st.width, st.height = w - w % 2, h - h % 2
                st.pix_fmt = 'yuv420p'
                st.time_base = tb
                st.options = {'crf': str(CLIP_CRF), 'bf': '0'}
            fr = av.VideoFrame.from_ndarray(a[:st.height, :st.width], format='rgb24')
            pts = int(round((ts[i] - ts[0]) * 90000)) if i < len(ts) else last + 1
            pts = max(pts, last + 1)
            fr.pts, fr.time_base = pts, tb
            last = pts
            for p in st.encode(fr):
                out.mux(p)
            n += 1
        if st is not None:
            for p in st.encode():
                out.mux(p)
    return n


def export_fiftyone(result: dict, dataset_path: str, out_dir: str, project=None, media: str = 'clips',
                    progress=None, cancel=None) -> dict:
    """Write the audit as a FiftyOne dataset folder (module docstring). Needs the audit run with per_frame on.
    Returns {'folder', 'samples', 'frames', 'media', 'files'}."""
    from ..datasets import open_dataset
    if media not in ('clips', 'reference'):
        raise StudyError('The FiftyOne media must be "clips" or "reference".')
    eps = result.get('episodes') or []
    if not eps:
        raise StudyError('There is no audited episode to export.')
    if not any('per_frame' in c for e in eps for c in e['cameras'].values()):
        raise StudyError('The audit was run without per-frame scores; run it with per_frame on to export to FiftyOne.')
    if os.path.exists(out_dir) and os.listdir(out_dir):
        raise StudyError(f'{os.path.basename(out_dir)} is not empty: choose a new or empty folder.')
    os.makedirs(os.path.join(out_dir, 'frames'), exist_ok=True)
    if media == 'clips':
        os.makedirs(os.path.join(out_dir, 'data'), exist_ok=True)
    report = progress or (lambda f, t='': None)
    samples, nframes = [], 0
    ds_format = (result.get('dataset') or {}).get('format', '')
    with open_dataset(dataset_path) as ds:
        if media == 'reference' and getattr(ds, 'hub', None) is not None:
            raise StudyError('media="reference" needs the videos on this computer; this dataset is read from the '
                             'Hugging Face Hub: export clips instead.')
        if media == 'reference' and not (ds.format == 'lerobot' and ds.version == 'v2.1'):
            raise StudyError('media="reference" needs one video per episode (LeRobot v2.1); this dataset is '
                             f'{ds.format} {ds.version}: export clips instead.')
        total = sum(len(e['cameras']) for e in eps)
        k = 0
        for e in eps:
            if not e['cameras']:                       # an unreadable episode (audit_episode said why): no media
                continue
            ep = ds.load(e['index'])
            for key, cam in e['cameras'].items():
                if cancel is not None and cancel():
                    raise StudyError('The export was cancelled.')
                report(k / max(1, total), f'episode {e["index"]}: {key}')
                k += 1
                name = f'episode_{e["index"]:06d}_{_safe(key)}'
                if media == 'clips':
                    rel = f'data/{name}.mp4'
                    _write_clip(ep.cameras[key], os.path.join(out_dir, rel), cancel)
                    filepath = rel
                else:
                    src = (ep.cameras[key].meta or {}).get('file') or (ep.cameras[key].meta or {}).get('path')
                    if not src:
                        raise StudyError(f'The video file of {key} is not known to the reader.')
                    src = src if os.path.isabs(src) else os.path.join(getattr(ds, 'root', None) or ds.path, src)
                    try:                                   # the importer prepends the export folder to a relative path
                        filepath = os.path.relpath(src, out_dir)
                    except ValueError:                     # another drive (Windows): the absolute path
                        filepath = os.path.abspath(src)
                pf = cam.get('per_frame') or {}
                M = len(pf.get('t') or [])
                frames = {}
                for i in range(M):
                    fd = {'frame_number': i + 1}
                    for f in FRAME_FIELDS:
                        v = (pf.get(f) or [None] * M)[i] if i < len(pf.get(f) or []) else None
                        fd['t_s' if f == 't' else f] = v
                    frames[str(i + 1)] = _clean(fd)
                nframes += M
                fname = f'frames/{name}.json'
                with open(os.path.join(out_dir, fname), 'w', encoding='utf-8') as fh:
                    json.dump({'frames': frames}, fh)
                sd = {'filepath': filepath, 'tags': _tags(e, key, cam), '_media_type': 'video', 'frames': fname}
                sd.update(_sample_fields(e, key, cam, ds_format))
                samples.append(sd)
    from .. import provenance as PV
    info = {'source': 'DexVantage dataset audit (core/audit)', 'format_read_from': FIFTYONE_SOURCE,
            'dataset': (result.get('dataset') or {}).get('name'), 'summary': _clean(result.get('summary') or {}),
            'requirements': _clean(result.get('requirements_used') or {})}
    if project is not None:
        info['provenance'] = PV.scrub(PV.block(project, result))
    meta = {'name': f'dexvantage-audit-{_safe((result.get("dataset") or {}).get("name") or "dataset")}',
            'media_type': 'video', 'info': info}
    with open(os.path.join(out_dir, 'metadata.json'), 'w', encoding='utf-8') as fh:
        json.dump(meta, fh, indent=1, ensure_ascii=False)
    with open(os.path.join(out_dir, 'samples.json'), 'w', encoding='utf-8') as fh:
        json.dump({'samples': samples}, fh, indent=1, ensure_ascii=False)
    report(1.0, 'done')
    return {'folder': out_dir, 'samples': len(samples), 'frames': nframes, 'media': media,
            'files': ['metadata.json', 'samples.json'] + [s['frames'] for s in samples] +
            ([s['filepath'] for s in samples] if media == 'clips' else [])}


def _safe(s: str) -> str:
    import re
    return re.sub(r'[^A-Za-z0-9_.-]+', '_', str(s)).strip('_') or 'x'
