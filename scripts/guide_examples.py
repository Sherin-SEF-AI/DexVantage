"""Run every worked example of the user guide (docs/user-guide.md) and the sample tours (docs/guide/samples.md) on
the bundled samples, and check that the figures the documents print are the ones the code gives now.

Each example is a function registered with @example(id, doc): it runs what the guide's worked example does (through
the Python API, `import dexvantage as dv`, or the core functions the window calls) and returns its figures, each
formatted as the document writes it (`Fig`: the value, the text, and an optional tolerance for figures that depend on
the GPU's rasteriser against numpy's within the occlusion model's own agreement). In the document, a worked example
starts with an invisible marker `<!-- example: <id> -->` and runs to the next marker or heading; every figure's text
must appear in that span (whitespace and line breaks do not matter), or, for a figure with a tolerance, some number in
the span must lie within it. So a figure in the guide is either produced by running the example now or the check
fails.

    python scripts/guide_examples.py                      # run every example, print its figures
    python scripts/guide_examples.py --only tour,grasp    # some of them
    python scripts/guide_examples.py --check              # run them and compare with the documents (exit 1 on a miss)
    python scripts/guide_examples.py --json out/figs.json # also write the figures as JSON

Run memory-capped like the other heavy scripts; the examples write only into a temporary folder (TMPDIR) that is
removed at the end. tests/test_guide_examples.py runs --check.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import re
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
os.environ.setdefault('QT_QPA_PLATFORM', 'offscreen')
GUIDE = 'docs/user-guide.md'
SAMPLES = 'docs/guide/samples.md'


@dataclass
class Fig:
    value: object
    text: str
    tol: float | None = None          # absolute: a number in the span within tol of value also matches
    unit: str | None = None           # with tol: only numbers followed by this unit count (not "4 episodes")


@dataclass
class Example:
    id: str
    doc: str
    fn: object
    title: str


EXAMPLES: dict[str, Example] = {}


def example(eid: str, doc: str = GUIDE, title: str = ''):
    def deco(fn):
        EXAMPLES[eid] = Example(eid, doc, fn, title or (fn.__doc__ or '').strip().split('\n')[0])
        return fn
    return deco


def num(v, d: int = 1) -> str:
    """A number as the guide writes it: d decimals, thin groups of three separated by a space ("57 600")."""
    s = f'{float(v):,.{d}f}'
    return s.replace(',', ' ')


def pct(frac, d: int = 1) -> str:
    return num(100.0 * float(frac), d)


def F(v, d: int = 1, tol: float | None = None, scale: float = 1.0, unit: str | None = None) -> Fig:
    return Fig(v, num(float(v) * scale, d), tol, unit)


def T(text: str) -> Fig:
    return Fig(text, text)


_TMP: list = []


def tmpdir(name: str) -> str:
    base = _TMP[0] if _TMP else None
    if base is None:
        base = tempfile.mkdtemp(prefix='dexvantage-guide-')
        _TMP.append(base)
    d = os.path.join(base, name)
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)
    return d


def _cam(doc, name):
    return next(c for c in doc['cameras'] if c['name'] == name)


# ====================================================================== the ten-minute tour (ALOHA 2)
@example('tour')
def ex_tour():
    """The ten-minute tour on ALOHA 2: grasps, a moved camera, depth, VLA tokens, bandwidth, exports, a bracket."""
    import dexvantage as dv
    from dexvantage.core.studies import bandwidth as BW
    rig = dv.sample('aloha')
    out = {'robots': T(f'{len(rig.robots)} robots'), 'sensors': T(f'{len(rig.sensors)} cameras')}
    g = rig.run('grasp')
    s = g['summary']
    out.update({
        'grasps': T(f'{s["grasps"]} grasps'), 'reachable': T(f'{s["feasible"]} reachable'),
        'out_of_reach': T(f'{s["infeasible"].get("out of reach", 0)} out of reach'),
        'cvs': T(f'{num(s["cvs_pct"], 0)} %'), 'wilson_low': F(100 * s['wilson_low'], 1, tol=0.6),
        'head': T(f'{100 * s["cvs"]:.1f} [{100 * s["wilson_low"]:.1f}–{100 * s["wilson_high"]:.0f}]'),
    })
    lw = _cam(g, 'cam_left_wrist')
    hi = _cam(g, 'cam_high (overhead)')
    rw = _cam(g, 'cam_right_wrist')
    out.update({
        'lw_left': F(100 * lw['fingertips']['left/left_finger_link'], 0, tol=1.0),
        'lw_right': F(100 * lw['fingertips']['left/right_finger_link'], 0, tol=1.0),
        'hi_own': F(100 * hi['points']['behind_own_finger'], 0, tol=1.0),
        'hi_scene': F(100 * hi['points']['hidden_by_scene'], 0, tol=1.0),
        'hi_seen': F(100 * hi['points']['visible'], 0, tol=1.0),
        'rw_outside': F(100 * rw['points']['outside_image'], 0, tol=1.0),
        'weakest': F(100 * min(x['score'] for x in g['gallery']), 0, tol=1.0),
    })
    moved = rig.with_factors({'sensor:cam_left_wrist.pitch': -75.0})
    m = moved.run('grasp', sensitivity=False)
    ms = m['summary']
    mlw = _cam(m, 'cam_left_wrist')
    out.update({'moved_cvs': F(ms['cvs_pct'], 1, tol=1.2), 'moved_meets': T(f'{ms["meets"]} of {ms["evaluated"]}'),
                'moved_head': T(f'{100 * ms["cvs"]:.1f} [{100 * ms["ci_low"]:.0f}–{100 * ms["ci_high"]:.0f}]'),
                'moved_word': T(ms['word']),
                'moved_outside': F(100 * mlw['points']['outside_image'], 0, tol=1.0)})
    d = rig.run('depth_validity')
    out['depth'] = T(d.headline['text'])
    v = rig.run('vla_tokens')
    out['vla_word'] = T(v['summary']['word'])
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import TaskObject
    p = rig.project
    p.workspace.objects.append(TaskObject(name='Box 1', size=(0.06, 0.06, 0.06), pose=Pose([0.45, 0.0, 0.03])))
    v2 = dv.Rig(p).run('vla_tokens')
    out['vla_box'] = T(v2.headline['text'])
    bw = rig.run('bandwidth')
    out['payload'] = F(bw['headline']['total_payload_mbps'], 1)
    BW.apply_suggestion(p, bw['suggestion'])
    bw2 = dv.Rig(p).run('bandwidth')
    out['bw_after'] = T(bw2.headline['text'].split(':', 1)[1].split('(')[0].strip())
    out['bw_word'] = T(bw2.headline['word'])
    out['bw_load'] = T(f'{bw2["headline"]["busiest_load_pct"]:.0f} %')
    br = rig.run('brackets')
    for x in br['suggestion']:
        p.variant.sensor(x['sensor_uid']).mount.bracket = x['kind']
    out['brackets'] = T(dv.Rig(p).run('brackets').headline['text'])
    m2 = dv.Rig(p).export(tmpdir('tour-mjcf-lerobot'), ['mjcf', 'lerobot'])
    n_ok = sum(1 for r in m2['formats'].values() if r['status'] == 'ok')
    out['export_hint'] = T(f'{n_ok} of {len(m2["formats"])} formats written; {len(m2["files"])} files.')
    folder = tmpdir('tour-export')
    man = dv.Rig(p).export(folder, ['mjcf', 'lerobot', 'brackets'])
    files = {k: v for k, v in man.get('formats', {}).items()}
    stls = [f for f in os.listdir(folder) if f.endswith('.stl')] + \
        [f for sub in ('brackets',) if os.path.isdir(os.path.join(folder, sub))
         for f in os.listdir(os.path.join(folder, sub)) if f.endswith('.stl')]
    out['stl'] = T(f'{len(stls)} STL')
    out['_formats'] = Fig(sorted(files), '')
    return out


# ====================================================================== getting started and the window
@example('window')
def ex_window():
    """The project summary of a new project with a 1.5 × 1.5 × 1.5 m task volume."""
    import dexvantage as dv
    rig = dv.new('Tabletop cell')
    vol = rig.project.workspace.volumes[0]
    out = {'size': T(' × '.join(f'{1000 * x:.0f}' for x in vol.size) + ' mm')}
    vol.size = (1.5, 1.5, 1.5)
    s = rig.run('summary')
    v = s['volumes'][0]
    out['dense'] = T(f'{v["voxels_dense_1mm"] / 1e9:.1f} × 10⁹')
    out['dense5'] = T(f'{v["voxels_dense"] / 1e6:.0f} × 10⁶')
    out['limit'] = T(f'{s["dense_limit_voxels"] / 1e7:.0f} × 10⁷')
    return out


# ====================================================================== robots
@example('robot')
def ex_robot():
    """The UR5e's gripper frame, manipulability and condition number at home."""
    import numpy as np

    import dexvantage as dv
    from dexvantage.core.frames import quat_from_matrix, rpy_from_quat
    from dexvantage.core.robot import manipulability as MP
    from dexvantage.core.robot.model import get_model
    rig = dv.sample('ur5e')
    r = rig.robots[0]
    m = get_model(rig.project, r.uid)
    T0 = m.link_pose(m.home, r.gripper_frame)
    mask = MP.chain_mask(m, r.gripper_frame)
    part = 'full' if int(np.count_nonzero(mask)) >= 6 else 'position'
    me = MP.measures(m, m.home, r.gripper_frame, part=part)
    rpy = np.degrees(rpy_from_quat(quat_from_matrix(T0[:3, :3])))
    q = m.home.copy()
    q[m.joint_names.index('elbow_joint')] = 0.0
    T1 = m.link_pose(q, r.gripper_frame)
    me1 = MP.measures(m, q, r.gripper_frame, part=part)
    return {
        'pos': T('(' + ', '.join(f'{1000 * x:+.1f}'.replace('-', '−') for x in T0[:3, 3]) + ') mm'),
        'rpy': T('(' + ', '.join(f'{(abs(x) if abs(abs(x) - 180) < 1e-6 else x) + 0.0:.0f}'.replace('-0', '0')
                                 for x in rpy) + ')°'),
        'w': T(f'{me["yoshikawa"]:.4f}'), 'cond': T(f'{me["condition"]:.1f}'),
        'nq': T(f'{m.nq} joints'),
        'straight': T('(' + ', '.join(f'{1000 * x:+.1f}'.replace('-', '−') for x in T1[:3, 3]) + ') mm'),
        'straight_singular': Fig(not math.isfinite(me1['condition']) or me1['condition'] > 1e12, ''),
    }


# ====================================================================== sensors and coverage
@example('coverage')
def ex_coverage():
    """Coverage, depth validity and a factor sweep of the overhead D435 on the UR5e sample."""
    import dexvantage as dv
    rig = dv.sample('ur5e')
    cov = rig.run('coverage')
    wr, ov = _cam(cov, 'Wrist D405'), _cam(cov, 'Overhead D435')
    out = {'headline': T(cov.headline['text']),
           'wr_seen': F(100 * wr['seen'], 1, tol=0.3), 'ov_seen': F(100 * ov['seen'], 1, tol=0.3),
           'wr_outside': F(100 * wr['outside_image'], 1, tol=0.3),
           'wr_ppmm': F(wr['ppmm_median'], 2, tol=0.02), 'ov_ppmm': F(ov['ppmm_median'], 2, tol=0.02)}
    dep = rig.run('depth_validity')
    out['depth'] = T(dep.headline['text'])
    up = rig.with_factors({'sensor:Overhead D435.dz': 0.10}).metrics(['coverage.covered', 'depth.valid']).figures()
    out['up_cov'] = F(up['coverage.covered'], 1, tol=0.3)
    out['up_depth'] = F(up['depth.valid'], 1, tol=0.3)
    cost = rig.metrics(['rig.cost']).figures().get('rig.cost')
    if cost is not None:
        out['cost'] = T(f'{num(cost, 0)} USD')
    return out


# ====================================================================== the studies
@example('grasp')
def ex_grasp():
    """Grasp self-occlusion on the UR5e sample's 200 tabletop grasps."""
    import dexvantage as dv
    rig = dv.sample('ur5e')
    g = rig.run('grasp')
    s = g['summary']
    out = {'cvs': F(s['cvs_pct'], 1, tol=1.0), 'reachable': T(f'{s["feasible"]} of {s["grasps"]}'),
           'meets': T(f'{s["meets"]} of {s["evaluated"]}'), 'word': T(s['word'])}
    for k, v in (s.get('infeasible') or {}).items():
        out[f'inf_{k}'] = T(f'{v} {k}')
    if g.get('best_move'):
        out['best'] = T(g['best_move']['text'])
    return out


@example('vla')
def ex_vla():
    """VLA token resolution for the UR5e sample's π0."""
    import dexvantage as dv
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import TaskObject
    rig = dv.sample('ur5e')
    none = rig.run('vla_tokens')
    rig.project.workspace.objects.append(TaskObject(name='Soda can', shape='cylinder', size=(0.033, 0.115, 0),
                                                    pose=Pose([-0.134, 0.492, 0.0575], [0, 0, 0, 1])))
    v = rig.run('vla_tokens')
    return {'none': T(none.headline['word']), 'headline': T(v.headline['text'])}


@example('dexhand')
def ex_dexhand():
    """The multi-finger occlusion matrix and the handoff on the dexhand sample."""
    import dexvantage as dv
    d = dv.sample('dexhand').run('dex_occlusion')
    return {'headline': T(d.headline['text'])}


@example('teleop')
def ex_teleop():
    """Teleop overlap on the SO-100 sample through its leader arm."""
    import dexvantage as dv
    t = dv.sample('so100').run('teleop')
    return {'headline': T(t.headline['text'])}


@example('lighting')
def ex_lighting():
    """Glare on the UR5e sample with a soda can and an Elgato Key Light 1.2 m above the task volume."""
    import dexvantage as dv
    from dexvantage.core import lights as LT
    from dexvantage.core.frames import Pose
    from dexvantage.core.project import TaskObject
    rig = dv.sample('ur5e')
    p = rig.project
    p.workspace.objects.append(TaskObject(name='Soda can', shape='cylinder', size=(0.033, 0.115, 0),
                                          pose=Pose([-0.134, 0.492, 0.0575], [0, 0, 0, 1]), materials=['metallic']))
    c = p.workspace.volumes[0].center
    p.workspace.lights.append(LT.from_library('elgato-key-light', {'position': [c[0], c[1], c[2] + 1.2],
                                                                    'quaternion': [1.0, 0.0, 0.0, 0.0]}))
    r = dv.Rig(p).run('lighting')
    return {'headline': T(r.headline['text'])}


@example('timing')
def ex_timing():
    """Motion blur, time sync and action-chunk latency along the UR5e sample's scripted pick."""
    import dexvantage as dv
    rig = dv.sample('ur5e')
    return {'blur': T(rig.run('motion_blur').headline['text']), 'sync': T(rig.run('time_sync').headline['text']),
            'chunk': T(rig.run('action_chunk_sync').headline['text'])}


@example('hardware')
def ex_hardware():
    """Bandwidth with the suggested host, cable flex and brackets on the UR5e sample."""
    import dexvantage as dv
    from dexvantage.core.studies import bandwidth as BW
    rig = dv.sample('ur5e')
    bw = rig.run('bandwidth')
    out = {'payload': F(bw['headline']['total_payload_mbps'], 1), 'no_host': T(bw.headline['word'])}
    p = rig.project
    BW.apply_suggestion(p, bw['suggestion'])
    bw2 = dv.Rig(p).run('bandwidth')
    out['host'] = T(bw2.headline['text'])
    out['cables'] = T(rig.run('cables').headline['text'])
    br = rig.run('brackets')
    out['brackets_before'] = T(br['rig']['text'])
    for x in br['suggestion']:
        p.variant.sensor(x['sensor_uid']).mount.bracket = x['kind']
    br2 = dv.Rig(p).run('brackets')
    out['brackets_after'] = T(br2.headline['text'])
    return out


@example('canonical')
def ex_canonical():
    """The canonical view warp from the UR5e's wrist D405 to its overhead D435."""
    import dexvantage as dv
    return {'headline': T(dv.sample('ur5e').run('canonical_view').headline['text'])}


@example('groundtruth')
def ex_groundtruth():
    """MuJoCo ground truth on the UR5e sample."""
    import dexvantage as dv
    g = dv.sample('ur5e').run('groundtruth')
    return {'headline': T(g.headline['text'])}


# ====================================================================== calibration
@example('handeye')
def ex_handeye():
    """A hand-eye plan for the UR5e's wrist D405, and a solve from 12 rendered board images at the planned poses."""
    import cv2
    import numpy as np

    import dexvantage as dv
    from dexvantage.core import jobs
    from dexvantage.core.calib import capture as CAP
    from dexvantage.core.calib import planner as PL
    from dexvantage.core.calib import records as RC
    from dexvantage.core.calib import synthetic as S
    rig = dv.sample('ur5e')
    plan = rig.run('handeye_plan')
    out = {'plan': T(plan.headline['text'])}
    p = rig.project
    p.calibration['boards'].append(RC.default_board_record(p, 'eye_in_hand', p.robots[0].uid))
    cr = PL.rig_from_project(p)
    pl = PL.plan(cr, PL.PlanSettings(n_poses=12))
    folder = tmpdir('handeye')
    img = os.path.join(folder, 'img')
    os.makedirs(img)
    rows = ['t,' + ','.join(cr.model.joint_names)]
    for k, ps in enumerate(pl.doc['poses']):
        q = np.asarray(ps['q'])
        im = S.render_view(cr.board, cr.camera, cr.T_cam_board(q), supersample=2, noise=1.0, seed=k)
        t = 100.0 + 5 * k
        cv2.imwrite(os.path.join(img, f'pose_{k:02d}_{t:.3f}.png'), im)
        rows.append(f'{t + 0.004},' + ','.join(f'{v:.10f}' for v in q))
    joints = os.path.join(folder, 'joints.csv')
    with open(joints, 'w') as fh:
        fh.write('\n'.join(rows) + '\n')
    sess = CAP.capture_folder(p, p.variant.sensors[0], img, joints)
    sess.pop('board', None)
    p.calibration['sessions'].append(sess)
    doc = jobs.run_study('handeye_solve', p)
    ap = doc['apply']
    out.update({'detected': T(f'{sess["source"]["detected"]} images'), 'status': T(doc['status_word']
                if 'status_word' in doc else doc['status']),
                'change_mm': F(ap['change_mm'], 2), 'change_deg': F(ap['change_deg'], 3)})
    return out


@example('data')
def ex_data():
    """Synthetic SO-100 episodes with every fault, audited; plan against recording; the silhouette calibration."""
    import dexvantage as dv
    rig = dv.sample('so100')
    folder = tmpdir('so100-synth')
    rig.synth_episodes(folder, episodes=2, frames=48, max_width=160, seed=7)
    ds = dv.datasets.open(folder)
    out = {'recording': T(repr(ds).strip('<>').split("'")[0].strip()), 'episodes': T(f'{len(ds.episodes())} episodes')}
    a = rig.run('audit', dataset=folder)
    out['audit'] = T(a.headline['text'])
    pv = rig.run('plan_vs_recording', dataset=folder)
    out['pvr'] = T(pv.headline['text'])
    ds.close()
    return out


@example('silhouette')
def ex_silhouette():
    """The silhouette calibration of the UR5e's overhead D435 over the synthetic episodes' defaults."""
    import dexvantage as dv
    rig = dv.sample('ur5e')
    folder = tmpdir('ur5e-synth')
    rig.synth_episodes(folder)
    sc = rig.run('silhouette_calib', dataset=folder)
    h, text = sc['headline'], sc.headline['text']
    # the offset depends on the silhouettes' edge pixels, which differ between renderers (measured 2026-11-08: 4.21 mm
    # on the development machine's NVIDIA EGL, 4.14 mm on its Mesa llvmpipe, 4.28 mm on Mesa llvmpipe on the first
    # GitHub CI run; the estimate's own σ 0.86 to 1.01 mm): a figure in mm in the guide within the estimate's 1 σ of
    # the one computed here matches
    return {'title': T(text.split(':', 1)[0] + ':'),
            'offset_mm': F(h['offset_mm'], 2, tol=h['sigma_mm'], unit='mm'),
            'word': T('(' + text.rsplit('(', 1)[1]) if '(' in text else T(''), '_sigma_mm': F(h['sigma_mm'], 3)}


@example('sim2real')
def ex_sim2real():
    """The sim-to-real gap on a UR5e recording rendered with a known +20 DN brightness gap."""
    import dexvantage as dv
    rig = dv.sample('ur5e')
    folder = tmpdir('s2r')
    rig.gap_dataset(folder, {'camera': 'observation.images.overhead_d435', 'brightness_dn': 20},
                    episodes=1, frames=24, size=[320, 180], seed=4)
    r = rig.sim2real(folder)
    return {'headline': T(r.headline['text'])}


# ====================================================================== optimisation and requirements
@example('requirements')
def ex_requirements():
    """A requirement, the tolerance Monte Carlo, a designed experiment and the optimiser on the UR5e sample."""
    import dexvantage as dv
    rig = dv.sample('ur5e')
    sc = rig.check([{'name': 'Task volume seen', 'metric': 'coverage.covered', 'op': '>=', 'threshold': 90}])
    out = {'scorecard': T(sc.headline['text'])}
    tol = rig.run('tolerance', n=16, seed=1, metrics=['coverage.covered'])
    out['tolerance'] = T(tol.headline['text'])
    return out


@example('experiment')
def ex_experiment():
    """A Latin hypercube of 8 runs over the overhead D435's height and pitch on the UR5e sample."""
    import dexvantage as dv
    from dexvantage.core import experiments as EX
    rig = dv.sample('ur5e')
    design = {'schema': 'dexvantage.experiment/1', 'name': 'Overhead camera height and tilt',
              'factors': [{'path': 'sensor:Overhead D435.dz', 'lo': -0.05, 'hi': 0.05},
                          {'path': 'sensor:Overhead D435.dpitch', 'lo': -5, 'hi': 5}],
              'responses': ['coverage.covered', 'depth.valid'], 'objective': 'coverage.covered',
              'design': {'method': 'lhs', 'n': 8, 'seed': 2}}
    res = EX.run(rig.project, design)
    h = res.get('headline') or {}
    return {'runs': T(f'{res["execution"]["runs_done"]} runs'), 'method': T(res['method_label']),
            'word': T(h.get('word', '')), 'text': T(h.get('text', ''))}


@example('optimise')
def ex_optimise():
    """The optimiser on the UR5e sample, budget 12 designs, seed 1."""
    import dexvantage as dv
    r = dv.sample('ur5e').run('optimise', budget=12, seed=1)
    return {'headline': T(r.headline['text'])}


# ====================================================================== export, assistant, MCP
@example('export')
def ex_export():
    """Every export format on the UR5e sample, with the manifest."""
    import dexvantage as dv
    rig = dv.sample('ur5e')
    folder = tmpdir('export')
    fmts = ['mjcf', 'urdf', 'opencv_yaml', 'masks', 'plucker', 'isaac', 'lerobot']
    man = rig.export(folder, fmts)
    nfiles = sum(len(v.get('files', [])) for v in man['formats'].values()) if isinstance(man.get('formats'), dict) \
        else len(man.get('files', []))
    return {'formats': T(f'{len(fmts)} formats'), 'files': T(f'{nfiles} files')}


@example('mcp')
def ex_mcp():
    """The MCP server in process: initialize, tools/list, open the ALOHA 2 sample and read its rig cost."""
    from dexvantage.mcp import server as SV
    srv = SV.Server(SV.Router(sample='aloha'))
    init = srv.handle({'jsonrpc': '2.0', 'id': 1, 'method': 'initialize',
                       'params': {'protocolVersion': '2025-11-25', 'capabilities': {}}})
    tl = srv.handle({'jsonrpc': '2.0', 'id': 2, 'method': 'tools/list'})
    n = len(tl['result']['tools'])
    return {'protocol': T(init['result']['protocolVersion']), 'tools': T(f'{n} tools')}


# ====================================================================== research workflows
@example('stats')
def ex_stats():
    """Two success rates compared, and the episodes a 70 % against 95 % difference needs."""
    import dexvantage as dv
    r = dv.stats.compare_rates([('ACT', 14, 20), ('Diffusion', 19, 20)])
    n = dv.stats.episodes_needed(0.7, 0.95)['n_per_group']
    return {'word': T(r.headline['word']), 'n': T(f'{n} episodes per condition')}


@example('campaign')
def ex_campaign():
    """The ALOHA 2 handover campaign of guide/campaign.md."""
    import dexvantage as dv
    spec = {'schema': 'dexvantage.campaign_spec/1', 'name': 'ALOHA handover, two lighting presets',
            'operators': ['Ana', 'Ben', 'Chen'], 'lighting': [{'name': 'Bright'}, {'name': 'Dim'}],
            'demos': {'mode': 'power', 'p_base': 0.5, 'p_target': 0.7}, 'episode_s': 30,
            'start_date': '2026-11-16',
            'budget': {'disk_gb': 4000, 'days': 10, 'upload_mbps': 1000, 'upload_hours_per_day': 8}}
    r = dv.campaign.plan(dv.sample('aloha'), spec)
    return {'headline': T(r.headline['text'])}


@example('versioning')
def ex_versioning():
    """A diff of the UR5e sample against a copy with the overhead D435 raised 10 cm, and its BibTeX."""
    import dexvantage as dv
    a = dv.sample('ur5e')
    b = a.with_factors({'sensor:Overhead D435.dz': 0.10})
    d = dv.diff(a, b)
    bib = dv.bibtex(a)
    ch = d['changes'][0]['text']
    return {'changes': T(d.text.split('\n')[1].split(':')[0]), 'change': T(ch),
            'bib': T(f'{bib.count("@")} BibTeX entries')}


@example('cells')
def ex_cells():
    """The UR5e sample compared with the SO-100 and ALOHA 2 samples as three cells."""
    import dexvantage as dv
    r = dv.cells.compare(dv.sample('ur5e'), 'sample:so100', 'sample:aloha')
    return {'headline': T(r.headline['text'])}


@example('paper')
def ex_paper():
    """The paper kit of the SO-100 sample."""
    import dexvantage as dv
    k = dv.publication.kit(dv.sample('so100'))
    st = k['status']
    return {'word': T(st['word']), 'traced': T(f'{st["traced"]} of {st["numbers"]}'), 'text': T(st['text'])}


# ====================================================================== the CLI and API references
@example('reference')
def ex_reference():
    """How many options the batch CLI has and how many names the API exports."""
    import dexvantage.api as api
    sys.path.insert(0, os.path.join(ROOT, 'scripts'))
    import gen_cli_reference as G
    ap = G.capture_parser()
    n = sum(1 for a in ap._actions if a.option_strings)
    return {'options': T(f'{n} options'), 'names': T(f'{len(api.__all__)} names')}


# ====================================================================== the sample tours (docs/guide/samples.md)
def _tour(name, grasp=True, extra=()):
    import dexvantage as dv
    rig = dv.sample(name)
    out = {'summary': T(f'{len(rig.robots)} robot' + ('s' if len(rig.robots) != 1 else '') +
                        f', {len(rig.sensors)} sensor' + ('s' if len(rig.sensors) != 1 else ''))}
    out['coverage'] = T(rig.run('coverage').headline['text'])
    if grasp:       # the sensitivity search ("what to move") is left out here: it does not change the score
        out['grasp'] = T(rig.run('grasp', sensitivity=False).headline['text'])
    for sid in extra:
        out[sid] = T(rig.run(sid).headline['text'])
    return out


@example('tour-so100', SAMPLES)
def ex_tour_so100():
    """The SO-100 tour."""
    return _tour('so100', extra=('teleop',))


@example('tour-ur5e', SAMPLES)
def ex_tour_ur5e():
    """The UR5e tour."""
    return _tour('ur5e', extra=('depth_validity', 'time_sync'))


@example('tour-dexhand', SAMPLES)
def ex_tour_dexhand():
    """The UR5e and Shadow Hand tour."""
    return _tour('dexhand', grasp=False, extra=('dex_occlusion',))


@example('tour-aloha', SAMPLES)
def ex_tour_aloha():
    """The ALOHA 2 tour."""
    return _tour('aloha')


@example('tour-panda-bimanual', SAMPLES)
def ex_tour_panda():
    """The bimanual Panda tour."""
    return _tour('panda-bimanual')


@example('tour-g1', SAMPLES)
def ex_tour_g1():
    """The Unitree G1 tour."""
    return _tour('g1', extra=('dex_occlusion',))


# ====================================================================== running and checking
def run(ids=None, progress=print) -> dict:
    """{example id: {figure name: Fig}}, running the examples in order (all, or those in ids)."""
    want = list(EXAMPLES) if not ids else [i for i in EXAMPLES if i in ids]
    out = {}
    try:
        for eid in want:
            t0 = time.time()
            out[eid] = EXAMPLES[eid].fn()
            if progress:
                progress(f'{eid}: {len(out[eid])} figures, {time.time() - t0:.1f} s')
    finally:
        for d in _TMP:
            shutil.rmtree(d, ignore_errors=True)
        _TMP.clear()
    return out


_NUM = re.compile(r'[−-]?\d[\d  ]*(?:\.\d+)?')


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s.replace(' ', ' ').replace(' ', ' ')).strip()


def spans(text: str) -> dict:
    """{example id: the normalised text of its span} from the markers in a document."""
    out = {}
    marks = list(re.finditer(r'<!-- example: ([\w-]+) -->', text))
    for i, m in enumerate(marks):
        end = marks[i + 1].start() if i + 1 < len(marks) else len(text)
        h = re.search(r'^#{1,3} ', text[m.end():end], re.M)
        if h:
            end = m.end() + h.start()
        out[m.group(1)] = out.get(m.group(1), '') + ' ' + _norm(text[m.end():end])
    return out


def _numbers(span: str, unit: str | None = None) -> list:
    vals = []
    for m in _NUM.finditer(span):
        if unit and not span[m.end():].lstrip().startswith(unit):
            continue
        try:
            vals.append(float(m.group(0).replace('−', '-').replace(' ', '').replace(' ', '')))
        except ValueError:
            pass
    return vals


def check(results: dict, root: str = ROOT) -> list:
    """Plain sentences, one per figure a document does not show as the example gives it (empty: all match)."""
    problems = []
    cache = {}
    for eid, figs in results.items():
        doc = EXAMPLES[eid].doc
        if doc not in cache:
            cache[doc] = spans(open(os.path.join(root, doc), encoding='utf-8').read())
        span = cache[doc].get(eid)
        if span is None:
            problems.append(f'{doc} has no worked example marked <!-- example: {eid} -->.')
            continue
        for name, f in figs.items():
            if not f.text or name.startswith('_'):
                continue
            if _norm(f.text) in span:
                continue
            if f.tol is not None and isinstance(f.value, (int, float)) and \
                    any(abs(x - float(f.value)) <= f.tol for x in _numbers(span, f.unit)):
                continue
            problems.append(f'{doc}, example {eid}: the figure {name} reads "{f.text}" now, and the guide does not '
                            'show it.')
    return problems


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Run the user guide\'s worked examples and check their figures.')
    ap.add_argument('--only', help='comma-separated example ids (default: all)')
    ap.add_argument('--check', action='store_true', help='compare the figures with the documents; exit 1 on a miss')
    ap.add_argument('--json', metavar='OUT.json', help='write the figures as JSON')
    ap.add_argument('--list', action='store_true', help='list the examples and exit')
    args = ap.parse_args(argv)
    if args.list:
        for e in EXAMPLES.values():
            print(f'{e.id:22} {e.doc:24} {e.title}')
        return 0
    ids = [x.strip() for x in args.only.split(',')] if args.only else None
    unknown = [i for i in ids or () if i not in EXAMPLES]
    if unknown:
        print(f'guide_examples: there is no example {", ".join(unknown)} (--list lists them).', file=sys.stderr)
        return 2
    res = run(ids)
    for eid, figs in res.items():
        print(f'== {eid}')
        for k, f in figs.items():
            if f.text:
                print(f'   {k:16} {f.text}')
    if args.json:
        with open(args.json, 'w', encoding='utf-8') as fh:
            json.dump({e: {k: {'value': f.value, 'text': f.text, 'tol': f.tol} for k, f in figs.items()}
                       for e, figs in res.items()}, fh, indent=1, ensure_ascii=False, default=str)
    if args.check:
        problems = check(res)
        for p in problems:
            print(p)
        print(f'{sum(len(f) for f in res.values())} figures from {len(res)} examples; '
              f'{len(problems)} not shown as computed.')
        return 1 if problems else 0
    return 0


if __name__ == '__main__':
    sys.exit(main())
