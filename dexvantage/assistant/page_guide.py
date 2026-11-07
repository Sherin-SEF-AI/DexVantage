"""What every page of the app is for, what it needs, and how far the user has got with it: the assistant's
get_page_context reads this, so it knows the page the user is on and cites the figures shown there.

No study runs here and nothing is computed: a page's state is read off what the window already holds (the page's
page_context() hook in ui/pages/base.py: its headline as shown; the analysis controller's last result of its study
and whether that result is older than the design; whether the study is running). Ported from Vantage
assistant/page_guide.py (its Guide, Need and page_state pattern) for DexVantage's pages; the recommendations of the
Vantage module belong to the Assistant dock.

    GUIDES                       page id -> Guide(purpose, needs, before, after)
    page_state(pid, window)      {'page', 'label', 'purpose', 'status', 'headline', 'shown', 'figures', 'missing',
                                  'before', 'after', 'ms'}; JSON-able, no paths, no keys
    current_page_state(window)   the page shown in the Analysis dock (or None)

status is one of needs | running | never_run | stale | done | ready (a page with no study).

Known limits: the needs are cheap checks on the project (a robot, a sensor of a kind, a policy, a dataset, a
grasp set, a trajectory, a host); a study can still refuse for a reason they do not see, and its own sentence then
shows on the page.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

STATUSES = ('needs', 'running', 'never_run', 'stale', 'done', 'ready')


@dataclass(frozen=True)
class Need:
    id: str
    text: str                    # what is missing, as it reads after "needs": "a robot"
    check: object                # fn(project) -> bool
    how: str = ''                # where to get it


@dataclass(frozen=True)
class Guide:
    page: str
    purpose: str
    needs: tuple = ()
    before: tuple = ()
    after: tuple = ()
    extra: dict = field(default_factory=dict)


def _sensors(p, kind=None):
    return [s for s in p.variant.sensors if s.enabled and (kind is None or s.kind == kind)]


ROBOT = Need('robot', 'a robot', lambda p: bool(p.robots), 'Robot ▸ Import robot…, or open a sample')
SENSOR = Need('sensor', 'a camera or depth sensor on the rig', lambda p: bool(_sensors(p)), 'Sensors ▸ Add a sensor')
DEPTH = Need('depth', 'a depth sensor on the rig', lambda p: bool(_sensors(p, 'depth')),
             'add a depth camera from the Library (Sensors page)')
TACTILE = Need('tactile', 'a tactile sensor', lambda p: bool(_sensors(p, 'tactile')), 'add one from the Library')
POLICY = Need('policy', 'a policy the data is for', lambda p: bool(p.policies), 'Tasks ▸ Policies')
GRASPS = Need('grasps', 'a grasp set', lambda p: bool(p.tasks.grasp_sets), 'Tasks page: sample or import grasps')
TRAJ = Need('trajectory', 'a trajectory', lambda p: bool(p.tasks.trajectories), 'Tasks page: import a q(t) CSV')
TELEOP = Need('teleop', 'a teleop device', lambda p: bool(p.tasks.teleop_devices), 'the Teleop overlap page')
LINKED = Need('link_sensor', 'a sensor mounted on a robot link', lambda p: any(s.mount.on_link for s in _sensors(p)),
              'mount a sensor on a link (Sensors page)')
DATASET = Need('dataset', 'a recorded dataset', lambda p: bool((p.data or {}).get('datasets')),
               'Data ▸ Audit a dataset…')

GUIDES = {g.page: g for g in [
    Guide('robot', 'The robot: its joints, limits, gripper and contact pads, read from its model file.',
          after=('sensors',)),
    Guide('sensors', 'The rig: each sensor\'s library model, mount and stream mode.', (ROBOT,), ('robot',),
          ('coverage', 'depth')),
    Guide('tasks', 'The tasks: grasp sets, trajectories and teleop devices the studies read.', (ROBOT,),
          ('sensors',), ('grasp', 'teleop')),
    Guide('policies', 'The policies the data is for: input size, crop rule, patch size and inference time, cited.',
          (), ('tasks',), ('vla', 'action_chunk')),
    Guide('coverage', 'How much of the task volume each camera sees, and at what resolution.', (SENSOR,),
          ('sensors',), ('depth', 'grasp')),
    Guide('depth', 'Where the depth sensors return valid depth in the task volume (min-Z, range, projector, '
                   'occlusion, materials).', (DEPTH,), ('coverage',), ('grasp',)),
    Guide('grasp', 'Whether the fingertips and the object stay visible at each grasp of a grasp set.',
          (ROBOT, SENSOR, GRASPS), ('tasks', 'coverage'), ('vla', 'dexhand')),
    Guide('dexhand', 'Which fingertip contacts each camera and tactile sensor sees on a multi-finger hand.',
          (ROBOT, SENSOR), ('grasp',), ('vla',)),
    Guide('vla', 'How many ViT tokens a critical object covers in each policy input.', (SENSOR, POLICY),
          ('policies', 'grasp'), ('action_chunk',)),
    Guide('action_chunk', 'How far the gripper close lands from contact given the policy\'s loop latency.',
          (POLICY, TRAJ), ('vla', 'time_sync'), ()),
    Guide('teleop', 'Where a teleop device can lead the robot through the task volume without a dead zone.',
          (ROBOT,), ('tasks',), ('time_sync',)),
    Guide('cables', 'Whether each cable across a joint pulls, kinks or snags over the joint ranges.',
          (ROBOT, LINKED), ('sensors',), ('brackets', 'bandwidth')),
    Guide('brackets', 'The bracket each sensor mounts on and its clearance from the robot.', (SENSOR,),
          ('sensors',), ('cables',)),
    Guide('bandwidth', 'Data rates, host controllers, links, power and disk for the rig.', (SENSOR,),
          ('sensors',), ('time_sync',)),
    Guide('time_sync', 'The skew between each camera stream and joint_states, and the label error it makes.',
          (SENSOR,), ('bandwidth',), ('action_chunk',)),
    Guide('lighting', 'Glare and exposure from the lab lights on the task objects.', (SENSOR,), ('coverage',),
          ('motion_blur',)),
    Guide('motion_blur', 'Pixel smear and rolling-shutter skew along a trajectory.', (SENSOR, TRAJ),
          ('lighting',), ('time_sync',)),
    Guide('groundtruth', 'The predicted visibility against MuJoCo renders of the same scene.', (ROBOT, SENSOR),
          ('coverage', 'grasp'), ()),
    Guide('handeye_plan', 'A hand-eye calibration pose set and its predicted uncertainty.', (ROBOT, SENSOR),
          ('sensors',), ('handeye_solve',)),
    Guide('handeye_solve', 'The hand-eye solve from captured board images.', (ROBOT, SENSOR), ('handeye_plan',)),
    Guide('silhouette', 'Camera mount calibration from the robot\'s silhouette in a recorded dataset.',
          (ROBOT, SENSOR, DATASET), ('audit',)),
    Guide('audit', 'Whether a recorded dataset meets the plan: drops, desync, mount slip.', (DATASET,), (),
          ('plan_vs_recording',)),
    Guide('plan_vs_recording', 'The recorded dataset\'s cameras, rates and mounts against the plan.',
          (ROBOT, DATASET), ('audit',), ('silhouette',)),
]}


def _label(pid) -> str:
    try:
        from ..ui.pages import PAGES
        info = PAGES.get(pid)
        return info.title if info is not None else str(pid)
    except ImportError:                                  # pragma: no cover - no Qt
        return str(pid)


def page_state(pid: str, window) -> dict:
    """The page's state as the window holds it (module docstring)."""
    from . import figures as F
    t0 = time.perf_counter()
    g = GUIDES.get(pid) or Guide(pid, '')
    pages = getattr(window, 'pages', None) or {}
    pg = pages.get(pid)
    ctx = getattr(window, 'ctx', None)
    project = ctx.store.project if ctx is not None else None
    missing = []
    for n in g.needs:
        try:
            ok = bool(n.check(project)) if project is not None else True
        except Exception:                                # noqa: BLE001 — a check must not take the tool down
            ok = True
        if not ok:
            missing.append({'id': n.id, 'text': n.text, 'how': n.how})
    shown = {}
    if pg is not None and hasattr(pg, 'page_context'):
        try:
            shown = dict(pg.page_context() or {})
        except Exception:                                # noqa: BLE001
            shown = {}
    sid = getattr(pg, 'STUDY', None) if pg is not None else None
    doc, running, stale = None, False, False
    if sid and ctx is not None:
        running = ctx.analysis.busy(sid)
        doc = ctx.analysis.results.get(sid)
        stale = doc is not None and doc.get('revision', ctx.store.revision) != ctx.store.revision
    if running:
        status = 'running'
    elif not sid:
        status = 'needs' if missing else 'ready'
    elif doc is None:
        status = 'needs' if missing else 'never_run'
    elif stale:
        status = 'stale'
    else:
        status = 'done'
    figs = None
    if doc is not None:
        try:
            figs = F.brief(sid, doc)
        except Exception:                                # noqa: BLE001 — the figures are a bonus here
            figs = None
    out = {'page': pid, 'label': _label(pid), 'purpose': g.purpose or shown.get('about', ''), 'study': sid,
           'status': status, 'shown': F.compact(shown, max_depth=3), 'figures': figs, 'missing': missing,
           'before': [{'page': b, 'label': _label(b)} for b in g.before],
           'after': [{'page': a, 'label': _label(a)} for a in g.after]}
    out['headline'] = headline(out)
    out['ms'] = round((time.perf_counter() - t0) * 1000, 2)
    return out


def headline(st: dict) -> str:
    """One line: where the user is with the page."""
    s = st.get('status')
    if s == 'running':
        return 'Running now.'
    if s == 'needs':
        return 'Cannot run yet: it needs ' + '; '.join(m['text'] for m in st.get('missing', [])[:3]) + '.'
    if s == 'never_run':
        return 'Not run yet on this design.'
    if s == 'stale':
        return 'Has a result older than the design (run it again to update it).'
    if s == 'done':
        h = (st.get('shown') or {}).get('headline') or {}
        if h.get('figure') and h.get('figure') != '—':
            return f'Shows {h["figure"]}{(" " + h["unit"]) if h.get("unit") else ""}: {h.get("word") or ""}.'.replace(
                ': .', '.')
        return 'Has a result.'
    return 'A page with no study to run.'


def current_page_state(window) -> dict | None:
    """The page shown in the window's Analysis dock (None when no page shows or there is no window)."""
    if window is None or not hasattr(window, 'current_page'):
        return None
    w = window.current_page()
    if w is None:
        return None
    pid = next((k for k, v in (getattr(window, 'pages', None) or {}).items() if v is w), None)
    if pid is None:
        return None
    dock = getattr(window, 'docks', {}).get('bottom')
    out = page_state(pid, window)
    out['visible'] = bool(dock is not None and dock.isVisible())
    return out
