"""Checks of the robot catalogue (core/robot/catalogue.py): a DATA-CONSISTENCY check, not a model check.

The catalogue holds makers' figures and a record of what DexVantage measured when it imported each robot's official
model once. Two things can be compared without the network:

1. Each entry against itself and its maker: the verified record's DoF must follow from its own counts (joints in q
   minus the excluded joints, or the actuators, by its stated rule), and must equal the DoF the maker states, unless
   the entry says why they differ (dof_differs, in the maker's and the model's words). An entry whose model did not
   import must say why.
2. The verified records against an independent measurement now: the Menagerie models bundled with DexVantage
   (core/robot/samples.py BUNDLED, copied from the same Menagerie commit) are imported again through
   core/robot/importer.py and compiled with MuJoCo, and their joints in q, actuators and links are counted again and
   compared with the catalogue's records for the same folder and file. A bundled file the catalogue does not name
   (a combined model such as the G1 with hands) is listed, not compared.

Tolerance: 0. Every quantity is a count; a difference of one joint is a data error, not a numerical one.
"""
from __future__ import annotations

from .suite import Outcome, check

# the bundled models whose folder and file the catalogue also names must be at least these many (the SO-ARM100, UR5e,
# 2F-85 and Shadow Hand of Phase 1-3; later phases bundle more)
MIN_RECOUNTED = 4


def _bundled_counts() -> list:
    """[(folder, file, {'joints_in_q', 'actuators', 'links'} or error text)] of the bundled Menagerie models."""
    from ..core.robot.importer import import_robot
    from ..core.robot.model import RobotModel
    from ..core.robot.samples import BUNDLED, robot_file
    out = []
    for folder, info in sorted(BUNDLED.items()):
        try:
            robot, sources = import_robot(robot_file(folder, info['main']))
            m = RobotModel(robot, sources)
            out.append((folder, info['main'], {'joints_in_q': m.nq, 'actuators': int(m.m.nu),
                                               'links': len(robot.links)}))
        except Exception as e:                       # noqa: BLE001 — reported as a disagreement, in words
            out.append((folder, info['main'], f'{type(e).__name__}: {e}'))
    return out


@check('robots.catalogue_dof', 'robots',
       'Robot catalogue DoF against the verified import and the maker (data consistency)',
       'entries and bundled models in disagreement: (1) each catalogue entry\'s verified DoF against its own joint and '
       'actuator counts and against the DoF its maker states (a difference the entry explains is listed, not '
       'counted); (2) for the Menagerie models bundled with DexVantage under a file the catalogue names, the joints '
       'in q, actuators and links counted again now by importing the bundled file, against the catalogue\'s record '
       'of the same folder and file',
       'the makers\' datasheets each entry cites, and MuJoCo\'s compiled model of the bundled Menagerie files', 0,
       'entries', refs=('robot_datasheets', 'mujoco_menagerie', 'mujoco_todorov2012'))
def _catalogue_dof(ctx):
    from ..core.robot import catalogue as CAT
    es = CAT.entries()
    problems, explained, compared = [], [], 0
    for e in es:
        problems += CAT.dof_problems(e)
        v = CAT.verified(e)
        stated = CAT.figure(e, 'dof')[0]
        meas = CAT.measured_dof(v)
        ok = CAT.dof_matches(stated, meas)
        if ok is not None:
            compared += 1
        if ok is False and v.get('dof_differs'):
            explained.append({'entry': e.id, 'maker': stated, 'model': meas, 'why': v['dof_differs']})
    by_folder = {}
    for e in es:
        m = CAT.model(e)
        if m.get('source') == 'menagerie':
            by_folder.setdefault((m.get('folder'), m.get('file')), e)
    rows, other = [], []
    for folder, main, got in _bundled_counts():
        e = by_folder.get((folder, main))
        if e is None:                                # bundled under another file (a combined model): not comparable
            other.append(f'{folder}/{main}')
            continue
        if isinstance(got, str):
            problems.append(f'the bundled model {folder}/{main} did not import: {got}')
            continue
        v = CAT.verified(e)
        diff = {k: (v.get(k), n) for k, n in got.items() if v.get(k) != n}
        rows.append({'entry': e.id, 'file': f'{folder}/{main}', 'now': got, 'recorded': {k: v.get(k) for k in got}})
        if diff:
            problems.append(f'{e.id}: recorded {", ".join(f"{k} {a}" for k, (a, _b) in diff.items())}, counted now '
                            f'{", ".join(f"{k} {b}" for k, (_a, b) in diff.items())}')
    detail = {'entries': len(es), 'compared_with_maker_dof': compared, 'explained_differences': explained,
              'bundled_recounted': rows, 'bundled_not_in_catalogue': other, 'problems': problems,
              'kind': 'data consistency (catalogue records against makers\' figures and a fresh import), not a model '
                      'accuracy figure'}
    return Outcome(float(len(problems)), detail, ok=bool(es) and len(rows) >= MIN_RECOUNTED,
                   note=f'{len(es)} entries; {compared} with a maker DoF to compare, {len(explained)} explained '
                        f'differences; {len(rows)} bundled models counted again.')
