import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor

from dexvantage import validation as VA

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')

# One model's quick run in a process of its own: what the checks build (compiled robot models, EGL contexts, render
# workers, the audit campaign, cube maps) goes back to the system when the process ends. In this process the quick run
# of every check grew the suite by 3 GB that the allocator kept (6.2 GB peak at the start of the suite, 3.3 GB left
# over), measured 2026-11-07. Alone, formats peaks at 4.9 GB, handeye at 3.7 GB and groundtruth at 2.4 GB (quick run,
# 2026-11-07): those run one at a time first, then the others (each under 1.8 GB) two at a time.
_CHILD = r"""
import json, sys
from dexvantage import validation as VA
res = VA.run(settings=VA.ValidationSettings(quick=True, models=(sys.argv[1],)))
print('@@' + json.dumps([{'id': c.id, 'model': c.model, 'passed': bool(c.passed), 'error': c.error,
                          'exception': (c.exception or '')[-800:], 'reference': c.reference, 'quantity': c.quantity,
                          'unit': c.unit, 'refs': list(c.refs)} for c in res.checks], default=str))
"""
WORKERS = 2
HEAVY = ('formats', 'handeye', 'groundtruth')


def _quick_run(model: str) -> list:
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, '-c', _CHILD, model], capture_output=True, text=True, env=env, cwd=ROOT,
                       timeout=900)
    line = next((x for x in r.stdout.splitlines() if x.startswith('@@')), None)
    assert r.returncode == 0 and line is not None, (model, r.returncode, r.stderr[-2000:])
    return json.loads(line[2:])


def test_every_check_passes_and_names_its_reference():
    # every check, in the quick run (fewer samples, the same tolerances), one model per process: the full run
    # (dexvantage --validate) takes over four minutes
    from dexvantage.validation import suite
    checks = [c for m in HEAVY for c in _quick_run(m)]
    with ThreadPoolExecutor(WORKERS) as ex:
        checks += [c for cs in ex.map(_quick_run, [m for m in suite.MODEL_IDS if m not in HEAVY]) for c in cs]
    failed = [(c['id'], c['error'], c['exception']) for c in checks if not c['passed']]
    assert not failed, failed
    assert len(checks) >= len(suite.specs(VA.ValidationSettings(quick=True)))
    for c in checks:
        assert c['reference'] and c['quantity'] and c['unit'], c['id']
        assert all(r in VA.REFERENCES for r in c['refs']), c['id']


def test_report_files(tmp_path):
    res = VA.run(settings=VA.ValidationSettings(quick=True, cheap_only=True))     # the report files, not the checks
    paths = VA.export(res, None, str(tmp_path))
    names = {p.split('/')[-1] for p in paths}
    assert {'validation.json', 'validation.csv', 'validation.html'} <= names
    doc = json.loads((tmp_path / 'validation.json').read_text())
    assert doc['schema'] == 'dexvantage.validation/1' and doc['summary']['ok']
    assert 'DexVantage validation report' in (tmp_path / 'validation.html').read_text()


def test_settings_are_tolerant_and_exit_codes():
    st = VA.ValidationSettings.from_dict({'models': 'frames,nonsense', 'seed': -5})
    assert st.models == ('frames',) and st.seed == 0
    res = VA.run(settings=st)
    assert {c.model for c in res.checks} == {'frames'} and VA.exit_code(res) == 0
    res.checks[0].passed = False
    assert VA.exit_code(res) == 1


def test_catalogue_lists_without_running():
    cat = VA.catalogue()
    assert {c['id'] for c in cat} >= {'frames.quat_to_matrix', 'formats.schema'}


def test_a_raising_check_fails_without_stopping_the_suite():
    from dexvantage.validation import suite
    spec = suite.Spec('x.boom', 'frames', 'Boom', 'q', 'r', 1.0, 'u', (), lambda ctx: 1 / 0)
    ctx = suite.Ctx(None, None, None, VA.ValidationSettings())
    c = suite.run_one(spec, ctx)
    assert not c.passed and 'ZeroDivisionError' in c.exception


def test_cheap_only_selects_the_cheap_checks():
    from dexvantage.validation import suite
    st = VA.ValidationSettings.from_dict({'cheap_only': True, 'quick': True})
    sel = suite.specs(st)
    assert sel and all(s.cost == 'cheap' for s in sel)
    assert len(sel) < len(suite.specs(VA.ValidationSettings()))
    assert st.to_dict()['cheap_only'] is True


def test_every_registered_check_declares_its_unit_reference_and_quantity():
    """Without running them (the full run takes minutes): a unit, a reference and a quantity on every check."""
    from dexvantage.validation import suite
    suite.load_checks()
    bad = [s.id for s in suite.CHECKS if not (s.unit and s.reference and s.quantity)]
    assert not bad, bad
