"""The definition-of-done review of Phase 11 (spec §16 type hints, §17 a demo script at the gate, §18 the product
walk): the Python API's other public modules annotate every parameter and public return, ruff checks them, and the
Phase 11 demo script walks every bullet of §18 in batch mode. The demo itself runs for minutes (the self-test, the
grasp study twice, the validation suite), so these tests check its text and its Python, not a run of it."""
import ast
import os
import re
import shutil
import subprocess

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEMO = os.path.join(ROOT, 'scripts', 'demo-phase11.sh')

# the modules whose names `import dexvantage as dv` gives (dv.diff, dv.merge, dv.figure, dv.cluster.*) or that a
# plugin calls (plugin_sdk.PluginAPI), beside the six that tests/test_phase11_polish.py checks
API_FILES = ('dexvantage/api_research.py', 'dexvantage/api_cluster.py', 'dexvantage/plugin_sdk.py')


def _unannotated(rel: str) -> list:
    tree = ast.parse(open(os.path.join(ROOT, rel), encoding='utf-8').read())
    bad = []
    for n in ast.walk(tree):
        if not isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        a = n.args
        for x in a.posonlyargs + a.args + a.kwonlyargs + [y for y in (a.vararg, a.kwarg) if y]:
            if x.annotation is None and x.arg not in ('self', 'cls'):
                bad.append(f'{rel}:{n.lineno} {n.name}({x.arg})')
        if not n.name.startswith('_') and n.returns is None:
            bad.append(f'{rel}:{n.lineno} {n.name} returns')
    return bad


@pytest.mark.parametrize('rel', API_FILES)
def test_api_module_annotates_every_parameter_and_public_return(rel):
    assert not _unannotated(rel)


def test_ruff_checks_the_annotations_of_those_modules():
    tomllib = pytest.importorskip('tomllib', reason='tomllib needs Python 3.11 or newer')
    with open(os.path.join(ROOT, 'pyproject.toml'), 'rb') as fh:
        lint = tomllib.load(fh)['tool']['ruff']['lint']
    neg = [k for k, v in lint['per-file-ignores'].items() if k.startswith('!') and v == ['ANN']]
    assert len(neg) == 1
    for rel in API_FILES:
        assert rel.split('dexvantage/', 1)[1] in neg[0], rel


def test_api_hints_name_only_what_the_modules_can_resolve():
    """Postponed annotations are strings: every name a hint uses is imported (at run time or under TYPE_CHECKING),
    a builtin or a typing name, so the generated API reference never shows a name a reader cannot look up."""
    import builtins
    known_typing = {'Callable', 'Iterable', 'Sequence', 'ModuleType', 'EntryPoint'}
    for rel in API_FILES:
        tree = ast.parse(open(os.path.join(ROOT, rel), encoding='utf-8').read())
        imported = set()
        for n in ast.walk(tree):
            if isinstance(n, ast.ImportFrom):
                imported |= {a.asname or a.name for a in n.names}
            elif isinstance(n, ast.Import):
                imported |= {(a.asname or a.name).split('.')[0] for a in n.names}
            elif isinstance(n, ast.ClassDef):
                imported.add(n.name)
        used = set()
        for n in ast.walk(tree):
            if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
                hints = [x.annotation for x in n.args.posonlyargs + n.args.args + n.args.kwonlyargs
                         + [y for y in (n.args.vararg, n.args.kwarg) if y]] + [n.returns]
                for h in hints:
                    if h is not None:
                        used |= {x.id for x in ast.walk(h) if isinstance(x, ast.Name)}
        missing = {u for u in used if u not in imported and not hasattr(builtins, u) and u not in known_typing}
        assert not missing, (rel, sorted(missing))


# ====================================================================== the Phase 11 demo script (spec §17, §18)
# The phase demo scripts are the lead engineer's, kept out of git (.gitignore: scripts/demo-*.sh), so a clone (CI)
# has none: the first GitHub CI run failed these four tests with "No such file or directory". They run where the
# script is.
_NO_DEMO = pytest.mark.skipif(not os.path.isfile(DEMO), reason='scripts/demo-phase11.sh is not here (the phase demo '
                              'scripts are kept out of git: .gitignore scripts/demo-*.sh)')


def _demo() -> str:
    return open(DEMO, encoding='utf-8').read()


@_NO_DEMO
def test_demo_script_is_valid_bash():
    bash = shutil.which('bash')
    if bash is None:
        pytest.skip('bash is not on PATH (the demo is a bash script)')
    r = subprocess.run([bash, '-n', DEMO], capture_output=True, text=True, timeout=30)
    assert r.returncode == 0, r.stderr


@_NO_DEMO
def test_demo_script_walks_every_bullet_of_the_definition_of_done():
    s = _demo()
    # §18.1: ALOHA 2; grasps and why; a camera moved and the score again; depth validity, VLA tokens, bandwidth;
    # the cameras to MuJoCo and LeRobot; a printable bracket
    for needle in ('--sample aloha', '--grasp', "rig.run('grasp')", 'behind_own_finger', 'hidden_by_scene',
                   "'sensor:cam_left_wrist.pitch'", "moved.run('grasp'", '--depth-validity', '--vla-tokens',
                   '--bandwidth', '--formats mjcf,lerobot', "export(f'{out}/export-brackets', 'brackets')"):
        assert needle in s, needle
    # §18.2 provenance (the export manifest) is written by every --export; §18.3: at least 40 checks pass
    assert '--validate' in s and ">= 40" in s
    # §18.4: a dataset audited with per-episode results
    assert '--synth-episodes' in s and '--audit' in s and "doc['episodes']" in s
    # the generated docs and the website are checked
    for g in ('gen_cli_reference.py --check', 'gen_api_reference.py --check', 'gen_overview_refs.py --check',
              'build_site.py --check'):
        assert g in s, g
    # batch mode does not create folders: the demo makes the ones it writes into
    assert re.search(r'mkdir -p "\$OUT/tour" "\$OUT/audit"', s)


@_NO_DEMO
def test_demo_script_python_steps_compile():
    blocks = re.findall(r"<<'PYEOF'\n(.*?)\nPYEOF", _demo(), flags=re.S)
    assert len(blocks) == 3
    for b in blocks:
        compile(b, 'demo-phase11.sh', 'exec')


@_NO_DEMO
def test_demo_script_uses_only_flags_the_parser_knows():
    """Every --flag of the demo is one docs/reference/cli.md lists (generated from the real parser and kept current
    by tests/test_docs_reference.py), or --check of the generator and site scripts."""
    cli = open(os.path.join(ROOT, 'docs', 'reference', 'cli.md'), encoding='utf-8').read()
    known = set(re.findall(r'`(--[a-z][a-z0-9-]+)', cli)) | {'--check'}
    used = set(re.findall(r'(?<![\w-])(--[a-z][a-z0-9-]+)', _demo()))
    assert used and not (used - known), sorted(used - known)
