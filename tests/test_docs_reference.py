"""The documentation stays in step with the code (spec §15): the generated command-line and Python API references,
the overview's model sections and references, the project format against its schema, the user guide's coverage of
every page and worked-example file, its links, and the sample tours' screenshots."""
import ast
import json
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DOCS = os.path.join(ROOT, 'docs')
# models whose overview section is about data formats or software, with no equation to write out
NO_EQUATION = {'formats', 'assistant', 'copilot', 'versioning', 'samples', 'robot_io', 'tracking', 'publication',
               'robots'}


def _read(*parts) -> str:
    with open(os.path.join(ROOT, *parts), encoding='utf-8') as fh:
        return fh.read()


def _norm(s: str) -> str:
    return re.sub(r'\s+', ' ', s)


def _script(name, *args):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    return subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', name), *args], capture_output=True, text=True,
                          env=env, cwd=ROOT, timeout=600)


def _models():
    src = _read('dexvantage', 'validation', 'suite.py')
    return ast.literal_eval(src.split('MODELS = ', 1)[1].split('\nMODEL_IDS', 1)[0])


def _section(text, anchor):
    m = re.search(r'^## [^\n]*\{#' + re.escape(anchor) + r'\}[^\n]*$', text, re.M)
    if not m:
        return None
    n = re.search(r'^## ', text[m.end():], re.M)
    return text[m.start(): m.end() + n.start()] if n else text[m.start():]


# ---------------------------------------------------------------------- generated references
@pytest.mark.parametrize('script', ['gen_cli_reference.py', 'gen_api_reference.py', 'gen_overview_refs.py'])
def test_generated_documents_are_up_to_date(script):
    r = _script(script, '--check')
    assert r.returncode == 0, r.stdout + r.stderr
    assert 'Traceback' not in r.stderr


def test_cli_reference_lists_every_study_flag_and_the_exit_codes():
    text = _read('docs', 'reference', 'cli.md')
    from dexvantage.core import jobs
    jobs.load_studies()
    for info in jobs.STUDIES.values():
        if info.batch and not getattr(info, 'plugin', None):
            assert f'`--{info.batch} OUT.json`' in text, info.batch
    assert '## Exit codes' in text and '| 2 |' in text
    for flag in ('--sample NAME', '--self-test', '--validate OUT_DIR', '--export OUT_DIR', '--check'):
        assert flag in text, flag


def test_api_reference_names_every_exported_name():
    text = _read('docs', 'reference', 'api.md')
    import dexvantage.api as api
    for name in api.__all__:
        assert re.search(r'^#{3,4} `dv\.' + re.escape(name) + r'\b', text, re.M), name
    assert '`plugin_sdk.PluginAPI' in text


# ---------------------------------------------------------------------- the overview's models
def test_every_validation_model_has_an_overview_section_with_its_parts():
    text = _read('docs', 'overview.md')
    problems = []
    for mid, title, _mods, anchor in _models():
        sec = _section(text, anchor)
        if sec is None:
            problems.append(f'{mid}: no section with {{#{anchor}}}')
            continue
        s = _norm(sec)
        for part in ('**Validation:**', '**Known limits:**', '**References (checks):**'):
            if part not in s:
                problems.append(f'{mid}: no {part}')
        if mid not in NO_EQUATION and not re.search(r'[^\s(]\s*[=≤≥]\s*[^\s]', re.sub(r'\[[^\]]*\]\([^)]*\)', '', s)):
            problems.append(f'{mid}: no equation')
    assert not problems, problems


def test_overview_references_resolve():
    from dexvantage.validation import references as R
    text = _read('docs', 'overview.md')
    keys = set(re.findall(r'^- `([\w.\-]+)`: ', text.split('## References {#references}', 1)[1], re.M))
    cited = set()
    for line in re.findall(r'^\*\*References \(checks\):\*\* (.*)$', text, re.M):
        cited.update(re.findall(r'`([\w.\-]+)`', line))
    assert cited and cited <= keys and keys <= set(R.REFERENCES)


def test_overview_internal_anchors_exist():
    text = _read('docs', 'overview.md')
    anchors = set(re.findall(r'\{#([\w\-]+)\}', text))
    for a in re.findall(r'\]\(#([\w\-]+)\)', text):
        assert a in anchors, a


# ---------------------------------------------------------------------- the project format
def test_project_format_documents_every_top_level_schema_key():
    schema = json.loads(_read('docs', 'project.schema.json'))
    doc = _read('docs', 'project-format.md')
    missing = [k for k in schema['properties'] if f'"{k}"' not in doc and f'`{k}`' not in doc]
    assert not missing, missing
    for k in schema.get('required', []):
        assert f'"{k}"' in doc, k


# ---------------------------------------------------------------------- the user guide
def test_user_guide_covers_every_page_and_links_every_worked_example():
    guide = _norm(_read('docs', 'user-guide.md'))
    titles = {}
    for f in os.listdir(os.path.join(ROOT, 'dexvantage', 'ui', 'pages')):
        if f.endswith('.py'):
            for pid, title in re.findall(r"register\('(\w+)', '([^']+)'", _read('dexvantage', 'ui', 'pages', f)):
                titles[pid] = title
    assert len(titles) >= 30
    missing = sorted(t for t in titles.values() if t not in guide)
    assert not missing, missing
    linked = set(re.findall(r'\]\(guide/([\w\-]+)\.md', guide))
    files = {f[:-3] for f in os.listdir(os.path.join(DOCS, 'guide')) if f.endswith('.md')}
    assert not files - linked, sorted(files - linked)


def test_user_guide_chapters_and_reference_anchors():
    guide = _read('docs', 'user-guide.md')
    for anchor in ('ten-minute-tour', 'getting-started', 'window', 'robots', 'sensors', 'studies', 'calibration',
                   'data', 'requirements', 'export', 'assistant', 'research', 'batch', 'python-api',
                   'troubleshooting'):
        assert '{#' + anchor + '}' in guide, anchor
    for ref in ('cli.md', 'api.md'):
        assert '../user-guide.md#' in _read('docs', 'reference', ref)


@pytest.mark.parametrize('doc', ['docs/user-guide.md', 'docs/guide/samples.md', 'docs/project-format.md'])
def test_relative_links_resolve(doc):
    text = _read(doc)
    base = os.path.dirname(os.path.join(ROOT, doc))
    overview = _read('docs', 'overview.md')
    ov_anchors = set(re.findall(r'\{#([\w\-]+)\}', overview))
    own = set(re.findall(r'\{#([\w\-]+)\}', text))
    bad = []
    for target in re.findall(r'\]\(([^)\s]+)\)', text):
        if re.match(r'[a-z]+://', target):
            continue
        path, _, anchor = target.partition('#')
        if not path:
            if anchor not in own:
                bad.append(target)
            continue
        full = os.path.normpath(os.path.join(base, path))
        if not os.path.exists(full):
            bad.append(target)
        elif anchor and full.endswith('overview.md') and anchor not in ov_anchors:
            bad.append(target)
    assert not bad, bad


# ---------------------------------------------------------------------- the sample tours
def test_every_sample_has_a_tour_with_screenshots_under_300_kb():
    from dexvantage.core.robot.samples import SAMPLES
    text = _read('docs', 'guide', 'samples.md')
    for name in SAMPLES:
        assert f'<!-- example: tour-{name} -->' in text, name
        assert f'`--sample {name}`' in text, name
    images = re.findall(r'!\[[^\]]*\]\(\.\./media/([\w\-.]+)\)', text)
    assert len(images) >= 6
    for img in images:
        path = os.path.join(DOCS, 'media', img)
        assert img.endswith('.png') and os.path.isfile(path), img
        assert os.path.getsize(path) < 300 * 1024, img
        with open(path, 'rb') as fh:
            assert fh.read(8) == b'\x89PNG\r\n\x1a\n', img
    for name in SAMPLES:
        assert any(i.startswith(f'tour-{name}-') for i in images), name
