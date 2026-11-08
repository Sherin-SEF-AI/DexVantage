"""Every figure in the user guide's worked examples and the sample tours is the one the code gives now
(scripts/guide_examples.py): each example runs in its own process (the samples it opens are freed when it ends) and its
figures are compared with the span of the document its marker opens, within the rounding the document prints.

The examples run the studies on the bundled samples at their stated settings, so this module takes several minutes
(about 6 on the RTX 5080 machine, 2026-11-08; the grasp and ground-truth examples use the GPU when there is one and
numpy's exact rasteriser otherwise)."""
import os
import re
import subprocess
import sys

import pytest

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
sys.path.insert(0, os.path.join(ROOT, 'scripts'))
import guide_examples as GE  # noqa: E402


def test_every_example_has_its_marker_and_every_marker_its_example():
    docs = {GE.GUIDE, GE.SAMPLES}
    marked = {}
    for doc in docs:
        with open(os.path.join(ROOT, doc), encoding='utf-8') as fh:
            for eid in re.findall(r'<!-- example: ([\w-]+) -->', fh.read()):
                marked.setdefault(eid, []).append(doc)
    assert set(marked) == set(GE.EXAMPLES), (set(marked) ^ set(GE.EXAMPLES))
    for eid, ex in GE.EXAMPLES.items():
        assert marked[eid] == [ex.doc], eid                          # one marker, in the example's own document


def test_the_check_finds_a_changed_figure(tmp_path):
    doc = tmp_path / 'docs' / 'user-guide.md'
    doc.parent.mkdir()
    doc.write_text('# x\n\n<!-- example: window -->\nA 500 × 600 × 300 mm volume, 3.3 × 10⁹ voxels.\n\n## next\n'
                   '27 × 10⁶\n', encoding='utf-8')
    res = {'window': {'size': GE.T('500 × 600 × 300 mm'), 'dense': GE.T('3.4 × 10⁹'), 'dense5': GE.T('27 × 10⁶'),
                      'near': GE.Fig(46.4, '46.4', tol=0.3)}}
    problems = GE.check(res, root=str(tmp_path))
    assert len(problems) == 3 and any('3.4 × 10⁹' in p for p in problems)    # the figure past the heading misses
    assert any('27 × 10⁶' in p for p in problems) and any('46.4' in p for p in problems)
    doc.write_text('<!-- example: window -->\nA 500 ×\n600 × 300 mm volume, 3.4 × 10⁹ and 27 × 10⁶ voxels, 46.6 %.\n',
                   encoding='utf-8')
    assert GE.check(res, root=str(tmp_path)) == []                   # line breaks and a figure within its tolerance


@pytest.mark.parametrize('eid', list(GE.EXAMPLES))
def test_the_documents_show_what_the_example_computes(eid):
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen')
    r = subprocess.run([sys.executable, os.path.join(ROOT, 'scripts', 'guide_examples.py'), '--only', eid, '--check'],
                       capture_output=True, text=True, env=env, cwd=ROOT, timeout=1200)
    assert 'Traceback' not in r.stderr, r.stderr[-3000:]
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-2000:]
    assert '0 not shown as computed' in r.stdout
