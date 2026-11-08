"""Build the DexVantage bundle for this platform, smoke-test it and pack it for download.

    python packaging/build.py [--dist DIR] [--work DIR] [--smoke] [--full-smoke] [--no-archive] [--no-build]

Needs PyInstaller and the bundle's requirements in the same Python (pip install -r packaging/requirements-bundle.txt,
or in a development environment pip install -e ".[bundle,lerobot,bags,assistant]"). Writes into DIR (default
out/dist, ignored by git; work files in out/build):
    Linux    DexVantage/ and DexVantage-<version>-linux-<arch>.tar.gz (the folder plus install.sh, uninstall.sh, the
             .desktop entry, the MIME type and the icon)
    Windows  DexVantage/, DexVantage-<version>-windows-x64.zip and, when Inno Setup's iscc is on PATH or in its default
             folder, DexVantage-<version>-windows-x64-setup.exe (a per-user installer: no administrator rights)
    macOS    DexVantage.app and DexVantage-<version>-macos-arm64.zip (unsigned; docs/packaging.md has the codesign and
             notarisation steps)
and prints each archive's size and SHA-256.

--smoke runs the bundled app (QT_QPA_PLATFORM=offscreen, throw-away XDG folders, so no window opens and nothing of
the user's is read): --version; "-m dexvantage --version" (the worker launch path); the start-up time to the main
window's event loop; a batch run of three studies on the SO-100 sample; the tolerance Monte Carlo with --jobs 2 (the
cluster's spawned local pool) and a designed experiment with two workers (the experiments' spawned process pool);
--self-test, where every line must pass without a warning except those in selftest.BUNDLE_MAY_WARN (OpenGL: build
machines may have no GPU), so a library or module that did not make it into the bundle fails the build.
--full-smoke adds --validate --quick (every validation model; minutes, several GB of memory).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
import release  # noqa: E402

VERSION = release.version(ROOT)
NAME = 'DexVantage'
LINUX_FILES = ('install.sh', 'uninstall.sh', 'dexvantage.desktop', 'dexvantage-mime.xml')
SMOKE_STUDIES = (('--coverage', 'coverage.json'), ('--depth-validity', 'depth.json'), ('--bandwidth', 'bandwidth.json'))


def arch():
    m = platform.machine().lower()
    if os.name == 'nt':
        return {'amd64': 'x64', 'x86_64': 'x64', 'arm64': 'arm64'}.get(m, m)
    return {'amd64': 'x86_64', 'x64': 'x86_64', 'aarch64': 'arm64'}.get(m, m)


def run(cmd, **kw):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    return subprocess.run(cmd, check=True, **kw)


def build(dist, work):
    run([sys.executable, '-m', 'PyInstaller', '--noconfirm', '--clean', '--distpath', dist, '--workpath', work,
         os.path.join(HERE, 'dexvantage.spec')])


def bundle_dir(dist):
    if sys.platform == 'darwin':
        return os.path.join(dist, f'{NAME}.app')
    return os.path.join(dist, NAME)


def cli_path(dist):
    if sys.platform == 'darwin':
        return os.path.join(dist, f'{NAME}.app', 'Contents', 'MacOS', 'dexvantage-cli')
    return os.path.join(dist, NAME, 'dexvantage-cli' + ('.exe' if os.name == 'nt' else ''))


def gui_path(dist):
    if sys.platform == 'darwin':
        return os.path.join(dist, f'{NAME}.app', 'Contents', 'MacOS', NAME)
    return os.path.join(dist, NAME, NAME + ('.exe' if os.name == 'nt' else ''))


def folder_mb(path):
    total = 0
    for d, _, files in os.walk(path):
        for f in files:
            p = os.path.join(d, f)
            if not os.path.islink(p):
                total += os.path.getsize(p)
    return total / 1e6


TEXT_SUFFIXES = ('.py', '.json', '.txt', '.cfg', '.toml', '.ini', '.xml', '.md', '.yaml', '.yml', '.desktop', '.plist',
                 '.cff', 'METADATA', 'RECORD', 'INSTALLER', 'WHEEL')


def personal_paths(folder, home=None) -> list[str]:
    """Text files in the bundle that name the build machine's home folder (a development install's direct_url.json,
    a path written into a config): DEVELOPING.md rule 7 holds for the downloads too."""
    home = os.path.abspath(home or os.path.expanduser('~'))
    if home in ('/', '') or not os.path.isdir(home):
        return []
    base = home.rstrip('\\/')
    # as written natively, and on Windows also with forward slashes and with JSON's doubled backslashes
    needles = {(base + os.sep).encode(), (base.replace('\\', '/') + '/').encode(),
               (base.replace('\\', '\\\\') + '\\\\').encode()}
    hits = []
    for d, _, files in os.walk(folder):
        for f in files:
            p = os.path.join(d, f)
            if f.endswith(TEXT_SUFFIXES) and not os.path.islink(p):
                with open(p, 'rb') as fh:
                    data = fh.read()
                if any(n in data for n in needles):
                    hits.append(os.path.relpath(p, folder))
    return hits


def self_test_problems(text: str) -> list[str]:
    """What a bundle's self-test output says is wrong: a FAILED or (outside BUNDLE_MAY_WARN) warning line, a check
    that did not run, or no final verdict."""
    sys.path.insert(0, ROOT)
    from dexvantage.selftest import BUNDLE_MAY_WARN, FAIL, NAMES, OK
    lines = [ln.rstrip() for ln in text.splitlines()]
    seen, bad = set(), []
    for ln in lines:
        if not ln.startswith('  '):
            continue
        rest = ln[2:]
        for name in sorted(NAMES, key=len, reverse=True):          # 'software 3d' before a shorter prefix
            if rest.startswith(name + ' ') and rest[len(name):].strip():
                status = rest[16:].split()[0] if len(rest) > 16 else ''
                seen.add(name)
                if status == FAIL or (status != OK and name not in BUNDLE_MAY_WARN):
                    bad.append(ln.strip())
                break
    bad += [f'{n}: did not run' for n in NAMES if n not in seen]
    if not any(ln.startswith('self-test passed') for ln in lines):
        bad.append('no "self-test passed" line')
    return bad


def smoke_env(tmp):
    """The bundle's environment for the smoke test: offscreen Qt and throw-away settings, data and cache folders."""
    env = dict(os.environ, QT_QPA_PLATFORM='offscreen', TMPDIR=tmp, TMP=tmp, TEMP=tmp)   # TMP, TEMP: Windows
    for k, sub in (('XDG_CONFIG_HOME', 'config'), ('XDG_DATA_HOME', 'data'), ('XDG_CACHE_HOME', 'cache')):
        env[k] = os.path.join(tmp, sub)
        os.makedirs(env[k], exist_ok=True)
    return env


def _timed(cmd, env, cwd, timeout, what, ok_codes=(0,)):
    print('+', ' '.join(str(c) for c in cmd), flush=True)
    t0 = time.perf_counter()
    r = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, encoding='utf-8', errors='replace',
                       timeout=timeout)
    sec = time.perf_counter() - t0
    if r.returncode not in ok_codes:
        sys.stdout.write(r.stdout[-4000:])
        sys.stderr.write(r.stderr[-4000:])
        raise SystemExit(f'{what} failed (exit {r.returncode})')
    return r, sec


def startup_seconds(dist, env, runs=3):
    """Wall time from starting the window launcher to its event loop running with the main window shown
    (run_dexvantage.py's DEXVANTAGE_STARTUP_PROBE prints and quits there); the median of runs, and the first."""
    times = []
    probe = os.path.join(env.get('TMPDIR') or tempfile.gettempdir(), 'startup-probe.txt')
    for _ in range(runs):
        if os.path.exists(probe):
            os.remove(probe)
        t0 = time.perf_counter()
        r = subprocess.run([gui_path(dist)], env=dict(env, DEXVANTAGE_STARTUP_PROBE='1',
                                                      DEXVANTAGE_STARTUP_PROBE_OUT=probe),
                           capture_output=True, text=True, timeout=300)
        sec = time.perf_counter() - t0
        said = r.stdout
        if os.path.exists(probe):            # the window launcher without a console (Windows) writes it here
            with open(probe, encoding='utf-8') as fh:
                said += fh.read()
        if r.returncode or 'startup-probe' not in said:
            raise SystemExit(f'the window did not start (exit {r.returncode}): {r.stderr.strip()[-2000:]}')
        times.append(sec)
    return sorted(times)[len(times) // 2], times[0]


def smoke(dist, work, full=False):
    cli = cli_path(dist)
    tmp = tempfile.mkdtemp(prefix='dexvantage-smoke-', dir=work)
    env = smoke_env(tmp)
    report = {'version': VERSION, 'platform': f'{platform.system()} {arch()}'}
    try:
        r, sec = _timed([cli, '--version'], env, tmp, 300, '--version')
        if r.stdout.strip() != f'dexvantage {VERSION}':
            raise SystemExit(f'--version printed {r.stdout.strip()!r}, not dexvantage {VERSION}')
        report['version_seconds'] = round(sec, 2)
        r, _ = _timed([cli, '-m', 'dexvantage', '--version'], env, tmp, 300, '-m dexvantage --version')
        if r.stdout.strip() != f'dexvantage {VERSION}':
            raise SystemExit(f'-m dexvantage --version printed {r.stdout.strip()!r}')
        med, first = startup_seconds(dist, env)
        report['startup_seconds'] = {'median_of_3': round(med, 2), 'first': round(first, 2)}
        print(f'start-up to the main window: {med:.2f} s (median of 3; first {first:.2f} s)', flush=True)

        cmd = [cli, '--sample', 'so100']
        for flag, name in SMOKE_STUDIES:
            cmd += [flag, os.path.join(tmp, name)]
        r, sec = _timed(cmd, env, tmp, 900, 'the batch run of three studies', ok_codes=(0, 2))
        for _, name in SMOKE_STUDIES:
            with open(os.path.join(tmp, name), encoding='utf-8') as fh:
                json.load(fh)
        report['batch_three_studies_seconds'] = round(sec, 1)
        print(r.stdout.strip()[-1500:], flush=True)

        # the cluster's local pool (spawned workers started through freeze_support)
        out = os.path.join(tmp, 'tol.json')
        r, sec = _timed([cli, '--sample', 'so100', '--tolerance-mc', out, '--jobs', '2'], env, tmp, 900,
                        'the tolerance Monte Carlo on 2 processes', ok_codes=(0, 2))
        with open(out, encoding='utf-8') as fh:
            json.load(fh)
        report['cluster_pool_jobs2_seconds'] = round(sec, 1)
        print(r.stdout.strip()[-800:], flush=True)

        # the designed experiments' own process pool (concurrent.futures, spawn)
        design = os.path.join(tmp, 'design.json')
        with open(design, 'w', encoding='utf-8') as fh:
            json.dump({'schema': 'dexvantage.experiment/1', 'name': 'bundle smoke test',
                       'factors': [{'path': 'sensor:Overhead D435.dpitch', 'lo': -5, 'hi': 5}],
                       'responses': ['coverage.covered'], 'objective': 'coverage.covered',
                       'design': {'method': 'lhs', 'n': 4, 'seed': 1}, 'execution': {'workers': 2}}, fh)
        r, sec = _timed([cli, '--sample', 'ur5e', '--experiment', design], env, tmp, 900,
                        'the designed experiment on 2 workers', ok_codes=(0, 2))
        with open(os.path.join(tmp, 'design.results.json'), encoding='utf-8') as fh:
            res = json.load(fh)
        mode = json.dumps(res).count('process pool of 2')
        if not mode:
            raise SystemExit('the designed experiment did not run in a process pool of 2')
        report['experiments_pool_workers2_seconds'] = round(sec, 1)
        print(r.stdout.strip()[-800:], flush=True)

        print('+', cli, '--self-test', flush=True)
        t0 = time.perf_counter()
        r = subprocess.run([cli, '--self-test'], env=env, cwd=tmp, capture_output=True, text=True, encoding='utf-8',
                           errors='replace', timeout=1800)
        report['self_test_seconds'] = round(time.perf_counter() - t0, 1)
        sys.stdout.write(r.stdout)
        if r.stderr.strip():
            print('stderr:\n' + r.stderr.strip()[-4000:], flush=True)
        bad = self_test_problems(r.stdout)
        if r.returncode or bad:
            raise SystemExit(f'self-test exit {r.returncode}; in a bundle every check must pass:\n  '
                             + '\n  '.join(bad))
        report['self_test'] = next((ln for ln in r.stdout.splitlines() if ln.startswith('self-test ')), '')

        if full:
            vdir = os.path.join(tmp, 'validation')
            r, sec = _timed([cli, '--validate', vdir, '--quick'], env, tmp, 7200, '--validate --quick',
                            ok_codes=(0, 1))
            print(r.stdout.strip()[-3000:], flush=True)
            report['validate_quick'] = {'exit': r.returncode, 'seconds': round(sec, 1),
                                        'last_line': (r.stdout.strip().splitlines() or [''])[-1]}
        report['bundle_mb'] = round(folder_mb(bundle_dir(dist)), 1)
        with open(os.path.join(dist, 'smoke-report.json'), 'w', encoding='utf-8') as fh:
            json.dump(report, fh, indent=2)
        print(json.dumps(report, indent=2), flush=True)
        print('smoke test passed', flush=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, 'rb') as fh:
        for block in iter(lambda: fh.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def archive(dist):
    tag = f'{NAME}-{VERSION}'
    if os.path.isdir(os.path.join(dist, NAME)):                 # DexVantage's own licence at the top of the folder
        shutil.copy2(os.path.join(ROOT, 'LICENSE'), os.path.join(dist, NAME, 'LICENSE.txt'))
    if sys.platform.startswith('linux'):
        folder = os.path.join(dist, NAME)
        for f in LINUX_FILES:
            shutil.copy2(os.path.join(HERE, 'linux', f), folder)
        for f in ('install.sh', 'uninstall.sh'):
            os.chmod(os.path.join(folder, f), 0o755)
        out = os.path.join(dist, f'{tag}-linux-{arch()}.tar.gz')
        with tarfile.open(out, 'w:gz') as tf:
            tf.add(folder, arcname=f'{tag}-linux-{arch()}')
        return [out]
    if os.name == 'nt':
        outs = [shutil.make_archive(os.path.join(dist, f'{tag}-windows-{arch()}'), 'zip', dist, NAME)]
        iscc = shutil.which('iscc') or next((p for p in (r'C:\Program Files (x86)\Inno Setup 6\ISCC.exe',
                                                          r'C:\Program Files\Inno Setup 6\ISCC.exe')
                                             if os.path.exists(p)), None)
        if iscc:
            run([iscc, f'/DAppVersion={VERSION}', f'/DBundleDir={os.path.join(dist, NAME)}',
                 f'/O{dist}', os.path.join(HERE, 'windows', 'dexvantage.iss')])
            outs.append(os.path.join(dist, f'{tag}-windows-x64-setup.exe'))
        else:
            print('Inno Setup (iscc) not found: no installer, only the zip', flush=True)
        return outs
    if sys.platform == 'darwin':
        out = os.path.join(dist, f'{tag}-macos-{arch()}.zip')
        # ditto keeps the app's symlinks, permissions and code signature (zip -r does not)
        run(['ditto', '-c', '-k', '--sequesterRsrc', '--keepParent', os.path.join(dist, f'{NAME}.app'), out])
        return [out]
    return []


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--dist', default=os.path.join(ROOT, 'out', 'dist'))
    ap.add_argument('--work', default=os.path.join(ROOT, 'out', 'build'))
    ap.add_argument('--smoke', action='store_true', help='run the bundled app after building')
    ap.add_argument('--full-smoke', action='store_true', help='--smoke and --validate --quick')
    ap.add_argument('--no-archive', action='store_true')
    ap.add_argument('--no-build', action='store_true', help='smoke-test and pack the bundle already in --dist')
    args = ap.parse_args()
    dist, work = os.path.abspath(args.dist), os.path.abspath(args.work)
    os.makedirs(work, exist_ok=True)
    if not args.no_build:
        build(dist, work)
    print(f'bundle: {bundle_dir(dist)}  {folder_mb(bundle_dir(dist)):.0f} MB', flush=True)
    leaks = personal_paths(bundle_dir(dist))
    if leaks:
        raise SystemExit('the bundle names the build machine\'s home folder in: ' + ', '.join(leaks))
    if args.smoke or args.full_smoke:
        smoke(dist, work, full=args.full_smoke)
    if not args.no_archive:
        for p in archive(dist):
            print(f'{os.path.basename(p)}  {os.path.getsize(p) / 1e6:.0f} MB  sha256 {sha256(p)}', flush=True)


if __name__ == '__main__':
    main()
