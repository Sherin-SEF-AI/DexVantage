# PyInstaller spec for DexVantage on Linux (x86_64 and arm64; DexVantage supports Linux only): one folder with two
# launchers that share one copy of Python, Qt, NumPy, MuJoCo, OpenCV, PyAV (FFmpeg) and pyarrow.
#
#   DexVantage              the window (the .desktop entry starts it)
#   dexvantage-cli          the same program for a terminal: batch mode, --self-test, --version, error output
#
#   pyinstaller packaging/dexvantage.spec        (or: python packaging/build.py)
#
# Ported from Vantage packaging/vantage.spec. Build from packaging/requirements-bundle.txt (it pins what the bundles
# carry and takes OpenCV without its own GUI, opencv-python-headless). What goes in, and why (docs/packaging.md):
#   - the dexvantage package: compiled into the archive AND as .py files (module_collection_mode pyz+py), so the
#     folder scans that find studies, pages, exporters, readers and checks (pkgutil over the package's folders), the
#     provenance code digest (SHA-256 over the .py files) and --reproduce work in a bundle as from source;
#   - every file of the package that is not Python (release.data_files): fonts, icons, the library JSON and its
#     schema, the project schema, the bundled robots and USD robots, JSON next to modules; LICENSE, CITATION.cff and
#     THIRD_PARTY.md at the top;
#   - MuJoCo: libmujoco and its extension modules (found by the analysis), its plugin libraries (mujoco/plugin, loaded
#     by mujoco/__init__.py from that folder), its OpenGL backends mujoco.egl, mujoco.osmesa and mujoco.glfw (chosen
#     by MUJOCO_GL at run time) with the PyOpenGL platform modules they load by name, and the glfw package's own
#     libglfw (DexVantage itself renders MuJoCo only through EGL). MuJoCo's wheel ships no EGL or OSMesa library:
#     those are the system's (a GPU driver or Mesa), loaded by PyOpenGL with ctypes, and never bundled;
#   - Qt: PyQt6's hook collects the platform plugins (xcb, wayland, offscreen), image formats,
#     icon engines (SVG) and TLS backends; the Qt modules DexVantage does not use are left out;
#   - PyAV with its FFmpeg libraries (av.libs) and pyarrow, through pyinstaller-hooks-contrib's hooks;
#   - the optional extras dexvantage/extras.py marks bundled (lerobot, bags, assistant) when installed; the others
#     (usd, rerun, step, research, tracking, pin) are excluded even when the build machine has them, and the app says
#     so in one sentence when a feature needs one (extras.bundle_note);
#   - validation references that are small (jsonschema, mcap, protobuf) when installed, so --validate runs the same
#     checks in a bundle.
import glob
import importlib.util
import os
import sys

from PyInstaller.utils.hooks import collect_data_files, collect_dynamic_libs, collect_submodules, copy_metadata

ROOT = os.path.dirname(SPECPATH)            # noqa: F821  (SPECPATH is set by PyInstaller)
HERE = os.path.join(ROOT, 'packaging')
if not sys.platform.startswith('linux'):
    raise SystemExit('dexvantage.spec: DexVantage supports Linux only, so its bundle is built on Linux.')
sys.path.insert(0, HERE)
import release                              # noqa: E402  (packaging/release.py)

VERSION = release.version(ROOT)
EXTRAS = release.extras_table(ROOT)


def have(mod):
    try:
        return importlib.util.find_spec(mod) is not None
    except (ImportError, ValueError):
        return False


# every module of the app, whether imported statically or by name (studies, pages, exporters, readers, checks, the
# assistant, the MCP server, the batch mode)
hiddenimports = collect_submodules('dexvantage')
# MuJoCo picks its OpenGL backend by MUJOCO_GL when mujoco.rendering.classic.gl_context is first imported
hiddenimports += ['mujoco.rendering.classic.gl_context', 'mujoco.rendering.classic.renderer', 'mujoco.egl',
                  'mujoco.egl.egl_ext', 'mujoco.osmesa', 'mujoco.glfw', 'glfw', 'glfw.library']
# PyOpenGL picks its platform module (EGL, GLX, OSMesa) and array handlers by name at run time
for mod in ('OpenGL.platform', 'OpenGL.arrays', 'OpenGL.raw.EGL', 'OpenGL.raw.GLX', 'OpenGL.raw.osmesa', 'OpenGL.EGL',
            'OpenGL.osmesa'):
    hiddenimports += collect_submodules(mod, on_error='ignore')
hiddenimports += ['OpenGL.EGL', 'OpenGL.GLX', 'OpenGL.osmesa']

# the optional extras the bundle carries (when the build environment has them) and the small validation references
REFERENCES = {'jsonschema': 'library entries checked against their JSON Schema; validation references',
              'mcap': 'validation reference for the ROS 2 bag reader', 'google.protobuf': 'validation reference '
              '(RLDS, MCAP protobuf)', 'certifi': "the assistant's HTTPS certificates"}
carried, missing, left_out = [], [], []
for name, e in EXTRAS.items():
    for mod in e.modules:
        (carried if e.bundled and have(mod) else missing if e.bundled else left_out).append((mod, name))
for mod in REFERENCES:
    (carried if have(mod) else missing).append((mod, 'reference'))
hiddenimports += [m for m, _ in carried] + (['lz4.frame'] if have('lz4') else [])
# the Claude provider imports the SDK only when it is chosen, and the SDK loads parts of itself lazily: collect all of
# it (and its HTTP client and models) so the bundle never meets a missing module on a user's machine
if have('anthropic'):
    for mod in ('anthropic', 'httpx2', 'httpcore2', 'httpx', 'httpcore', 'pydantic', 'pydantic_core', 'jiter',
                'anyio'):
        hiddenimports += collect_submodules(mod, on_error='ignore')
print('dexvantage.spec: bundled optional modules: ' + (', '.join(f'{m} ({n})' for m, n in carried) or 'none')
      + ''.join(f'; MISSING {m} ({n}: not installed here)' for m, n in missing)
      + '; left out by decision: ' + ', '.join(sorted({n for _, n in left_out})))

excludes = ['tkinter', 'matplotlib', 'IPython', 'pytest', '_pytest', 'ruff', 'setuptools', 'pip',
            'PyQt6.QtWebEngineCore', 'PyQt6.QtWebEngineWidgets', 'PyQt6.QtQml', 'PyQt6.QtQuick', 'PyQt6.QtMultimedia',
            'PyQt6.QtBluetooth', 'PyQt6.QtSql', 'PyQt6.QtDesigner', 'PyQt6.QtPdf', 'PyQt6.QtPdfWidgets',
            'PyQt6.QtTest', 'PyQt6.Qt3DCore', 'PyQt6.QtCharts', 'PyQt6.QtDataVisualization', 'PyQt6.QtPositioning',
            'PyQt6.QtRemoteObjects', 'PyQt6.QtSensors', 'PyQt6.QtSerialPort', 'PyQt6.QtSpatialAudio',
            'PyQt6.QtTextToSpeech', 'PyQt6.QtWebChannel', 'PyQt6.QtWebSockets', 'PyQt6.QtNfc', 'PyQt6.QtHelp',
            # other Qt bindings (PyInstaller refuses to bundle two) and big packages an optional import elsewhere
            # would pull in from a well-stocked build machine
            'PyQt5', 'PySide2', 'PySide6', 'torch', 'tensorflow', 'jax', 'jaxlib', 'cupy', 'dask', 'sparse', 'numba',
            'llvmlite', 'sklearn', 'sympy', 'onnxruntime', 'gi', 'notebook', 'jupyter_client', 'lerobot', 'curobo',
            'fiftyone', 'fast_simplification',
            # gsplat (the splat views' optional fast back end, scripts/build_gsplat.sh): its CUDA library links
            # PyTorch's, and the splat extra (PyTorch) is left out of the bundle (dexvantage/extras.py)
            'gsplat',
            # scikit-image: only the SSIM reference check imports it (the test extra; requirements-bundle.txt does not
            # install it), so a development environment would otherwise give the bundle a fragment of it
            'skimage',
            # MuJoCo's parts DexVantage does not use: the interactive viewer, the experimental studio (41 MB), the
            # Filament renderer, system identification, MJX
            'mujoco.viewer', 'mujoco._simulate', 'mujoco.experimental', 'mujoco.rendering.filament', 'mujoco.mjrf',
            'mujoco._render_filament', 'mujoco.sysid', 'mujoco.mjx', 'mujoco.minimize',
            # zstandard's cffi backend (11 MB) is for PyPy; CPython uses its C backend
            'zstandard.backend_cffi', 'zstandard._cffi',
            # pyarrow's parts DexVantage does not use (it reads and writes local Parquet only): Flight RPC, the S3,
            # GCS, Azure and HDFS filesystems, ORC, Substrait, CUDA; and fsspec, which pyarrow imports only if present
            'pyarrow.flight', 'pyarrow._flight', 'pyarrow._s3fs', 'pyarrow._gcsfs', 'pyarrow._azurefs',
            'pyarrow._hdfs', 'pyarrow.orc', 'pyarrow._orc', 'pyarrow._dataset_orc', 'pyarrow.substrait',
            'pyarrow._substrait', 'pyarrow.cuda', 'pyarrow._cuda', 'pyarrow.conftest', 'fsspec',
            # Pillow: only PyAV's VideoFrame.to_image / from_image import it, and DexVantage uses arrays (to_ndarray)
            'PIL',
            # optional integrations of the anthropic SDK that collect_submodules('anthropic') would pull in from a
            # build machine that has them: Google Vertex credentials, the OpenTelemetry SDK and exporters (the SDK
            # needs only opentelemetry-api), and what they bring (requests, pyasn1, click, dotenv, pygments)
            'google.auth', 'google.oauth2', 'opentelemetry.sdk', 'opentelemetry.exporter', 'opentelemetry.proto',
            'requests', 'pyasn1', 'pyasn1_modules', 'click', 'dotenv', 'pygments', 'psutil']
excludes += sorted({m for m, _ in left_out})     # the extras left out by decision, even when installed here

binaries = []
# MuJoCo's plugin libraries: mujoco/__init__.py loads every library in its plugin folder
import mujoco                               # noqa: E402
_plug = os.path.join(os.path.dirname(mujoco.__file__), 'plugin')
binaries += [(p, 'mujoco/plugin') for p in sorted(glob.glob(os.path.join(_plug, '*')))
             if p.endswith(('.so', '.dll', '.dylib'))]
binaries += collect_dynamic_libs('glfw')    # libglfw (x11 and wayland builds on Linux, glfw3.dll, libglfw.3.dylib)

datas = release.data_files(ROOT)
# SciPy's Sobol direction numbers (scipy.stats.qmc.Sobol loads the .npz from its folder; the SciPy hook leaves it out):
# the designed experiments' Sobol designs and indices need it
datas += collect_data_files('scipy', includes=['**/*.npz'], excludes=['**/tests/**'])
# the versions the provenance block, the publication kit and its check read with importlib.metadata
for dist in ('mujoco', 'numpy', 'PyQt6', 'PyQt6-Qt6'):
    datas += copy_metadata(dist)

a = Analysis(                               # noqa: F821
    [os.path.join(HERE, 'run_dexvantage.py')],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=excludes,
    module_collection_mode={'dexvantage': 'pyz+py'},
    noarchive=False,
)

# every module of the package must be in: the studies, pages, exporters and checks are imported by name, so check
# here rather than finding out on a user's machine
_collected = {name for name, _, _ in a.pure}
_expected = []
for d, subdirs, files in os.walk(os.path.join(ROOT, 'dexvantage')):
    subdirs[:] = [s for s in subdirs if s != '__pycache__']
    for f in files:
        if f.endswith('.py'):
            rel = os.path.relpath(os.path.join(d, f), ROOT)[:-3].replace(os.sep, '.')
            _expected.append(rel[:-len('.__init__')] if rel.endswith('.__init__') else rel)
_missing = sorted(m for m in _expected if m not in _collected)
if _missing:
    raise SystemExit('dexvantage.spec: modules missing from the bundle: ' + ', '.join(_missing))
_present = sorted({n.split('.')[0] for n, _, _ in a.pure} & {m for m, _ in left_out})
if _present:
    raise SystemExit('dexvantage.spec: extras left out by decision were collected anyway: ' + ', '.join(_present))


def _dest(entry):
    return entry[0].replace('\\', '/')


# Qt's translations of its own dialogs (DexVantage's interface is English and loads none); OpenCV's own copy of Qt 5's
# plugins and fonts when the build environment has opencv-python rather than -headless (its highgui is never used)
a.datas = [d for d in a.datas if not _dest(d).startswith(('PyQt6/Qt6/translations/', 'cv2/qt/'))]
# DexVantage's own install metadata (a development build's editable install records the checkout's path in
# direct_url.json; the bundle needs none of it), any other direct_url.json, and pyarrow's C++ headers and Cython
# sources (for building extensions against pyarrow, not for running it)
a.datas = [d for d in a.datas
           if not (_dest(d).split('/')[0].startswith('dexvantage-') and _dest(d).split('/')[0].endswith('.dist-info'))
           and not _dest(d).endswith('/direct_url.json')
           and not _dest(d).startswith(('pyarrow/include/', 'pyarrow/includes/', 'pyarrow/tests/'))
           and not (_dest(d).startswith('pyarrow/') and _dest(d).endswith(('.pyx', '.pxd', '.pxi', '.h', '.cc')))]
a.binaries = [b for b in a.binaries if not _dest(b).startswith('cv2/qt/')]

# the licences of everything bundled, and a notice listing it (licenses/THIRD-PARTY.txt)
from PyInstaller.config import CONF                     # noqa: E402
_top = {name.split('.')[0] for name, _, _ in a.pure}
_top |= {b[0].replace('\\', '/').split('/')[0].split('.')[0] for b in a.binaries if b[2] == 'EXTENSION'}
a.datas += [(os.path.join(dest, os.path.basename(src)), src, 'DATA')
            for src, dest in release.license_files(_top, os.path.join(CONF['workpath'], 'THIRD-PARTY.txt'))]


def drop_unused_libraries(binaries, plugins):
    """Leave out the Qt plugins given (dest path suffixes) and the libraries only they need. Linux only: the GTK 3
    platform theme pulls in ~35 libraries (GTK, GLib/GIO, cairo, pango, ...) from the build machine, and a bundled
    GTK loads the user's own GTK modules and themes, a known source of crashes on other distributions (DexVantage
    styles itself; file dialogs still go through the desktop portal); the PDF image-format plugin pulls in QtPdf."""
    from PyInstaller.depend.bindepend import get_imports
    gone = [b for b in binaries if _dest(b).endswith(plugins)]
    if not gone:
        return binaries
    by_name = {os.path.basename(b[0]): b for b in binaries}

    def reach(entries):
        seen, todo = set(), list(entries)
        while todo:
            e = todo.pop()
            if e[0] in seen:
                continue
            seen.add(e[0])
            todo += [by_name[n] for n, _ in get_imports(e[1]) if n in by_name]
        return seen
    # everything loaded on its own (Python extensions, Qt plugins, MuJoCo plugins: entries in subfolders, but not
    # Qt's own libraries, which are only loaded as something's dependency) except those
    roots = [b for b in binaries if '/' in _dest(b) and not _dest(b).startswith('PyQt6/Qt6/lib/') and b not in gone]
    drop = (reach(gone) - reach(roots)) | {g[0] for g in gone}
    print(f'dexvantage.spec: leaving out {", ".join(os.path.basename(_dest(g)) for g in gone)} and '
          f'{len(drop) - len(gone)} libraries only they need')
    return [b for b in binaries if b[0] not in drop]


def drop_unreached(binaries, folder):
    """Leave out the shared libraries in folder (a package's own, collected wholesale by its hook) that no extension
    module, plugin or other kept library loads: pyarrow's hook collects every libarrow_* although the Flight, S3 and
    Substrait libraries are loaded only by the extension modules excluded above."""
    from PyInstaller.depend.bindepend import get_imports
    by_name = {os.path.basename(b[0]): b for b in binaries}
    ext = {b[0] for b in binaries if b[2] == 'EXTENSION'}
    cand = {b[0] for b in binaries if _dest(b).startswith(folder) and b[0] not in ext
            and os.path.basename(_dest(b)).startswith('lib')
            and not any(t in os.path.basename(_dest(b)) for t in ('.cpython-', '.abi3.', '.pyd'))}
    if not cand:
        return binaries
    seen, todo = set(), [b for b in binaries if b[0] not in cand]
    while todo:
        e = todo.pop()
        if e[0] in seen:
            continue
        seen.add(e[0])
        try:
            todo += [by_name[os.path.basename(n)] for n, _ in get_imports(e[1]) if os.path.basename(n) in by_name]
        except Exception:                   # noqa: BLE001 — a file binary analysis cannot read is kept, not followed
            pass
    gone = cand - seen
    if gone:
        print(f'dexvantage.spec: leaving out {len(gone)} unreached libraries of {folder}: '
              + ', '.join(sorted(os.path.basename(g) for g in gone)))
    return [b for b in binaries if b[0] not in gone]


a.binaries = drop_unused_libraries(a.binaries, ('platformthemes/libqgtk3.so', 'imageformats/libqpdf.so'))
a.binaries = drop_unreached(a.binaries, 'pyarrow/')


def drop_dangling_links(entries, others):
    """Leave out the SYMLINK entries whose target was left out above (PyInstaller links each library of a package
    folder from the top folder; a link to a dropped library would dangle)."""
    kept = {_dest(e) for e in list(entries) + list(others) if e[2] != 'SYMLINK'}
    links = {_dest(e): e for e in entries if e[2] == 'SYMLINK'}

    def target(e):
        return os.path.normpath(os.path.join(os.path.dirname(_dest(e)), e[1].replace('\\', '/'))).replace('\\', '/')
    good, changed = set(kept), True
    while changed:                          # a link to a kept link is kept
        changed = False
        for d, e in links.items():
            if d not in good and target(e) in good:
                good.add(d)
                changed = True
    gone = sorted(d for d in links if d not in good)
    if gone:
        print('dexvantage.spec: leaving out links to libraries left out: ' + ', '.join(gone))
    return [e for e in entries if not (e[2] == 'SYMLINK' and _dest(e) in gone)]


a.binaries = drop_dangling_links(a.binaries, a.datas)
a.datas = drop_dangling_links(a.datas, a.binaries)
pyz = PYZ(a.pure)                           # noqa: F821

common = dict(exclude_binaries=True, debug=False, strip=False, upx=False, disable_windowed_traceback=False)
# Python's UTF-8 mode: output to a pipe or a file (batch mode, the self-test on CI) is UTF-8 whatever the locale (a
# C or POSIX locale would otherwise make it ASCII, which cannot encode the arrows and symbols the reports print)
options = [('X utf8', None, 'OPTION')]
gui = EXE(pyz, a.scripts, options, name='DexVantage', console=False, **common)          # noqa: F821
cli = EXE(pyz, a.scripts, options, name='dexvantage-cli', console=True, **common)       # noqa: F821
coll = COLLECT(gui, cli, a.binaries, a.datas, strip=False, upx=False, name='DexVantage')   # noqa: F821

