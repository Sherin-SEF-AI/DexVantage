"""Checks of the materials (core/materials.py) and lights (core/lights.py) the lighting study (core/studies/lighting.py)
is built on, each against a reference written out again here.

lighting.ggx_normalised — the GGX distribution's projected area: ∫ D(h) cos θ_h dω_h = 1 for every roughness, by
adaptive quadrature over θ_h (scipy.integrate.quad), not by DexVantage's sampler.

lighting.white_furnace — the weak white-furnace test of a microfacet model (Heitz 2014, §6): with F = 1 and the masking
of the outgoing direction only, ∫ G1(ωo, h) D(h) / (4 |n · ωo|) dωi = 1 for every ωo and roughness. With dωi =
4 |ωo · h| dωh this is G1(ωo) ∫ max(0, ωo · h) D(h) dωh / cos θo = 1, evaluated by adaptive quadrature over θ_h
(scipy.integrate.quad) with the φ integral of max(0, A cos φ + B) written out in closed form; tolerance 10⁻⁶ (the
quadrature resolves it to about 10⁻⁹). Monte Carlo with microfacet normals drawn from D(h) cos θ_h, the reflected ωi,
is reported beside it and must lie within four standard errors (it tests the sampler; alone, its band of up to 0.024
at grazing view would let a 2 % error through, which is why the quadrature is the measured figure).

lighting.energy — energy conservation of the full BRDF of every material tag: the directional albedo
ρ(ωo) = ∫ f_r cos θ_i dωi ≤ 1 at incidence angles 0–89°, by Monte Carlo with multiple importance sampling
(cosine-weighted and GGX-normal samples, the balance heuristic; Veach 1997, §9.2). The error is how far the largest
albedo exceeds 1 (0 when it does not), less three standard errors.

lighting.reciprocity — Helmholtz reciprocity f_r(ωi, ωo) = f_r(ωo, ωi) on random direction pairs for every tag.

lighting.peak_closed_form — the BRDF at normal incidence and normal view against its closed form written out again:
D = 1 / (π α²), G = 1, F = F0 = ((n − 1)² + k²) / ((n + 1)² + k²), so f_s = F0 / (4π α²), plus the coupled diffuse
term (28 ρ / 23π)(1 − F0)(1 − 2⁻⁵)².

lighting.schlick_vs_fresnel — Schlick's Fresnel approximation with F0 from the index against the exact Fresnel
equations (Hecht, Optics, §4.6.2) for the dielectrics (PMMA and BK7), over the incidence angles; the aluminium
figure is reported, not tested (Schlick is poorer for metals, a known limit).

lighting.inverse_square — the point-light law E = I / d² (I = L A for a small Lambertian square of side a facing the
point) against Monte Carlo integration of ∫ L cos θ_e cos θ_r / r² dA over the square; at d ≥ 10 a the two must agree
within 0.5 % (the far-field error is about (a / d)² / 3). The series over d / a shows the far-field convergence.

lighting.area_vs_closed_form — an area light split into patches (core/lights.emitters, 4 × 4 by default) against the
closed-form irradiance of a uniform Lambertian rectangle on a parallel receiving element (the corner configuration
factor F = (1 / 2π)[X / √(1 + X²) atan(Y / √(1 + X²)) + Y / √(1 + Y²) atan(X / √(1 + Y²))], X = a / d, Y = b / d,
summed with signs over the four corner rectangles; E = π L F; Howell's catalogue of configuration factors), and that
closed form against Monte Carlo, at receivers half the panel's size away and further, on and off its axis.
"""
from __future__ import annotations

import math

import numpy as np

from .suite import Outcome, check, figure, series

ALPHAS = (0.02, 0.05, 0.1, 0.2, 0.5, 0.9)
THETAS_DEG = (0.0, 30.0, 60.0, 80.0, 89.0)


def _frame_dirs(theta: float) -> np.ndarray:
    return np.array([math.sin(theta), 0.0, math.cos(theta)])


@check('lighting.ggx_normalised', 'lighting', 'GGX normal distribution integrates to one (projected area)',
       'largest |∫ D(h) cos θ_h dω_h − 1| over α = 0.02 … 0.9',
       'adaptive quadrature over θ_h (scipy.integrate.quad), D written out from Walter et al. 2007 eq. 33', 1e-6,
       'absolute', refs=('walter2007', 'scipy_quad'))
def _ggx_normalised(ctx):
    from scipy.integrate import quad

    from ..core import materials as MT
    worst, rows = 0.0, []
    for a in ALPHAS:
        # substitution t = tan θ spreads the narrow lobe; the integrand is D cos θ sin θ 2π dθ
        val, _err = quad(lambda th: 2 * math.pi * float(MT.ggx_d(math.cos(th), a)) * math.cos(th) * math.sin(th),
                         0.0, math.pi / 2, points=[math.atan(a) * k for k in (0.5, 1, 2, 4)], limit=400,
                         epsabs=1e-12, epsrel=1e-12)
        worst = max(worst, abs(val - 1.0))
        rows.append(val)
    return Outcome(worst, {'alphas': list(ALPHAS), 'integrals': [round(v, 12) for v in rows]},
                   note='Quadrature to 1e-12; the integral is exactly 1 for the GGX distribution, so 1e-6 leaves room '
                        'only for the quadrature itself.')


@check('lighting.white_furnace', 'lighting', 'Weak white furnace of the GGX microfacet BRDF (F = 1, G1 of the view)',
       'largest |∫ G1(ωo) D / (4 |n·ωo|) dωi − 1| over α = 0.02 … 0.9 and θo = 0 … 89°, by adaptive quadrature; the '
       'Monte Carlo estimate with h ~ D cos θ_h must also lie within four standard errors of 1',
       'the exact value 1 (Heitz 2014, the weak white furnace test): scipy.integrate.quad over θ_h with the φ integral '
       'in closed form', 1e-6,
       'absolute', refs=('heitz2014', 'walter2007', 'scipy_quad'))
def _white_furnace(ctx):
    from ..core import materials as MT
    rng = ctx.rng('lighting.white_furnace')
    N = 50_000 if ctx.quick else 400_000
    worst, rows, errs = 0.0, [], []
    for a in ALPHAS:
        out = []
        for td in THETAS_DEG:
            wo = _frame_dirs(math.radians(td))
            h = MT.sample_ggx_h(a, rng.random(N), rng.random(N))
            oh = h @ wo
            co = wo[2]
            # h drawn with pdf D cos θ_h: the estimator of ∫ G1 D / (4 |n·ωo|) dωi (over every reflected ωi, the
            # weak furnace's domain) is G1(ωo, h) (ωo·h) / (|n·ωo| |n·h|)
            g1 = MT.smith_g1(co, a, oh)
            wall = np.where(oh > 0, g1 * oh / (co * h[:, 2]), 0.0)
            est = float(wall.mean())
            se = float(wall.std() / math.sqrt(N))
            out.append(est)
            errs.append(se)
            worst = max(worst, abs(est - 1.0))
        rows.append(out)
    band = 4 * max(errs)
    quad = [[_furnace_quad(MT, a, math.radians(td)) for td in THETAS_DEG] for a in ALPHAS]
    qworst = max(abs(v - 1.0) for r in quad for v in r)
    return Outcome(qworst, {'alphas': list(ALPHAS), 'theta_deg': list(THETAS_DEG), 'samples': N,
                            'quadrature': [[round(v, 12) for v in r] for r in quad],
                            'monte_carlo_estimates': [[round(v, 5) for v in r] for r in rows],
                            'monte_carlo_largest_error': round(worst, 6),
                            'monte_carlo_band': round(band, 6),
                            'largest_standard_error': round(max(errs), 6)}, ok=worst <= band,
                   figure=figure('Weak white furnace', 'view angle θo (°)', '∫ G1 D / (4 cos θo) dωi',
                                 [series(f'α = {a:g}', THETAS_DEG, r) for a, r in zip(ALPHAS, rows)], hline=1.0,
                                 hline_label='exact'),
                   note='The measured figure is the quadrature (exact to about 10⁻⁹). The Monte Carlo estimator, '
                        'shown in the figure, grows as 1 / cos θo at grazing view on rough surfaces, so its standard '
                        'error is largest at 89° and α 0.9; it must lie within four standard errors of 1.')


def _furnace_quad(MT, alpha: float, tho: float) -> float:
    """G1(ωo) ∫ max(0, ωo·h) D(h) dωh / cos θo by quadrature over θ_h; the φ integral of max(0, A cos φ + B), A =
    sin θo sin θ_h, B = cos θo cos θ_h, is 2πB when A ≤ |B| (and B > 0), else 2(A sin φ0 + B φ0), φ0 = acos(−B / A)."""
    from scipy import integrate

    so, co = math.sin(tho), math.cos(tho)

    def inner(th):
        A, B = so * math.sin(th), co * math.cos(th)
        if A <= abs(B):
            v = 2 * math.pi * B if B > 0 else 0.0
        else:
            p0 = math.acos(-B / A)
            v = 2 * (A * math.sin(p0) + B * p0)
        return v * float(MT.ggx_d(math.cos(th), alpha)) * math.sin(th)

    val, _err = integrate.quad(inner, 0.0, math.pi / 2, limit=500, points=[math.atan(alpha)], epsabs=1e-13,
                               epsrel=1e-11)
    return float(MT.smith_g1(co, alpha)) * val / co


def _albedo(MT, mat, wo, N, rng):
    """(directional albedo, standard error) of mat at view wo by MIS over cosine and GGX-normal samples."""
    n = np.array([0.0, 0.0, 1.0])
    half = N // 2
    u = rng.random(half)
    p = 2 * np.pi * rng.random(half)
    r = np.sqrt(u)
    wc = np.stack([r * np.cos(p), r * np.sin(p), np.sqrt(np.maximum(1 - u, 0))], 1)
    h = MT.sample_ggx_h(mat.alpha, rng.random(half), rng.random(half))
    oh = h @ wo
    wg = 2 * oh[:, None] * h - wo
    W = np.concatenate([wc, wg])

    def pdf(wi):
        hh = wi + wo
        hh /= np.maximum(np.linalg.norm(hh, axis=1, keepdims=True), 1e-300)
        pc = np.maximum(wi[:, 2], 0) / np.pi
        ohh = np.abs(hh @ wo)
        pg = MT.ggx_d(hh[:, 2], mat.alpha) * np.maximum(hh[:, 2], 0) / np.maximum(4 * ohh, 1e-300)
        return 0.5 * pc + 0.5 * pg
    f = MT.brdf(np.broadcast_to(n, W.shape), W, np.broadcast_to(wo, W.shape), mat)
    pw = pdf(W)
    val = np.where((W[:, 2] > 0) & (pw > 0), f * np.maximum(W[:, 2], 0) / np.maximum(pw, 1e-300), 0.0)
    return float(val.mean()), float(val.std() / math.sqrt(len(val)))


@check('lighting.energy', 'lighting', 'Energy conservation of every material\'s BRDF (directional albedo ≤ 1)',
       'how far the largest directional albedo ∫ f_r cos θ_i dωi, less three standard errors, exceeds 1, over the '
       'material tags and θo = 0 … 89°',
       'the bound 1 (a passive surface reflects no more than arrives), Monte Carlo with multiple importance sampling',
       0.0, 'albedo', refs=('veach1997', 'walter2007', 'ashikhmin2000'))
def _energy(ctx):
    from ..core import materials as MT
    rng = ctx.rng('lighting.energy')
    N = 100_000 if ctx.quick else 600_000
    worst, rows, top = 0.0, {}, (0.0, '', 0.0)
    for tag, mat in MT.MATERIALS.items():
        out = []
        for td in THETAS_DEG:
            a, se = _albedo(MT, mat, _frame_dirs(math.radians(td)), N, rng)
            out.append(a)
            worst = max(worst, a - 3 * se - 1.0)
            if a > top[0]:
                top = (a, tag, td)
        rows[tag] = out
    return Outcome(max(worst, 0.0), {'albedo': {k: [round(v, 4) for v in r] for k, r in rows.items()},
                                     'theta_deg': list(THETAS_DEG), 'samples': N,
                                     'largest': {'albedo': round(top[0], 4), 'tag': top[1], 'theta_deg': top[2]}},
                   figure=figure('Directional albedo of each material', 'view angle θo (°)', 'albedo',
                                 [series(k, THETAS_DEG, r) for k, r in rows.items()], hline=1.0,
                                 hline_label='energy bound'),
                   note='The tolerance is 0: any albedo above 1 by more than three standard errors fails.')


@check('lighting.reciprocity', 'lighting', 'Helmholtz reciprocity of the BRDF',
       'largest |f_r(ωi, ωo) − f_r(ωo, ωi)| / max f_r over 20 000 random pairs per material tag',
       'the identity f_r(ωi, ωo) = f_r(ωo, ωi) (Helmholtz reciprocity)', 1e-12, 'relative',
       refs=('walter2007', 'ashikhmin2000'))
def _reciprocity(ctx):
    from ..core import materials as MT
    rng = ctx.rng('lighting.reciprocity')
    N = 20_000

    def hemi(k):
        v = rng.normal(size=(k, 3))
        v[:, 2] = np.abs(v[:, 2])
        return v / np.linalg.norm(v, axis=1, keepdims=True)
    worst = 0.0
    n = np.broadcast_to([0.0, 0.0, 1.0], (N, 3))
    for mat in MT.MATERIALS.values():
        a, b = hemi(N), hemi(N)
        f1, f2 = MT.brdf(n, a, b, mat), MT.brdf(n, b, a, mat)
        worst = max(worst, float(np.max(np.abs(f1 - f2)) / max(float(np.max(f1)), 1e-300)))
    return Outcome(worst, {'pairs_per_tag': N, 'tags': list(MT.MATERIALS)},
                   note='Both terms are symmetric by construction; the tolerance allows rounding only.')


@check('lighting.peak_closed_form', 'lighting', 'Specular peak at normal incidence against its closed form',
       'largest relative difference between f_r(n, n) and F0 / (4π α²) + (28 ρ / 23π)(1 − F0)(1 − 2⁻⁵)², over the '
       'material tags and α = 0.02 … 0.9',
       'the closed form written out again, F0 from the index by ((n − 1)² + k²) / ((n + 1)² + k²)', 1e-9, 'relative',
       refs=('walter2007', 'schlick1994', 'ashikhmin2000', 'hecht2017'))
def _peak(ctx):
    from dataclasses import replace

    from ..core import materials as MT
    worst, rows = 0.0, []
    n = np.array([[0.0, 0.0, 1.0]])
    for tag, mat in MT.MATERIALS.items():
        for a in ALPHAS:
            m = replace(mat, alpha=a)
            ix = MT.INDEX[m.index]['vis']
            f0 = ((ix.real - 1) ** 2 + ix.imag ** 2) / ((ix.real + 1) ** 2 + ix.imag ** 2)
            ref = f0 / (4 * math.pi * a * a) + 28 * m.albedo / (23 * math.pi) * (1 - f0) * (1 - 0.5 ** 5) ** 2
            got = float(MT.brdf(n, n, n, m)[0])
            worst = max(worst, abs(got - ref) / ref)
            rows.append((tag, a, got))
    return Outcome(worst, {'cases': len(rows), 'example': {'tag': rows[0][0], 'alpha': rows[0][1],
                                                           'f_r_per_sr': rows[0][2]}},
                   note='Exact arithmetic but for rounding (the GGX D at cos θ_h = 1 is 1 / (π α²)).')


@check('lighting.schlick_vs_fresnel', 'lighting', 'Schlick\'s Fresnel against the exact Fresnel equations',
       'largest |F_Schlick(θ) − (R_s + R_p) / 2| over θ = 0 … 90° for the dielectrics (PMMA, BK7); aluminium '
       'reported',
       'the Fresnel equations for a smooth interface (Hecht, Optics, §4.6.2), complex index', 0.04, 'reflectance',
       refs=('schlick1994', 'hecht2017', 'sultanova2009', 'schott_bk7', 'mcpeak2015'))
def _schlick(ctx):
    from ..core import materials as MT
    th = np.linspace(0, math.pi / 2, 1801)
    c = np.cos(th)
    out, curves = {}, []
    for key in ('pmma', 'bk7', 'aluminium'):
        nn = MT.INDEX[key]['vis']
        rs, rp = MT.fresnel_exact(c, nn)
        ex = (rs + rp) / 2
        sc = MT.schlick_f(c, MT.f0_from_index(nn))
        out[key] = float(np.max(np.abs(ex - sc)))
        curves.append(series(f'{key}: Schlick − exact', np.degrees(th[::30]), (sc - ex)[::30]))
    err = max(out['pmma'], out['bk7'])
    return Outcome(err, {k: round(v, 5) for k, v in out.items()},
                   figure=figure('Schlick against exact Fresnel (550 nm)', 'incidence angle (°)', 'difference',
                                 curves),
                   note='Schlick\'s approximation is exact at 0° and 90° and largest in between near grazing; for '
                        'n ≈ 1.5 it stays below 0.04 (measured here), which the tolerance states. For aluminium the '
                        f'difference reaches {out["aluminium"]:.3f}, a known limit of the model (docs).')


def _mc_square(L, a, d, offset, N, rng):
    X = (rng.random(N) - 0.5) * a
    Y = (rng.random(N) - 0.5) * a
    r = np.stack([X - offset[0], Y - offset[1], np.full(N, -d)], 1)
    r2 = np.einsum('ij,ij->i', r, r)
    c = d / np.sqrt(r2)
    v = L * c * c / r2 * a * a
    return float(v.mean()), float(v.std() / math.sqrt(N))


@check('lighting.inverse_square', 'lighting', 'Point-light inverse-square law against Monte Carlo over an area light',
       'relative difference between I / d² (I = L a²) and the Monte Carlo irradiance of a Lambertian square of side '
       'a, on its axis, at d ≥ 10 a (largest)',
       'Monte Carlo integration of ∫ L cos θ_e cos θ_r / r² dA over the square', 0.005, 'relative',
       refs=('mcluney2014',))
def _inverse_square(ctx):
    from ..core import lights as LT
    rng = ctx.rng('lighting.inverse_square')
    N = 200_000 if ctx.quick else 2_000_000
    a, L = 0.1, 1000.0
    ratios = (1.0, 2.0, 3.0, 5.0, 10.0, 20.0)
    rows, worst = [], 0.0
    for k in ratios:
        d = k * a
        pt = LT.clean_light({'kind': 'point', 'pose': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1]},
                             'luminous_flux_lm': {'value': 4 * math.pi * L * a * a, 'source': 'check'}})
        em = LT.emitters(pt, 'vis')
        _wi, E = LT.irradiance(em, np.array([[0.0, 0.0, d]]))
        e_pt = float(E[0, 0])
        e_mc, _se = _mc_square(L, a, d, (0.0, 0.0), N, rng)
        rel = abs(e_pt - e_mc) / e_mc
        rows.append(rel)
        if k >= 10:
            worst = max(worst, rel)
    return Outcome(worst, {'d_over_a': list(ratios), 'relative_difference': [round(r, 6) for r in rows],
                           'samples': N},
                   figure=figure('Point light against an area light', 'distance / source size', 'relative difference',
                                 [series('|I/d² − MC| / MC', ratios, rows, 'points')], logy=True, hline=0.005,
                                 hline_label='tolerance'),
                   note='The point law is the far-field limit: expanding the closed form of the square, its relative '
                        'error is about (a / d)² / 3, 1.3 % at d = 5 a and 0.33 % at d = 10 a, so 0.5 % from ten '
                        'source sizes on states it; the Monte Carlo error at this sample count is below 10⁻⁴.')


def _corner(X, Y):
    return (X / np.sqrt(1 + X * X) * np.arctan(Y / np.sqrt(1 + X * X))
            + Y / np.sqrt(1 + Y * Y) * np.arctan(X / np.sqrt(1 + Y * Y))) / (2 * math.pi)


def rect_irradiance(L, w, h, dx, dy, d) -> float:
    """Closed-form irradiance on an element parallel to a uniform Lambertian w × h rectangle of luminance L, at
    distance d, its foot offset (dx, dy) from the rectangle's centre (four signed corner rectangles)."""
    def G(x, y):
        return math.copysign(1, x) * math.copysign(1, y) * _corner(abs(x) / d, abs(y) / d)
    x1, x2, y1, y2 = -w / 2 - dx, w / 2 - dx, -h / 2 - dy, h / 2 - dy
    return math.pi * L * (G(x2, y2) - G(x1, y2) - G(x2, y1) + G(x1, y1))


@check('lighting.area_vs_closed_form', 'lighting', 'Area light split into patches against the closed form',
       'largest relative difference between the 4 × 4-patch area light (core/lights) and the closed-form irradiance '
       'of a 0.6 m square panel, receivers 0.3–2 m away and up to 0.5 m off its axis; and the closed form against '
       'Monte Carlo',
       'the configuration factor of a rectangle to a parallel element at its corner, summed over four corner '
       'rectangles (Howell\'s catalogue), and Monte Carlo', 0.025, 'relative', refs=('howell_catalog', 'mcluney2014'))
def _area(ctx):
    from ..core import lights as LT
    rng = ctx.rng('lighting.area_vs_closed_form')
    N = 200_000 if ctx.quick else 1_000_000
    w = h = 0.6
    L = 1000.0
    lt = LT.clean_light({'kind': 'panel', 'pose': {'position': [0, 0, 0], 'quaternion': [0, 0, 0, 1]},
                         'size_m': [w, h], 'luminance_cd_m2': {'value': L, 'source': 'check'}})
    worst, worst_mc, rows = 0.0, 0.0, []
    for n_area in (1, 2, 4, 8):
        em = LT.emitters(lt, 'vis', n_area)
        errs = []
        for d in (0.3, 0.6, 1.0, 2.0):
            for dx in (0.0, 0.2, 0.5):
                wi, E = LT.irradiance(em, np.array([[dx, 0.1, d]]))
                got = float(np.sum(E[:, 0] * np.maximum(-wi[:, 0, 2], 0)))
                ref = rect_irradiance(L, w, h, dx, 0.1, d)
                errs.append(abs(got - ref) / ref)
                if n_area == LT.DEFAULT_N_AREA:
                    worst = max(worst, errs[-1])
                if n_area == 1 and dx in (0.0, 0.5):
                    mc, se = _mc_square(L, w, d, (dx, 0.1), N, rng)
                    worst_mc = max(worst_mc, abs(mc - ref) / ref - 3 * se / ref)
        rows.append(max(errs))
    return Outcome(max(worst, worst_mc), {'largest_by_patches': {str(k): round(v, 5) for k, v in
                                                                  zip((1, 2, 4, 8), rows)},
                                          'default_patches_per_side': LT.DEFAULT_N_AREA,
                                          'closed_form_vs_monte_carlo': round(max(worst_mc, 0.0), 6)},
                   figure=figure('Area light split against the closed form', 'patches per side',
                                 'largest relative difference', [series('split − closed form', (1, 2, 4, 8), rows,
                                                                        'points')], logy=True, hline=0.025,
                                 hline_label='tolerance'),
                   note='The split converges as 1 / n² (patches per side n); with the default 4 × 4 the largest '
                        'difference is about 2 % at half the panel\'s size, where the patch is a quarter of the '
                        'distance; 2.5 % states that bound.')


def lit_sample():
    """The UR5e sample with a metallic can at the task volume's centre, an Elgato Key Light 1.2 m above it (library
    figures) and a window with example figures (the checks' and the docs' worked scene)."""
    from ..core import lights as LT
    from ..core.frames import Pose
    from ..core.project import TaskObject
    from ..core.robot.samples import sample_project
    p = sample_project('ur5e')
    c = np.asarray(p.workspace.volumes[0].center, float)
    p.workspace.objects.append(TaskObject(name='Soda can', shape='cylinder', size=(0.033, 0.115, 0),
                                          pose=Pose(np.array([c[0], c[1], 0.0575]), np.array([0, 0, 0, 1.0])),
                                          materials=['metallic']))
    p.workspace.lights = [
        LT.from_library('elgato-key-light', {'position': [float(c[0]), float(c[1]), float(c[2]) + 1.2],
                                             'quaternion': [1, 0, 0, 0]}),
        LT.clean_light({'kind': 'window', 'name': 'Window', 'size_m': [1.2, 1.0],
                        'pose': {'position': [float(c[0]), 1.4, 0.8], 'quaternion': [0.7071068, 0, 0, 0.7071068]},
                        'luminance_cd_m2': {'value': 2000.0, 'source': 'example figure, not a measurement'},
                        'ir_irradiance_w_m2': {'value': 10.0, 'source': 'example figure, not a measurement'}})]
    return p


@check('lighting.visibility_vs_raycast', 'lighting', 'The study\'s camera and light visibility against exact ray '
       'casting',
       'share of the sampled object, gripper and table area (area-weighted) where the study\'s visibility (cube maps '
       'with its slope-scaled tolerance) disagrees with exact ray casting, over every camera and light at the '
       'working poses of the UR5e sample with a can, a Key Light and a window',
       'core/raycast first hits along each line of sight against the full posed scene (no cube map), to the same '
       'lifted points', 0.015, 'area share', refs=('moller1997', 'amanatides_woo1987', 'williams1978'), cost='study')
def _visibility(ctx):
    from ..core import lights as LT
    from ..core import raycast as RC
    from ..core.scenegeom import sensor_occluders
    from ..core.studies import lighting as LS
    p = lit_sample()
    p.analysis.studies['lighting'] = {'max_poses': 2 if ctx.quick else 6}
    st = LS.settings_of(p)
    ev = LS.Evaluator(p, st, [])
    ev.lights = LT.lights_of(p)
    tot = bad = 0.0
    n_pts = n_bad = 0
    rows = []
    for k in range(len(ev.poses)):
        d = ev.pose(k)
        W = d['W']

        def exact(occ, o):
            v = o - d['P']
            cn = np.einsum('ij,ij->i', d['N'], v)
            nn = d['N'] * np.where(cn >= 0, 1.0, -1.0)[:, None]
            Pl = d['P'] + st['lift_mm'] / 1000.0 * nn
            dirs = Pl - o
            dist = np.linalg.norm(dirs, axis=1)
            return RC.Grid.build(np.asarray(occ.tris, float)).first_hit(np.broadcast_to(o, Pl.shape), dirs) >= \
                dist - 1e-3
        pairs = []
        for lt in ev.lights:
            o = np.asarray(lt['pose']['position'], float)
            pairs.append((lt['name'], ev.shadow(k, lt), exact(d['occ'], o)))
        for cam in ev.cams:
            vis, _wo, _n, _R, o = ev.seen(k, cam)
            occ = sensor_occluders(p, cam.sensor, d['qd'], base=d['occ'], origins=o[None])
            pairs.append((cam.name, vis, exact(occ, o)))
        for name, got, ref in pairs:
            dis = got != ref
            tot += float(W.sum())
            bad += float(W[dis].sum())
            n_pts += len(dis)
            n_bad += int(dis.sum())
            rows.append({'pose': ev.poses[k][0], 'from': name, 'area_share': round(float(W[dis].sum() / W.sum()), 5)})
    err = bad / max(tot, 1e-300)
    return Outcome(err, {'poses': len(ev.poses), 'cube_n': st['cube_n'], 'points_compared': n_pts,
                         'points_disagreeing': n_bad, 'per_view': rows[:24]},
                   note='The disagreements are within about a texel of a silhouette or a contact (the cube map is '
                        'sampled per texel; at 512 texels one spans 3.9 mrad at a face centre) and where the slope '
                        'tolerance lets a point just behind an oblique edge through. Measured on the RTX 5080: 0.66 % '
                        'of the area at 512 texels, 0.85 % at 256, 0.31 % at 1024.')
