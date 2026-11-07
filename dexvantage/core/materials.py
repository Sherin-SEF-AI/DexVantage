"""Surface materials for the lighting study (spec §7.8): a microfacet BRDF per material tag, and transmissive objects.

    MATERIALS                           tag -> Material (the six project.MATERIAL_TAGS), every parameter with its source
    material_for(tags, overrides=None)  the Material of a task object's tag list (the rule below)
    brdf(n, wi, wo, mat, band='vis')    f_r (N,) in 1/sr: the diffuse and specular terms summed
    brdf_terms(n, wi, wo, mat, band)    (diffuse (N,), specular (N,), cos_d (N,)) for the polariser model
    ggx_d, smith_g1, schlick_f          the microfacet pieces (vectorised)
    fresnel_exact(cos_i, n_complex)     (R_s, R_p) of a smooth interface: the Fresnel equations, complex index
    f0_from_index(n_complex)            normal-incidence reflectance ((n − 1)² + k²) / ((n + 1)² + k²)
    polariser_factors(...)              what an ideal linear polariser leaves of the diffuse and specular light

The BRDF. f_r(ωi, ωo) = f_d + f_s with the microfacet specular term of Walter et al. 2007 (eq. 20),

    f_s = F(ωi · h) G(ωi, ωo) D(h) / (4 |n · ωi| |n · ωo|),   h = (ωi + ωo) / |ωi + ωo|,

the GGX (Trowbridge-Reitz) normal distribution D(h) = α² / (π cos⁴θ_h (α² + tan²θ_h)²) (Walter et al. 2007, eq. 33;
Trowbridge and Reitz 1975), Smith's masking-shadowing as the separable product G = G1(ωi) G1(ωo) with the GGX
G1(v) = 2 / (1 + √(1 + α² tan²θ_v)) (Walter et al. 2007, eqs. 23 and 34; Smith 1967), and Schlick's approximation of
the Fresnel factor F(c) = F0 + (1 − F0)(1 − c)⁵ (Schlick 1994, eq. 15). The diffuse term is the energy-coupled one of
Ashikhmin and Shirley 2000 (eq. 5),

    f_d = (28 ρ_d / 23π) (1 − F0) (1 − (1 − |n·ωi|/2)⁵) (1 − (1 − |n·ωo|/2)⁵),

which is symmetric in ωi and ωo (so the BRDF is reciprocal) and gives way to the specular lobe at grazing angles, so
the total reflects no more than arrives (the white-furnace check in validation/checks_lighting.py measures it). α is
Walter's roughness α_g (not squared here).

Parameters per tag. F0 comes from a measured complex refractive index at 550 nm (visible) and 850 nm (the D4xx
projector's band) through f0_from_index; the indices are tabulated measurements from the refractiveindex.info
database (public domain, CC0; files listed in INDEX_SOURCES). The roughness α, the diffuse albedo ρ_d and the IR
albedo are **DexVantage defaults** (labelled so wherever they are shown): measured BRDFs of real samples (the MERL
database, Matusik et al. 2003, fitted with GGX by Löw et al. 2012 and others) spread over a wide range within each of
these classes, so no single measured value stands for "glossy" or "matte". Set a measured value per object
(settings 'materials' of the lighting study) when the surface is known.

    tag            index (550 / 850 nm)               α      ρ_d    IR albedo   transmissive  depth reads through
    matte          PMMA (Sultanova 2009), 1.493/1.483  0.5    0.5    0.5         no            —
    glossy         PMMA (Sultanova 2009)               0.1    0.5    0.5         no            —
    metallic       Al (McPeak 2015), 0.789+5.852i …    0.2    0      0 (metal)   no            —
    black          PMMA (Sultanova 2009)               0.4    0.04   0.04        no            —
    transparent    N-BK7 glass (SCHOTT 2017)           0.02   0      0           yes           yes (default)
    ir_absorbing   a modifier: the IR albedo becomes 0.03 (default); the other parameters come from the other tags

The rule for a tag list (material_for): the first of transparent, metallic, glossy, black, matte present gives the
material ('matte' when none is); 'ir_absorbing' then lowers the IR albedo (and the IR F0 of a dielectric is kept:
absorbing pigments still reflect at the surface). Transmissive surfaces reflect only their Fresnel part (ρ_d = 0);
whether a depth sensor reads the surface or what lies behind it is the flag depth_reads_through (default true for
'transparent': active stereo and ToF see through clear plastic and glass and return the background, or nothing).

Known limits: one index per tag (PMMA for the dielectrics, aluminium for every metal, BK7 for every clear part) at
two wavelengths, not spectra; Schlick's F with the complex-index F0 (exact Fresnel is used only for the polariser);
the separable Smith G is slightly darker than the height-correlated form at grazing angles (Heitz 2014); no
anisotropy, no subsurface scattering, no clear-coat layers; the diffuse and IR albedos and the roughness are defaults,
not measurements; transmission is a flag (no refraction, no caustics).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import numpy as np

TAG_ORDER = ('transparent', 'metallic', 'glossy', 'black', 'matte')      # material_for: the first present wins
DEFAULT_IR_ABSORBING_ALBEDO = 0.03          # the IR albedo of an 'ir_absorbing' surface: a DEFAULT
WAVELENGTHS_NM = {'vis': 550.0, 'ir': 850.0}


# ====================================================================== the measured indices (CC0 data)
def _sellmeier2(lam_um: float, coeffs) -> float:
    """refractiveindex.info 'formula 2': n² − 1 = C0 + Σ B λ² / (λ² − C)."""
    c = list(coeffs)
    l2 = lam_um * lam_um
    s = c[0]
    for i in range(1, len(c) - 1, 2):
        s += c[i] * l2 / (l2 - c[i + 1])
    return math.sqrt(1.0 + s)


_PMMA = (0.0, 1.1819, 0.011313)                                       # Sultanova et al. 2009, 20 °C, 0.437–1.052 µm
_BK7 = (0.0, 1.03961212, 0.00600069867, 0.231792344, 0.0200179144, 1.01046945, 103.560653)   # SCHOTT 2017-01-20b

INDEX = {
    # material: {band: complex index}
    'pmma': {'vis': complex(_sellmeier2(0.55, _PMMA), 0.0), 'ir': complex(_sellmeier2(0.85, _PMMA), 0.0)},
    'bk7': {'vis': complex(_sellmeier2(0.55, _BK7), 0.0), 'ir': complex(_sellmeier2(0.85, _BK7), 0.0)},
    # McPeak et al. 2015, tabulated: 0.55 µm 0.789405353 5.851936501; 0.85 µm 2.204898553 7.188085811
    'aluminium': {'vis': complex(0.789405353, 5.851936501), 'ir': complex(2.204898553, 7.188085811)},
}
INDEX_SOURCES = {
    'pmma': ('N. Sultanova, S. Kasarova, I. Nikolov, "Dispersion properties of optical polymers", Acta Physica '
             'Polonica A 116, 585–587 (2009), Sellmeier fit by M. Polyanskiy: refractiveindex.info database '
             'data/organic/(C5H8O2)n - poly(methyl methacrylate)/nk/Sultanova.yml (CC0), read 2026-11-07'),
    'bk7': ('SCHOTT Zemax catalogue 2017-01-20b, N-BK7 Sellmeier coefficients: refractiveindex.info database '
            'data/specs/schott/optical/N-BK7.yml (CC0), read 2026-11-07'),
    'aluminium': ('K. M. McPeak et al., "Plasmonic films can easily be better: rules and recipes", ACS Photonics 2, '
                  '326–333 (2015), evaporated Al, tabulated n and k: refractiveindex.info database '
                  'data/main/Al/nk/McPeak.yml (CC0), read 2026-11-07'),
}
DEFAULT_SOURCE = 'a DexVantage default (not a measurement): set a measured value per object when the surface is known'


def f0_from_index(n) -> float:
    """Normal-incidence reflectance of a smooth interface from air: ((n − 1)² + k²) / ((n + 1)² + k²) for the complex
    index n + ik (Born and Wolf; Hecht, Optics, §4.6)."""
    n = complex(n)
    a, k = n.real, n.imag
    return float(((a - 1) ** 2 + k * k) / ((a + 1) ** 2 + k * k))


def fresnel_exact(cos_i, n) -> tuple[np.ndarray, np.ndarray]:
    """(R_s, R_p) power reflectances of a smooth interface from air into a medium of complex index n, at incidence
    cosines cos_i (the Fresnel equations; Hecht, Optics, §4.6.2)."""
    c = np.clip(np.asarray(cos_i, float), 0.0, 1.0).astype(complex)
    n = complex(n)
    s2 = 1.0 - c * c
    ct = np.sqrt(1.0 - s2 / (n * n))                      # cos θ_t, complex for an absorbing medium
    rs = (c - n * ct) / (c + n * ct)
    rp = (n * c - ct) / (n * c + ct)
    return np.abs(rs) ** 2, np.abs(rp) ** 2


@dataclass(frozen=True)
class Material:
    tag: str
    title: str
    index: str                  # a key of INDEX
    alpha: float                # GGX roughness α (Walter et al. 2007)
    albedo: float               # diffuse albedo ρ_d (visible)
    ir_albedo: float            # diffuse albedo at the projector's band
    metal: bool = False
    transmissive: bool = False
    depth_reads_through: bool = False
    sources: dict = field(default_factory=dict)      # parameter -> where it comes from

    def f0(self, band: str = 'vis') -> float:
        return f0_from_index(INDEX[self.index][band])

    def n(self, band: str = 'vis') -> complex:
        return INDEX[self.index][band]

    def rho(self, band: str = 'vis') -> float:
        return self.albedo if band == 'vis' else self.ir_albedo

    def to_dict(self) -> dict:
        return {'tag': self.tag, 'title': self.title, 'alpha': self.alpha, 'albedo': self.albedo,
                'ir_albedo': self.ir_albedo, 'f0_vis': round(self.f0('vis'), 5), 'f0_ir': round(self.f0('ir'), 5),
                'index': self.index, 'metal': self.metal, 'transmissive': self.transmissive,
                'depth_reads_through': self.depth_reads_through, 'sources': dict(self.sources)}


def _src(index_key: str, **defaults) -> dict:
    out = {'f0': f'from the refractive index: {INDEX_SOURCES[index_key]}'}
    for k in ('alpha', 'albedo', 'ir_albedo'):
        out[k] = defaults.get(k, DEFAULT_SOURCE)
    return out


MATERIALS = {
    'matte': Material('matte', 'Matte plastic or paint', 'pmma', 0.5, 0.5, 0.5, sources=_src('pmma')),
    'glossy': Material('glossy', 'Glossy plastic or lacquer', 'pmma', 0.1, 0.5, 0.5, sources=_src('pmma')),
    'metallic': Material('metallic', 'Bare metal (aluminium)', 'aluminium', 0.2, 0.0, 0.0, metal=True,
                         sources=_src('aluminium', albedo='0: a metal has no diffuse term (its light is all '
                                      'reflected at the surface)', ir_albedo='0: a metal has no diffuse term')),
    'black': Material('black', 'Black plastic or paint', 'pmma', 0.4, 0.04, 0.04, sources=_src('pmma')),
    'transparent': Material('transparent', 'Clear plastic or glass', 'bk7', 0.02, 0.0, 0.0, transmissive=True,
                            depth_reads_through=True,
                            sources=_src('bk7', albedo='0: a clear part transmits what it does not reflect',
                                         ir_albedo='0: a clear part transmits what it does not reflect')),
}
# 'ir_absorbing' is a modifier, not a material of its own (material_for)
MATERIALS['ir_absorbing'] = replace(MATERIALS['matte'], tag='ir_absorbing', title='IR-absorbing matte surface',
                                    ir_albedo=DEFAULT_IR_ABSORBING_ALBEDO,
                                    sources=dict(MATERIALS['matte'].sources, ir_albedo=DEFAULT_SOURCE))

OVERRIDABLE = ('alpha', 'albedo', 'ir_albedo', 'depth_reads_through')


def material_for(tags, overrides: dict | None = None) -> Material:
    """The Material of a tag list (module docstring), with per-object overrides {'alpha', 'albedo', 'ir_albedo',
    'depth_reads_through'} applied (each then sourced 'your setting')."""
    tags = [str(t) for t in (tags or [])]
    base = next((t for t in TAG_ORDER if t in tags), 'matte')
    m = MATERIALS[base]
    if 'ir_absorbing' in tags and not m.metal:
        m = replace(m, ir_albedo=min(m.ir_albedo, DEFAULT_IR_ABSORBING_ALBEDO),
                    sources=dict(m.sources, ir_albedo='IR-absorbing: ' + DEFAULT_SOURCE),
                    title=m.title + ', IR-absorbing')
    ov = {k: v for k, v in (overrides or {}).items() if k in OVERRIDABLE and v is not None}
    if ov:
        clean = {}
        for k, v in ov.items():
            if k == 'depth_reads_through':
                clean[k] = bool(v)
            else:
                x = float(v)
                if not math.isfinite(x):
                    continue
                clean[k] = min(max(x, 1e-3), 1.0) if k == 'alpha' else min(max(x, 0.0), 1.0)
        m = replace(m, **clean, sources=dict(m.sources, **{k: 'your setting' for k in clean}))
    return m


# ====================================================================== the microfacet pieces
def ggx_d(cos_h, alpha):
    """GGX / Trowbridge-Reitz D(h) for cos θ_h (Walter et al. 2007, eq. 33), 0 where cos θ_h ≤ 0."""
    c = np.asarray(cos_h, float)
    a2 = np.asarray(alpha, float) ** 2
    c2 = c * c
    den = np.pi * (c2 * (a2 - 1.0) + 1.0) ** 2
    return np.where(c > 0, a2 / np.maximum(den, 1e-300), 0.0)


def smith_g1(cos_v, alpha, cos_vh=None):
    """Smith's GGX masking G1(v) = 2 / (1 + √(1 + α² tan²θ_v)) (Walter et al. 2007, eq. 34), with the χ⁺(v·h / v·n)
    factor when cos_vh is given."""
    c = np.clip(np.asarray(cos_v, float), 1e-12, 1.0)
    a2 = np.asarray(alpha, float) ** 2
    tan2 = (1.0 - c * c) / (c * c)
    g = 2.0 / (1.0 + np.sqrt(1.0 + a2 * tan2))
    if cos_vh is not None:
        g = np.where(np.asarray(cos_vh, float) * np.asarray(cos_v, float) > 0, g, 0.0)
    return g


def schlick_f(cos_d, f0):
    """Schlick's Fresnel F0 + (1 − F0)(1 − cos)⁵ (Schlick 1994, eq. 15)."""
    c = np.clip(np.asarray(cos_d, float), 0.0, 1.0)
    return f0 + (1.0 - f0) * (1.0 - c) ** 5


def _unit(v):
    v = np.asarray(v, float)
    return v / np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-300)


def brdf_terms(n, wi, wo, mat: Material, band: str = 'vis'):
    """(f_d (N,), f_s (N,), cos_d (N,) = ωi · h) in 1/sr for unit normals n, light directions wi (towards the light)
    and view directions wo (towards the viewer), all (N, 3) or broadcastable; 0 where either is below the surface."""
    n, wi, wo = _unit(n), _unit(wi), _unit(wo)
    ci = np.einsum('...k,...k->...', n, wi)
    co = np.einsum('...k,...k->...', n, wo)
    up = (ci > 0) & (co > 0)
    h = _unit(wi + wo)
    ch = np.einsum('...k,...k->...', n, h)
    cd = np.einsum('...k,...k->...', wi, h)
    a = float(mat.alpha)
    f0 = mat.f0(band)
    D = ggx_d(ch, a)
    G = smith_g1(ci, a, cd) * smith_g1(co, a, np.einsum('...k,...k->...', wo, h))
    F = schlick_f(cd, f0)
    cic, coc = np.clip(ci, 1e-12, 1.0), np.clip(co, 1e-12, 1.0)
    fs = F * G * D / (4.0 * cic * coc)
    rho = mat.rho(band)
    fd = (28.0 * rho / (23.0 * np.pi)) * (1.0 - f0) * (1.0 - (1.0 - cic / 2) ** 5) * (1.0 - (1.0 - coc / 2) ** 5)
    return np.where(up, fd, 0.0), np.where(up, fs, 0.0), cd


def brdf(n, wi, wo, mat: Material, band: str = 'vis'):
    """f_r = f_d + f_s (1/sr), see brdf_terms."""
    fd, fs, _cd = brdf_terms(n, wi, wo, mat, band)
    return fd + fs


def polariser_factors(cos_d, mat: Material, band: str = 'vis', k: float = 1.0) -> tuple[float, np.ndarray]:
    """(diffuse factor, specular factor (N,)) that an ideal linear polariser with transmission k along its axis leaves
    of unpolarised diffuse light (k / 2: Malus's law averaged over the polarisation angles; Hecht, Optics, §8.2) and
    of the specular reflection at microfacet incidence cosines cos_d, turned to pass the weaker of its s and p
    components: k min(R_s, R_p) / (R_s + R_p) of the unpolarised specular light (Fresnel, fresnel_exact). The turn is
    taken as the best one for every point, so the specular factor is an upper bound on what one polariser achieves."""
    rs, rp = fresnel_exact(cos_d, mat.n(band))
    tot = np.maximum(rs + rp, 1e-300)
    return 0.5 * float(k), float(k) * np.minimum(rs, rp) / tot


# ====================================================================== sampling (for the checks)
def sample_ggx_h(alpha: float, u1, u2):
    """Microfacet normals drawn with pdf D(h) cos θ_h (Walter et al. 2007, eqs. 35–36), in the frame with n = +z."""
    u1, u2 = np.asarray(u1, float), np.asarray(u2, float)
    t = np.arctan(alpha * np.sqrt(u1 / np.maximum(1.0 - u1, 1e-300)))
    p = 2 * np.pi * u2
    return np.stack([np.sin(t) * np.cos(p), np.sin(t) * np.sin(p), np.cos(t)], axis=-1)
