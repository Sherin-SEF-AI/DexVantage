# Lighting, glare and IR wash-out

Spec §7.8. Code: `dexvantage/core/materials.py` (the BRDF and the material tags), `dexvantage/core/lights.py` (light
sources and their photometry), `dexvantage/core/studies/lighting.py` (the study, batch flag `--lighting`),
`dexvantage/ui/pages/lighting.py` (the page). Checks: `dexvantage/validation/checks_lighting.py`. Tests:
`tests/test_lighting.py`. The worked example is in [../guide/lighting.md](../guide/lighting.md).

The study answers two questions for the rig's cameras over the gripper's working poses:

1. How much of the object and the gripper does each camera see **blown out by specular highlights** at its exposure?
2. How much **valid depth** does an active-stereo sensor lose because ambient IR washes its projector's pattern out,
   or because the surface is dark, IR-absorbing or clear?

It then tries the usual fixes (a polariser, moving or dimming a light, tilting the camera, blinding a window) and
reports each one's change in the predicted figure.

## Lights

Lights live in `project.workspace.lights`. Each is a dict:

```
{'uid', 'name', 'kind': 'panel' | 'ring' | 'window' | 'point' | 'spot',
 'pose': {'position', 'quaternion' (x, y, z, w)},        # the face looks along its local +z
 'size_m': [w, h] | null,
 'luminance_cd_m2': {'value', 'source'} | null,
 'luminous_flux_lm': {'value', 'source'} | null,
 'ir_irradiance_w_m2': {'value', 'source', 'at_m'?} | null,
 'cct_k': {'value', 'source'} | null,
 # optional: 'library_id', 'beam_deg' {'value', 'source'}, 'inner_m', 'dim' (0..1), 'enabled', 'sun' (a window)}
```

Each figure carries its source: a datasheet with its URL and the date it was read, or "my meter reading". A figure
nobody states is null. The study then names the light and says which figure is missing. It does not fill the gap.

**Photometry** (the visible band, in candela and lux):

| kind | model | intensity / luminance |
|---|---|---|
| point | isotropic | I = Φ / 4π (or L · A when a luminance and a size are given) |
| spot | uniform cone, full angle β | I = Φ / (2π (1 − cos β/2)); no β stated → not computed |
| panel, window, ring | Lambertian area emitter | L as stated, else L = Φ / (π A); A = w h, a ring's annulus π(R² − r²) |

An area light is split into n × n patches (`n_area`, 4 by default; a ring into 4n patches round the annulus). Each
patch is a point emitter with I(θ) = L · A_patch · cos θ on its front side. The irradiance on a plane facing an emitter
is E⊥ = I(θ) / r² (the inverse-square law [McCluney 2014]). On a surface with normal n it is E⊥ max(0, n · ωi).

**IR** (W/m², the band the depth sensors' projectors use). `ir_irradiance_w_m2` is the in-band irradiance on a
surface facing the light, on its axis, at `at_m`. A point-like light then has J = E · at_m², and an area light has
radiance J / A. If an area light has no `at_m`, the figure is taken as the irradiance at its own face (for a window,
the IR coming through the glass), with radiance E / π.

**Sun through a window.** A window's optional `sun` block gives a parallel beam: its direction, illuminance and IR
irradiance. The beam reaches a point when the line from the point towards the sun passes through the window's
rectangle and nothing in the scene is in the way (`core/raycast`).

**`dim`** scales every figure of the light. The stated figure itself stays as stated.

### The `light` library kind

Three products are in the library, with their figures from the makers' pages (read 2026-11-07):

| entry | flux | CCT | face | source |
|---|---|---|---|---|
| Elgato Key Light | 2800 lm (maximum) | 2900–7000 K | 350 × 250 mm | elgato.com/us/en/p/key-light |
| Elgato Key Light Air MK.2 | up to 2100 lm ("depending on power source") | 2900–7000 K | 319 × 207 mm | elgato.com/us/en/p/key-light-air |
| Elgato Ring Light (discontinued) | 2500 lm | 2900–7000 K | Ø 432 mm outer, 310 mm inner | elgato.com/us/en/p/ring-light |

None of the three makers states a luminance, a beam profile or an IR output. Those figures are null. The emitting
face is taken as the panel's front, or the ring's annulus: this is an **assumption**, said in each entry's note.
Machine-vision IR illuminators state radiant intensity per LED. No lab-scale IR source was found whose public page
states an irradiance at a distance, so none is in the library. Enter a meter reading instead.

**Projectors.** Intel's D400 datasheet (337029-017, March 2026, Tables 3-16 and 3-17) states "Optical Power 360 mW
average" and "850 nm ± 10 nm" for the D41x and the D43x/D45x projectors. These figures are in the D415, D435, D435i
and D455 entries (`projector.optical_power_w`, `wavelength_nm`, `optical_power_what`). The D457's own datasheet does
not state the figure, so its entry has none. No other maker in the library states its projector's power.

## Materials: the BRDF

f_r(ωi, ωo) = f_d + f_s. The specular term is the microfacet model of Walter et al. 2007 (eq. 20):

    f_s = F(ωi · h) G(ωi, ωo) D(h) / (4 |n · ωi| |n · ωo|),   h = (ωi + ωo) / |ωi + ωo|

- D: GGX / Trowbridge-Reitz, D(h) = α² / (π cos⁴θ_h (α² + tan²θ_h)²) [Walter 2007 eq. 33; Trowbridge and Reitz 1975].
- G: Smith's masking-shadowing as the separable product G1(ωi) G1(ωo), with
  G1(v) = 2 / (1 + √(1 + α² tan²θ_v)) [Walter 2007 eqs. 23 and 34; Smith 1967].
- F: Schlick's approximation F0 + (1 − F0)(1 − c)⁵ [Schlick 1994, eq. 15].
- f_d: the energy-coupled diffuse term of Ashikhmin and Shirley 2000 (eq. 5),
  (28 ρ_d / 23π)(1 − F0)(1 − (1 − |n·ωi|/2)⁵)(1 − (1 − |n·ωo|/2)⁵). It is symmetric, so the BRDF is reciprocal, and it
  gives way to the specular lobe at grazing angles, so the total stays within the energy bound.

**Parameters per tag.** F0 comes from measured complex refractive indices through ((n − 1)² + k²) / ((n + 1)² + k²),
at 550 nm (visible) and 850 nm (the D4xx projector). The indices are tabulated measurements from the
refractiveindex.info database, which is public domain (CC0); all were read 2026-11-07.

| tag | index source | n at 550 / 850 nm | F0 vis / IR | α | ρ_d | IR albedo |
|---|---|---|---|---|---|---|
| matte | PMMA, Sultanova et al. 2009 | 1.4926 / 1.4835 | 0.0391 / 0.0379 | 0.5 | 0.5 | 0.5 |
| glossy | PMMA | as matte | as matte | 0.1 | 0.5 | 0.5 |
| metallic | Al, McPeak et al. 2015 | 0.789 + 5.852i / 2.205 + 7.188i | 0.9157 / 0.8576 | 0.2 | 0 | 0 |
| black | PMMA | as matte | as matte | 0.4 | 0.04 | 0.04 |
| transparent | N-BK7, SCHOTT 2017 | 1.5185 / 1.5098 | 0.0424 / 0.0413 | 0.02 | 0 | 0 |
| ir_absorbing | a modifier on the other tags | | | | | 0.03 |

α, ρ_d and the IR albedo are **DexVantage defaults**, not measurements. Measured BRDFs (the MERL database of Matusik
et al. 2003, fitted with microfacet models by Löw et al. 2012 and others) spread widely within each of these classes.
No single measured sample stands for "glossy" or "matte". Set measured values per object with the study setting
`materials` {object uid: {'alpha', 'albedo', 'ir_albedo', 'depth_reads_through'}}.

**The rule for a tag list.** The first of transparent, metallic, glossy, black and matte that is present gives the
material ('matte' when none is). `ir_absorbing` then lowers the IR albedo. A transparent part is a flag,
`depth_reads_through`. Its default is true: active stereo and ToF see through clear plastic and glass, and return
the background or nothing. Set it per object.

**The polariser.** An ideal linear polariser passes k/2 of unpolarised diffuse light (Malus's law averaged over the
angles [Hecht 2017, §8.2]). Turned to pass the weaker component of the specular reflection, it passes
k min(R_s, R_p) / (R_s + R_p) of that light. R_s and R_p come from the exact Fresnel equations with the complex index
[Hecht 2017, §4.6.2], at each microfacet's incidence angle. The model turns the polariser to its best for every
point, so its gain is an **upper bound** on what one polariser achieves. A metal reflects both polarisations about
equally, so a polariser removes little of a metal's highlight.

## The study

**Working poses.** If the project has a grasp set, the study picks up to `max_poses` (6) grasps evenly from it. The
grasp study's IK solves them, with its own settings and obstacle checks. The reachable solutions are the poses.
Without a grasp set, the study uses the joint state of the settings `q`, else home. The page's "Pose on screen" uses
the 3D view's joint state.

**Sampling.** Points are drawn uniformly by area, with a seed. There are 400 per task object, 1200 over the gripper
(the gripper frame's link and every link below it), and 500 per surface. Each point stands for its share of its
part's area. Robot links take `robot_material` ('matte' by default, labelled) or `link_materials`. Model files carry
no material tags.

**Visibility comes from core/occlusion.** A point is seen when it projects inside the imager's image and the camera's
cube map of the posed scene (512 texels, as in the grasp study) has nothing nearer. A point is lit when the cube map
at the light's centre has nothing nearer. These are hard shadows. The camera's seat on its mount link is left out by
scenegeom's rule. Three details make the test agree with exact ray casting:

- A point on a closed part (an object, the gripper) whose outward normal faces away from the camera or the light is on
  the part's far side, so it is hidden. Surfaces are one-sided sheets, so either side of them shows.
- The point is lifted `lift_mm` (1 mm, a default) along its normal, turned towards the viewer.
- The tolerance is slope-scaled: eps + ¾ · texel · distance · tan θ, with θ the angle from the normal, capped at 80°.
  A surface seen obliquely changes depth across a texel. Without the slope term, such surfaces shadow themselves
  (shadow-map "acne" [Williams 1978]). With 256-texel maps and a flat 1 mm tolerance, the first version of this study
  disagreed with ray casting on 20 % of the sampled points; it also gave 22 % on the GPU and 30 % with numpy for the
  same scene.

The check `lighting.visibility_vs_raycast` measures the remaining disagreement.

**Radiance.** L = Σ f_r E⊥ max(0, n · ωi) over the lights' emitters and the sun, in cd/m².

**Saturation, absolute.** This model runs when the imager's sensitivity S (DN per lux second at the sensor, unity
gain) and f-number N are stated and an exposure t is given (the library entry, or the settings `sensors[uid]`). It
uses the camera equation E = (π / 4) T L cos⁴α / N², then DN = S g t E. A point is saturated when DN ≥ 2^bits − 1.
The lens transmittance T is 1, an **assumption**, because no maker states it. The gain g defaults to 1, and bits to 8.
No library entry states S today, so this model runs only on your figures.

**Saturation, relative.** This is the default, and it is labelled "relative" wherever it is shown. It models an
auto-exposure that meters the mean luminance L̄ of the visible sample as 18 % grey. ISO 12232's saturation-based
speed puts that grey at 18 %/√2 = 12.7 % of saturation, which leaves half a stop of headroom. A point therefore
saturates when L ≥ L̄ / 0.127 = 7.86 L̄ (`sat_ratio`, a default taken from that definition). The glare index L / L̄ is
relative, not absolute. Two consequences follow. Scaling every light by the same factor changes nothing (the test
`test_relative_model_is_labelled_and_scale_free`). Dimming a single light changes nothing either.

**The headline** is the worst camera's saturated share of the visible object and gripper area, averaged over the
poses, with the worst pose named. Surfaces (the table) are reported beside it. Status: good under 1 %, warning under
5 %, serious under 15 %, else critical. These thresholds are defaults (`good_pct`, `warn_pct`, `serious_pct`).
When an active depth sensor's predicted depth loss is worse (warning from 5 %, serious from 15 %, critical from 30 %,
defaults `depth_good_pct`, `depth_warn_pct`, `depth_serious_pct`), it sets the status and the word names both, for
example "low glare; depth washed out", so a green glare word never hides a washed-out depth sensor.

**Glare maps.** Each camera's image is cut into 32 × 24 cells (`grid`). Each cell keeps the largest saturated share
of the visible area in it over the poses, and the largest level (glare index, or DN / full scale). The page draws the
map. The Camera view outlines saturated cells in the critical colour and cells near saturation (half the threshold)
in the warning colour.

**IR wash-out.** This runs for active IR stereo and structured light. At each pose the study takes the points the
reference imager sees inside the projector's stated cone. The pattern's irradiance there is E_p = (P / Ω) cos θ / d².
P is the projector's stated optical power, spread uniformly over its rectangular field Ω = 4 arcsin(sin(a/2) sin(b/2)).
The ambient IR E_a comes from the lights' and the sun's IR figures, with shadows. The pattern's contrast is
C = E_p / (E_p + E_a). This is the same for any Lambertian albedo, because both are reflected alike. A point is washed
out when C < `contrast_min` (0.5 by default: the ambient is as bright as the pattern). A point on a surface with IR
albedo under `ir_albedo_min` (0.1, a default) is "dark". A point on a transmissive object the depth reads through is
"clear". The predicted loss of valid depth is the share of the visible object and gripper area with any of the
three. If the projector's power is not stated, wash-out is not computed and the result says so. The dark and clear
shares are still computed.

**Recommendations.** Each one re-runs the same evaluation with one change:

- a polariser on the worst camera;
- the light that lights most saturated points moved ±`move_m` (0.3 m) along world x or y, or upwards;
- that light dimmed to `dim_to` (50 %);
- the camera tilted ±5° or ±10° about its optical x axis (the rotation is about the optical centre, so its cube maps
  are reused);
- for wash-out, the IR source that adds most ambient IR switched off or blinded.

The suggestion is the one that lowers the headline most. If no camera saturates, it is the one that lowers the depth
loss most. Apply makes it as one undo step.

## Validation (measured, 2026-11-07, full run; the visibility check on the RTX 5080 through EGL device 0)

| check | reference | measured | tolerance |
|---|---|---|---|
| `lighting.ggx_normalised` | ∫ D cos θ_h dω = 1 by adaptive quadrature (scipy) | 1.4 × 10⁻¹³ | 10⁻⁶ |
| `lighting.white_furnace` | weak white furnace = 1 [Heitz 2014], adaptive quadrature over θ_h with the φ integral in closed form; the Monte Carlo estimate (400 000 samples per case, 0.0065 from 1) must also lie within 4 standard errors (0.024 at α 0.9, 89°) | 8.1 × 10⁻¹² | 10⁻⁶ (the Monte Carlo band alone would let a 2 % error at grazing view through) |
| `lighting.energy` | directional albedo ≤ 1 for every tag, θo 0–89°, MIS Monte Carlo | 0 (none above 1) | 0 |
| `lighting.reciprocity` | f_r(ωi, ωo) = f_r(ωo, ωi), 20 000 pairs per tag | 1.3 × 10⁻¹⁴ | 10⁻¹² |
| `lighting.peak_closed_form` | F0 / (4π α²) + diffuse closed form at normal incidence | 2.2 × 10⁻¹³ | 10⁻⁹ |
| `lighting.schlick_vs_fresnel` | exact Fresnel, PMMA and BK7 | 0.036 (aluminium: 0.080, reported) | 0.04 |
| `lighting.inverse_square` | Monte Carlo over a Lambertian square, d ≥ 10 a | 0.0033 | 0.005 |
| `lighting.area_vs_closed_form` | rectangle configuration factor [Howell B-3], and Monte Carlo | 0.022 (4 × 4 patches at half the panel's size) | 0.025 |
| `lighting.visibility_vs_raycast` | exact ray casting (core/raycast) of every camera's and light's line of sight, UR5e sample with a can, a Key Light and a window, 6 poses | 0.0066 of the area (2.1 % of the points, mostly table points at contacts and silhouettes) | 0.015 |

The split area light converges as 1/n². Its largest error at 0.3 m from a 0.6 m panel is 22 % with one patch, 11 %
with 2 × 2, 2.2 % with 4 × 4 and 0.5 % with 8 × 8. Schlick's approximation is poorest for metals, and the study uses
it for F; the exact Fresnel equations are used only for the polariser.

## Performance

The worked example is the UR5e sample with one Key Light and a metallic can, 6 working poses and two cameras. There
the study takes 1.3–1.4 s on the NVIDIA GeForce RTX 5080 (EGL device 0, the discrete GPU; driver 595.99.02). This
includes the IK of up to 18 grasps and the recommendations, which re-evaluate up to 13 variants. With the numpy
rasteriser (no GPU) it takes 18 s, and the headline is the same, 34.9 %. The visibility agreement at other map sizes
(area share against ray casting): 0.85 % at 256 texels and 0.31 % at 1024.

The study runs on the analysis controller's worker thread. Its UI-thread work was measured on the page, offscreen:
showing a result 0.5 ms, the Camera view's glare-map outlines 0.5 ms, the light glyphs 0.2 ms, the lights table
0.6 ms, painting the glare map 17 ms, and Apply 8–17 ms.

## Known limits

- **Light transport.** Only direct light is modelled: no inter-reflections and no ambient term. Shadowed parts are
  darker than they would be in a room. The relative model meters only the sampled object, gripper and surfaces, not
  the whole image.
- **Lights.** Shadows are hard and cast from each light's centre. Emitters are Lambertian or uniform cones, with no
  measured beam profile and no IES file. A window is a diffuse emitter plus an optional sun beam; there is no sky
  model.
- **Spectra.** The camera's spectral response is taken as photometric (V(λ)). The IR figures are taken to be for the
  band the depth imagers see; no maker in the library states the imagers' filter.
- **Projector.** Its field is taken as uniform over the stated FOV, but the real dot pattern is not. Its own shadow is
  not modelled. Intel halves the laser power above 60 °C module temperature, which is not modelled either. The
  360 mW is the datasheet's projector power and the top of the laser-power control (0–360 mW in 30 mW steps,
  Table 4-23): the wash-out is predicted at full laser power, and a lower setting weakens the pattern.
- **Wash-out and passive stereo.** Passive stereo still matches textured surfaces when the pattern is washed out, so
  the predicted loss applies to textureless surfaces.
- **Materials.** Roughness and albedos are defaults. Every metal is aluminium, every dielectric is PMMA, and every
  clear part is BK7, at two wavelengths. G is the separable Smith form, slightly darker than the height-correlated one
  at grazing angles. Transmission is a flag, with no refraction.
- **Polariser.** Its gain is an upper bound, because it is turned at its best for every point.
- **Sampling.** Points are sampled, not pixels. A highlight smaller than the sample spacing can be missed; raise
  `object_points` and `gripper_points` to catch it.
- **Visibility.** It is decided per cube-map texel, with a slope tolerance. A point within about a texel of a
  silhouette or a contact can be judged either way, and a point just behind an oblique edge can leak through
  (0.66 % of the area in the check).
- **Robot materials.** Model files carry no material tags, so a robot's link materials are a setting.

## References

Walter, Marschner, Li and Torrance 2007 (EGSR); Williams 1978 (SIGGRAPH, shadow maps); Trowbridge and Reitz 1975 (JOSA 65(5)); Smith 1967 (IEEE TAP 15(5));
Schlick 1994 (CGF 13(3)); Ashikhmin and Shirley 2000 (JGT 5(2)); Heitz 2014 (JCGT 3(2)); Veach 1997 (PhD thesis);
Matusik et al. 2003 (ACM TOG 22(3)); Löw et al. 2012 (ACM TOG 31(1)); Sultanova et al. 2009; SCHOTT 2017; McPeak et
al. 2015; Hecht, Optics, 5th ed. 2017; McCluney 2014; Howell's catalogue of configuration factors; ISO 12232 as
described in Wikipedia, "Film speed" (read 2026-11-07); Intel RealSense D400 datasheet 337029-017 (read
2026-11-07). Keys in `dexvantage/validation/references.py`.
