# The workspace check

`dexvantage/core/workspace_check.py` (no Qt), shown on the Workspace page (`dexvantage/ui/pages/workspace.py`,
Workspace ▸ Workspace). Validation model `workspace`.

## What it answers

The coverage, depth-validity, grasp and VLA-token studies measure inside the project's **task volumes**
(`project.workspace.volumes`: axis-aligned boxes in the world frame). A task object outside every task volume is
therefore not measured by them. The check asks one question: does every task object lie inside a task volume? If
not, it suggests the smallest change that makes it true.

## The model

For each task object the world triangles come from `core/scenegeom.object_triangles` (a box's 12 triangles, a
cylinder's and a sphere's tessellation of 24 segments around and 12 rings pole to pole, a mesh object's own mesh).
Its axis-aligned world bounding box is

    lo = min over vertices v of v,    hi = max over vertices v of v.

The object is **inside** volume V (centre c, size s) when

    lo ≥ c − s/2 − τ   and   hi ≤ c + s/2 + τ   (each axis),

with τ = `INSIDE_TOL_M` = 1 mm (a DexVantage default: rounding of typed positions). The headline is the number of
task volumes with a status word:

| Word | When |
|---|---|
| ready | at least one volume, and every object (with geometry) inside one |
| objects outside | some object lies outside every volume |
| no task volume | the project has none |

**The suggestion (Apply, one undo step).** With objects outside, let B be the union of their boxes grown by
`MARGIN_M` = 20 mm (a DexVantage default); the volume whose centre is nearest B's centre grows to the union of its
own box and B. With no volume, a new tabletop volume is B around every object, or the default volume of
`core/project.Volume` (500 × 600 × 300 mm centred at (0.45, 0, 0.15) m, a default to move to your table) when there
are no objects. When the objects stand on or above the floor (z = 0), the margin stops at the floor.

## Validation (measured, 2026-11-07)

`dexvantage --validate out --models workspace`:

| Check | Reference | Tolerance and why | Measured |
|---|---|---|---|
| `workspace.object_extent` | the exact box of the true shape: a turned box's half-extents \|R\| h (Arvo 1990); a cylinder's r √(1 − a_i²) + (h/2)\|a_i\| per world axis i, with a the cylinder's axis in the world; a sphere's r; rotations drawn by SciPy, not `core/frames` | 1.0 of the tessellation bound: a box is exact to rounding (10⁻¹² m); a tessellated cylinder or sphere lies inside the true shape, so its box may fall short by up to r (1 − cos²(π/24)) = 1.70 % of the radius and never stick out | 0.861 of the bound over 300 random objects (largest shortfall 0.66 mm, a sphere; cylinders 0.49 mm; boxes 1.1 × 10⁻¹³ mm); quick run 0.861 over 30 |
| `workspace.inside_classification` | the exact classification of the true shapes' boxes in three random volumes | 0 objects: every object is clear of each face by at least 3 mm, more than τ plus the 1.7 % shortfall of a 60 mm radius, so any disagreement is a fault | 0 of 240 (132 inside, 108 outside); quick run 0 of 30 |

The self-test line `workspace` runs both quick checks.

## Known limits

- Volumes are axis-aligned boxes, so a suggested volume around a turned object is larger than the object.
- The inside test uses the object's axis-aligned bounding box: a long thin object turned 45° can be called outside
  although its own surface lies inside.
- An object whose geometry cannot be read (an empty or missing mesh) is not checked; the advice names it.
- The cylinder's and sphere's tessellation makes their boxes up to 1.7 % of the radius small (measured above).

## References

- J. Arvo, "Transforming axis-aligned bounding boxes", in A. S. Glassner (ed.), *Graphics Gems*, Academic Press,
  548–550, 1990 (`arvo1990`).
- SciPy `scipy.spatial.transform.Rotation` (`scipy_rotation`).
