# Sparse voxels with adaptive refinement {#voxels}

Module: `dexvantage/core/voxels.py`. Tests: `tests/test_voxels.py`. Spec §7.1.

## The grid

`voxels.SparseGrid.from_project(project, voxel_mm=None, refine_mm=None, margin_m=0.02)` covers each task volume
(an axis-aligned box, `project.workspace.volumes`) with cubic base voxels of edge `voxel_mm` (default
`project.analysis.voxel_mm`, 5 mm by default): ceil(edge / voxel) per axis, as `core.jobs.voxel_count` counts them,
the grid centred on the volume so it overhangs each face by less than half a voxel.

A base voxel is **refined**, split into k³ voxels with k = ceil(voxel / refine) so that the refined edge voxel / k is
at most `refine_mm` (default `project.analysis.refine_mm`, 2 mm by default: 5 mm / 3 = 1.67 mm), when

- its volume is a refinement region (`Volume.refine`), or
- it overlaps a task object's bounding box grown by `margin_m` (20 mm, a default): the object and its contact zone.

The grid gives `.centers` (N, 3) float32 world metres, `.size` (N,) float32 edge metres, `.volume_index` (N,) int32
into `.volumes`, `.refined` (N,) bool, `.count`, `.volume_m3` and `.chunks(n)`, which yields
`Chunk(start, stop, centers, size, volume_index)` views of at most n voxels so a study's per-voxel temporaries stay
bounded. The voxels' volumes add up to the base grid's volume exactly (refinement splits, it never adds).

## The limit and its message

`voxels.estimate(project, voxel_mm=None, refine_mm=None)` counts the voxels before anything is built (a byte per base
cell for the refinement mask) and returns `count`, `dense_count` (a dense grid at the refined size over the same
volumes), `feasible` (count ≤ `core.jobs.DENSE_LIMIT`, 5 × 10⁷ voxels) and a plain-English `message`.
`SparseGrid.from_project` refuses a grid over the limit with `core.jobs.StudyError(message)`. Spec §7.1's example
reads, for one 1.5 × 1.5 × 1.2 m volume at 1 mm:

> A dense 1 mm grid over 1.5 × 1.5 × 1.2 m is 2.7 × 10⁹ voxels, which is over the limit of 5 × 10⁷ and is not
> built: use a coarser voxel size and keep the fine size to refinement regions around the objects or the contact
> zone.

At the default 5 mm the same volume is 2.2 × 10⁷ voxels and fits; the message then says so, and that a dense 1.7 mm
grid would be 5.8 × 10⁸ voxels, over the limit. `voxels.dense_message(size_m, voxel_m)` writes the sentence for any
box.

Sizes on the samples: the UR5e tabletop volume (0.50 × 0.40 × 0.30 m) is 480,000 voxels at 5 mm; the SO-100's
(0.30 × 0.20 × 0.12 m) is 57,600.

## Known limits

- One level of refinement (base → k³), not a full octree.
- The refined zone around an object is its bounding box grown by the margin, not its shape.
- Volumes are axis-aligned boxes; where two volumes overlap, the shared space has voxels for each (each volume's
  statistics stay whole), so a sum over all voxels counts it twice.
- The 5 × 10⁷ limit (`core.jobs.DENSE_LIMIT`) is a memory bound of this build (about 1 GB for the centres, sizes,
  volume indices and refinement flags at 21 bytes per voxel), not a property of the method.
