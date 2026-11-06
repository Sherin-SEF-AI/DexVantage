# Swept volume of a trajectory {#swept}

`core/robot/swept.py`, drawn by the Tasks page (**Swept volume**, spec §13 "the ghosted swept volume of a
trajectory"). A picture of where the arm goes along a joint trajectory, not a measurement: nothing is computed from
it (the cable study tests snags at its own sampled poses, `core/cables.py`; the grasp study tests collisions at each
grasp, `core/studies/grasp_visibility.py`).

## The model

- **Meshes.** The robot's visual triangles (`RobotModel.local_triangles`, each in its own link's frame) are simplified
  once per compiled model by vertex clustering (`core/scenegeom.decimate`, Rossignac and Borrel 1993), link by link
  in the link's own frame, with a cell of 10 mm (`GHOST_CELL_M`, a default). On the samples: UR5e 193 113 triangles,
  SO-100 60 682, the dexhand sample 179 519 before; the decimation takes about 0.36, 0.08 and 0.32 s (measured
  2026-11-07, Ryzen 9 9900X), so the page runs it as a background job the first time and caches it per model.
- **Poses.** `GHOST_POSES` (8, a default; at most 64) times evenly spaced over the trajectory's time range, its first
  and last sample included; at each the trajectory is mapped onto the model's joints (`trajectory.on_model`: joints
  it does not name at home, values clipped to the limits) and each link's simplified triangles are carried by its
  forward-kinematics transform: `p_world = R_link(q(t)) p_link + t_link(q(t))`.
- **Drawing.** One translucent `TriDraw` in the trajectory colour, alpha rising linearly from 0.10 at the first time
  to 0.35 at the last (fainter early, stronger late, so the direction of motion reads).

## Validation

No validation check: the ghosts are a display, and the only computation is forward kinematics, which the kinematics
checks hold against closed-form DH and finite differences. `tests/test_spec_review_p367.py` checks that each ghost
is its link's simplified mesh carried by `fk_array` at that time (10⁻⁵ m, float32 storage) and lies within a cell of
the full meshes' bounds at that pose.

## Known limits

- A set of poses, not the continuous swept volume: what the arm passes through between two ghosts is not drawn.
- The simplified meshes are coarser than the cell near thin parts (a clamp, a finger, a cable) and can lose them.
- The Camera view does not show the ghosts.
