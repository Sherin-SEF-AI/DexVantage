# DexVantage user guide

A step-by-step guide with a worked example for every feature. Each phase adds its chapter.

## 1. The window (Phase 0)

1. Run `dexvantage`. An empty project opens with a default tabletop task volume (500 × 600 × 300 mm, labelled
   default).
2. **Left:** the Outliner (project, robots, sensors of the active rig variant, workspace, tasks) and the Library (it
   fills in Phase 1 and 2). **Centre:** the 3D view with the pose scrubber under it. **Right:** the Inspector and the
   Camera view; Ctrl+J opens the Assistant column. **Bottom:** Analysis.
3. Orbit with the middle button or a left drag on empty space (about the point under the cursor); Shift pans; the
   wheel zooms towards the cursor. 7, 1 and 3 look from the top, the front and the side; 5 toggles orthographic;
   Home frames everything. The control in the top right does the same with a mouse or touchpad only.
4. **Worked example.** Sensors ▸ Add camera on a tripod adds a camera looking at the task volume. Select the task
   volume in the Outliner and type `1.5` into each Size field (or `1500 mm`): the Inspector warns that a dense 1 mm grid
   would need 2.7 × 10⁹ voxels, and the Analysis summary updates. Ctrl+Z undoes each change as one step.
5. Ctrl+K opens the command palette; Ctrl+F finds an object; `?` lists every shortcut.
6. Window ▸ workspaces switch the panel layout (Placement, Grasp study, Teleop, Calibration, Data audit, Review);
   Window ▸ Save workspace layout keeps your arrangement, Reset puts the default back.
7. File ▸ Save writes a `.dexv` file; File ▸ Render image (F12) writes a 2× render; Export ▸ Export the project
   summary writes JSON with its provenance block.
8. Help ▸ Coordinate frames shows the world and optical frames; Help ▸ Validate the models runs the validation suite
   and opens its report.

## 2. Batch mode

```bash
dexvantage cell.dexv --analyse summary.json        # the summary, with provenance
dexvantage cell.dexv --reproduce summary.json      # re-run and compare: "Reproduces exactly …"
dexvantage --validate report/                      # validation report (exit 0 pass, 1 fail, 2 error)
dexvantage --self-test                             # check this installation
```
