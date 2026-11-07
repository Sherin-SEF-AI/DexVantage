# Versioning a project: diff, review, merge and Git

A worked example on the UR5e sample. The model, the merge rules and the validation are in
[docs/models/versioning.md](../models/versioning.md).

## 1. Keep the project as a folder Git can read

A `.dexv` file is a zip: Git sees a binary blob. Write the project as an unpacked folder instead, one JSON file per
section:

    dexvantage --sample ur5e --unpack cell.dexv.d
    Unpacked UR5e + Robotiq 2F-85 sample: 43 files written, 0 unchanged, 0 removed → cell.dexv.d

    cell.dexv.d/
        project.json  robots.json  variants.json  workspace.json  tasks.json  requirements.json  analysis.json
        rig.json  cables.json  policies.json  calibration.json  data.json  log.json
        sources/…     the robot model files (MJCF, meshes, licences)

In the window: File ▸ Save as folder (for Git)… and File ▸ Open folder…; Save then keeps writing the folder, and only
the files whose content changed are rewritten. `dexvantage cell.dexv.d --pack cell.dexv` makes a `.dexv` file again
(for sending by mail, say). Every command that takes a project also takes the folder.

## 2. Or let `git diff` read .dexv files

If you keep `.dexv` files in Git, give Git a text form of them (a textconv driver: Git runs the program with the
file's name and diffs what it prints):

    # .gitattributes (in the repository)
    *.dexv diff=dexvantage

    # once per clone
    git config diff.dexvantage.textconv "dexvantage --textconv"
    git config diff.dexvantage.cachetextconv true

`git diff`, `git log -p` and `git show` then show changed sections as JSON lines, and mesh arrays and model files as
one line each with their SHA-256. The text does not include the save time, so saving without a change shows nothing.

## 3. Review what changed

Move the wrist camera up 15 mm and swap the overhead D435 for a D455 (in the window, or in Python), save as
`edited.dexv.d`, then:

    dexvantage cell.dexv.d --diff edited.dexv.d --diff-metrics rig.camera_count,bandwidth.margin
    UR5e + Robotiq 2F-85 sample (cell.dexv.d) → UR5e + Robotiq 2F-85 sample (edited.dexv.d)
    2 changes: sensors 2

    section  op       change
    sensors  changed  Sensor Wrist D405: mount pose moved 15.0 mm (x +0.0, y +0.0, z +15.0 mm) and turned 0.00°
    sensors  changed  Sensor Overhead D435: model "intel-realsense-d435" → "intel-realsense-d455"

    Figures of merit (a → b):
      Cameras                     2 count → 2 count  +0 count (same)
      Smallest controller margin  — → —  not stated on both sides

The exit code is 0 when the projects are the same and 1 when they differ (as `diff`), 2 on an error. Add an output
file for the JSON (`--diff edited.dexv.d diff.json`, with provenance) and `--diff-html diff.html` for a page to attach
to a review. "Not stated on both sides": the sample has no host topology, so the bandwidth study states no margin;
nothing is filled in.

In the window, File ▸ Compare with… picks the other project and shows the changes (filter them by section), Swap
reverses the direction, Compare figures runs the metrics of the studies you have run on the open project on both,
and Export writes JSON or HTML.

## 4. Merge two people's edits

Two people edited copies of `base.dexv`: one moved the wrist camera, the other added a side camera.

    dexvantage ours.dexv --merge base.dexv theirs.dexv merged.dexv
    Merged: 1 change of ours, 1 of theirs → merged.dexv

If both changed the same thing differently, nothing is written, the exit code is 1 and each conflict is named:

    dexvantage: The two edits cannot be merged: 1 place was changed differently on both sides: (1) variants ▸ Wrist
    D405 + overhead D435 (sample) ▸ sensors ▸ Wrist D405 ▸ mount ▸ pose (base (…) m, yours (…) m, theirs (…) m).
    Make the change on one side only, or merge these by hand.

To use it as Git's merge driver for `.dexv` files:

    # .gitattributes
    *.dexv merge=dexvantage
    # once per clone (%O base, %A ours and the result, %B theirs)
    git config merge.dexvantage.driver "dexvantage %A --merge %O %B %A"

With the folder form Git merges the JSON files line by line itself; `--merge` is the semantic check (it refuses a
sensor left on a robot the other side removed, which a line merge would accept).
