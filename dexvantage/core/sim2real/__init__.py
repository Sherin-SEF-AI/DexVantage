"""The sim-to-real gap (research tools part 2): a recorded episode rendered again through the project's rig and
compared with the recording, camera by camera, with what explains the gap and the change that closes it.

    gap.analyse(project, dataset, settings=None, progress=None, cancel=None) -> dict   the study's document
    gap.apply_change(project, change) -> [sections]     one suggestion's change (Apply)
    measures                                            the image, depth and distribution measures (no rendering)
    inject.make_gap_dataset(project, folder, gaps)      synthetic recordings with known gaps (for the checks)

The study is core/studies/sim2real.py (batch --sim2real OUT.json --dataset PATH), the page ui/pages/sim2real.py
(Data ▸ Sim-to-real gap), the model docs/models/sim2real.md.
"""
