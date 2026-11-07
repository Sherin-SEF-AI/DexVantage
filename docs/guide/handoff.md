# The visual-to-tactile handoff: a worked example

The Multi-finger occlusion page (**Analysis ▸ Multi-finger occlusion**) shows, under its sweep, the handoff chart:
for each approach, how much of each fingertip's contact patch the rig sees against the distance to contact, and
whether a tactile pad takes over at contact. The model is in `docs/models/handoff.md`.

## 1. The dexhand sample as it is

1. **File ▸ Open sample ▸ dexhand** (the UR5e with the Shadow Hand and three candidate cameras).
2. Open **Analysis ▸ Multi-finger occlusion** and press **Run**. The handoff runs with the occlusion matrix (about 6 s
   more; settings `handoff`, default on).
3. Under "Visual-to-tactile handoff", choose an approach (each grasp posture fully closed). The chart has the distance
   to contact on its x axis, contact at the right; one line per fingertip (% of its patch seen by any camera); the
   dashed line at 50 % is the "seen" threshold (a default). The head camera sees every fingertip up to contact: "no
   blind gap". The note says no tactile sensor is on the fingers, so nothing would be touched at contact.

## 2. A blind gap and the pads that close it

1. Add a task object between the head camera and the hand (a thin plate whose top edge is a little above the
   fingertips' line of sight). Along the approach the objects occlude (a default), so fingertips disappear behind it.
2. Run again. The chart shades each flagged fingertip's blind gap; the headline line reads **"contact unsensed"**
   (critical): those fingertips are hidden for the last part of the approach and nothing touches them at contact.
3. Add a **DIGIT** from **Library ▸ Tactile sensors** for each fingertip (each goes on a fingertip link; leave its
   mount pose as added and the study assumes it sits centred on the patch, which the result labels). Run again: the
   points at contact show each fingertip's pad coverage. The first three fingers are touched (72–78 %); the little
   finger (48 %) and the thumb (11 %) are not: the pad is smaller than their curved patches. Those two remain
   unsensed at contact and are flagged; move or add pads there, or a camera that sees them.

The blind gap and its flags are in the exported JSON under `handoff` (`approach_visibility`, `tactile_coverage`,
`blind_gap_m`, `flags`, `status`), and in batch:

    dexvantage --sample dexhand --dex-occlusion dex.json
