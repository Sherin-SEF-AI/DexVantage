# Visual-to-tactile handoff and the sensory blind gap (spec §7.15)

Part 2 of `dexvantage/core/studies/dex_hand_tactile.py` (the multi-finger occlusion study, `--dex-occlusion`; the
result's `handoff` key; `tactile_handoff(project)` runs it alone) · the chart on **Analysis ▸ Multi-finger
occlusion** · check `dexhand.blind_gap_bruteforce` (`dexvantage/validation/checks_chunk.py`).

The question: as the hand closes in on the object, when do the cameras lose sight of each fingertip's contact region,
and does a tactile pad take over at contact? The distance before contact over which a contact is neither seen nor
touched is the **sensory blind gap**.

## The approaches

They end at the study's contact cases: each grasp of a grasp set reached by IK, or each grasp posture fully closed
(at most `handoff_max_cases`, default 6, evenly spaced). Along each, the gripper frame is backed off along −approach
(the grasp's approach, else the gripper's approach axis, `core/grasps.py` `gripper_geometry`, turned with the frame)
by a distance d, and the arm is solved by IK from the solution at the nearer distance (`nearest_q0`, one seed,
self-collision not checked; the hand keeps its contact posture), as the grasp study poses its pre-contact steps.

At each d, part 1's kernel (`visibility_at`: exact ray casting by default, owner groups, the cameras' lens models)
gives each fingertip's share of its contact patch that some camera sees. Along the approach the task objects occlude
(`handoff_objects_occlude`, default on: the object is what hides the contact as the fingers close on it). A fingertip
is seen from `min_visible_pct` (50 %, a default).

Distances: 0 to `handoff_max_mm` (default 50 mm, the grasp study's pre-contact default) every `handoff_step_mm`
(10 mm), then every `handoff_refine_mm` (2 mm) inside each coarse interval where a fingertip's seen state changes.

## The tactile pads at contact

Every tactile sensor in use on a link of a finger covers that finger's patch points within `pad_gap_mm` (3 mm) of its
sensing plane and inside its pad, `pad_mm` from the library entry (GelSight Mini 18.6 × 14.3 mm, DIGIT 19 × 16 mm,
both cited; not stated → coverage not known). A pad left at its link's origin (the mount pose not set, as the Library
adds one) is **assumed** centred on the patch, facing along its mean normal, its longer side along the patch's longer
extent; the result labels it. A placed pad senses along its mount's body x axis (REP-103 forward), its sides along
body y and z. A fingertip is touched from `min_touch_pct` (50 %) of its patch on a pad.

## The gap

From (distance, seen) samples of a fingertip:

    gap = 0                                  when seen at contact
        = (d_last_hidden + d_first_seen) / 2  otherwise
        = d_max ("or more")                   when never seen

and the hidden total counts every stretch not seen, a change of state at the midpoint between samples. Touch begins
only at contact, so a fingertip neither seen nor touched at contact is **unsensed at contact** whatever its gap.
Flags: a gap above `blind_gap_max_mm` (default 10 mm) or an unsensed contact. Status: good with no flag, warning for
flagged gaps up to 2 × the threshold, serious above, critical when a contact is unsensed (defaults).

## Validation (measured 2026-11-07)

`dexhand.blind_gap_bruteforce`: the dexhand sample with only its head camera and a 4 mm plate between it and the
hand, its top edge where the line of sight to the patches' centre passes 20 mm before contact; the fully closed power
posture, 0 to 40 mm. The adaptive scan (13 visibility evaluations) against a scan of every distance at 2 mm (21
evaluations): **0 mm difference** over the five fingertips, two transitions (the middle finger lost at 19 mm, the ring
finger at 37 mm; the first finger, little finger and thumb not seen over the whole 40 mm); tolerance one refine step,
2 mm, because both read the same kernel at the same distances where they meet and can differ only where a short
seen or hidden stretch falls between two coarse samples.

## On the samples

The dexhand sample (no task object, no tactile sensor): its three closed postures are seen by the head camera up to
contact (gap 0 mm, status good), and the notes say no tactile sensor is on the fingers. With a DIGIT on every
fingertip and the plate above, the first three fingers are touched at contact (72–78 % of the patch on the pad), the
little finger's patch is 48 % covered and the thumb's 11 % (its patch wraps round the distal link, wider than the 19
× 16 mm pad), so those two stay unsensed at contact: critical, and the flags name them. Time: about 6 s for three
approaches (18 evaluations, exact ray casting on the CPU; no GPU is used by the default backend), on top of part 1's
4.5 s; `--dex-occlusion` on the sample takes about 12 s with start-up.

## Known limits

The approach is a straight line in the gripper frame's position with the hand already in its contact posture (the
fingers do not close as it comes in); a seen or hidden stretch shorter than the coarse step inside one interval can
be missed (the check measures it on one scene); a distance IK cannot reach is left out and counted; touch is judged
only at contact and by geometry (a pad's footprint on the patch), not by contact force or the sensor's sensitivity;
an assumed pad placement is a guess the result labels, and only the pad's mount pose says where it really is.
