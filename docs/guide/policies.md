# Policies: a worked example on the UR5e sample

Name the policy your data is for, choose the cameras it takes, see what part of each image it uses, and replace a
figure the source does not state with one you measured. The model, the library's sources and the validation are in
[../models/policies.md](../models/policies.md).

## 1. Add a policy

1. **File ▸ Open sample ▸ UR5e and Robotiq 2F-85 with a wrist D405 and an overhead D435**.
2. Open the **Policies** page (Tasks ▸ Policies, or Ctrl+K: "Policies"). The sample comes with one policy,
   "π0 (sample)"; to follow these steps from the start select it and press **Remove**. The headline then reads
   **— · no policy**.
3. Pick **π0 (pi0) · Physical Intelligence** in the combo and press **Add policy** (one undo step). The table shows
   `π0 (pi0) · all (2) · 224 × 224 px · 0`: it takes every camera and depth sensor of the rig, its input is
   224 × 224, and no figure the studies need is missing.
4. Select it. The fields list each figure and its source: action chunk H `50 steps`, "cited: K. Black et al., π0 …,
   arXiv:2410.24164, read 2026-11-06: "we use H=50 for our tasks"". Hover over a value for the full quote.

The headline now reads **0 not stated · camera count**: π0 takes 3 images (a base camera and two wrist cameras)
and the rig has 2. That is information, not an error: π0 masks a missing slot. Tell DexVantage so (step 3).

## 2. See what the policy sees

Select **Overhead D435** and look at the **Camera view** while the Policies page is shown: the policy's input
region is drawn on the image, labelled `π0 (pi0) input 224 × 224 · resized with padding · 44 % padding`. For π0 the
whole 1920 × 1080 colour image is used and shrunk to 224 × 126 (0.117 input pixels per camera pixel), with black
bars above and below. Add **GR00T N1 2B** instead and the rectangle shrinks to the centred 95 % crop
(48, 27)–(1872, 1053), stretched to 224 × 224; **Diffusion Policy** draws its centred 81 % crop.

## 3. Enter your own figure

1. Select the **π0** row. In **Override** pick *Cameras*, enter `2` and the source `this rig has two cameras; the
   third slot is masked`. Press **Set** (one undo step). The field now reads `2` with "yours: measured by the user:
   this rig has two cameras; the third slot is masked", and its tooltip keeps the library's figure. The headline turns
   **0 not stated · all cited**.
2. Add **π0-FAST**: the headline reads **1 not stated · measure them**, "control rate is not stated by its source"
   (the paper lists 5–50 Hz across its robots). Measure your controller's rate, enter it as an override with how you
   measured it, and the headline turns good again.
3. A value that cannot be a figure (`fast`, `224`) or a figure with no source is refused with the reason under the
   buttons; nothing changes. **Clear** brings back the library's figure. Ctrl+Z undoes each step.

## 4. Choose the cameras

Untick **Every camera and depth sensor of the rig** and tick the cameras the policy takes. The token and latency
studies use these cameras; the Camera view draws the input region only on them.

## 5. Export

**Export…** writes every policy, its cameras and each resolved field with its source and origin (library, yours or
not stated), with the project's provenance block, as JSON.

From code, the same figures:

```python
from dexvantage.core import policies
policies.resolve(project, uid)['inference_ms']      # (73, 'K. Black et al., … read 2026-11-06: 73 ms on-board …')
policies.input_geometry(project, uid, (1920, 1080)).dst_rect   # (0.0, 49.0, 224.0, 175.0)
```
