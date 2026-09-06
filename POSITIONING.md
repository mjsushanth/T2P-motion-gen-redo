# POSITIONING — what this project is actually worth

Stage 2, Part C, promoted to a first-class deliverable per `docs/DECISIONS.md` D-17: Joel's own
words rejecting "eval-first" as the project's agenda were "the whole agenda is a proper redo,
full redo, it might involve full business idea changes, research changes, architecture changes."
This document takes that literally — the business question gets the same evidentiary standard
as F1 got. No claim here is asserted without the licence/evidence chain that supports it.

---

## 1. The headline, not a caveat: on these datasets, there is no commercial product at the end of this road

**Stated as plainly as the evidence supports, per review (`reviews/REVIEW_QUEUE.md`
SUP-20260906-04): this is not "consult a lawyer." The prohibition is explicit, and it covers the
trained model, not just the data.**

**VERIFIED, this session:** HumanML3D's own upstream README states "Due to the distribution
policy of AMASS dataset, we are not allowed to distribute the data directly" — its motion data
inherits AMASS/SMPL's licence terms, which are (`LANDSCAPE.md` §4.1/4.2, both VERIFIED by direct
fetch of the licence pages): non-commercial research/education/art only, no redistribution,
**explicit prohibition on "training methods/algorithms/neural networks/etc. for commercial use."**
**VERIFIED, review, SUP-20260906-03:** PoseScript's poses are SMPL+H G format — genuine SMPL
parameters — so it inherits the identical chain, stacked on top of its own CC BY-NC-SA 4.0. The
`smplx` Python package itself is bundled under the same non-commercial terms as the model data,
not separately permissively licensed (`LANDSCAPE.md` §4.3).

**Consequence, stated plainly:** any model trained on HumanML3D-derived or PoseScript-derived
data, however well engineered, is a non-commercial research/portfolio artifact by licence, not by
engineering choice — this project's work cannot design around it, because it is a property of
the training data's provenance, not of the model.

**This makes the business question a routing question, with exactly three exits:**

- **(a) Commercial licensing.** Negotiate directly — Meshcapade.com or
  `smpl@max-planck-innovation.de` (`LANDSCAPE.md` §4.1). A real, available option; not something
  this project's engineering resolves, a business/legal decision for Joel to make separately if
  Track A's output is ever wanted commercially.
- **(b) A differently-licensed dataset.** Not found in this landscape survey — every static-pose
  or motion dataset surveyed here that has real benchmarks and tooling (HumanML3D, PoseScript)
  routes through AMASS/SMPL. Not ruled out in general, just not identified as of this pass.
- **(c) Abandon SMPL, work in 2D keypoints.** The one released, working, MIT-licensed artifact in
  this whole landscape survey — Bonnet et al.'s text-to-pose (`LANDSCAPE.md` §3.1) — uses
  DWpose-style 2D keypoints, **explicitly not SMPL.** That may be exactly why it's the one thing
  here unencumbered end to end. This is Track B, below — **not a recommendation to pick this
  exit over (a) or (b)**, but the one this document can actually build toward and test cheaply,
  since it requires no licence negotiation and no different dataset search to start with.

**The one part of the landscape that is NOT under this restriction:** Bonnet et al.'s
text-to-pose work (`LANDSCAPE.md` §3.1) generates **2D keypoints** (18 body + 42 hand + 68 face,
DWpose-style), not SMPL parameters, and is MIT-licensed end to end
([clement-bonnet/text-to-pose](https://github.com/clement-bonnet/text-to-pose)). Stable Diffusion
XL, which their pose adapter conditions, is under Stability AI's CreativeML Open RAIL++-M licence
— a behavior-restriction licence (bars certain output uses) rather than a non-commercial-only
licence like SMPL's. **This is the one track in this whole landscape survey that is not
licence-blocked from commercial framing**, because it never touches SMPL/AMASS at all. This
matters enough to shape §4 below.

---

## 2. As a learning artifact — real, and already partly delivered

Four separately-verified findings (F1-F4, `FORENSICS.md`) plus two more found by a review's
deeper audit (`docs/LANDMINES.md` §11, §12 — CFG folded into the training loss making the
objective trivially satisfiable by ignoring conditioning; per-batch normalisation and a
mis-indexed timestep corrupting the anatomy-loss computation) constitute a genuinely rare
artifact: a *documented, evidence-chained account of exactly how a three-month project produced
zero real measurement*, with each defect independently verified rather than asserted. This has
standalone value as a case study regardless of what Stage 3/4 produce — `docs/LANDMINES.md` is
already a reusable checklist ("traps that produce plausible wrong answers with no error raised")
that generalizes past this specific project (§1's generalisation note: "this applies to any
packed motion/pose format — SMPL parameter vectors, AMASS, MotionX").

**Established, not assumed:** the forensics alone are worth keeping and showing, independent of
whether Stage 3/4 ever produce a competitive model.

---

## 3. As an interview artifact — strong, with a specific caution

**Strong, because:** "I found a decode bug that misread 4 root-motion scalars as position data,
verified it two independent ways (primary-source code + a bone-length invariant test with the
decision rule stated before running), and it explains six of the ten failure modes in the
project" is a genuinely good story — process, not just outcome, and it demonstrates exactly the
kind of scepticism ("does the arithmetic summing to 263 actually mean the layout is right?
No — validate with an invariant") that a technical interviewer wants to see.

**The specific caution, already surfaced by review and explicitly not mine to act on:**
the "99.995% loss reduction" framing that this whole project exists to debunk was found, by the
review pass, sitting in the Obsidian deep-dive notes **inside a scripted 60-second
interview-answer passage** (`LANDMINES.md` §4). That means the number this project disproves is
specifically one Joel has been rehearsing to say out loud. This document is not the place to
decide what to do about that — the review already raised it with Joel directly, separately
from this repo's content, and explicitly said not to act on it without instruction. Recorded here
only because it's directly relevant to "interview artifact" value: **the corrected story (a real
bug, found and fixed, with a measured before/after) is a strictly better interview answer than
the original one, not just a more honest one.**

---

## 4. As a product / business artifact — two different tracks, evaluated separately

### Track A: 3D pose/motion generation itself (the research track, `REBUILD_SPEC.md` §0-§6)

**No commercial business case, stated in those words.** The training data (HumanML3D and/or
PoseScript) is non-commercial-restricted by licence at the source, not by this project's choice.
A model trained on it cannot be commercialized without separately negotiating SMPL/Max Planck's
commercial licence terms (`LANDSCAPE.md` §4.1: "Commercial licensing is routed through
Meshcapade.com or smpl@max-planck-innovation.de") — a real, separate business decision this
project's engineering work does not and cannot resolve. **This track's honest ceiling is
research/portfolio value (§2, §3), not a product.**

### Track B: text -> 2D pose -> image/animation (the demonstrability track)

**Established, not assumed, that this track is different:** Bonnet et al.'s approach (§1) does
not touch SMPL/AMASS licensing at all. A from-scratch or fine-tuned version of *this specific
pipeline shape* — text -> 2D keypoints -> ControlNet-conditioned image (or, further out, Champ/
Animate-Anyone-style conditioned video) — is not licence-blocked from commercial framing the way
Track A is. **Who would use this and for what:** concrete, verifiable use cases already exist in
the wild for exactly this pipeline shape — pose-guided character art/illustration tools, virtual
try-on and fashion visualization (Animate Anyone's own reported benchmark domain,
`LANDSCAPE.md` §3.2), and rapid pose-reference generation for illustrators/animators who currently
pose reference figures by hand or search stock photos for a matching pose.

**What would have to be true for Track B to be worth more than a portfolio piece:**
1. The text-to-2D-pose step needs to beat or match a KNN/retrieval baseline meaningfully (Bonnet
   et al.'s own bar: 78% win rate on their CLaPP score) — otherwise "generate a pose from text" is
   not adding value over "retrieve a similar existing pose."
2. The downstream image quality needs to be competitive with just using ControlNet+OpenPose on a
   manually-posed or hand-drawn reference — i.e., the value has to be in the *text-to-pose* step
   specifically, not merely in re-plumbing an existing ControlNet demo.
3. A real target user (illustrator, animator, game-pose-reference tool) would need to actually
   prefer this over existing tools (manually posing a 3D mannequin app, or searching stock pose
   references) — this project has not talked to any such user, so this is a real, unresolved
   assumption, not a validated one.

**Honest verdict on Track B: plausible, not established.** The licence path is clear; the
user-value path is not yet tested. This is the right thing to build the "smallest demonstrable
end-to-end thing" (§5) around, precisely because it's cheap to test whether points 1-2 above hold
— and testing them would be the actual next step toward resolving whether this is a real business
idea or a well-scoped demo, rather than assuming either answer.

---

## 5. Who would use a text-to-pose system, and for what — separated by track

- **Track A (3D pose/motion, research-only by licence):** other researchers benchmarking against
  a documented, evaluation-first HumanML3D baseline; Joel himself, as a portfolio/interview
  artifact; students or engineers learning from `docs/LANDMINES.md` as a generalizable checklist.
- **Track B (2D pose -> image, commercially unblocked):** illustrators and character artists
  wanting a quick pose reference from a text description; indie game developers prototyping
  character poses without a full 3D rig; hobbyist/small-studio content creators doing
  pose-guided image or short-animation generation, per the Animate Anyone / Champ precedent.

---

## 6. The smallest demonstrable end-to-end thing a non-expert can look at and immediately get

**Recommendation: a text box, and an image of a person in the described pose, coming out the
other side.** Concretely: type "a person kicking a soccer ball," see a rendered figure in
approximately that pose. This is Track B, using the ControlNet+OpenPose route (`LANDSCAPE.md`
§3.2) as the fastest path to a working end-to-end demo (no training required — pretrained SDXL +
a public OpenPose ControlNet checkpoint), with the project's own text-to-pose model (once E2/E3
in `REBUILD_SPEC.md`'s ablation ladder produce something reasonable) swapped in as the pose
source once it exists.

**Why this, not a bone-length table or an FID number:** every other output this project produces
(FID, R-Precision, bone-length CV) requires domain knowledge to interpret. An image of a person
in a described pose requires none — anyone can look at it and judge, correctly, whether it
worked. This is the artifact to actually show someone, separate from the research record they'd
have to be a specialist to evaluate.

**Not yet done:** the specific ControlNet SDXL-OpenPose checkpoint's licence has not been
independently re-verified in this pass (`REBUILD_SPEC.md` §5, last table row) — check before
building on it, not after.

---

## 7. Summary verdict

- **Learning artifact:** real value, already partly delivered, generalizes past this project.
- **Interview artifact:** strong, with one caution already surfaced to Joel directly by peer
  review (not repeated as an action item here).
- **Business artifact, Track A (3D research):** no business case — stated in those words, because
  the training data's own licence forecloses it, not because the engineering is weak.
- **Business artifact, Track B (2D pose -> image):** plausible but unestablished — the licence
  path is genuinely clear, unlike Track A, but real user value has not been tested. Worth
  building the demo specifically to find out, not worth claiming a business case exists yet.
