# POSITIONING — what this project is actually worth

Stage 2, Part C, promoted to a first-class deliverable per `docs/DECISIONS.md` D-17: Joel's own
words rejecting "eval-first" as the project's agenda were "the whole agenda is a proper redo,
full redo, it might involve full business idea changes, research changes, architecture changes."
This document takes that literally — the business question gets the same evidentiary standard
as F1 got. No claim here is asserted without the licence/evidence chain that supports it.

---

## 1. The licence question is moot — corrected 2026-09-06, per D-20

**This section previously argued Track A (3D pose/motion) was licence-capped to non-commercial
use and Track B (2D keypoints) was "the licence-clear exit." That framing was correct as an
analysis and is now the wrong conclusion, because the premise it answered no longer applies.**

Author's own words (`docs/DECISIONS.md` D-20): *"I WON'T GET any business profit from this, I'm a
student post-grad... I just need EDUCATIONAL value, self value, interview value, deep AI research
value. ZERO business as I have NO money or funding."* There was never commercial intent for this
work to route around.

**VERIFIED, this session, and still true — but no longer load-bearing for the business
question:** HumanML3D's own upstream README states "Due to the distribution policy of AMASS
dataset, we are not allowed to distribute the data directly" — its motion data inherits
AMASS/SMPL's licence terms (`LANDSCAPE.md` §4.1/4.2): non-commercial research/education/art only,
no redistribution, explicit prohibition on "training methods/algorithms/neural networks/etc. for
commercial use." PoseScript's poses are confirmed SMPL+H G format (SUP-20260906-03), so it
inherits the identical chain.

**The correction:** SMPL/SMPL-X's terms **explicitly permit** exactly what this project is —
non-commercial scientific research, education, and personal projects. HumanML3D, PoseScript,
`smplx`, and the SMPL body models are all **fully available for this project's actual purpose.**
There is no wall to route around, because this project was never trying to walk through the door
the wall blocks. **Stop routing around SMPL for licence reasons** — if SMPL-based rendering,
`smplx`'s differentiable body model, or full mesh output makes the demonstrator (§6) better, use
it. The 2D-keypoint route (Bonnet et al., `LANDSCAPE.md` §3.1) is now one legitimate option among
several on its own technical merits, not the only unencumbered path.

**Unaffected by this correction:** D-11 (representation) and D-12 (dataset/task) were argued on
reproducibility and evaluator-maturity grounds, not licence grounds — see `REBUILD_SPEC.md` §1,
§3. They stand on their own merits regardless of commercial intent.

**Still real, for the record:** if this project's output were ever wanted commercially in the
future, the licence chain above still applies and commercial licensing would still route through
Meshcapade.com or `smpl@max-planck-innovation.de` (`LANDSCAPE.md` §4.1). That is simply not this
project's question right now.

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

## 4. As a product / business artifact — corrected per D-20: there is no business track, and that's fine

**D-20 makes this section simpler than it was.** There is no commercial intent, so "which track
has a business case" is not this project's question. What replaces it, per D-20's own framing —
*"people want to see a usable PRODUCT out of AI projects... move past research AFTER FINISHING
research architectures, and do an outcome, productive show"* — is: **which track makes the better
Stage 5 demonstrator** (§6), not which one avoids a licence wall that was never actually blocking
anything for this project's real purpose.

Both tracks remain live on their technical merits:

- **Track A (3D pose/motion, the research track, `REBUILD_SPEC.md` §0-§6):** fully available
  under SMPL/AMASS's non-commercial terms (§1). Its output is joint positions/rotations over a
  skeleton — harder to make legible to a non-specialist without rendering, but closer to the
  actual research question D-11/D-12 argue for.
- **Track B (text -> 2D pose -> image, `LANDSCAPE.md` §3.1-§3.2):** Bonnet et al.'s MIT-licensed
  approach, or ControlNet+OpenPose more generally. Immediately legible (a picture of a person in a
  pose), cheapest to stand up (no training required for the image half), but a layer removed from
  the actual research question.

**Neither is "the exit" any more — the choice is a demonstrator-design question, not a licence
one.** See §6 for what Stage 5 actually requires, which is a sharper and more useful question than
"which track has a business case."

---

## 5. Who would use a text-to-pose system, and for what — separated by track

- **Track A (3D pose/motion):** other researchers benchmarking against a documented,
  evaluation-first HumanML3D baseline; the author, as a portfolio/interview artifact; students or
  engineers learning from `docs/LANDMINES.md` as a generalizable checklist.
- **Track B (2D pose -> image):** illustrators and character artists wanting a quick pose
  reference from a text description; indie game developers prototyping character poses without a
  full 3D rig; hobbyist/small-studio content creators doing pose-guided image or short-animation
  generation, per the Animate Anyone / Champ precedent. (Both tracks are equally available under
  D-20 — this split is about who benefits from each, not who's licence-permitted.)

---

## 6. Stage 5 — the Demonstrator (per D-20; not started, design-constrained now)

**Not optional, and the project does not end at `RESULTS.md`** — a metrics table is evidence, not
an outcome; nobody outside the field can look at an FID and see anything. **Explicitly not a
deployed service** — author's words: *"Something local is fine too."* Do not start building it
until E0-E4 (`REBUILD_SPEC.md` §6) are done — but every design decision between here and there
should keep it reachable: if a choice makes the demonstrator harder for no research gain, take
the other one.

**Five requirements, all must hold (per the director session's framing, recorded here since it
directly shapes this document's earlier "smallest demonstrable thing" recommendation):**
1. **Thirty-second legibility.** A non-specialist gets it with no explanation — type a sentence,
   watch a human move (or a rendered pose appear). This is why Track B (an image/render) reads
   more immediately than Track A's raw joint output alone — but Track A can satisfy this too if
   rendered.
2. **The value is legible, not asserted.** Show the generated result **beside a nearest-neighbour
   retrieval from the training set** — that baseline is what "just look it up" already gives you
   for free. If the model isn't beating that, the demo makes it visible rather than hides it. This
   is the sharpest requirement in the list: it forecloses a demo that only shows good-looking
   cherry-picked outputs.
3. **Surfaces real metrics, seed spread, and failure cases in the interface itself** — not just a
   polished happy-path. The FID/R-Precision numbers and their seed spread (`LANDMINES.md` §7)
   belong in the UI, not just in `RESULTS.md`.
4. **Runs from a checkpoint in one command, no GPU, no retraining.** CPU-inference-only, matching
   how this project's own E0a evaluator already runs (`docs/EXPERIMENT_LOG.md`).
5. **Reproducible by someone else.** Not a personal script — an artifact a stranger could clone
   and run.

**Likely shape:** a Gradio or Streamlit app rendering an animated/static skeleton (or, if Track B
is chosen, the ControlNet-rendered image), with the nearest-neighbour comparison and metrics panel
built in from the start rather than added after. **Build a demonstration, not a product** — per
D-20, this is explicitly not a deployed web service.

**Why this replaces the earlier "text box -> image" recommendation, not just extends it:** the
earlier version of this section proposed a single generated image as sufficient. Requirement 2
above (nearest-neighbour comparison) makes clear that a single output, however good-looking, is
not actually legible evidence of value — a demo needs to show the model earning its result against
the trivial baseline, not just produce something plausible.

**Not yet done:** the specific ControlNet SDXL-OpenPose checkpoint's licence has not been
independently re-verified in this pass (`REBUILD_SPEC.md` §5, last table row) — check before
building on it. Nothing in this section is built yet; recorded here so design choices in Stage 3/4
stay compatible with it.

---

## 7. Summary verdict — corrected 2026-09-06 for D-20

- **Learning artifact:** real value, already partly delivered, generalizes past this project.
- **Interview artifact:** strong, with one caution already surfaced to Joel directly by review
  (not repeated as an action item here).
- **Business artifact:** not this project's question. Zero commercial intent (D-20) — the licence
  analysis in §1 is preserved for the record (it's correct and would matter if that ever changed)
  but no longer decides anything here.
- **What actually replaces it:** Stage 5, a required local demonstrator (§6) satisfying five
  concrete legibility/honesty requirements — not started, not optional, design decisions from
  Stage 3/4 onward should keep it reachable.
