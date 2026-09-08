# POSITIONING — what this project is actually worth

This document takes the business question as seriously as any technical one — no claim here is
asserted without the same licence/evidence chain that supports a technical finding elsewhere in
this repo.

---

## 1. The licence question is moot

HumanML3D's own upstream README states "Due to the distribution policy of AMASS dataset, we are
not allowed to distribute the data directly" — its motion data inherits AMASS/SMPL's licence
terms (`LANDSCAPE.md` §4.1/4.2): non-commercial research/education/art only, no redistribution,
explicit prohibition on "training methods/algorithms/neural networks/etc. for commercial use."
PoseScript's poses are confirmed SMPL+H G format, so it inherits the identical chain.

This project has no commercial intent — it is self-funded, non-commercial research and education,
with no funding or revenue plan attached. **That premise makes the licence question moot rather
than blocking:** SMPL/SMPL-X's terms **explicitly permit** exactly what this project is —
non-commercial scientific research, education, and personal projects. HumanML3D, PoseScript,
`smplx`, and the SMPL body models are all **fully available for this project's actual purpose.**
There is no wall to route around, because this project was never trying to walk through the door
the wall blocks. If SMPL-based rendering, `smplx`'s differentiable body model, or full mesh output
makes the demonstrator (§6) better, use it — a 2D-keypoint alternative (`LANDSCAPE.md` §3.1) is
one legitimate option among several on its own technical merits, not the only unencumbered path.

**Unaffected by this:** the project's representation and dataset/task choices were argued on
reproducibility and evaluator-maturity grounds, not licence grounds, and stand on their own merits
regardless of commercial intent.

**Still real, for the record:** if this project's output were ever wanted commercially in the
future, the licence chain above still applies and commercial licensing would still route through
Meshcapade.com or `smpl@max-planck-innovation.de` (`LANDSCAPE.md` §4.1). That is simply not this
project's question right now.

---

## 2. As a learning artifact — real, and already partly delivered

Four separately-verified findings (`FORENSICS.md`) plus two more found by a deeper audit
(`docs/LANDMINES.md` §11, §12 — CFG folded into the training loss making the objective trivially
satisfiable by ignoring conditioning; per-batch normalisation and a mis-indexed timestep
corrupting the anatomy-loss computation) constitute a genuinely rare artifact: a *documented,
evidence-chained account of exactly how a three-month project produced zero real measurement*,
with each defect independently verified rather than asserted. This has standalone value as a case
study regardless of what any later research stage produces — `docs/LANDMINES.md` is already a
reusable checklist ("traps that produce plausible wrong answers with no error raised") that
generalizes past this specific project (its own generalisation note: "this applies to any packed
motion/pose format — SMPL parameter vectors, AMASS, MotionX").

**Established, not assumed:** the forensics alone are worth keeping and showing, independent of
whether a competitive model ever gets built on top of them.

---

## 3. As an interview artifact — strong, with a specific caution

**Strong, because:** "I found a decode bug that misread 4 root-motion scalars as position data,
verified it two independent ways (primary-source code + a bone-length invariant test with the
decision rule stated before running), and it explains six of the ten failure modes in the
project" is a genuinely good story — process, not just outcome, and it demonstrates exactly the
kind of scepticism ("does the arithmetic summing to 263 actually mean the layout is right?
No — validate with an invariant") that a technical interviewer wants to see.

**The specific caution:** the "99.995% loss reduction" framing this whole project exists to
debunk was found sitting in a separate set of deep-dive study notes, **inside a scripted
60-second interview-answer passage** (`docs/LANDMINES.md` §4) — meaning the number this project
disproves is specifically one that was being rehearsed to say out loud. Recorded here only because
it's directly relevant to interview-artifact value: **the corrected story (a real bug, found and
fixed, with a measured before/after) is a strictly better interview answer than the original one,
not just a more honest one.**

---

## 4. As a product / business artifact — there is no business track, and that's fine

There is no commercial intent, so "which track has a business case" is not this project's
question. What replaces it: **which track makes the better demonstrator** (§6), not which one
avoids a licence wall that was never actually blocking anything for this project's real purpose.

Both tracks remain live on their technical merits:

- **Track A (3D pose/motion, the research track):** fully available under SMPL/AMASS's
  non-commercial terms (§1). Its output is joint positions/rotations over a skeleton — harder to
  make legible to a non-specialist without rendering, but closer to the actual research question
  this project's representation and dataset choices argue for.
- **Track B (text -> 2D pose -> image, `LANDSCAPE.md` §3.1-§3.2):** Bonnet et al.'s MIT-licensed
  approach, or ControlNet+OpenPose more generally. Immediately legible (a picture of a person in a
  pose), cheapest to stand up (no training required for the image half), but a layer removed from
  the actual research question.

**Neither is "the exit" — the choice is a demonstrator-design question, not a licence one.** See
§6 for what the demonstrator actually requires, which is a sharper and more useful question than
"which track has a business case."

---

## 5. Who would use a text-to-pose system, and for what — separated by track

- **Track A (3D pose/motion):** other researchers benchmarking against a documented,
  evaluation-first HumanML3D baseline; a portfolio/interview artifact; students or engineers
  learning from `docs/LANDMINES.md` as a generalizable checklist.
- **Track B (2D pose -> image):** illustrators and character artists wanting a quick pose
  reference from a text description; indie game developers prototyping character poses without a
  full 3D rig; hobbyist/small-studio content creators doing pose-guided image or short-animation
  generation, per the Animate Anyone / Champ precedent. (Both tracks are equally available under
  §1 — this split is about who benefits from each, not who's licence-permitted.)

---

## 6. The demonstrator

**Not optional, and the project does not end at a metrics table** — a table of numbers is
evidence, not an outcome; nobody outside the field can look at an FID and see anything.
**Explicitly not a deployed service** — a local, runnable artifact is the goal. Design decisions
throughout the research work should keep the demonstrator reachable: if a choice makes it harder
for no research gain, take the other one.

**Five requirements, all must hold:**
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
   polished happy-path. The FID/R-Precision numbers and their seed spread (`docs/LANDMINES.md` §7)
   belong in the UI, not just in a separate results document.
4. **Runs from a checkpoint in one command, no GPU, no retraining.** CPU-inference-only, matching
   how this project's own evaluator sanity check already runs.
5. **Reproducible by someone else.** Not a personal script — an artifact a stranger could clone
   and run.

**Likely shape:** a Gradio or Streamlit app rendering an animated/static skeleton (or, if Track B
is chosen, the ControlNet-rendered image), with the nearest-neighbour comparison and metrics panel
built in from the start rather than added after. **Build a demonstration, not a product.**

**Why this matters more than a single generated image:** a single output, however good-looking, is
not actually legible evidence of value — a demo needs to show the model earning its result against
the trivial baseline, not just produce something plausible. This project's own demo (`demo/`)
follows exactly this requirement.

**Not yet done:** the specific ControlNet SDXL-OpenPose checkpoint's licence has not been
independently re-verified — check before building on it, if Track B is ever pursued.

---

## 7. Summary verdict

- **Learning artifact:** real value, already partly delivered, generalizes past this project.
- **Interview artifact:** strong, with one caution already noted in §3.
- **Business artifact:** not this project's question. Zero commercial intent — the licence
  analysis in §1 is preserved for the record (it's correct and would matter if that ever changed)
  but no longer decides anything here.
- **What actually replaces it:** a required local demonstrator (§6) satisfying five concrete
  legibility/honesty requirements.
