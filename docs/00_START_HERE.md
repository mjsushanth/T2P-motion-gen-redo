# START HERE — orientation for a fresh session (human or agent)

> Read this first. Five minutes. It tells you what this project is, what is actually known
> versus merely suspected, and the specific ways this domain produces wrong answers that
> raise no error. If you read nothing else, read sections 1, 2 and 7.

---

## 1. The one thing to know

**This project exists because a three-month research project produced no measurement.**

I built a CLIP-conditioned diffusion model for text-to-pose
generation on HumanML3D, as a graduate course project. It ran. It trained. It produced a
9-page NeurIPS-format report. And every number in that report is a **training loss**.

There is no validation loop, no test-split evaluation, no seeding, no experiment tracking,
and not one of the field's standard metrics — no R-Precision, no FID, no MM-Dist, no
Diversity, no MultiModality, no MPJPE. The much-quoted "99.995% loss reduction" compares three
numbers produced by three *different loss functions*, the first of which was a bug artifact of
order 1e15. (That figure appears nowhere in the report itself — it is a later summary of it.)

**So the results were never subpar. They were never taken.** That is the thing to
internalise before you form any other impression of this work.

**The second thing:** there is a strong suspicion, not yet empirically confirmed at the time
of writing, that the project sliced the wrong 66 numbers out of HumanML3D's 263-dimensional
motion vector — and that this single bug is the root of the entire documented struggle. See
`LANDMINES.md` §1 and `../BRIEFING.md` F1. **Check `FORENSICS.md` for the verdict before
you build anything on top of it.**

---

## 2. What the task even is, in one diagram

```
   TEXT-TO-POSE GENERATION

   input                              output
   "a person raising                  22 joints in 3D space
    their right arm"        ==>        (a single static human pose)

                                            o     <- head
                                          / |
                                     ----+  |     <- right arm raised
                                        /|\
                                         |       <- torso
                                        / \      <- legs
                                       /   \

   Three things must be true at once:
     1. it must LOOK LIKE A HUMAN     (anatomically valid: bones the right length,
                                       joints connected, no impossible angles)
     2. it must MATCH THE TEXT        (semantically aligned)
     3. it must not be a MEMORISED    (generalises to descriptions never seen)
        training example
```

The hard part is **not** the generative model. The hard part is that most 66-dimensional
vectors are not human poses, most text does not specify a pose uniquely ("raising an arm" —
which arm? how high? standing or sitting?), and the standard datasets encode poses in a
format that is easy to decode almost-correctly.

---

## 3. Where to go next, by what you want

| I want to... | Go to |
|---|---|
| know what the original project did and what is wrong with it | `../BRIEFING.md` |
| know which of those suspicions actually held up | `../FORENSICS.md` (Stage 1 output) |
| **avoid producing a plausible wrong number** | `LANDMINES.md` ← read before writing code |
| know why this repo is shaped the way it is | `DECISIONS.md` |
| find any document here | `DOCUMENTATION_INDEX.md` |
| look up a term or acronym | `GLOSSARY.md` |
| see what has actually been measured | `EXPERIMENT_LOG.md` |
| know what every file and function is for | `CODE_MAP.md` |
| know what was run, when, and what broke | `../LEDGER.md` |
| run an unattended research session | `../AUTONOMOUS_RUN_PROMPT.md` |

---

## 4. Project stages

This project runs in stages, and **each stage is gated on the previous one's verdict.**

| stage | question it answers | output | status |
|---|---|---|---|
| **1 — Forensics** | Which of the four suspected defects are real? | `FORENSICS.md` | in progress |
| **2 — Landscape + spec** | What does the field do now, and what should we build? | `LANDSCAPE.md`, `REBUILD_SPEC.md` | not started |
| **3 — Harness + baseline** | Can we measure anything honestly at all? | `src/t2p/`, `tests/`, `RESULTS.md` | not started |
| **4 — Experiments** | Which changes actually move a metric? | `EXPERIMENT_LOG.md` E-series | not started |

**Stage 3 is evaluation-first and this is not negotiable.** The metric harness is built and
validated against a *published* baseline number before any new model is trained. A harness
nobody checked is worth nothing, and the original project is the proof.

**But evaluation-first is a method, not this project's identity.** The scope is a genuine redo:
the architecture, the dataset, the problem framing, and the business case are all open to being
replaced rather than repaired (`DECISIONS.md` D-17). Stage 2 is where those get decided, and it
is expected to propose changes the original never considered — not to rebuild it correctly.

---

## 5. What is real vs what is not

### Real
```
../BRIEFING.md              forensic read of the original project by the review pass, 2026-09-05.
                            Findings F1-F5, each with a confidence label and a test recipe.
<ARCHIVE>/...           the original project. READ-ONLY. 2 notebooks, a PDF report,
                            3 READMEs, a Windows/CUDA conda env, EDA artifacts.
Obsidian .../091 AI...      "DL - T2P Deep Dive.md", 86 KB of the author's own notes.
```

### Not real — do not assume it exists
- **No environment has been created here.** The original's env is win-64 + CUDA 11.8 and
  cannot be reproduced on this Mac. An existing conda env is being used for forensics.
- **No dataset has been downloaded here** unless `LEDGER.md` says otherwise.
- **No model, no checkpoint, no metric, no baseline number.** Nothing has been trained in
  this repository.
- **No published benchmark figure has been fetched.** Until one is, quote none.

---

## 6. The epistemic convention

Every factual claim in this repository is labelled:

- **VERIFIED** — fetched, ran, or observed *in this repo*, with the output saved.
- **UNVERIFIED** — plausible, probably right, not checked here. Treat with suspicion.

This is load-bearing, not decoration. The project this one replaces failed precisely by
letting a plausible belief (the 263-dim layout) stand unchecked for three months because it
happened to sum to the right total. **If you are unsure, it is UNVERIFIED.**

---

## 7. The traps that will bite you

Short version; full detail with evidence in `LANDMINES.md`.

1. **HumanML3D's `motion[:66]` is almost certainly not 22 joints of XYZ.** The real layout
   starts with 4 root scalars. Slicing 66 gives you a shifted, spliced mess that still
   reshapes cleanly to `(22, 3)` and still plots as a vaguely humanoid blob.
2. **`4 + 63 + 126 + 66 + 4` and `66 + 66 + 126 + 5` both equal 263.** Arithmetic agreement
   is not verification. This is how the original bug survived.
3. **The "pose" and the "caption" may describe different things.** Frame 0 of *"a person
   does a cartwheel"* is a person standing still.
4. **A falling training loss means nothing.** Especially across phases whose loss functions
   differ. Never compare them.
5. **CLIP cannot reliably tell left from right.** The original measured a cosine distance of
   0.03 between "raising left arm" and "raising right arm."
6. **Anatomical validity should be structural, not a loss term.** Parameterise in rotation
   space with forward kinematics and bone lengths are correct for free.
7. **MPS is non-deterministic.** Same seed, different numbers. Report seed spread or the
   comparison is not a comparison.

---

## 8. If you are an AI agent picking this up

In order:

1. Read this file, then `LANDMINES.md`. Ten minutes, and together they prevent the single
   most likely failure here: producing a plausible wrong number with no error raised.
2. Read `../FORENSICS.md` if it exists. Its verdicts constrain everything downstream. If a
   suspicion in `BRIEFING.md` was REFUTED, stop repeating it.
3. Read `DECISIONS.md` before changing any design. Most obvious improvements were already
   considered and rejected for a stated reason, and each entry says what would reverse it.
4. **Do not re-derive numbers.** Every measurement lives in `../artifacts/*_record.json`.
   Read it. One source of truth.
5. If you run an experiment: write JSON to `../artifacts/`, append to `EXPERIMENT_LOG.md`,
   append to `../LEDGER.md`. All three.
6. Label everything VERIFIED or UNVERIFIED.
7. Do not touch anything under `<ARCHIVE>/`. It is the archive being studied.

---

## 9. The honest state of the science

**Updated 2026-09-06 — this section previously said nothing had been measured. That is no longer
true.** Two findings now exist, both from the corrected pipeline.

**1. Caption truncation costs alignment in proportion to volume, not selection.** The original
project truncated each caption to its first action clause. Measured in a validated retrieval space
with no model and no training: **R-Precision-top3 falls 0.145-0.157 corpus-wide, ~0.27 on captions
the rule actually altered.** Three controls decompose it: **~93% of the cost is how much text is
removed, ~7% is which part** — the position component resolved at 5.9 sigma, and only after
averaging over 8 random placement draws. **The original's "first-action segmentation" was neither
clever nor uniquely harmful. It was one of many ways to discard 35% of the words.**

**2. A healthy loss curve on a bad model.** The first model trained here reached ~60% of the way to
a fully-converged reference loss at **0.63%** of that reference's training budget — while scoring
**13x worse on FID**. Loss measures training, not quality. That is F6's failure mode reproduced
deliberately, and it is the clearest available demonstration of why this project is organised
around measurement.

**What is NOT established.** The harness gate (D-03) is **UNRESOLVED**: the evaluator is validated
against ground truth — reproducing it to within 0.003 on five independent runs — but reproducing a
published *generated-model* number did not succeed at affordable sample sizes. **So every number
here is internally-comparable-only** (D-22). No claim of comparability to published results is
supported. Nothing has been established about generation quality under truncated conditioning; that
is E1B and it has not run.

**And the process is part of the record.** Six of the supervising session's own findings were
withdrawn or corrected by measurement during a single day, including the original specification of
the headline experiment — which was a tautology, caught before it consumed ~10 hours of compute.
The front-loading result was called refuted, then downgraded to unresolvable, then confirmed. Those
reversals are documented in `../reviews/REVIEW_QUEUE.md` rather than tidied away, because the
sequence is what makes the surviving numbers worth believing.
