# DECISIONS — why this project is shaped this way

> **Purpose.** Every non-obvious technical choice, with the reason, what was rejected, and
> **what evidence would reverse it**.
>
> Format: **D-nn — decision** · *status* · reasoning · rejected · what would reverse it.
>
> **FORCED** = a fact or a constraint removed the alternatives.
> **JUDGEMENT** = a genuine call that could reasonably have gone the other way. These are the
> ones worth revisiting.
>
> Append new decisions; do not renumber existing ones.

---

## Evaluation and evidence

### D-03 — The harness must reproduce a published number before it is trusted · JUDGEMENT
A metric implementation that has never agreed with an external reference is an unvalidated
instrument (`docs/LANDMINES.md` §10). Gate: reproduce one published HumanML3D figure to a stated
tolerance, or explicitly downgrade every number to internally-comparable-only.
**Rejected:** unit tests alone — they prove self-consistency, not correctness of protocol.
**Status:** R-Precision half satisfied at full scale (tight reproduction); FID half:
**non-rejection achieved, tight reproduction not achieved** — kept open, not counted as met.
**Primary evidence, and the harder check:** `docs/EXPERIMENT_DESIGN_E3.md` §9.1/§9.6 — this
project's harness scores the released MDM checkpoint's *generated* motion at 0.6172 (n=4,640)
against the paper's own published 0.611±.007 — +0.87σ, inside ordinary sampling noise, on a
checkpoint this project did not train. This is the decisive version of the gate: it validates
evaluator + checkpoint loading + generation + scoring end to end, not only the evaluator's ability
to encode already-correct motion.
**Supporting, ground-truth-only:** `artifacts/e0/e0b_mdm_reproduction_record.json` (E0b, n=128)
measured ground-truth R-Precision-top3 at 0.7969, against the released checkpoint's own bundled
20-replication log
(`checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/eval_humanml_trans_enc_512_000475000_gscale2.5_wo_mm.log`,
which reports 0.7977, CInterval 0.0022, over 20 replications) — a difference of 0.0008, well
inside that reference's own reported interval. (An earlier "0.06σ" figure for this same comparison
could not be independently re-derived from these two numbers under a standard SE calculation and
is not repeated here as a checked fact — the raw values above are what is verified.)
**A ground-truth measurement that does NOT match, and why it is not a contradiction:** this
project also holds `artifacts/e0/e0_evaluator_sanity_check_record_seed0.json` (E0a, n=2,080
disjoint subset), which measured ground-truth R-Precision-top3 at 0.7202 — several σ from 0.797
under a naive binomial SE. Traced to source: E0a's script (`scripts/e0_evaluator_sanity_check.py`)
hand-builds its own config `Namespace` and data-loading path from scratch; that reconstruction had
a bug not present in E0b's script (`scripts/e0b_mdm_reproduction.py`), which calls MDM's own
upstream `get_dataset_loader`/`evaluation_parser` directly instead of reconstructing them. E0b's
number is why this gate cites E0b, not E0a, as ground-truth support — E0a's miss is an artifact of
that script's own hand-rolled loader, not evidence the ground-truth quantity itself is unstable.
**The stricter remaining piece** — reproducing MDM's own published FID (0.544±.044, 20
replications) on its own checkpoint — remains open. `docs/EXPERIMENT_DESIGN_E3.md` §9.7
bootstrap-resampled E3's own single-replication FID (point estimate 0.4858) and found the
published **0.544 falls inside that interval — a result not distinguishable from the published
figure under an uncertainty this test cannot tighten**: the accepted interval (point estimate ±
2 bootstrap-std) is [0.335, 0.637], a ±15.5% window — against R-Precision's ±1.2%
(0.6172±0.0071, above). **A non-rejection, not a
tight reproduction, and the two halves of this gate are not the same strength of evidence**; lead
with R-Precision wherever both are cited. This bootstrap does not close the gate for a specific,
named reason: it resamples one fixed generation and one fixed reference pool, so it measures
resampling stability, not the paper's own between-replication variance from independent
generations (different sampled noise), and the reference-construction protocol was never verified
to match the paper's sample-for-sample. **What would actually tighten it, priced in measured
terms:** E3's one full-scale generation replication (4,640 samples) measured **4.03h wall-clock on
MPS** (`artifacts/e3/generate.log`). Three total replications (two more independent runs) would
cost **~8.1h of additional MPS compute** and yield a genuine, if small, between-replication
spread — a real tightening, unlike another bootstrap. A full 20-replication protocol matching the
paper's own would need 19 more runs beyond the one that exists — **19 × 4.03h ≈ 76.6h of
additional MPS compute** (≈80.6h total across all 20, if counted from scratch with none already
done) — superseding the earlier pre-MPS CPU estimate, "~5 CPU-hours... roughly 100", this line
previously carried. Both figures are priced here as the next candidate experiment for this gate,
not undertaken — whether either is worth spending against other candidate work is the author's
own call. Numbers from the generation-comparison work onward stay labelled
internally-comparable-only specifically on the FID axis; the R-Precision axis no longer needs
that downgrade for this checkpoint.
**Would close the FID half if:** the ~76.6h-additional-MPS 20-replication protocol (or a smaller number of
additional independent generation runs, enough to estimate genuine between-replication variance
directly rather than via bootstrap) is actually spent and lands within the paper's own reported
spread — this has not been attempted, only priced, and the bootstrap result above is not a
substitute for it.
**Would reverse if:** a published number turns out unreproducible for a reason other than sample
size, in which case the downgrade path is stated even more broadly.

### D-08 — Torch on MPS, never CUDA-assuming code · FORCED
Apple Silicon, no NVIDIA GPU. Consequence: MPS is non-deterministic (`docs/LANDMINES.md` §7), so
every comparison needs multiple seeds.
**Would reverse if:** a rented GPU becomes the primary compute; even then the code must stay
device-agnostic.

### D-24 / D-27 / D-28 — MPS is usable after all, at different speedups for training and generation; the resulting affordable experiment was still not worth running · CORRECTION, twice
An early measurement found training crashed on MPS (`_extract_into_tensor` moves a float64 array
to device before casting; MPS refuses float64) and concluded "training is CPU-only, MPS is
unusable for this codebase." **That inference was wrong, not the reproduction.** The failing line
also names its own fix — cast before the device transfer, not after, which is bit-identical since
indexing is a pure gather (mechanism, and a second identical bug found the same way in the
evaluator wrapper: `docs/LANDMINES.md` §22-23). Once fixed: **training is 9.95x faster on MPS,
generation only 5.47x** (launch-overhead-bound at this batch size) — training and generation
speedups do not transfer to each other and are reported separately from here on.

At the corrected MPS rate, the generation-comparison experiment this project had called
unaffordable (needing ~1,780 generated samples per arm to resolve a 0.0469 gap at 3σ) turned out
to cost about an overnight run, not an impossibility. **It was still not run**, because the
target itself was circular: 0.0469 was the experiment's own noise reading, at 0.80σ, reproducible
from seed alone — not the actual hypothesis-motivated effect size, which an earlier retrieval-only
measurement had already put at 0.145-0.157. Against that real target, the n=128 run already run
is 70-80% powered, not 7%, and already establishes a bounded result: **effects ≥0.175 are excluded
at 3σ, ≥0.117 at 2σ; whether a smaller effect (0.05-0.10) exists is open**, and resolving it would
need roughly 400-1,600 samples per arm.
**Decision: this comparison is closed at n=128.** Affordability was never the real blocker, and a
tighter bound on a bounded null about one dead project's specific bug is not worth further compute
regardless of how cheap MPS made it.
**Would reverse if:** a properly-powered propagation estimate becomes independently worth
running — at which point the MPS rate above already makes it cheap.

### D-25 — FID-decisive gating is regime-dependent; small-sample rungs gate on R-Precision instead · CORRECTION
FID was established as this project's decisive metric because R-Precision is saturated at the
published frontier (StableMoFusion 0.841, MoMask 0.807, both exceeding the paper's own "Real"
ground-truth row of 0.797). That is correct **about the frontier** and was wrongly generalized to
every rung: three FID values computed on one bit-identical set of 128 generated motions (1.0731 /
1.3997 / 3.2909, `docs/LANDMINES.md` §14) show FID's covariance estimate is rank-deficient and
unusable as a decision criterion at generated sample counts this hardware can afford.
**The fix:** for early, small-sample rungs, R-Precision-top3 is decisive (it estimates no
covariance, so small n does not wreck it); FID is reported as secondary, with its instability
stated inline. Any later rung whose affordable sample count approaches the published frontier
should re-apply the original FID-decisive rule.
**Would reverse if:** a later rung's generated sample count can affordably reach the range FID
needs (roughly n in the high hundreds to low thousands per arm/seed).

### D-29 — No training-comparison experiments until the evaluation instrument is trusted · JUDGEMENT
Every training-comparison experiment available right now shares the same shape as the one closed
in D-24/27/28: a comparison at a training budget nobody has shown is adequate, measured with an
instrument this project has already found reasons to distrust in two independent ways — R-Precision
saturates at the published frontier and stops discriminating exactly where a converged model would
need it to (D-25's own regime note), and the same evaluator's text-motion embedding space collapses
to ~1% at full-corpus retrieval, resolving only within its trained 32-candidate protocol
(`docs/LANDMINES.md` §13). Running a bigger version of the same comparison before either question
is settled would spend compute re-deriving the same "was the instrument the problem or the model"
doubt already resolved once. The instrument work needed here requires no training at all: probing
the frozen text encoder already in use, and finding a second, independent evaluator to cross-check
the primary one against.
**Would reverse if:** both instrument questions resolve and a specific, pre-registered training
comparison is designed against a stated, defensible budget.

**A related, standing consequence: new deliverable format, executed Jupyter notebooks.** Every
earlier deliverable in this project was a script + a JSON record + a markdown write-up — right
for a measurement, wrong for something meant to be read and be convincing on its own. A notebook
that shows its own internals (the actual embeddings, the actual similarity matrix, the actual
distribution, not a number quoted from elsewhere) is a stronger form of the same append-only,
re-verify-before-trusting discipline this project follows generally, applied to presentation
rather than only to computation. A notebook that has never been executed, with no committed
outputs, is not a deliverable.

---

## Task and representation

### D-11 — Redundant-vector baseline, rotation+FK as an ablation · JUDGEMENT
Predicting joint positions forces anatomical validity to be a *loss term* that fights the data
term and still admits invalid poses; predicting rotations over a fixed skeleton makes bone lengths
correct by construction (`docs/LANDMINES.md` §8's underlying logic). **Decision, argued from first
principles rather than field precedent** (an earlier version of this argument incorrectly claimed
published models use rotation-only prediction; checked directly, they don't — both predict the
same redundant vector HumanML3D encodes): the redundant vector is the **baseline** (matches the
field, matches the published-number target this project's evaluator is validated against);
rotation+FK is an **ablation**, testing whether making bone-length correctness structural — a
real, F1-grounded hypothesis — measurably beats the baseline.
**Rejected:** position output plus a bone-length loss (the original project's design, dominated
either way). Rotation-only as the baseline (deviates from the reproducible-published-number
strategy).
**Would reverse if:** the rotation+FK ablation beats the baseline decisively.

### D-12 / D-18 — Full-sequence text-to-motion on corrected HumanML3D, not static pose · JUDGEMENT
The original project's task (text -> one static pose, frame 0 of a motion sequence) is not
inherited. **Decision: the task becomes text -> full motion sequence**, on corrected HumanML3D
rather than PoseScript. This eliminates the caption/frame mismatch (F3) by construction — a
sequence's caption describes the whole sequence — gives access to a mature, MIT-licensed evaluator
with a clear published-number ladder to target for D-03, and lets the F1 decode fix transfer
directly (same representation, same decode). Both datasets inherit the identical AMASS/SMPL
non-commercial licence chain, so licence terms do not distinguish between them; the evaluator's
relative maturity does. The original static-pose framing is not discarded — it becomes a
controlled ablation, converting F3's dispersion-ratio finding into an actual measured delta.
**Rejected:** PoseScript as primary (weaker published-number target). Continuing the original's
static-pose framing as the primary task — F1 and F6 together mean there is little reason to
inherit the original task alongside fixing its bugs.
**Would reverse if:** the evaluator turns out unreproducible for a reason unrelated to sample size,
in which case PoseScript's own native metrics become the more pragmatic fallback target.

### D-13 — Text encoder: CLIP token-level baseline, DistilBERT ablation, T5 explicitly rejected · JUDGEMENT
CLIP ViT-B/32's pooled embedding is a known-weak signal for laterality/spatial relations —
evidenced by external CLIP literature (arXiv:2311.11477, arXiv:2305.14897), not by the original
project's own claim, which is doubly unverifiable given its training objective never required the
model to use text at all (F6). **Decision:** CLIP token-level (per-token) conditioning as
baseline — the cheapest fix targeting the specific documented failure (the pooled bottleneck);
DistilBERT as an ablation, following HumanML3D-adjacent precedent; **T5 explicitly rejected** — a
HumanML3D-scale model that tried it saw no gain, domain-specific counter-evidence to general
T5-over-CLIP findings elsewhere. A laterality-specific metric is mandatory regardless of encoder
choice.
**Rejected:** unchanged pooled CLIP. Full T5 swap. Sentence-transformer embeddings (no source
found evaluating spatial/lateral behaviour).
**Would reverse if:** the laterality metric shows token-level CLIP still fails badly — DistilBERT
promotes to primary, or explicit mirror augmentation is added regardless of encoder choice.

### D-14 — Reproduce the original as ONE reference run, not a phase-by-phase ablation ladder · JUDGEMENT
The original project's three-phase narrative does not describe three independently interpretable
configurations — it describes three bugs compounding (CFG folded into the training loss with a
zero-loss degenerate solution, F6; one timestep stepping an entire batch and corrupting the
anatomy-loss input for ~95/96 samples, F7). Faithfully reproducing all three phases as ablation
rungs would measure the cost of that compounding, not a clean, interpretable delta.
**Decision:** keep exactly one original-configuration run as a documented historical reference
point, and organize the actual ablation ladder around the corrected pipeline instead, isolating
the comparison that is actually useful (task framing, correctly instrumented, free of the CFG/
timestep/normalisation bugs entirely).
**Rejected:** faithfully reproducing all three original phases as ablation rungs. Deleting the
original configuration as dead code (kept, as one reference run).
**Would reverse if:** a way is found to cleanly separate the three phases' bugs from each other
well enough that a faithful three-phase reproduction would isolate one variable at a time.

### D-23 — The A-vs-B conditioning-mismatch design, replacing a tautological E1 · CORRECTION
An earlier design specified this project's first real comparison as: the original's frame-0-only
static-pose output against full-sequence generation, on the same corrected pipeline, deciding on
FID. This was a tautology, not a measurement — scoring a single static pose through a
sequence-only evaluator guarantees the static-pose arm scores badly for evaluator-shape reasons
alone, independent of whether the actual conditioning defect (caption truncation, frame selection)
is present.

**What the underlying finding actually claims:** the original project's defect was a *conditioning
mismatch* — the caption paired with a single frame does not describe that frame alone, because
captions were truncated to a first-action clause and the frame selected was not representative of
the described action. Both are properties of the *input pairing*, not the *output space*.
Measuring them means holding the output space fixed (full sequence throughout) and varying only
the caption/target pairing:

| arm | caption | target | isolates |
|---|---|---|---|
| **A (control)** | full caption | full sequence | the corrected pipeline, no defect present |
| **B** | first-action-clause only (the original's actual truncation rule) | full sequence | caption truncation |
| **C (stretch)** | full caption | frame-0-representative conditioning | frame selection |

All three arms share one output space and one evaluator, so the comparison is apples-to-apples —
the only variable is the conditioning/pairing under test. A-vs-B is the minimum viable version; C
runs only if a feasibility check shows budget for it.
**Would reverse if:** a reason emerges that output-space mismatch, not conditioning/pairing, is
actually what the underlying finding claims — re-reading it does not support this; no such reason
has surfaced.

---

## Deliverable shape

### D-15 — Notebooks are narrative, `src/t2p/` is machinery · FORCED by policy
The predecessor put its entire system in four notebook cells of 44k, 46k, 52k and 101k characters,
with three near-duplicate re-implementations of the same classes coexisting in one file. That is
the anti-pattern this rule exists to prevent: if logic would be pasted twice, it moves to a module
and gets a test.

### D-20 — Representation choice is not licence-constrained · FORCED
SMPL/AMASS terms bar training networks for *commercial* use, which had been treated as a
constraint to route around when choosing a representation (favoring, e.g., a 2D-keypoint
alternative). This project has no commercial intent, and SMPL's licence explicitly permits
non-commercial research, education, and personal projects — HumanML3D, PoseScript, `smplx`, and
the SMPL body models are all fully available for the actual purpose here. D-11's redundant-vector
decision therefore stands on its own technical merits; the licence wall does not favor one
representation over another.
