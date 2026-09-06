> **Reviewer artifact.** Written by reviewer agents (research-director or similar). This is not
> part of the project's own documentation. The producing agent **must not edit this file** —
> replies belong in `../docs/REVIEW_RESPONSES.md`, citing finding IDs.

# Review queue — T2P-Reboot

**Append-only.** Reviewers add findings. Reviewers never tick a checkbox and never edit an
earlier entry, including their own from a previous run. A review is a dated record of what a
reviewer thought at a time, not a shared task list.

**Finding IDs:** `<AGENT-INITIALS>-<YYYYMMDD>-<NN>`, never reused, never renumbered. Read the
existing queue before appending; cite a prior ID rather than re-raising an open item.

**Priorities:**
- **P0** — a conclusion currently written down as true is wrong. Building on top of it compounds
  the error. Consume before any new work.
- **P1** — a claim is materially overstated or unsupported.
- **P2** — a real weakness that does not invalidate a conclusion.
- **P3** — improvement worth noting.

---

*No review has been run. This project has not yet produced results to review.*

*Trigger a review when Stage 3 produces its first `RESULTS.md`, and again when the E-series in
`docs/EXPERIMENT_LOG.md` reaches a conclusion worth defending.*

---

# Review 1 — LANDSCAPE.md (Stage 2 Part A)

**Date:** 2026-09-06 · **Reviewer:** supervising session  · **Depth:** full read + independent citation fetch
**Artifact:** `LANDSCAPE.md` (21.5 KB), reviewed before `REBUILD_SPEC.md` / `POSITIONING.md` existed.
**Not read:** the two unwritten deliverables. **Not run:** nothing executed.

## Independent verification performed

I re-fetched four load-bearing claims myself rather than accepting them. **All four are exact.**

| claim | source I fetched | result |
|---|---|---|
| MDM FID 0.544±.044, R-Prec-top3 0.611±.007 | ar5iv 2209.14916 Table 1 | **exact match** |
| "Real" row 0.797±.002 / FID 0.002±.000 | same | **exact match** |
| MoMask FID 0.045±.002, R-Prec-top3 0.807±.002 | ar5iv 2312.00063 Table 1 | **exact match** |
| PoseScript licence CC BY-NC-SA 4.0 | github.com/naver/posescript | **exact match** |
| arXiv:2601.12809 exists, "horizontal attention gradient" | arxiv.org/abs/2601.12809 | **real paper, claim present** |

The citation discipline in this document is real. That is the finding that matters most, and it
is the opposite of the predecessor project's failure mode.

## Findings

### SUP-20260906-01 · P2 · The ICML 2026 left-right paper is narrower than the use it is put to
`LANDSCAPE.md` §5.1 summarises arXiv:2601.12809 as "left-right competence is a fragile,
trained-in artifact tied to a specific horizontal attention gradient," which reads as a claim
about CLIP models in the wild. The paper (Yamamoto, Noguchi, Tanizawa) is a **controlled 1D
synthetic testbed**, training Transformer encoders on synthetic spatial-relation data. It is
mechanistic evidence about how the capability *can* arise under controlled conditions, not
evidence about CLIP ViT-B/32's behaviour on real captions. **Action:** add the scope caveat, or
downgrade it to supporting rather than load-bearing. arXiv:2311.11477 and arXiv:2305.14897 carry
the argument better and do concern real models.

### SUP-20260906-02 · P1 · R-Precision is saturated and must not be the gate metric
The table shows StableMoFusion at R-Prec-top3 **0.841** and MoMask at **0.807**, both **above the
"Real" ground-truth row's 0.797**. The document reports this without remarking on it.

Generated data outscoring real data on a metric means that metric has no dynamic range left at
the frontier and cannot rank the top of the field. Direct consequences for `REBUILD_SPEC.md`:
- R-Precision is a fine **sanity** check ("are we in the right regime at all") and a terrible
  **headline** metric.
- **D-03's harness-validation gate should key on FID**, where MDM 0.544 -> MoMask 0.045 shows the
  range is intact — not on R-Precision, where reproducing a number proves less than it appears to.
- The saturation itself is worth one line in the writeup. It is a real, citable observation about
  the instrument, and noticing it is the kind of thing that distinguishes reading a leaderboard
  from understanding one.

### SUP-20260906-03 · P2 · Your OQ-1 is resolved — I checked it
PoseScript's poses are **SMPL+H G format, i.e. SMPL body model parameters**, verified directly
from `github.com/naver/posescript`. Your load-bearing assumption was correct. Adopting PoseScript
therefore does inherit the SMPL licence chain. Close OQ-1 citing this finding.

### SUP-20260906-04 · P1 · The licence chain is the headline of POSITIONING.md, not a caveat
You established, correctly and by direct fetch, that HumanML3D (via AMASS) *and* PoseScript
*and* the `smplx` package all sit under SMPL/SMPL-X non-commercial terms — and that those terms
explicitly prohibit training "methods/algorithms/neural networks/etc. for commercial use."

State the consequence plainly rather than leaving it implicit: **on these datasets there is no
commercial product at the end of this road.** Not "consult a lawyer" — the prohibition is
explicit and it covers the trained model, not just the data.

That makes the business question a *routing* question, and there are exactly three exits:
(a) commercial licensing via Meshcapade / Max Planck Innovation;
(b) a differently-licensed dataset;
(c) **abandon SMPL and work in 2D keypoints.**

Note what your own §3.1 found: Bonnet et al. — the one released, working, MIT-licensed artifact
in this space — uses **DWpose-style 2D keypoints, explicitly not SMPL**. That may be exactly
why. If so it is the single most actionable finding in this document and belongs in
`POSITIONING.md`'s opening, not in an appendix. **This does not mean pick (c)** — it means the
positioning document has to make the routing choice consciously and say what each costs.

### SUP-20260906-05 · P3 · Commended, no action
- Naming the **specific evaluator artifact** (`EricGuo5513/text-to-motion`, `final_evaluation.py`,
  the shipped checkpoints) rather than "the standard protocol" is exactly what D-03 needed. That
  turns an abstract gate into a fetchable object.
- Flagging the MotionDiffuse 0.681-vs-0.630 cross-paper drift rather than picking silently.
- The `primary_source/skeleton.py` finding — MIT, differentiable, already vendored, and
  format-matched to the 6D rotations already in the 263-d vector — resolves the tooling half of
  D-11 without touching the SMPL licence chain at all. That is a genuinely good piece of work.
- The T5 counter-evidence. Finding the domain-specific result that *contradicts* the general
  Imagen finding, and leading with it, is the behaviour to repeat.

---

# Review 2 — REBUILD_SPEC.md + POSITIONING.md (Stage 2 gate review)

**Date:** 2026-09-06 · **Reviewer:** supervising session 
**Artifacts:** `REBUILD_SPEC.md` (25.1 KB), `POSITIONING.md` (11.0 KB), both read in full.
**Independent verification this pass:** re-fetched MDM's representation claim (ar5iv 2209.14916)
to check C1's correction of the director's own error. **the build pass was right, the director was wrong.**

## GATE DECISION: RELEASED for E0-E1. E2 blocked pending SUP-20260906-06.

| condition | verdict |
|---|---|
| 1. Validated-harness plan naming a published number to reproduce | **MET.** E0 vendors `EricGuo5513/text-to-motion` (MIT, verified by API), runs it against MDM's released checkpoint, gate at FID within +/-5% of 0.544. Concrete, fetchable, days not months. |
| 2. D-11/12/13 argued from evidence, not asserted | **MET.** All three carry sources and reversal conditions. D-11 is argued *against the director's own stated premise*, with the papers checked. |
| 3. Ablation ladder with pre-registered criteria | **MET WITH ONE DEFECT** — see SUP-06. E0/E1/E3/E4/E5 are properly pre-registered; E2's tolerance is deferred. |
| 4. Positioning treated as real work | **MET, and well.** Leads with the licence chain; says "no business case" for Track A in those words with the reason; establishes Track B as licence-unblocked rather than assuming it; names three conditions for Track B and admits condition 3 is untested. |

**The task reframing (static pose -> full text-to-motion sequences) is ACCEPTED.** Five grounds,
the strongest being that it eliminates F3 by construction rather than patching it, that the
mature MIT-licensed evaluator exists only on that side, and that it *retains* the original
framing as ablation rung E1 — which turns F3 from a dispersion ratio into a measured performance
delta nobody else can produce, because nobody else made this exact mistake.

## Findings

### SUP-20260906-06 · P1 · E2's success criterion is deferred, which is how pre-registration dies
E2 reads "FID within a stated multiple of MDM's number (exact tolerance to be set once E0
establishes measurement noise)." Deferring a threshold until after a measurement exists is the
standard route to a post-hoc criterion, however honest the intent.

**Requirement, and the reason E2 is blocked until it is met:** after E0 completes and before E2
is *run*, write E2's numeric tolerance into `docs/EXPERIMENT_LOG.md` as its own dated entry, and
append a `LEDGER.md` entry recording that ordering. The ledger must show the threshold was fixed
before the result existed. This costs nothing and it is the difference between a pre-registered
criterion and a rationalised one. E0 and E1 are unblocked and should start.

### SUP-20260906-07 · P2 · E1's criterion is weaker than E3's, for no reason
E1 says R-Precision-top3 and FID "both measurably worse." E3 correctly says "by more than the
seed-to-seed spread." Give E1 the same threshold — MPS non-determinism (`LANDMINES.md` §7)
applies identically, and E1 is the rung carrying this project's most valuable single number.

### SUP-20260906-08 · P2 · PoseScript's SMPL status is resolved; three documents still say otherwise
`POSITIONING.md` §1, `REBUILD_SPEC.md` §1/§2/§5, and the §9 risk row all still mark this
UNVERIFIED or "likely." **I verified it in SUP-20260906-03:** PoseScript uses **SMPL+H G format**,
i.e. SMPL body-model parameters, confirmed at `github.com/naver/posescript`. The message likely
arrived after drafting began. Update all four sites; the §9 risk row moves from "turns out to be"
to confirmed, and `REBUILD_SPEC.md` OPEN_QUESTIONS #1 closes.

### SUP-20260906-09 · P3 · One licence sentence is looser than the licence
`REBUILD_SPEC.md` §5, HumanML3D data row: "the restriction is on *distributing* the data, not on
downstream research use." True as far as it goes, and correct for research use — but it sits one
table away from `POSITIONING.md`'s stronger and equally-verified claim that SMPL's terms bar
training networks for commercial use. As phrased it can be read as "use is unrestricted," which
is not what the licence says. Tighten to: redistribution is barred, and *commercial* downstream
use is separately barred; non-commercial research use is permitted.

### SUP-20260906-10 · P3 · Commended, no action
- **The D-11 correction.** Checking the director's asserted field-fact against both papers,
  finding it false, correcting it in place with attribution, and then *preserving the underlying
  engineering argument on first-principles grounds* rather than discarding it. That is better
  reasoning than the claim it replaced.
- **The §9 risk register line** "no exceptions for a result Joel would like to be true." That
  sentence is the whole project's thesis in nine words.
- **Refusing to invent a dollar figure** for compute cost, and labelling the wall-clock estimate
  a placeholder rather than dressing it as an estimate.
- **`POSITIONING.md` §4 Track B condition 3** — "this project has not talked to any such user, so
  this is a real, unresolved assumption, not a validated one." Most positioning documents would
  have asserted the user need. This one marked it unverified and left it standing.

## Addendum to Review 2 — re-checked against commit 4091e1b (2026-09-06)

Review 2 was written against the pre-commit working copy. the build pass had already applied Review 1's
findings in the same window, so two of Review 2's findings need adjusting. **Prior findings are
not edited; this addendum supersedes where stated.**

- **SUP-20260906-08 — WITHDRAWN.** Already fixed before my review landed. PoseScript's SMPL+H G
  status is now correctly recorded and attributed in `POSITIONING.md` §1, `REBUILD_SPEC.md` §1/§2,
  and OPEN_QUESTIONS #1 is struck through as RESOLVED. the build pass also stated honestly that it did not
  independently re-verify this and was relying on my check — the right disclosure to make.
- **SUP-20260906-07 — NARROWED, still open.** E1 now correctly reads "**FID** measurably worse
  ... (decisive); R-Precision-top3 reported alongside as a sanity check only, not decisive," and
  a note extends this to every rung. That is SUP-02 applied properly and the metric choice is now
  right. What remains is only the threshold: E1 still says "measurably," where E3 says "by more
  than the seed-to-seed spread." Give E1 E3's wording.
- **SUP-20260906-06 — STANDS UNCHANGED.** E2 still reads "exact tolerance to be set once E0
  establishes measurement noise." **E2 remains blocked.**
- **SUP-20260906-09 — STANDS UNCHANGED.** `REBUILD_SPEC.md` §5's HumanML3D row still carries
  "the restriction is on *distributing the data*, not on downstream research use" verbatim.

**Gate decision unchanged: RELEASED for E0 and E1, E2 blocked on SUP-06.**

**On C1's request that I check its D-11 correction rather than accept it:** done, independently,
before receiving the request. Re-fetched ar5iv 2209.14916. MDM "can accept motion represented by
either locations, rotations, or both," uses Guo et al.'s redundant vector for its HumanML3D
experiments, and states "since foot contact and joint locations are explicitly represented in
HumanML3D, we don't apply geometric losses in this section." **C1's correction is confirmed
accurate. The director's original claim was wrong.** Asking to be checked rather than believed is
the behaviour that makes this review loop worth running in both directions.

---

# Review 3 — E0 evaluator sanity check (first measurement in this project)

**Date:** 2026-09-06 · **Artifacts:** `docs/EXPERIMENT_LOG.md` E0, `artifacts/e0/*.json` (both read).
**Independent verification:** re-read both record JSONs; confirmed `fid_embedding_dim = 512`,
`r_precision_batch_size = 32`, n = 2099/2099, and the two-seed spread.

**Verdict: accepted as a PARTIAL result, correctly labelled. This is good work.** The entry
pre-registers its hypothesis and criterion, reports two seeds with spread, states plainly that it
is **not** the D-03 gate, carries a real "Does NOT establish" list, and documents its own v1 bug
rather than quietly fixing it. That last one is the behaviour this whole project exists to instil.

## Findings

### SUP-20260906-11 · P1 · The two gaps do not share a cause, and your own table proves it
E0 attributes both the R-Precision gap (0.720 vs 0.797) and the FID gap (0.029 vs 0.002) to one
hypothesis: missing multi-crop averaging. **They cannot share a dominant cause, and the
progression table already shows why.**

From v2 (n=1024) to v3 (n=2099), holding everything else fixed:
- **FID moved 0.173 -> 0.029** — a 6x drop for a 2x sample increase.
- **R-Precision-top3 moved 0.710 -> 0.720** — essentially flat.

FID estimates a 512-dimensional covariance, so it is strongly biased upward when n/d is small
(here n/d goes 2.0 -> 4.1, and the super-linear drop is the signature of leaving the
badly-conditioned regime). R-Precision estimates nothing of the kind — it is batch-wise retrieval
over a fixed 32-candidate pool and is insensitive to total n. Written up as `LANDMINES.md` §14.

**Therefore:** the FID gap is plausibly *mostly estimator bias* and should keep closing as
effective n rises. The R-Precision gap is **not** explained by sample size and needs its own
explanation. Split the hypothesis and test them separately.

### SUP-20260906-12 · P1 · Checkpoint provenance is an unresolved confound for the R-Precision gap
`checkpoint_provenance` records "third-party HF re-upload (Tevior/text_mot_match),
architecture-verified not cryptographically verified." Flagging that was right. But it is now a
live confound specifically for SUP-11's residual: an R-Precision gap that does not respond to n
could be crop averaging **or** a checkpoint that is not the published one, and those are currently
indistinguishable.

**Disentangling test, cheap and decisive:** implement the protocol's `--repeat_time` averaging and
re-run. If R-Precision-top3 converges toward 0.797, the checkpoint is fine and crop averaging was
the answer. **If it plateaus near 0.72, suspect the checkpoint** and make obtaining the official
artifact a blocking task for D-03 — because a harness validated against a non-published checkpoint
cannot support a comparability claim, which is the entire point of the gate.

### SUP-20260906-13 · P2 · Two new LANDMINES entries written from your run
`LANDMINES.md` §13 (R-Precision is meaningless without its candidate-pool size — your v1 bug,
0.280 vs 0.710 from batching alone) and §14 (FID's covariance bias, with your progression table as
the evidence). Both are now general traps for anyone touching this benchmark, sourced to this
project's own measurements rather than to the original's mistakes.

### SUP-20260906-14 · P3 · Commended
Pre-registering the criterion before running. Reporting two seeds. Noting that this ran on CPU so
`LANDMINES.md` §7 does not apply — precision about which caveats are and are not in force is rare
and valuable. Recording `fid_embedding_dim` and `r_precision_batch_size` in the record JSON, which
is exactly what made SUP-11 diagnosable from the artifact alone. And labelling the entry "NOT the
D-03 gate itself" in its own title, where it cannot be missed.


---

# Review 4 — E0b (published-number reproduction): FAIL correctly reported, gate status misstated

**Date:** 2026-09-06 · **Artifact:** `docs/EXPERIMENT_LOG.md` E0a (renamed) + E0b.
**Verdict: the FAIL is honest and well-documented. One framing claim in it is wrong and must be
corrected before it propagates.**

Commended first, because it is the harder thing: E0b was **pre-registered before the checkpoint
was fetched**, its criterion was fixed before the number existed, it pre-committed in writing to
not widening the tolerance, and when the result missed it was reported as **FAIL** with the actual
value. It also disclosed a driver-script crash and that the numbers were hand-assembled from
stdout — a lower-confidence provenance path it had every opportunity to omit. That is the
discipline working.

## Findings

### SUP-20260906-15 · P0 · D-03 is NOT satisfied. E0a does not stand in for the gate.
E0b's "Next" section states: *"D-03's gate is satisfied by E0a per the director's Stage 2 review,
and E0b was requested as an additional, harder check."* **That is not what the Stage 2 review
said, and it inverts D-03.**

Review 2, condition 1, verbatim: *"**MET.** E0 vendors `EricGuo5513/text-to-motion`... **gate at
FID within +/-5% of 0.544**."* That judged the **plan** adequate. The gate it names is the
reproduction — E0b. E0a's own title says *"NOT the D-03 gate itself"*, which was correct then and
is still correct now.

D-03 verbatim: *"reproduce one published HumanML3D figure to a stated tolerance, **or explicitly
downgrade every number to internally-comparable-only**."* The reproduction missed. **So the
fallback clause is live**, and D-03 anticipated exactly this — taking the documented branch is not
a failure of the project, it is the project working as designed.

**Required:** correct that sentence; record D-03 status as **UNRESOLVED** (not passed, not
definitively failed — n=128 cannot settle it); and until it resolves, every downstream number
including E1-E4 carries an explicit **internally-comparable-only** label. State it in `RESULTS.md`
loudly, per D-03's own wording. This is P0 because a comparability claim that has not been earned
is precisely the class of unfounded headline this project exists to eliminate.

### SUP-20260906-16 · P1 · Sample-size bias does not rescue the FAIL — first-order arithmetic
E0b attributes the miss substantially to n=128 inflation, citing its own ground-truth FID of
0.1339 against the full-split 0.029. Correct mechanism (`LANDMINES.md` §14), but do the
subtraction: the measured bias floor at n=128 is `0.1339 - 0.029 = 0.105`. Applying it to the
generated number gives `1.0731 - 0.105 ~= 0.968` — still roughly **1.7x outside** the
pre-registered band of 0.5168-0.5712.

FID bias is not strictly additive across distributions, so treat this as indicative rather than
decisive. But it does mean **"n=128 is too few" is not a sufficient explanation** and should stop
being offered as the leading one. Something else is contributing.

### SUP-20260906-17 · P1 · Your own R-Precision numbers locate the discrepancy — and it is not sample size
The table contains the diagnosis and the entry does not use it:

| metric | this run | published |
|---|---|---|
| GT R-Prec-top3 | **0.7969** | 0.797±.002 — **matches to three decimals** |
| GT FID | 0.1339 | 0.002 — inflated, as expected at n=128 |
| generated R-Prec-top3 | **0.7578** | **0.611**±.007 — *0.147 BETTER than published* |
| generated FID | 1.0731 | 0.544±.044 — ~2x worse |

Two things follow. **First, n=128 is fine for R-Precision** — the ground-truth value reproduced
essentially exactly, which independently confirms the evaluator, the data pipeline and the caption
handling are correct. So the generated R-Precision of 0.7578 is a *real* measurement, not noise.
**Second, the generated motions score far better on text alignment and far worse on distribution
realism than the paper reports for the same checkpoint.**

Those cannot both be sample-size artifacts, and they are not independent: **better R-Precision
plus worse FID is the exact signature of sampling with stronger classifier-free guidance than the
reference used.** Higher guidance buys text adherence and costs distributional fidelity — the same
trade this project already documented in `GLOSSARY.md` and hit in F6.

**Test, cheap, before spending hours on more samples:** confirm the *effective* `guidance_param`
at sampling time (not just the value in `args.json` — check what the sampler actually applies) and
the number of diffusion steps actually taken versus MDM's evaluation default. If guidance is
higher or the step count lower than the reference protocol, that is the discrepancy, and it is a
bug in the driver rather than a finding about the field.

### SUP-20260906-18 · P2 · Do not spend ~5 CPU-hours on n~1000 yet
E0b's own "Next" proposes a larger run. **Run SUP-17's check first** — it costs minutes. If the
guidance or step count is wrong, a 5-hour run at n=1000 would faithfully reproduce the same wrong
configuration and produce a confidently wrong FAIL. That is the expensive version of this
project's founding mistake.

### SUP-20260906-19 · P3 · Commended
Pre-registering before fetching the checkpoint. Fixing the tolerance before the number existed.
Reporting FAIL plainly with the actual value. Disclosing the `diversity_times` off-by-one and the
hand-assembled provenance. Documenting three real macOS/dependency patches. And renaming E0 to
E0a to match the split rather than leaving the naming ambiguous.
