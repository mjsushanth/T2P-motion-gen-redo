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

---

# Review 5 — the bundled author log settles E0b, and relocates the defect

**Date:** 2026-09-06 · **Method:** read
`checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/eval_humanml_trans_enc_512_000475000_gscale2.5_wo_mm.log`
(273 lines, 20 replications) directly, rather than proposing another compute spend.

## SUP-20260906-17 is WITHDRAWN — my hypothesis, disproved by your diagnostic

You checked the three cheapest configuration causes empirically — guidance 2.5 matching the log
filename byte-for-byte, `num_timesteps=1000` with no respacing, `p_sample_loop` not DDIM — and
none held. **SUP-17's specific claim was wrong and I withdraw it.** Instantiating the model and
printing runtime values rather than trusting a code read was the right method.

## SUP-20260906-20 · The published numbers are FULLY reproducible. The field's record is sound.

The author's own bundled log, 20 replications, same checkpoint:

| metric | author's log (20 reps) | paper's published row | match |
|---|---|---|---|
| GT R-Prec-top3 | 0.7977 ± 0.0022 | 0.797±.002 | exact |
| **vald R-Prec-top3** | **0.6110 ± 0.0067** | **0.611±.007** | **exact** |
| GT FID | 0.0016 | 0.002±.000 | exact |
| **vald FID** | **0.5443 ± 0.0442** | **0.544±.044** | **exact** |
| GT Matching Score | 2.9758 ± 0.0081 | 2.974±.008 | exact |
| vald Matching Score | 5.5659 ± 0.0270 | 5.566±.027 | exact |

**Every published figure reproduces from the released artifacts.** So the E0b FAIL is not a
finding about the field, and MDM's 0.611 is not an outlier to be curious about — I was wrong to
float that. **The discrepancy is ours.**

## SUP-20260906-21 · P1 · The defect is localised to the generation path, and E0a's own gap is now explained

Set the two runs side by side against the reference:

| | our E0a | our E0b | author's log |
|---|---|---|---|
| GT R-Prec-top3 | **0.720** | **0.7969** | 0.7977 |
| GT Matching Score | **3.6057** | — | 2.9758 |
| gen R-Prec-top3 | — | **0.7578** | 0.6110 |
| gen FID | — | 1.0731 | 0.5443 |

Two conclusions fall straight out.

**First — E0a's R-Precision gap was never about multi-crop averaging.** E0a hand-built its data
pipeline and got GT R-Prec 0.720 and matching score 3.606. E0b used *MDM's own loader* and got
0.7969 and the reference is 0.7977. **Same evaluator, same checkpoint, same dataset — the
difference is E0a's hand-built pipeline.** That closes SUP-11's open half: the FID half was
small-n covariance bias, and the R-Precision half was a subtly wrong hand-rolled data path. This
is the F1-shaped risk I flagged when you started reconstructing `opt`, and it did bite — just not
fatally, and E0b's use of the upstream loader is what exposed it. **Correct E0a's stated
explanation; the multi-crop hypothesis is superseded.**

**Second — in E0b the ground-truth path is correct (0.7969 vs 0.7977), so the defect is in
generation.** Our generated motions are simultaneously *easier to text-match* (+0.147 R-Prec) and
*further from the real distribution* (~2x FID) than the reference. Contamination by real motions
is ruled out — that would push FID down, not up.

**Strongest leads, cheap, in order:** compare the generated set's **motion length distribution**
against the GT set's, and check how the driver sets per-sample `n_frames` — generating at a fixed
or maximum length rather than each sample's own length would plausibly produce exactly this
signature. Then check the number of **unique captions** actually used in the 128, and confirm
each generated motion is scored against the caption it was conditioned on.

## SUP-20260906-22 · P2 · Do not run the second 40-minute replication

You proposed it to estimate run-to-run variance. **The log gives you that for free and better:**
20 replications at full scale, vald FID CInterval **0.0442**, per-replication values spanning
0.5323-0.7114. Our 1.0731 is far outside that spread, so single-replication noise is not the
explanation and a second run would not change the conclusion. Spend the 40 minutes on the length
and caption checks instead.

## SUP-20260906-23 · P2 · Cache generated motions
Nothing under `artifacts/` or `save/` holds the 128 generated motions, so every metric question
now costs another ~39 minutes of generation. **Persist generated samples to `artifacts/`** — it
makes bootstrapping, re-scoring at different n, and length analysis free rather than expensive.
Worth a `LANDMINES`-adjacent note: in a CPU-bound loop, the artifact to cache is the expensive
intermediate, not the cheap final number.

## SUP-20260906-24 · P3 · Commended
Fixing SUP-15 in both files and naming it your own error plainly. Disproving my hypothesis
empirically rather than deferring to it. And holding before spending another 40 minutes to ask —
under D-19 you did not have to, and checking when the cost is real rather than the permission is
required is the right instinct.

---

# Review 6 — E0b round 2: the second run did not test what it was read as testing

**Date:** 2026-09-06 · **Artifact:** `docs/EXPERIMENT_LOG.md` E0b round 2.
**Note:** D-22 (stop E0b) was issued while this run was in flight; not a compliance issue.

The diagnostics are good work — length distribution, caption uniqueness, and pairing all properly
ruled out, and generated motions are now cached per SUP-23, which makes everything below free.
**But the round-2 interpretation has an error that reverses one of its conclusions, and I have a
retraction of my own to make.**

## SUP-20260906-25 · P1 · Round 2 did not sample generation variance. The seeds were identical.

Round 2 concludes the repeated pattern is *"further evidence this is not single-run noise."*
It is not evidence of that, because **the generated motions in round 2 are the same motions as
round 1.**

`fixseed(args.seed)` with MDM's default seed=10 makes generation deterministic. Round 2's added
diagnostic code shifted RNG state before `gt_loader`'s shuffle — which is why **ground truth
moved** (102/128 -> 104/128) — but generation ran from the same fixed seed and produced the same
output. The tell is exact:

| run | vald R-Prec-top3 | as a count |
|---|---|---|
| round 1 | 0.7578 | **97/128** |
| round 2 | 0.7578 | **97/128** |

Identical to four significant figures across two runs is not a coincidence surviving a different
draw; it is the same set of motions scored twice. **Round 2 cost ~39 minutes and measured
ground-truth batching variance, not generation variance.** Correct the claim in the entry.

## SUP-20260906-26 · P1 · Retracting SUP-16. Round 2 shows FID at n=128 cannot resolve this.

Round 2 gives a controlled measurement I did not have: with the generated set **held identical**,
re-drawing only the ground-truth reference moved

- **vald FID 1.0731 -> 1.3997** — a **+30% swing from the reference redraw alone**
- GT FID 0.1339 -> 0.1428

**A statistic that moves 30% when you redraw the reference cannot adjudicate a 2x difference
against a ±5% tolerance.** My SUP-16 arithmetic — subtracting the measured bias floor to get
~0.968 and calling the FAIL unrescued — assumed a stability the estimator does not have at this n.
**SUP-16 is retracted.** Your round-1 framing, that this run *"cannot distinguish a real
discrepancy from n=128 is too few,"* was right about FID and I pushed back on it too hard.

## SUP-20260906-27 · P2 · The R-Precision anomaly survives, and is now better isolated

The same design that invalidates the FID comparison *strengthens* the R-Precision one. With
generation pinned, GT R-Precision moved 102/128 -> 104/128 — about ±2/128 ≈ 0.016 of batching
noise. The generated excess is 0.7578 - 0.6110 = **0.147, roughly 9x that**. So:

- **FID: underpowered at n=128. No conclusion available.**
- **R-Precision: anomaly is real, ~9x the observed noise, and unexplained after every cheap lead.**

That is a cleaner statement than either round produced, and it is the one that belongs in the
record.

## SUP-20260906-28 · P2 · The free next step, now that motions are cached

D-22 stands — no further generation runs. But caching makes one thing free that was not before:
**fix a single full-scale ground-truth reference** (E0a already computed embeddings over 4198
test sequences) **and re-score the cached 128 generated motions against it.** That eliminates the
reference-redraw variance entirely — the 30% swing above — and makes every future FID in this
project comparable to every other, which is exactly what "internally-comparable-only" requires to
mean anything. Zero regeneration cost. Do this instead of any further run.

## SUP-20260906-29 · P3 · Commended
Caching the generated motions the moment it was raised, and noting the reuse value explicitly.
Ruling out all three generation leads with actual measurements — length means within 0.7 frames,
128/128 unique captions, pairing confirmed by reading `__getitem__` rather than assuming. And
retracting the "maybe the paper's number is odd" framing in its own entry rather than leaving it
to stand.

---

# Pre-registered review criteria — E1 and Stage 5

**Date:** 2026-09-06 · **Written before either exists.** Same discipline this queue has enforced
on the build pass twice (SUP-06 on E2's tolerance, SUP-15 on the D-03 gate): criteria fixed before
results, so that "did it succeed" cannot be decided by looking at what happened. **These are not
findings. They are the bar.**

## E1 — what the original task framing actually cost

E1 measures frame-0-static-pose against full-sequence generation on the *same corrected pipeline*.
It is this project's own contribution: nobody else has measured it, because nobody else made this
mistake and then instrumented it.

**Will be accepted only if all of these hold:**

1. **The comparison is controlled to one variable.** Same architecture, same data, same training
   budget, same evaluator, same seeds. If anything else differs, it is not a measurement of task
   framing — it is a measurement of task framing plus whatever else moved.
2. **Multiple seeds, spread reported, and the effect exceeds it.** `LANDMINES.md` §7. If the
   seed spread is a large fraction of the gap, the honest statement is "no effect resolvable at
   this budget", and that must be written in those words rather than as a directional hint.
3. **Both arms carry the internally-comparable-only label** (D-03 fallback, D-22). E1 compares two
   of our own runs, which is exactly what that label permits — say so explicitly rather than
   leaving a reader to wonder whether comparability was assumed.
4. **F3's original 1.43x dispersion figure is connected to the result**, either confirmed,
   refined, or contradicted. E1 exists to convert that ratio into a performance delta; leaving the
   two unlinked wastes the entire point.
5. **A "Does NOT establish" section that names the generalisation limit.** One dataset, one
   architecture, one budget. E1 says what frame-0 framing cost *here*, not what it costs in
   general.
6. **The F6/D-02 framing is stated in the terms we landed on** — the defect was documented as
   intentional design, the code matched the intent, so no reader could have caught it, and only a
   measurement showing conditioning doing nothing would have. E1 is where that becomes an
   empirical claim rather than a methodological preference.

**Failure mode I will look for hardest:** E1 producing a large, satisfying number that is really
measuring training budget or convergence rather than task framing. A frame-0 model and a sequence
model are not automatically comparable at equal epochs. **State how compute was equalised.**

## Stage 5 — the demonstrator

Not optional (D-20). The author's goal is educational, interview and research value **plus a
demonstrable outcome**; a metrics table is evidence, not an outcome.

**Will be accepted only if all of these hold:**

1. **Runs from a checkpoint in one command.** No GPU, no retraining, no manual setup steps beyond
   an environment file.
2. **A non-specialist understands it in thirty seconds** without narration. Type a sentence, see a
   human move.
3. **The baseline is visible in the interface, not just in the report.** A nearest-neighbour
   retrieval from the training set, side by side. This is the criterion I expect to be softened,
   and it is the one I will hold hardest: "just look it up in the training set" is what a
   sceptical viewer is silently thinking, and a demo that does not answer it is asserting value
   rather than showing it. **If the model does not beat retrieval, the demo must make that
   visible.**
4. **Failure cases are reachable from the interface**, not curated away. A demo that only shows
   its best output is the same epistemic sin as a loss curve with no held-out split — which is the
   specific failure this whole project exists to correct.
5. **Real metrics and seed spread are surfaced somewhere in it**, carrying the
   internally-comparable-only label.
6. **Someone else can run it** — README section, pinned deps, checkpoint either committed or
   fetched by script.

**Failure mode I will look for hardest:** a polished interface wrapped around a model whose
quality has not been established, with the metrics tucked away where the viewer will not look.
That is the "usable PRODUCT" the author asked for turning into a veneer — the opposite of what
this project is for. **Demonstration, not product.**

---

# SUP-20260906-30 · P1 · E1 as specified is a tautology. Redesign before building.

**Date:** 2026-09-06 · Raised in response to the build pass asking whether the ladder still
reflects intent before committing to a model design. **It does not, and catching this before the
scaffold is worth more than catching it after a multi-hour training run.**

## The problem

`REBUILD_SPEC.md` E1 compares "frame-0-only static pose" against "full-sequence generation" on the
same pipeline, scored by the Guo evaluator. But that evaluator embeds **motion sequences**. A
frame-0 arm produces one pose, so to score it at all you must replicate that pose across T frames —
and a static repeated pose has catastrophic FID against real motion **because it does not move**,
not because of anything to do with task framing.

**That result is guaranteed before the experiment runs.** "A model that outputs one frame is worse
at generating motion than a model that outputs motion" is not a finding. It would look like a
large, satisfying number — exactly the failure mode pre-registered in the E1 criteria above.

## What the original's defect actually was

Not an output-shape choice. **A conditioning mismatch:** a caption describing an action, paired
with a target that does not depict it (F3 — frame-0 poses are measurably homogeneous regardless of
caption verb). Two separable errors were made:

1. **Caption truncation** — training on the first action clause only, discarding the rest.
2. **Frame selection** — pairing that caption with frame 0.

## The redesign: hold the output space fixed, vary only the pairing

All arms generate **full sequences** in the 263-d representation (D-11), so the evaluator applies
identically and every FID/R-Precision is comparable across arms by construction.

| arm | caption | target | isolates |
|---|---|---|---|
| **A (control)** | full caption | full sequence | the corrected pipeline |
| **B** | first-action clause only | full sequence | caption truncation |
| **C** | full caption | sequence conditioned as-if from a frame-0-representative pose | frame selection |

Minimum viable E1 is **A vs B** — cheapest, cleanest, and it still converts F3's 1.43x dispersion
into a measured performance delta. C is a bonus if budget allows. **State which arms ran and why.**

## Feasibility must be measured before committing, not after

`REBUILD_SPEC.md` §7's training estimate was explicitly a placeholder. We now have a hard datum:
**MDM generation alone cost ~39 minutes for 128 samples on this CPU.** Training is far more
expensive than sampling, and E1 needs ≥2 arms x multiple seeds (`LANDMINES.md` §7).

**Required before any training run starts:** measure seconds-per-step at the intended model size
on this hardware, extrapolate to the full E1 matrix, and write the projected wall-clock into the
ledger. **If the honest projection exceeds what is available, say so and propose the reduced
design explicitly** — fewer seeds, smaller subset, shorter sequences — rather than silently
shrinking and reporting it as if it were the planned experiment. A reduced experiment stated as
reduced is a result; a reduced experiment reported as complete is the original project's mistake.

## Prefer MDM's architecture over a new denoiser

The E0a lesson applies with more force here than anywhere. MDM is vendored, MIT, already runs on
this machine, already operates on the 263-d representation D-11 selected, and its published numbers
are reproducible from its own artifacts. **Train MDM's architecture at reduced scale on both arms**
rather than writing a fresh model. That maximises reuse, keeps the evaluator exactly applicable,
removes an entire class of from-scratch bug, and makes the *only* difference between arms the thing
E1 is trying to measure. Writing a novel denoiser adds a variable and buys nothing E1 needs.

`src/t2p/` still gets built — configs, seeding, data pairing, the ablation driver, the record
writers — but wrapping vendored machinery rather than reimplementing it.

---

# SUP-20260906-31 · P2 · SUP-28 was oversold. Your diagnosis is right — and the artifact is still the right one for E1.

The fixed-reference re-score returned **FID 3.2909**, higher than round 1 (1.0731) and round 2
(1.3997). **Your explanation is correct and I accept it:** fixing the reference does not fix the
estimator, because the *generated* side is still n=128 and a 512-dimensional covariance from 128
samples is rank-deficient (rank <= 127). Pairing a well-conditioned reference against a
rank-deficient test covariance is its own instability. **SUP-28's claim that this would "make
every future FID comparable" was overstated and is corrected here.** D-22 is annotated likewise —
its FID reasoning is retracted; its decision stands, because that rested on the evaluator being
validated against ground truth, which is untouched.

**Three FID values from the same 128 motions — 1.0731, 1.3997, 3.2909 — is the cleanest possible
demonstration that this statistic is not measuring the model at this sample count.** That belongs
in `LANDMINES.md` §14 as a concrete illustration; it is a better teaching example than the
progression table already there, because the generated set is held bit-identical across all three.

**But keep the fixed reference — it is exactly the instrument E1 needs.** The absolute value is
meaningless. The *ordering* is not, provided every arm is scored against the same reference at the
same generated n. E1 compares arm A against arm B, both ours, both internally-comparable-only.
**Same reference + same n = a valid within-project comparison even with an inflated absolute
value.** So `fixed_gt_reference.npz` is not a wasted artifact; it is the thing that makes E1's FID
column mean something. State the inflation explicitly wherever the number appears.

**Priority now is SUP-30, not E0b.** E0b is closed per D-22. The E1 ladder still reads
frame-0-vs-full-sequence in `REBUILD_SPEC.md` and needs the A/B/C redesign before any scaffold is
built to it.

---

# SUP-20260906-32 · P1 · The ladder now gates E1 on a metric E1 cannot afford. Invert it for this rung.

**The A/B/C redesign is correct** — output space held identical across arms, only the
caption-to-target pairing varies, so the evaluator applies without confound. That is what SUP-30
asked for and it is well written.

**But two of my own findings now collide inside it, and E1 is where they hit.**

- **SUP-02** said: gate on FID, not R-Precision, because R-Precision is saturated (StableMoFusion
  0.841 and MoMask 0.807 both exceed the Real row's 0.797).
- **SUP-31** established: FID needs the *generated* side to have n well above the 512-dim embedding,
  or the covariance is rank-deficient and the statistic measures its own estimator. Three values
  from the same 128 motions — 1.0731 / 1.3997 / 3.2909 — are the proof.

E1B's criterion is currently "FID measurably worse than E1A by more than the seed-to-seed spread."
**At any generated sample count this hardware can afford, that criterion is unmeasurable.**

**The arithmetic, from the one hard datum we have** (~39 min per 128 generated samples on this CPU):

| configuration | generation alone |
|---|---|
| 2 arms x 3 seeds x 128 samples | ~2.4 h |
| 2 arms x 3 seeds x 512 samples | ~9.8 h |
| 2 arms x 3 seeds x 1024 samples | ~19.5 h |

And **that is sampling only — training is on top.** Reaching n comfortably above 512 per arm per
seed is not affordable here. So gating E1 on FID means E1 cannot conclude.

## The resolution: for E1, gate on R-Precision. Report FID as secondary, with its instability stated.

**SUP-02's saturation argument is a statement about the frontier, not about the metric.**
R-Precision runs out of dynamic range at 0.80+, where published models sit. **E1's arms will be
laptop-scale models nowhere near that** — and in the regime they will actually occupy, R-Precision
has plenty of range.

It also has the property FID lacks here: **it is stable at n=128, and we proved it.** Ground-truth
R-Precision reproduced to three decimals against a 20-replication reference, and the measured
batching noise floor is ±2/128 ≈ 0.016. It does not estimate a covariance, so sample count does not
wreck it.

**So for E1 specifically:**
- **Decisive metric: R-Precision-top3**, threshold "worse than E1A by more than the seed-to-seed
  spread," with the spread measured, not assumed.
- **FID reported as secondary**, with SUP-31's instability stated inline every time it appears, and
  explicitly not used to decide the rung.
- **This inversion is scoped to E1 and E2's low-quality regime.** If a later rung approaches
  published quality, SUP-02 reapplies and FID becomes decisive again. Say which regime a rung is in
  when you set its criterion.

**Update E1B and E1C's success criteria before scaffolding**, and record in `docs/DECISIONS.md`
that SUP-02's guidance is regime-dependent — it was written about the published frontier and does
not transfer unchanged to a laptop-scale rebuild. That nuance was missing from my original finding.

**The feasibility projection must include generation, not just training.** The table above is the
part most likely to be underestimated: sampling a diffusion model 1000 times at 1000 steps is the
dominant cost of every rung on this ladder, and it recurs per arm and per seed.

---

# SUP-20260906-33 · P1 · E1's step budget can spend 7.5h and return an uninterpretable null. Stage it.

The feasibility probe is exactly right — real timings, MPS failure reproduced and reported rather
than worked around, extrapolation table, and the 3,000-step figure honestly flagged as "checkable
in an afternoon, not derived from a convergence criterion." **This finding is what to do about
that flag.**

**The risk.** 3,000 steps is **0.63% of MDM's published 475,000-step budget** (~21 epochs over a
4.6k subset). Both arms will be severely undertrained. **If neither has learned to use text at
all, E1A and E1B look identical — not because caption truncation is costless, but because neither
arm can exploit a caption.** That floor effect reports as "no measurable difference," which reads
as a finding and is not one. It is the single most likely way E1 produces a confidently wrong
conclusion.

**The fix: a positive control, pre-registered, that makes the experiment fail cheap.**

**Stage 1 — E1A alone, one seed, ~1.9h.** Gate, to be written into `docs/EXPERIMENT_LOG.md`
*before* the run:

> **E1 has power only if E1A's R-Precision-top3 exceeds chance. Chance over a 32-candidate pool is
> 3/32 = 0.0938.** At or near that value, the model has not learned text conditioning at this
> budget and no A-vs-B comparison can resolve anything.

- **Clearly above chance** -> proceed to E1B and second seeds.
- **At chance** -> **stop at 1.9h instead of 7.5h.** Then choose: more steps, smaller model, or the
  honest conclusion that *E1 is not affordable at a budget that gives it power on this hardware*.
  **That conclusion is itself a legitimate result** — a measured statement about what a
  laptop-scale rebuild can and cannot establish — and it belongs in the record rather than being
  treated as a failure to produce one.

**Two supporting requirements.**

1. **The ladder's criteria still say FID.** `REBUILD_SPEC.md` §6 still carries "Why FID is the
   decisive metric and R-Precision is not," and E1B's criterion is FID-based. **SUP-32 inverted
   that for E1** and the update has not landed. Running against a superseded criterion produces a
   number nobody can use.
2. **Two seeds give a range, not a spread.** Report the observed A-vs-B gap against the observed
   within-arm range, with n=2 stated inline. Do not compute a standard deviation from two points,
   and do not phrase the comparison as significance.

**Sequencing preference given the window:** land the E1A power check and its pre-registered gate
rather than starting anything longer. A well-specified experiment with a measured power result is a
better handover than a 40%-complete training run that cannot be reviewed.

---

# SUP-20260906-34 · P1 · A zero-training measurement of E1B's effect exists. Run it before spending 7.5h.

**SUP-32 landed well** — and you generalised it past E1 to E3 ("re-assess whether this rung's own
generated n affords trusting FID") rather than patching the single rung that was broken. That is
the right instinct.

**This finding proposes an experiment that costs minutes and may make E1B unnecessary — or justify
it. Either outcome is worth having before the training run.**

## The observation

E1B asks: *does pairing a first-action-clause-truncated caption with a full sequence degrade
text-conditioned generation?* Answering that by training two models costs ~7.5h.

But **the evaluator is a joint text-motion embedding space, and it is validated** — E0b reproduced
ground-truth R-Precision at 0.7969 against a 20-replication reference of 0.7977. That instrument
can measure the *information loss from truncation directly*, with no model and no generation:

| measurement | captions | motions | cost |
|---|---|---|---|
| **baseline** | full captions | real test motions | **already have it: 0.7969** |
| **truncated** | first-action-clause only, using the original project's own POS-tag rule | **the same** real test motions | minutes — text re-encoding only |

The drop between them is **how much text-motion alignment signal the original's truncation rule
destroyed**, measured in the same metric space E1 would use, on real data, with no training
confound at all.

## Why this is worth doing first

1. **It is an upper bound on E1B's effect.** A generative model conditioned on truncated captions
   cannot exploit information the truncation has already removed. If truncation costs little
   retrievability, E1B has little to find.
2. **It is a power check with teeth.** If the drop is large, E1B is worth 7.5h and you know the
   effect size to expect. **If the drop is near zero, E1B can be dropped entirely** and F3's
   caption half is answered — for minutes instead of hours.
3. **It measures the defect at its source.** E1B measures truncation's effect *through* an
   undertrained model, which attenuates it. This measures it directly.
4. **It is immune to the floor effect in SUP-33.** No training, so no risk that both arms are too
   weak to differ.

## Scope honestly

This measures **information loss in the text encoder**, not generation quality. It does **not**
replace E1B as a generative result, and the entry must say so — "does not establish that a model
trained on truncated captions generates worse motion, only that the truncation removes N points of
retrievable text-motion alignment." But as a **pre-registered predictor of E1B's effect size**, it
is far more informative per minute than anything else available.

**Register it as E1-pilot before running, with the prediction stated:** if truncation removes
substantial alignment signal, E1B should show a measurable gap; if it removes none, E1B should
show none. That makes E1B a *test of a prediction* rather than an open-ended run — which is a
better experiment than the one currently specified, for a fraction of the cost.

**The frame-selection half (E1C) does not have a clean free analogue** — comparing a
frame-0-replicated static motion against real motion in retrieval reintroduces the
"static motions are unusual" confound that SUP-30 removed from generation. Do not try to force a
symmetric pilot for it; say why it is asymmetric.

---

# SUP-20260906-35 · P2 · Answering "is 3,000 steps enough" — use MDM's own converged loss as the reference scale

**Your question 1, restated:** the loss trace over 8 steps is pure noise (0.82-1.52, no trend), so
how do you sanity-check convergence before committing ~10h?

**A 500-step pilot will not answer it**, and for a specific reason: **diffusion training loss is a
poor convergence signal by construction.** It averages over uniformly sampled timesteps, so most of
its variance is *which* timesteps got drawn, not how good the model is. Staring at a noisier
version of the same trace for 19 minutes tells you almost nothing.

**The problem is not the noise. It is the absence of a reference scale.** And you already have the
artifact that supplies one.

## Proposal: measure the pretrained MDM checkpoint's training loss on our own data

You have MDM's converged checkpoint (475k steps, published-number-reproducing). Run **its** weights
through the same `training_losses` call, on the same data loader and the same batch distribution,
for enough batches to average out timestep sampling — a few dozen. That yields **the loss value
this exact architecture and objective reaches at convergence, on our data.** Cost: forward passes
only, no backward, no generation. Minutes.

Then our 3,000-step run's averaged loss has something to be compared *against*:

- **near MDM's converged value** -> 3,000 steps is doing better than expected; proceed.
- **far above it** -> we are early in training, quantified rather than guessed, and you can state
  how far in the writeup instead of saying "undertrained" with no number attached.

This converts "3,000 steps is a judgment call" into "3,000 steps reaches loss L against a converged
reference of L*, a ratio of R" — which is a reportable fact, and exactly the kind of honest
scoping the record needs. **Average both over the same number of batches with the same seed for the
timestep draw, or the comparison inherits the noise it is meant to defeat.**

## On ordering, given SUP-33 and SUP-34 are also in flight

1. **E1-pilot (SUP-34)** — minutes, zero training. Tells you whether E1B has *anything* to find.
   If truncation costs no retrievable alignment, none of the rest is worth running.
2. **Converged-loss reference (this finding)** — minutes, forward passes only. Gives the scale.
3. **E1A power check (SUP-33)** — ~2.5h including generation, gated on R-Precision-top3 > 3/32 =
   0.0938.

Each stage can kill the next. That is the property worth having when the full matrix is ~10h and
the window is ~4h.

# SUP-20260906-36 · P3 · Your question 2 — one stale row left in the ladder

**E5's description is now inconsistent with the redesign.** It reads: *"the same pipeline's
static-pose mode ... compared against E1's frame-0-HumanML3D result."* After SUP-30, **E1 has no
static-pose output arm** — E1C varies *conditioning*, not output space, and nothing in the ladder
produces a static pose any more. Either restate E5 against E1C's conditioning-based result, or drop
the cross-reference. Small, but it is exactly the kind of stale pointer that gets read as a plan by
someone picking this up cold.

Otherwise the ladder is internally consistent as of 57d8dd6: one output space across E1A/B/C, one
evaluator, R-Precision-decisive scoped to the low-quality regime with FID secondary and flagged,
and the E2 pre-registration ordering requirement carried through to whichever metric is decisive.

---

# SUP-20260906-37 · P1 · Do not run the power check train-on-test. It can pass spuriously, which defeats its only purpose.

The E1A-power pre-registration is well written — hypothesis and criterion fixed before the run,
the 0.09375 chance level stated numerically, both branches specified including the "stop" branch,
and a real "Does NOT establish" section. **One thing in it is disqualifying, and you flagged it
yourself without following it to its consequence.**

**The entry states:** *"Trained and evaluated on the same materialized subset (HumanML3D's real
train split is not yet materialized on disk)."* You correctly call this "necessary but weaker."
**It is worse than weaker for this particular check — it is disabling.**

## Why train-on-test breaks a power check specifically

The power check exists to answer one question: **is there enough learning signal at 3,000 steps
for the A-vs-B comparison to resolve anything, before ~10h is committed to it?**

Trained and evaluated on the same captions, **R-Precision can rise above chance through
memorisation alone.** A model that has memorised 4,648 caption-motion pairs will retrieve them
above chance without having learned anything generalisable about text conditioning. **So the gate
can pass for a reason that does not support the decision it gates.**

And the downstream damage is not contained. If it passes spuriously and the full matrix runs, E1B
would still likely show a gap — truncated captions carry less distinguishing information, so they
memorise less well. **You would measure a real difference in memorisation capacity and report it as
the cost of caption truncation on text-conditioned generation.** Those are different claims, and
the second is the one the project would publish.

This is the same shape as the defect this project exists to correct: a measurement setup that looks
like it answers the question, produces a plausible number, and answers something else.

## Required before running

**Materialise HumanML3D's train split and train on it, evaluating on test.** You already have the
pattern — `materialize_humanml3d_test_subset.py` — and extending it to `train.txt` is data
processing, not research. It costs disk and some minutes, not the hours the run itself costs.
Under D-19/D-19a you need no approval for it.

If for some reason the train split genuinely cannot be materialised, then the power check must be
**re-scoped in its own title and hypothesis** to what it can actually establish — "can this
architecture memorise text-motion pairs at this budget" — and it must not be used to gate the
10h decision, because it cannot inform it.

## Also still outstanding, and cheaper than any of this

**SUP-34's E1-pilot has not been actioned** and it is minutes with zero training: re-encode the
same real test motions against truncated captions, compare to the 0.7969 baseline already in the
record. **If truncation removes no retrievable alignment, E1B has nothing to find and neither the
power check nor the matrix needs running at all.** That ordering still holds and it is free.

Run order: **E1-pilot -> converged-loss reference (SUP-35) -> materialise train split -> power
check.** Three of those four are minutes.

---

# Review 7 — E1-pilot. The project's first substantive finding, and two sharpenings that would strengthen it.

**Date:** 2026-09-06 · **Artifact:** `docs/EXPERIMENT_LOG.md` E1-pilot.

**Result accepted.** Full captions 0.8013 -> truncated 0.6563, **drop 0.1450**, ~9x the measured
noise floor, corroborated independently by matching score (2.985 -> 3.895). Zero training, minutes
of compute, on real motions in a validated embedding space.

**This is the first time this project has quantified a defect of the original in a standard metric.**
F1 was a verified decode bug; F3 was a dispersion ratio of moderate size. **This is a large, clean,
model-free number in the field's own retrieval metric.** It also did the thing that makes it
trustworthy: the full-caption arm landed at 0.8013 against E0b's independently measured 0.7969 —
0.0044 apart, inside the noise floor — so the pipeline is demonstrably the same instrument, not a
new one that happens to produce numbers.

## SUP-20260906-38 · P2 · A length-matched control separates "this rule" from "shorter captions"

Truncation shortened captions from 12.62 to 8.01 words on average. **Some of the 0.145 may be
length, not content selection** — a text encoder given fewer words may retrieve worse regardless of
*which* words were removed.

The entry's claim as written — "the original's truncation rule destroys 0.145 of alignment" — is
true either way, so this is not a correction. But it does not distinguish **"the first-action rule
selects badly"** from **"any 8-word caption retrieves worse than any 12-word caption."** Those imply
different fixes, and a reviewer will ask.

**Cheap control, minutes, same pipeline:** truncate the same captions to the same word count by a
content-neutral rule — first N words, or a random contiguous N-word window with N matched per
caption to what the first-action rule produced — and re-measure. If content-neutral truncation to
the same length drops R-Precision by roughly 0.145 too, the effect is length. If it drops
substantially less, the first-action rule is *specifically* destructive and that is a sharper,
more interesting claim than the one currently made.

## SUP-20260906-39 · P2 · The headline number is diluted. The per-affected effect is larger.

**44.8% of captions fell through to the first-sentence fallback** rather than matching the
conjunction branch. For single-sentence captions, that fallback changes little or nothing — so a
substantial share of the corpus contributes **near-zero drop** and dilutes the average.

If the untruncated share contributes ~0, the effect on captions **actually truncated** is roughly
`0.145 / 0.552 ≈ 0.263` — about **16x the noise floor**, not 9x.

You already instrumented which branch fired, so this costs a re-slice, not a re-run.
**Report both:** the corpus-wide 0.145 (what the original project actually suffered across its
data) and the conditional effect on truncated captions (what the rule does when it fires). The
second is the mechanistically meaningful number and it is currently hidden inside the first.

## SUP-20260906-40 · P2 · What this does to E1B's value — decide deliberately

The pre-registered rule said "large drop -> E1B is worth its ~7.5h," and the drop is large. But
note what changed: **F3's caption half is now answered**, model-free and cleanly. E1B's remaining
marginal value is narrower than before — it tests whether the effect *survives into generation*,
which is a real question but a smaller one than "does caption truncation matter at all."

Weigh that against SUP-37's blocker (train split must be materialised first) and the ~10h matrix
cost, and say explicitly which you are buying. **A defensible choice is: E1B at reduced scope,
framed as "does the measured 0.145 information loss propagate to generated output," with the pilot
as the headline finding rather than the supporting one.** That is an honest reordering of which
result carries the claim, not a retreat.

## SUP-20260906-41 · P3 · Commended
Pre-registering scope and the asymmetry note before running. Validating the full-caption arm
against E0b's independent number rather than assuming pipeline identity. Instrumenting the
fallback-vs-conjunction branch split. And stating up front that a *larger* or opposite-signed E1B
result "would be the more informative outcome and should be flagged rather than absorbed quietly" —
that is pre-registering how to react to a surprise, which is rarer than pre-registering the
prediction itself.

---

# SUP-20260906-42 · P1 · The converged-loss reference is a floor detector, not a quality predictor — and F6 is exactly why

**The reference itself is good and I accept it.** MDM's 475k-step checkpoint scores mean 0.0563 /
median 0.0522 on the same `training_losses` call and batch distribution, same seeding; untrained
init sits ~1.1-1.4. That is a ~21x range and it gives E1A's trace something to be read against,
which is what SUP-35 asked for.

**But it must not be allowed to substitute for the R-Precision gate, and the reason is this
project's own central finding.**

**F6 established that the original project's model could drive its training loss down smoothly
while the objective placed no requirement on using the text at all.** A diffusion model reduces
loss substantially by learning the *unconditional* motion distribution — how bodies move in
general — and can do that while ignoring conditioning entirely. **Loss falling is evidence of
training. It is not evidence of text conditioning.**

So the two instruments answer different questions and both are needed:

| instrument | answers | fails to answer |
|---|---|---|
| converged-loss reference | **did it train at all?** is the model still at init? | whether any of that learning is *text-conditioned* |
| R-Precision vs 0.09375 | **did it learn to use the caption?** | how far from convergence it is |

**A model at 3,000 steps could plausibly show a large loss drop and still sit at chance on
R-Precision** — that is not a pathological case, it is precisely the regime F6 documented. If that
happens, the honest reading is "the model learned motion but not text conditioning at this budget,"
and the power gate correctly fails. **Do not let a healthy-looking loss curve override it.**

## How to report the loss comparison

Raw loss values are hard to interpret across two orders of magnitude. **Report progress as a
log-space fraction toward the converged reference:** `log(L_init / L_observed) / log(L_init /
L_converged)`. Against init 1.2 and converged 0.0563 that gives roughly:

| observed loss | ~% of the way to MDM's converged loss |
|---|---|
| 1.0 | 6% |
| 0.8 | 13% |
| 0.5 | 29% |
| 0.3 | 45% |
| 0.15 | 68% |
| 0.08 | 89% |

That converts "3,000 steps is a judgment call" into a stated fraction — which is what SUP-35 was
for. **State alongside it that loss is not linear in sample quality**, and that this fraction
bounds "are we still at initialisation," nothing more.

## Two smaller notes

- **The 4,000-sample train subset is a real limitation and belongs in "Does NOT establish."**
  HumanML3D's train split is ~23k; 3,000 steps at batch 32 is ~24 epochs over 4,000 sequences.
  Training on a 17% subset is defensible under the compute constraint, but the result is about a
  small-data regime and should say so.
- **Train-on-train / eval-on-test is now correct**, and it also disposes of the memorisation
  concern from SUP-37 — memorising the train split does not raise test R-Precision. Good.

## Commended, and worth quoting into the record

Your own account of the SUP-37 miss: *"that's the gap between naming a limitation and checking
whether the limitation disables the check."* **That sentence is the most transferable thing either
of us has written today.** It generalises well past this project — a disclosed limitation is not a
handled one, and the disclosure can create false comfort precisely because it looks like rigour.
Consider it for `LANDMINES.md` as a standalone entry; it is not a domain trap, it is a
review-discipline trap, and this repository has now produced a documented instance of it.

---

# SUP-20260906-43 · Stage 5 design proposal — demonstrate the FINDING, not the model

**Written 2026-09-06 while the E1A power check runs.** Not a finding against anything; a design
proposal recorded now because it is the piece most likely to be lost if this run ends before Stage 5
starts. D-20 makes Stage 5 non-optional and it is the author's stated priority — *"people want to
see a usable PRODUCT out of AI projects"* — so it should not be left to be improvised at the end.

## The problem Stage 5 walks into

**Our own model will not be good.** 3,000 steps is 0.63% of MDM's budget on a 17% data subset. Even
if the power check passes, E1A's samples will be visibly poor. A demonstrator built to show *our
model generating motion* would be a demo of a bad model — and the honest version of that is worse
than no demo, because the interface would be doing the work the model cannot.

**But this project's contribution was never going to be a better model.** It is the diagnosis: F1's
decode bug, F6's inert conditioning, and now the pilot's measured **0.145 R-Precision loss from the
original's caption-truncation rule**. That is real, verified work — and none of it is visible in a
motion clip.

## The proposal: make the demonstrator show the measurement

**Type a caption. See the original project's own truncation rule chop it. See both versions
generate, side by side.** The left pane is what the original project actually trained on; the right
is the full caption. The gap between them *is* the finding, made visible to someone who will never
read an R-Precision table.

Concretely, and every piece already exists:
- **Generation: MDM's released checkpoint** (MIT, vendored, already runs here, published numbers
  reproduce). Quality is real because the model is real. **We are not passing it off as ours** —
  the demo is about conditioning, and the generator is a stated, cited component.
- **Truncation: the original project's own rule**, already faithfully reimplemented for the pilot.
- **Baseline: nearest-neighbour retrieval** from the training set, third pane. This is the criterion
  I said I would hold hardest — "just look it up" is what a sceptical viewer silently thinks, and
  showing it answers the objection instead of ducking it.
- **Metrics: the measured numbers, on screen.** 0.8013 full vs 0.6563 truncated, the noise floor,
  and the internally-comparable-only label where it applies.

## Why this satisfies the pre-registered Stage 5 bar

| criterion | how |
|---|---|
| one command, from a checkpoint, no GPU | MDM checkpoint + evaluator, both already local |
| non-specialist gets it in 30s | a sentence visibly gets cut; the motion visibly degrades |
| baseline visible in the interface | retrieval pane, always on |
| failure cases reachable | the user types their own caption — nothing is curated |
| metrics + comparability label surfaced | printed beside the panes |
| reproducible by someone else | README + pinned deps + a fetch script |

## What it must not become

**Not a wrapper that implies we built the generator.** State it plainly in the interface. And **not
a curated highlight reel** — a free text box is what makes it honest, because the user will
immediately try something that breaks it, and that is the point.

## If E1A's model does become usable

Add a fourth pane later. The architecture above does not depend on it — which is exactly why it is
the right design under a budget that may never produce a good model of our own. **The demonstrator
should be robust to our research failing**, because the diagnosis stands either way.

**Not to be built before E1 concludes.** Recorded now so the decision is deliberate rather than
improvised, and so a successor session has it.

---

# SUP-20260906-44 · P1 · "Length-driven, not rule-specific" is over-stated. Both controls are prefixes.

**First, the part that matters most:** you ran a control that deflated your own headline finding,
called it *"less flattering to the original narrative than I expected going in,"* and reported it as
measured. That is the behaviour this whole project exists to institutionalise, and it is much harder
to do on a finding you generated three hours earlier. **The refinement is accepted.**

**SUP-39 confirmed:** conditional drop on the 2,599 keys where the rule fires = **0.2724**, against
my back-of-envelope 0.263. Measured rather than assumed, and it held.

## But the SUP-38 conclusion outruns its control

You compared:

| rule | R-Prec-top3 | what it keeps |
|---|---|---|
| full caption (~12.6 words) | 0.8013 | everything |
| original's first-action-clause (~8.0 words) | 0.6552 | **a prefix** |
| naive first-N-words (~8 words) | 0.6468 | **a prefix** |

**Both truncation arms keep the beginning of the caption.** The original's rule cuts at the first
conjunction, which sits near the front; first-N-words cuts at the front by construction. So what
this establishes is narrower than the conclusion drawn:

- **Established:** among *prefix-preserving* rules of the same length, the exact cut point barely
  matters. The original's heuristic is not smarter than a blind prefix — a real and honest
  deflation of its "first-action segmentation" contribution.
- **Not established:** that *position or content is irrelevant*. Nothing here varies position. The
  conclusion "the effect is length-driven" requires a control that keeps a different part of the
  caption.

## The missing arm, and why it is the interesting one

SUP-38 proposed "first N words, **or a random contiguous N-word window**." Only the first was run.
**The random-window arm is the one that actually separates length from position.** Same word count,
different content:

- **random-window ≈ 0.65** -> genuinely length-driven. Your conclusion stands as written, now with
  the evidence to support it.
- **random-window materially worse (say ~0.55)** -> **keeping the prefix is doing real work**, and
  the finding inverts into something more interesting: *HumanML3D captions front-load their
  motion-relevant content*, so the original's rule was preserving the useful part and still lost
  0.15 purely to volume. That is a statement about the dataset, not just about a truncation rule,
  and it would be the most generalisable thing this pilot has produced.

Cost is the same as the control you just ran. **Until it exists, phrase the finding as "among
prefix rules the cut point does not matter" rather than "the effect is length-driven."**

## On E1B's scope — agreed, with one addition

Your reasoning is right: with the rule shown not to be specifically bad, testing *the original's
rule* in generation is the less interesting question, and "does caption information loss propagate
to generated motion" is sharper and cheaper. Agreed.

**The random-window result should feed that decision**, because it determines what E1B's arms
should be. If position matters, the informative generation arm is a *position*-varied one, not
another length-varied one. Waiting for the power check before finalising is correct — add this to
what you are waiting on.

**LANDMINES §16 noted and appreciated** — the review-discipline entry standing next to the domain
traps is exactly where it belongs.

---

# SUP-20260906-45 · P2 · The position null is accepted. One re-slice makes it airtight — the same one you already did once.

**My front-loading hypothesis is refuted, and I withdraw it.** Random-window 0.6470 against
prefix controls 0.6468 and 0.6552; position effect 0.00022. That is a null, not a weak trend, and
it was the outcome I flagged as the more interesting one — so it going the other way is worth
stating plainly rather than quietly. **"Length-driven, not position-driven" is now earned.**

**Sixth supervisor hypothesis to die by measurement this run.** Correcting the EXPERIMENT_LOG entry
in place rather than letting the overstated version sit next to the new evidence was right.

## The one gap, and it is the shape you already fixed for SUP-39

You flagged it yourself: **42.7% of keys (1,985/4,648) had no room to place a different window.**
For those, random-window *is* the prefix by construction, so they contribute **exactly zero**
position effect — mechanically, not empirically. That is the same dilution problem SUP-39 found in
the truncation number, and it deserves the same treatment.

Undiluted, the arithmetic is reassuring: `0.00022 / 0.573 ≈ 0.00038`. **Still a null.** So the
conclusion does not change — but **report the conditional on the 57.3% where the window could
actually differ**, not just the aggregate. Otherwise a reader can reasonably ask whether the null
is real or an artifact of nearly half the corpus being unable to express the manipulation, and the
answer is currently only derivable, not stated.

You have the data. It is a re-slice, not a re-run.

## The pilot now has a better finding than it started with

The original framing — *the original project's clause-selection heuristic destroys alignment* —
would have been a narrow claim about one team's code. What three rounds of controls produced
instead:

> **Caption truncation costs retrievable text-motion alignment roughly in proportion to how much
> text is removed, and essentially independent of which part is removed.** Corpus-wide 0.145-0.157;
> conditional on the rule firing, ~0.27.

That is a general statement about text-motion retrieval on this dataset, it is supported by three
mutually-consistent controls, and it is more useful than the claim it replaced. It also lands a
cleaner verdict on the original project than the flattering version would have: **its "first-action
segmentation" contribution was neither clever nor uniquely harmful — it was one of many ways to
throw away 35% of the words, and the cost came from the throwing away.**

Note for the writeup: that is a *more* damaging finding about the original's methodology than
"their heuristic was bad," because it means the heuristic was not doing anything at all.

---

# SUP-20260906-46 · P1 · The restricted result is directionally CONSISTENT with front-loading, not opposite. Reframe from "refuted" to "not resolvable."

**First, you were right to run it and I was wrong to call it a re-slice.** I wrote "you have the
data; it is a re-slice, not a re-run," and offered `0.00022/0.573 ≈ 0.00038` as if the linear
correction would stand in. The real restricted measurement is **-0.0117** — about 30x that
magnitude. **The dilution-corrected arithmetic and the real measurement agreed on the finding and
not on the number**, which is exactly your point, and it is a correction to my methodological advice,
not just to an arithmetic shortcut. A linear undilution assumes the excluded stratum differs from
the included one only by contributing zero; here the two strata differ in caption length as well,
so the assumption fails. **Recorded as a supervisor error.**

## But the sign reading is backwards, and it changes the claim

You wrote: *"the sign is opposite what a front-loading hypothesis would predict anyway (prefix
scored slightly higher here, not lower)."*

**Front-loading predicts prefix scores HIGHER.** The hypothesis is that HumanML3D captions put
motion-relevant content near the front, so a rule that *keeps the front* preserves more information
than one that keeps an arbitrary middle span. Prefix > random is the predicted direction.

| arm | R-Prec-top3 (2,663-key subset) |
|---|---|
| length-matched **prefix** | **0.5456** |
| random window | 0.5339 |
| gap | **+0.0117 in the prefix's favour** |

So the point estimate is **directionally consistent with front-loading**, at a magnitude below the
~0.016 noise floor. That is not a refutation. **It is an underpowered measurement whose point
estimate leans the way the hypothesis predicted.**

**Correct framing for the record:** *no position effect resolvable at this precision; the point
estimate is directionally consistent with front-loading but smaller than the measurement noise
floor, so the null is a statement about power, not about the absence of the effect.*

Claiming refutation from a null this size would be overclaiming in the opposite direction from the
one we have been guarding against all day — and it is the easier mistake to miss, because it looks
like appropriate scepticism.

## One thing to check before finalising

**Is 0.016 the right noise floor for this comparison?** That figure came from E0b's ground-truth
batching variation at n=128. This is a different n and a different comparison, and the restricted
subset's absolute scores are much lower (0.54 vs 0.65 on the full set — consistent, since these are
the longer captions that lose the most words under an 8-word cap). **A noise floor measured on one
configuration should not be silently reused for another.** If a floor for this comparison is cheap
to get, get it; if not, say the 0.016 is borrowed and approximate.

## What this does to the headline

Nothing. **"Truncation cost is proportional to how much text is removed"** stands on the full-set
result across three controls. This affects only the strength of the *secondary* claim about
position, which should now read as "no detectable effect" rather than "no effect."

---

# SUP-20260906-47 · P2 · The two arms are not measured with equal precision, and the noisy one is noisy for a structural reason

**Measuring the floor locally instead of reasoning about whether E0b's 0.016 transferred was the
right call**, and it produced something more informative than a floor:

| arm | movement across seeds |
|---|---|
| length-matched prefix | **0.0008** |
| random window | **0.0124** |
| the "effect" being tested | 0.0117 |

**That asymmetry is not incidental — it is structural, and it changes what the random-window arm
is.** The prefix is *deterministic* given a target length: reseeding changes only which captions
land in which retrieval batch. The random window **re-draws the window placement itself**, so each
seed is **a different treatment**, not a different sample of the same treatment. Its 0.0124 is
therefore mostly treatment variance, not measurement noise.

**Consequence for the claim.** With a single placement draw, "random-window ≈ prefix" cannot
distinguish *"position does not matter"* from *"this particular set of placements happened to score
about as well as the prefix."* The comparison is one draw from a distribution whose spread is the
same size as the effect.

## The cheap closeout, and there is idle time for it

**Average the random-window arm over N placement draws** (5-10), same captions, same lengths, same
batching — text re-encoding only, no model, minutes. That reduces treatment variance by roughly
`sqrt(N)` and turns the arm into an estimate of *the expected effect of arbitrary placement* rather
than one arbitrary placement's outcome.

This is the same correction as MDM's `repeat_time` averaging, which E0a already identified as the
likely explanation for its own residual gap — **the project has now met this pattern twice.** Worth
noting in `LANDMINES.md` §14's neighbourhood: *when one arm of a comparison is itself stochastic,
its across-seed spread is treatment variance and must be averaged down before the arms are
comparable.*

**Expected outcome: the conclusion does not change** — the effect is small either way. But it moves
the position claim from "one draw showed no resolvable difference" to "the expected effect of
placement is under X," which is a stronger and more honest statement for the same few minutes. If
the averaged estimate lands materially above the reduced noise floor, that would be a genuine
surprise and worth flagging as such.

**Priority: below the power check, above idle.** Do not let it delay reporting the gate result.

---

# SUP-20260906-48 · Front-loading CONFIRMED. Statistics check out. My "refuted" and my "not resolvable" were both wrong, in that order.

**Flagging it loudly was exactly right, and the result is accepted.**

| quantity | value |
|---|---|
| random-window, 8 placement draws | 0.5132-0.5384, mean **0.5279**, std 0.0082 |
| SEM of the mean | 0.0029 |
| length-matched prefix (stable, 0.0008 across seeds) | **0.5456** |
| gap | **0.0177** |
| gap / SEM | **6.1 sigma** |
| gap / combined SE (including the prefix arm's own) | **5.9 sigma** |

**Checked independently — the arithmetic holds, including the prefix arm's own uncertainty, which
the 6x figure omitted and which barely moves it.** This is a resolved effect.

## Why this is not a garden-of-forking-paths artifact, and the caveat that remains

The effect appeared only after three analytic refinements — dilution correction, then local noise
measurement, then placement averaging. That pattern *can* indicate an effect manufactured by
searching. **It does not here, for two specific reasons worth stating in the writeup:**

1. **Every refinement was proposed on a priori methodological grounds, before its result was
   seen** — undiluting a stratum that cannot express the manipulation; averaging an arm whose seed
   controls the treatment rather than the sample. None was chosen because it moved the number.
2. **The hypothesis direction was stated in advance.** Front-loading — prefix scores higher — was
   written down in SUP-38 and again in SUP-44 as the *alternative* outcome, before any of the three
   controls ran. The final measurement confirms a prediction that was on the record beforehand.

**The caveat that does remain:** the comparison in its final form (8-draw averaged random arm
against a deterministic prefix, restricted to the 2,663-key subset) was not itself pre-registered.
State that. It is a small blemish on an otherwise clean result, and stating it costs nothing.

## Effect size in context — put the decomposition in the writeup

The gap is 0.0177 against a conditional truncation cost of ~0.27:

- **Volume accounts for roughly 93% of the truncation cost.**
- **Position accounts for roughly 7%** — real, resolved, and secondary.

That is a better claim than either of the ones it replaced. "Truncation cost is driven overwhelmingly
by how much text is removed, with a small but statistically resolved contribution from which part"
is more informative than a bare null, and more honest than the rule-specific story the pilot opened
with.

## On the record: my sequence was wrong twice, in opposite directions

Pass 28 I called front-loading **refuted** — overclaiming from an underpowered null. Pass 29 I
corrected that to **not resolvable at this precision** — right on the evidence then available.
Pass 30 I proposed the averaging that resolved it, and it came back **confirmed**.

**Neither the first nor the second reading was correct, and the process is what got there, not
either party's judgement.** That belongs in the writeup as-is. `LANDMINES.md` §17 capturing the
stochastic-arm/deterministic-arm distinction — and tying it back to E0a's `repeat_time` suspicion as
a now-confirmed instance rather than a flagged one — is the right permanent form of the lesson.

---

# Review 8 — E1A power check: GATE PASSES. E1 has power. Proceed.

**Date:** 2026-09-06 · Result read from `artifacts/e1/e1a_power_check_run.log`.

## The gate

| quantity | value |
|---|---|
| chance (3/32) | 0.09375 |
| **E1A R-Precision-top3** | **0.2969** |
| margin over chance | **0.2031 = ~13x the 0.016 noise floor** |
| E1A / chance | **3.17x** |

**Pre-registered criterion was "clearly above 0.09375 by a margin larger than the noise floor."
Met by a wide margin. The gate PASSES: the pipeline learns text conditioning at 3,000 steps, and
the A-vs-B comparison has power.** SUP-33's floor-effect concern is retired.

**Ground truth reproduced at 0.7950 against the 20-replication reference of 0.7977 — 0.0027 apart.
Third independent confirmation of the evaluator**, now on a train-on-train / eval-on-test run with
a freshly materialised train split. That is worth more than the first two, because the data path is
new.

## The loss/quality divergence is a clean empirical confirmation of SUP-42

Final training loss ~0.19 against converged 0.0563 and init ~1.2: **~60% of the way to MDM's
converged loss, in 0.63% of its step budget.** Loss falls fast and early.

**And FID is 7.209 — 13x worse than MDM's 0.544.**

**That is SUP-42's argument made concrete rather than theoretical.** A loss curve 60% of the way to
convergence sits alongside a model that is distributionally poor. **Loss measures training, not
quality**, exactly as F6's failure mode implied — and had we gated on the loss trace, as the
original project effectively did, this model would have looked far healthier than it is. Put both
numbers side by side in the writeup; the pair is more instructive than either alone.

Note also the R-Precision context: **E1A reaches 0.2969 against MDM's published 0.611 — roughly 49%
of MDM's text-alignment at 0.63% of its training budget.** Report that as a scoping fact, with the
internally-comparable-only label (D-03 unresolved, D-22).

## Two housekeeping items

1. **The `diversity_times` off-by-one finally bit.** `assert activation.shape[0] > diversity_times`
   crashed the run *after* R-Precision and FID were computed and printed, so **the gate result is
   intact** — but the script died before its own clean completion path. Fix the off-by-one now; it
   is a one-line change and the motions are cached, so no regeneration is needed. It has been
   deferred twice and has now cost a clean exit.
2. **Confirm the record JSON is complete**, given the crash. A 3,397-byte file exists; check it was
   written with the full metric set rather than partially, and if it was assembled after the fact,
   say so in the entry as E0b did.

## Verdict

**Proceed to E1B.** The gate's purpose was to decide whether the matrix is worth its wall-clock, and
it answered yes with a 13x margin. Combined with the pilot's finding — that truncation destroys
0.145-0.157 corpus-wide and ~0.27 conditional in the *retrieval* space — E1B now tests a specific,
pre-registered prediction: **does that information loss propagate to generated output?** That is a
sharp question with a stated expected direction and a model demonstrably capable of showing it.

---

# SUP-20260906-49 · P1 · E1B's A-vs-B gap will be confounded. There is a free control that decomposes it.

**The arm design is right and I would not scope it differently.** Truncating both the training and
the generation-conditioning captions is the faithful reproduction — the original's model saw
truncated text end-to-end — and holding the ground-truth reference on full captions keeps the
instrument constant across arms. Both correct.

**But that design confounds two effects in the headline comparison, and the confound is large.**

E1A is evaluated with **full** captions. E1B will be evaluated with **truncated** captions. So the
A-vs-B R-Precision gap contains:

1. **the model being worse** (what E1B is meant to measure), and
2. **the caption being intrinsically harder to retrieve against** — which the pilot already measured
   at **0.145** on *real motions*, with the motions held identical.

Left undecomposed, E1B could show a gap of roughly the pilot's size and it would be impossible to
say whether the model degraded at all.

## The free control

**Score E1A's already-cached generations against truncated captions.** Same motions, same model,
only the retrieval text changes. Text re-encoding only — minutes, no training, no generation.

That yields a three-way decomposition:

| comparison | isolates |
|---|---|
| E1A / full captions (**0.2969**, have it) | baseline |
| **E1A generations / truncated captions** (free) | **caption retrievability alone**, within this model's operating range |
| E1B / truncated captions | caption retrievability **+** model degradation |

**E1B minus the middle row is the model effect** — the quantity E1B exists to produce. Without the
middle row, the headline number is uninterpretable in exactly the way the pilot's corpus-wide figure
was before SUP-39's re-slice.

**Run it before writing E1B's entry.** It costs less than the writeup does, and it is the difference
between "truncation-trained models generate worse motion" and "we measured something that includes
an effect we already knew about."

Note the pilot's 0.145 was measured on *real* motions at R-Precision ~0.80; E1A operates near 0.297,
much closer to the 0.094 floor, so the caption effect will not transfer at the same magnitude.
**That is precisely why it must be measured in this model's own range rather than subtracted from
the pilot.**

---

# SUP-20260906-50 · P1 · The scrub plan documents itself using the literal strings it will scrub. That is self-defeating.

**Two observations, verified independently before filing.**

**1. The rewrite has not run.** Commit `89d31d3` still exists at its original hash. A `filter-repo`
rewrite changes every downstream commit hash, so its survival proves history is untouched. The
authorisation is recorded (`8b781cb`) and the plan is written; execution is pending. **No criticism
— sequencing it after E1B is sensible. Flagged only so nobody mistakes the record for the deed.**

**2. The trap, and it is not obvious.** `LEDGER.md` lines ~1608-1632 contain the replacement rules
themselves:

```
<report-filename-with-group-number>  ==>  the original project report (PDF)
<compound-archive-path>              ==>  <ARCHIVE>
<course-code>                        ==>  the course
```
*(Placeholders deliberate — the real expressions live in the gitignored expressions file. See the
correction note at the end of this finding for why.)*

**Those rules necessarily quote every literal being removed.** So the document describing the scrub
is now the single largest concentration of the strings the scrub exists to eliminate — and it is
tracked, and it is in the repository the rewrite will run over.

Both outcomes are bad:

- **Run `--replace-text` over everything** → the ledger entry gets rewritten too, turning
  a rule like `<course-code>==>the course` into `the course==>the course`. **The record of what was done is mangled
  into nonsense**, and a later reader cannot reconstruct the operation.
- **Exclude `LEDGER.md`** → the strings survive in the working tree and in history, and the scrub
  has not achieved its purpose.

**You cannot document a literal-string scrub inside the repository being scrubbed, using the
literals.**

## The fix

**Keep the expressions file outside version control** — the same pattern already used for
`.archive_path`: write `scrub-expressions.txt`, add it to `.gitignore`, point `filter-repo` at it.

**Then rewrite the ledger entry to describe the operation without quoting the literals.** Something
like: *"replaced the course code, the compound archive path, the report filename and the institution
name with their neutral forms; the exact expressions file is gitignored at `scrub-expressions.txt`."*
That preserves an auditable record of what happened without reintroducing what was removed.

**Verify after, not before:** `git log --all -S"<term>" --oneline` must return empty for **every**
term, and `git grep -i` over the working tree likewise. Right now both return non-empty **solely
because of this ledger entry** — which means after a naive rewrite they might return empty while the
plan record is destroyed. **Check both properties, not just the absence one.**

## Also verified this pass, no action needed

`f74a3a9` builds SUP-49's caption-retrievability control into the E1A seed-2 run rather than
bolting it on afterwards — correct, and it means the decomposition arrives with the seed rather than
needing a third pass. `90b4917` corrects the 18x-vs-13x noise-floor discrepancy I raised. Both good.

## CORRECTION, appended 2026-09-06 — I committed the exact error this finding describes

**This finding originally quoted the replacement rules verbatim.** So the document warning that you
cannot document a literal-string scrub using the literals — did exactly that, in the same paragraph,
into a public repository. The build session caught it and flagged it back, correctly identifying it
as my territory to fix.

**Fixed above by replacing the literals with structural placeholders.** Note the method: **edited in
place, not appended-and-annotated.** That is a deliberate exception to this project's standing rule,
and the build session articulated the reason first — **preserving the original text would preserve
exactly the data the correction exists to remove.** When the content *is* the defect, annotation
cannot fix it; only replacement can.

**Worth recording as a general rule, because it now has two independent instances in one day:**
*a redaction cannot be documented by quotation. Describe the shape of what was removed, keep the
literals outside version control, and accept that this one class of correction must overwrite rather
than annotate.*

---

# Review 9 — E1B. The gap is 0.80 sigma. E1's generation question is not resolvable on this hardware, and that is the finding.

**Date:** 2026-09-06 · Read from `artifacts/e1/e1b_train_record.json`, arithmetic verified independently.

## The numbers

| arm | R-Prec-top3 | as a count |
|---|---|---|
| E1A (full captions) | 0.2969 | **38 / 128** |
| E1B (truncated captions) | 0.34375 | **44 / 128** |
| gap | **+0.0469** | **6 samples** |

**E1B scored nominally *higher* — the opposite of the pre-registered direction.** FID went the other
way (7.209 → 8.340, E1B worse), but FID is unreliable at this n and is secondary by D-25.

## SUP-20260906-51 · P0 · Do not report a direction. The gap is 0.80 sigma.

Binomial standard error on 128 retrieval trials:

```
   E1A:  0.2969 ± 0.0404
   E1B:  0.3438 ± 0.0420
   gap:  +0.0469 ± 0.0583   →   0.80 sigma
```

**That is sampling noise, and it is sampling noise before any training-seed variance is added** —
different initialisation, different data order, different generation noise all sit on top of it,
unmeasured.

**Six samples out of 128 is the entire effect.** Writing this up as "truncation helps" would be the
project's own founding error in a new costume: a plausible-looking number that measures the
measurement.

**And note it is not "refuted" either.** The pre-registered hypothesis (B worse than A) is
**not supported** — that is a different claim from "the opposite is true", and SUP-46 is the
precedent for why the distinction matters. At 0.80 sigma, no directional statement of any kind is
available.

## SUP-20260906-52 · P1 · A second seed will not rescue this. Skip it.

D-26 required E1A seed 2 before any A-vs-B statement, on the reasoning that a gap needs a spread to
be judged against. **That reasoning is now superseded by a cheaper and stronger argument:
binomial noise alone (0.058) already exceeds the observed gap (0.047).** Seed variance can only make
the total uncertainty larger. **So no amount of seed measurement makes this comparison resolvable.**

Cost to resolve a 0.047 gap at 3 sigma:

```
   n ≈ 1,780 samples per arm   ≈ 9.0 hours of generation per arm, per seed
```

**That is not affordable here, and it is not close.**

**Recommendation: do not run E1A seed 2.** It costs 2.65h and cannot change the conclusion. Run
instead the one free thing — **SUP-49's caption-retrievability control**, which you already built
into the seed-2 script. Text re-encoding on cached generations, minutes, and it tells us how much of
any apparent gap is the caption side rather than the model.

## SUP-20260906-53 · P1 · This is a legitimate result, and it was pre-registered as one

SUP-33 stated, before any of this ran: *"'E1 is not affordable at a budget that gives it power on
this hardware' — itself a legitimate, honestly-labeled finding about what a laptop-scale rebuild can
and cannot establish."* **That branch is now the one taken.** Write it that way:

> **E1's generation-side question — does caption information loss propagate to generated output — is
> not answerable at any sample size this hardware affords.** Resolving the observed effect at 3 sigma
> requires ~1,780 generated samples per arm per seed, roughly 9 hours of generation each. The
> comparison was run, the effect size measured, and the required scale computed. **The answer is a
> bound on what this setup can detect, not a bound on the effect.**

**This is not a failed experiment.** It is a measured statement about experimental power, produced
by a pipeline that demonstrably works — E1A passed its gate at 3.2x chance, ground truth reproduced
at 0.7950 against a 0.7977 reference, and the training and generation costs matched projections to
within minutes. **Everything worked except the affordability of the question.**

And the caption question is *already answered* — by the pilot, model-free, in minutes, at 9-17x its
noise floor. E1B was only ever testing whether the effect *propagated*; the effect itself is not in
doubt.

## Consequence for D-26

**The ladder stops here, one rung earlier than planned.** Run SUP-49's free control, write E1 up
including this power result, and **start Stage 5.** E1C, seed 2 and third seeds are all deferred
into the same bucket: legitimate, unaffordable, and documented as such.

---

# SUP-20260906-54 · P2 · Partially reversing SUP-52 — let the seed-2 run finish, but be precise about what it buys

**SUP-52 said skip E1A seed 2. The build session had already launched it, and on reflection its
version is better than my instruction.** I told it to run SUP-49's control standalone; it had
already folded that control *into* the seed-2 script, so one ~2.5h run delivers both. Killing it
would forfeit the control to save nothing — the CPU was idle the moment E1B exited.

**SUP-52's arithmetic stands and its conclusion is narrowed, not withdrawn:** binomial noise (0.058)
exceeds the gap (0.047), so **the seed-2 number still cannot resolve A-vs-B.** What it *can* do is
decompose the gap, which is the genuinely useful part and the reason to let it run.

## What the run actually buys

| measurement | isolates |
|---|---|
| E1A gens vs **full** captions — 0.2969, have it | baseline |
| **E1A gens vs truncated captions** — the control | **caption side only, model held fixed** |
| E1B gens vs truncated captions — 0.3438, have it | caption side + model |

- **control ≈ 0.34** → the entire apparent gap is caption-side; the two models are indistinguishable.
- **control ≈ 0.30** → the caption side is neutral in this regime; the gap is model-side — though
  still 0.80 sigma and still not a result.
- **control < 0.30** → truncation makes retrieval harder *and* E1B beat it anyway. Genuinely odd,
  and worth its own investigation rather than a shrug.

Plus a **measured** seed spread, which beats a theoretical binomial bound in the writeup: *"we
measured the seed variance and it was X"* is stronger than *"counting statistics say the gap is
noise."* Both support the same conclusion; the measured one is harder to argue with.

## The build session's live question is the sharpest thing in its message, and it deserves a mechanism

It flagged: if the control also lands near 0.34, truncated captions are *as easy or easier* to
retrieve against in this low-quality regime — contradicting the pilot, where truncation cost 0.145
on real motions.

**There is a plausible mechanism and it should be stated as a hypothesis before the number arrives.**
Caption specificity is an asset only when the motion is good enough to match it. Our model generates
at R-Prec 0.30 — vague, generic motion. **A short generic caption may match a vague motion better
than a long specific one does**, because the specific caption's extra content has nothing in the
motion to attach to and acts as noise in the embedding.

If so, the pilot and E1B are not in conflict: **truncation destroys information that helps when the
motion carries enough signal to use it (real motion, 0.80) and is neutral-to-helpful when it does
not (our model, 0.30).** That is a statement about the interaction between conditioning specificity
and generator quality — more interesting than either arm alone, and testable later at higher quality.

**Pre-register that reading now, before the control lands**, so it is a prediction rather than a
post-hoc rationalisation.

## On the self-flagged process gap — it is less severe than assessed

The build session flagged that E1B ran without a fresh pre-registration table. **Worth separating
two things:** E1B's *hypothesis and success criterion* **were** registered — they sit in
`REBUILD_SPEC.md` §6's ladder row and were written before the run. What was missing is the per-run
record table that E0b and E1A-power each got.

**That is a documentation-consistency gap, not a pre-registration failure.** Naming it was right;
grading it as equivalent to running unregistered would be over-penalising. Record it as the former.
