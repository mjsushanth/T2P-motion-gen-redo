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
