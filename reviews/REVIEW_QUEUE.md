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

**Date:** 2026-09-06 · **Reviewer:** supervising session (Opus 5) · **Depth:** full read + independent citation fetch
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

**Date:** 2026-09-06 · **Reviewer:** supervising session (Opus 5)
**Artifacts:** `REBUILD_SPEC.md` (25.1 KB), `POSITIONING.md` (11.0 KB), both read in full.
**Independent verification this pass:** re-fetched MDM's representation claim (ar5iv 2209.14916)
to check C1's correction of the director's own error. **C1 was right, the director was wrong.**

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
