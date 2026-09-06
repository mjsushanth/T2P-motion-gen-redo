# Supervisor log

> Reviewer territory. Written by the supervising session. Append-only.
> `FOR JOEL` items collect at the bottom.

## [2026-09-06T03:20Z] Supervisor pass 1 — diffusion-core audit

**C1 state:** Stage 2 Part A in flight (3 research agents). Licence item closed correctly
(HumanML3D = MIT, verified from LICENSE file, vendored verbatim + PATCHES.md). No deliverable
to review; worked standing queue item 1 instead.

**Reviewed:** nothing from C1 (nothing new).

**Independent work:** audited `DL_T2P_IMPL.ipynb` cell 47 (2,558 lines, extracted via json).
Three findings, all VERIFIED by direct code reading:
- **F6** CFG applied to the training objective. `_compute_loss` MSEs `u + w(c-u)` against the
  true noise with `w` always > 1 in Phase 3. Zero-loss solution exists at `c = u = eps`, which
  requires no text dependence. No conditioning dropout anywhere. ROOT-CAUSE GRADE.
- **F7** `scheduler.step(timestep=t[0].item())` on a per-sample `t` vector — anatomy loss input
  is mis-stepped for all but sample 0.
- **F8** `normalize_batch` rescales `x_0` per batch before noising.

**Files written (my territory):** `docs/LANDMINES.md` §11, §12 (new); `BRIEFING.md` new section
"Findings added after Stage 1". No C1 file touched.

**Sent to C1:** one message — the three findings, plus four concrete consequences for
`REBUILD_SPEC.md`. Flagged that **D-14 was my error**: I told C1 to preserve the three phases as
an ablation ladder so the F1 fix would be a measured delta; F6 undermines that premise, since
the phases differ by three interacting bugs rather than by design choices. Proposed keeping one
reference run and dropping the ladder, invited disagreement.

**Also flagged:** "CLIP is weak at left/right" must be downgraded to UNVERIFIED. F6 means the
model was never required to use the text, so CLIP was never actually tested.

**Waiting on:** `LANDSCAPE.md`, `REBUILD_SPEC.md`, `POSITIONING.md` from C1.
**Next:** on delivery, review per SUPERVISOR_LOOP_PROMPT §4 — spot-check 3-4 load-bearing
citations by fetching them myself. Until then, standing queue item 3 (mine the Obsidian deep
dive for further docs-vs-code contradictions; five found so far).

---

## FOR JOEL

1. **Public-repo disclosure, your call.** `BRIEFING.md` and `docs/00_START_HERE.md` name the
   course code, institution and group number of a jointly-authored submission whose failures the
   repo documents. Co-author is not named. Scrubbing the three identifiers or making the repo
   private are both one-minute fixes. C1 told not to act.
2. **Your interview answer contains a claim that is now doubly wrong.** The deep dive's scripted
   60-second answer (line 3139) says the key innovation "reduced loss by 99.995% from baseline."
   That number is not in the paper, compares three different loss functions, and — with F6 — the
   phase it credits was trained under an objective that did not require the model to read the
   text. Worth rewriting before it gets said out loud.
3. **Docs-vs-code contradictions found so far: five.** 8-vs-10 clusters; 49.6% vs measured
   38.77%; KMeans on "reduced embeddings" vs raw 66-d; 99.995% absent from the PDF; "ensure 10%
   null conditioning" described in the notes, absent from the code.

## [2026-09-06T04:05Z] Supervisor pass 2 — LANDSCAPE.md reviewed

**C1 state:** `LANDSCAPE.md` written (21.5 KB). `REBUILD_SPEC.md` and `POSITIONING.md` not yet
present — C1 mid-synthesis, has not reported. Reviewed anyway (reading does not disturb it) so
the gate review turns around fast when the other two land.

**Independent verification — 5 load-bearing claims re-fetched by me, all exact:**
MDM FID 0.544±.044 / R-Prec-top3 0.611±.007 and the Real row 0.797±.002 / 0.002±.000 (ar5iv
2209.14916 Table 1); MoMask FID 0.045±.002 / R-Prec 0.807±.002 (ar5iv 2312.00063 Table 1, and
confirmed its table has **no** Real row, closing one of C1's own unconfirmed items); PoseScript
CC BY-NC-SA 4.0 (github.com/naver/posescript); arXiv:2601.12809 is a real paper with the
"horizontal attention gradient" claim present. **C1's citation discipline holds under audit.**

**Findings written to `reviews/REVIEW_QUEUE.md`:** SUP-20260906-01..05.
- **02 (P1)** R-Precision is saturated — StableMoFusion 0.841 and MoMask 0.807 both exceed the
  Real row's 0.797. D-03's gate must key on FID, not R-Precision. C1 reported the numbers without
  noticing the implication.
- **04 (P1)** The licence chain is POSITIONING.md's headline, not a caveat: SMPL/SMPL-X terms
  explicitly prohibit training networks for commercial use, and HumanML3D + PoseScript + smplx
  all inherit them. Three exits only (Meshcapade licence / different dataset / 2D keypoints), and
  C1's own §3.1 already found that Bonnet et al. — the one MIT-licensed working artifact — uses
  DWpose 2D keypoints, not SMPL. Possibly not a coincidence.
- **03 (P2)** Resolved C1's OQ-1 myself: PoseScript poses are SMPL+H G format = SMPL parameters.
- **01 (P2)** arXiv:2601.12809 is a controlled 1D synthetic testbed, narrower than the claim C1
  hangs on it. Two other cited papers carry that argument better.
- **05 (P3)** Commended: naming the concrete evaluator artifact; flagging the MotionDiffuse
  0.681-vs-0.630 drift rather than picking silently; leading with the T5 counter-evidence; and
  the `primary_source/skeleton.py` finding, which resolves D-11's tooling half MIT-licensed and
  format-matched without touching SMPL at all.

**Sent to C1:** one message, framed "do not stop, fold these in" — 02 and 04 change what the two
in-flight deliverables should say, so interrupting was cheaper than a rewrite.

**Monitor armed** (task b33lmn8g8, persistent): fires when `REBUILD_SPEC.md` and `POSITIONING.md`
both settle at a stable non-trivial size.

**Next:** on that event, run the Stage 3 gate review per SUPERVISOR_LOOP_PROMPT §4 — the four
conditions are a named published number to reproduce, D-11/12/13 argued from evidence, an
ablation ladder with pre-registered criteria, and a positioning document that took the question
seriously. **Gate still held.**

---

## FOR JOEL (running)

4. **A commercial product on this data is prohibited, not merely awkward.** SMPL/SMPL-X licences
   explicitly bar training networks for commercial use, and HumanML3D (via AMASS), PoseScript and
   the `smplx` package all inherit that. If the business angle matters to you, the decision is a
   routing one and it is best made now rather than after a model exists: licence commercially via
   Meshcapade, find differently-licensed data, or drop SMPL and work in 2D keypoints. Worth
   noting that the one released, working, MIT-licensed system in this space (Bonnet et al., your
   own paper's reference [1]) took the third route.

## [2026-09-06T04:40Z] Supervisor pass 3 — REBUILD_SPEC.md, and an error of mine

**Event:** monitor b33lmn8g8 fired — `REBUILD_SPEC.md` settled at 25,093 bytes.
`POSITIONING.md` still absent; monitor remains armed for it. **Gate still held** (condition 4
cannot be assessed without it).

**C1 corrected a false claim I wrote, and it was right.** `docs/LANDMINES.md` §8 asserted that
rotation-space + differentiable FK "is what MDM and MotionDiffuse do." C1 checked both papers and
found it is not. I re-verified independently (ar5iv 2209.14916): MDM states it "can accept motion
represented by either locations, rotations, or both," uses Guo et al.'s **redundant vector** for
its HumanML3D experiments, and explicitly notes "Since foot contact and joint locations are
explicitly represented in HumanML3D, we don't apply geometric losses in this section."
**CONFIRMED: my claim was wrong.** I asserted a field fact from memory — precisely what this
project's epistemics section forbids, in a document whose purpose is to prevent that.

C1's fix is better than what I would have written: it keeps the rotation-space argument as a
**first-principles** engineering case (bone lengths are a verified dataset constant, §1's
corollary) while removing the false appeal to precedent. That distinction is the correct one.

**A rule of mine was also wrong.** `SUPERVISOR_LOOP_PROMPT.md` §2 claimed `docs/LANDMINES.md` and
`docs/DECISIONS.md` for the supervisor. `CLAUDE.md` assigns all of `docs/` to the producer. C1
was inside its territory; my later document contradicted my earlier one. Amended: the operative
rule is now **"append or annotate, never silently delete or rewrite another agent's entry."**
Both sessions may write in `docs/`; `reviews/` and `guidance/` stay hard-walled, because that is
where the record that a defect existed lives.

**Reciprocal obligation recorded:** `BRIEFING.md` F6-F8 and `LANDMINES.md` §11-12 are supervisor
output that nobody has audited. C1 to be invited to audit them.

**Assessment of REBUILD_SPEC.md so far (conditions 1-3 of 4):**
- **Condition 1 (named published number to reproduce): MET, and well.** `EricGuo5513/text-to-motion`,
  MIT-verified by API, run against MDM's or T2M-GPT's released checkpoint. Concrete and fetchable.
- **Condition 2 (D-11/12/13 argued): MET.** All three argued from fetched evidence, each with a
  stated reversal condition, and D-11 argued *against* the director's own premise with sources.
- **Condition 3 (ablation ladder with pre-registered criteria): APPEARS MET**, E3/E4 rungs each
  carry a hypothesis and a deciding metric with a seed-spread threshold. Full read pending.
- **Condition 4 (positioning): CANNOT ASSESS — file absent.**
- **Major proposal: switch the primary task from static pose to full text-to-motion sequences.**
  Argued on five grounds, strongest being that it eliminates F3 by construction rather than
  patching it, and that the mature MIT evaluator only exists on that side. Retains the F3
  frame-0 case as a controlled ablation rung — which would produce a number nobody else has,
  because nobody else made this exact mistake. **Director's view: persuasive, likely accept.**
  Formal decision deferred to the gate review with POSITIONING.md in hand.

## [2026-09-06T05:05Z] Supervisor pass 4 — STAGE 2 GATE REVIEW. Gate released.

**Event:** monitor b33lmn8g8 fired on `POSITIONING.md` (10,964 bytes) and exited cleanly.
Both Stage 2 deliverables present and read in full.

**GATE DECISION: RELEASED for E0 and E1. E2 blocked on SUP-20260906-06.**

| condition | verdict |
|---|---|
| 1. harness plan naming a published number | MET — E0 vendors the MIT Guo evaluator, runs MDM's released checkpoint, gate at FID +/-5% of 0.544 |
| 2. D-11/12/13 argued from evidence | MET — all three sourced, all with reversal conditions |
| 3. ablation ladder pre-registered | MET except E2's deferred tolerance (SUP-06) |
| 4. positioning as real work | MET, and the strongest part of the deliverable |

**Task reframing accepted:** primary task moves from static pose to full text-to-motion
sequences. Decisive: eliminates F3 by construction rather than patching it; the mature
MIT-licensed evaluator exists only on that side; and the original framing is *retained* as
ablation rung E1, converting F3 from a 1.4x dispersion ratio into a measured performance delta
nobody else can produce. C1 asked to record this as D-18 with a reversal condition.

**Findings filed:** SUP-20260906-06 (P1, E2 blocked — pre-registration ordering must be visible
in the ledger), 07 (P2, E1 needs E3's seed-spread threshold), 08 (P2, PoseScript SMPL status
resolved, four sites stale), 09 (P3, one licence sentence looser than the licence), 10 (P3,
commendations).

**My error, confirmed and owned.** C1's D-11 correction was right; I re-verified against ar5iv
2209.14916 myself. My `LANDMINES.md` §8 claim that rotation+FK "is what MDM and MotionDiffuse do"
was asserted from memory in a document that exists to prevent that. C1's replacement is better
reasoning than the claim it replaced. Also amended my own contradictory territory rule and
invited C1 to audit `BRIEFING.md` F6-F8 and `LANDMINES.md` §11-12, which are supervisor output
that nobody has checked.

**Next:** Stage 3 (E0 harness reproduction) is C1's work; nothing to review until `RESULTS.md`
or an E0 outcome exists. Supervisor turns to standing queue item 3 — mining the Obsidian deep
dive for further docs-vs-code contradictions (five found so far).

---

## FOR JOEL (running)

5. **The primary task has changed, and you should know before you see the code.** C1 proposed,
   and I accepted, moving from *text -> single static pose* to *text -> motion sequence*. Your
   original project's framing survives as ablation rung E1 rather than being discarded, and the
   2D-pose demo track in `POSITIONING.md` is still pose-shaped, so "pose" has not left the
   project. But this is a real scope change and it is your project. Reversal is cheap right now
   and gets expensive once E2 trains — say so early if you dislike it. The argument is
   `REBUILD_SPEC.md` §0; my acceptance reasoning is Review 2 in `reviews/REVIEW_QUEUE.md`.
6. **The business answer is in, and it is a split verdict.** 3D pose/motion generation: **no
   commercial case**, because HumanML3D and PoseScript both inherit SMPL/AMASS terms that
   explicitly bar training networks for commercial use. Not a lawyer question. The one
   licence-clear route is text -> **2D** keypoints -> ControlNet image, which never touches SMPL
   — which is, notably, exactly what the only released MIT-licensed system in this space did.
   C1 rates that track "plausible, not established" and is honest that no potential user has
   been spoken to. `POSITIONING.md` §4.

## [2026-09-06T05:25Z] Supervisor pass 5 — crossed messages reconciled, gate confirmed

C1 reported Stage 2 complete (commit 4091e1b) while my gate review was in flight, and had applied
Review 1's five findings in the same window. Re-checked all of Review 2 against the committed
version and filed an addendum (append-only, prior findings not edited):

- **SUP-08 WITHDRAWN** — already fixed before my review landed; PoseScript SMPL+H G status now
  correct and attributed in all four sites. C1 disclosed that it had not independently
  re-verified and was relying on my check — correct disclosure, not a silent adoption.
- **SUP-07 NARROWED** — metric choice now right (E1 gates on FID, R-Precision demoted to sanity
  check, note extends this to every rung). Only the threshold wording remains.
- **SUP-06 STANDS** — E2's tolerance still deferred. **E2 remains blocked.**
- **SUP-09 STANDS** — the loose licence sentence is unchanged.

**Gate decision unchanged: RELEASED for E0/E1, E2 blocked.**

**Note on the review loop working in both directions.** C1 asked me to check its D-11 correction
rather than accept it; I had already done so independently. That is the second time in two stages
it has caught something I asserted from memory (the first being the D-14 phase-ladder premise,
which F6 undermined). The value of this arrangement is not that the supervisor is right — it is
that two sessions with different context are auditing each other against fetched sources.

**Monitor re-armed** (task b7nnio6j5, persistent): fires when `docs/EXPERIMENT_LOG.md` gains an
entry or `RESULTS.md` appears — i.e. when Stage 3 produces its first measurable outcome.

**Next:** standing queue item 3 — mine the Obsidian deep dive for further docs-vs-code
contradictions (five found so far). Nothing of C1's to review until E0 reports.

**Outstanding with C1:** flip D-11/12/13 out of PENDING and add D-18 (task reframing); close
SUP-06, 07, 09; audit the supervisor's own F6-F8 and LANDMINES §11-12.
