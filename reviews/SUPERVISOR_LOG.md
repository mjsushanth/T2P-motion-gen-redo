# Supervisor log

> Reviewer territory. Written by the supervising session. Append-only.
> `FOR JOEL` items collect at the bottom.

## [2026-09-06T03:20Z] Supervisor pass 1 — diffusion-core audit

**the build pass state:** Stage 2 Part A in flight (3 research agents). Licence item closed correctly
(HumanML3D = MIT, verified from LICENSE file, vendored verbatim + PATCHES.md). No deliverable
to review; worked standing queue item 1 instead.

**Reviewed:** nothing from the build pass (nothing new).

**Independent work:** audited `DL_T2P_IMPL.ipynb` cell 47 (2,558 lines, extracted via json).
Three findings, all VERIFIED by direct code reading:
- **F6** CFG applied to the training objective. `_compute_loss` MSEs `u + w(c-u)` against the
  true noise with `w` always > 1 in Phase 3. Zero-loss solution exists at `c = u = eps`, which
  requires no text dependence. No conditioning dropout anywhere. ROOT-CAUSE GRADE.
- **F7** `scheduler.step(timestep=t[0].item())` on a per-sample `t` vector — anatomy loss input
  is mis-stepped for all but sample 0.
- **F8** `normalize_batch` rescales `x_0` per batch before noising.

**Files written (my territory):** `docs/LANDMINES.md` §11, §12 (new); `BRIEFING.md` new section
"Findings added after Stage 1". No the build pass file touched.

**Sent to C1:** one message — the three findings, plus four concrete consequences for
`REBUILD_SPEC.md`. Flagged that **D-14 was my error**: I told the build pass to preserve the three phases as
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
   private are both one-minute fixes. the build pass told not to act.
2. **Your interview answer contains a claim that is now doubly wrong.** The deep dive's scripted
   60-second answer (line 3139) says the key innovation "reduced loss by 99.995% from baseline."
   That number is not in the paper, compares three different loss functions, and — with F6 — the
   phase it credits was trained under an objective that did not require the model to read the
   text. Worth rewriting before it gets said out loud.
3. **Docs-vs-code contradictions found so far: five.** 8-vs-10 clusters; 49.6% vs measured
   38.77%; KMeans on "reduced embeddings" vs raw 66-d; 99.995% absent from the PDF; "ensure 10%
   null conditioning" described in the notes, absent from the code.

## [2026-09-06T04:05Z] Supervisor pass 2 — LANDSCAPE.md reviewed

**the build pass state:** `LANDSCAPE.md` written (21.5 KB). `REBUILD_SPEC.md` and `POSITIONING.md` not yet
present — the build pass mid-synthesis, has not reported. Reviewed anyway (reading does not disturb it) so
the gate review turns around fast when the other two land.

**Independent verification — 5 load-bearing claims re-fetched by me, all exact:**
MDM FID 0.544±.044 / R-Prec-top3 0.611±.007 and the Real row 0.797±.002 / 0.002±.000 (ar5iv
2209.14916 Table 1); MoMask FID 0.045±.002 / R-Prec 0.807±.002 (ar5iv 2312.00063 Table 1, and
confirmed its table has **no** Real row, closing one of C1's own unconfirmed items); PoseScript
CC BY-NC-SA 4.0 (github.com/naver/posescript); arXiv:2601.12809 is a real paper with the
"horizontal attention gradient" claim present. **C1's citation discipline holds under audit.**

**Findings written to `reviews/REVIEW_QUEUE.md`:** SUP-20260906-01..05.
- **02 (P1)** R-Precision is saturated — StableMoFusion 0.841 and MoMask 0.807 both exceed the
  Real row's 0.797. D-03's gate must key on FID, not R-Precision. the build pass reported the numbers without
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

**the build pass corrected a false claim I wrote, and it was right.** `docs/LANDMINES.md` §8 asserted that
rotation-space + differentiable FK "is what MDM and MotionDiffuse do." the build pass checked both papers and
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
output that nobody has audited. the build pass to be invited to audit them.

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
nobody else can produce. the build pass asked to record this as D-18 with a reversal condition.

**Findings filed:** SUP-20260906-06 (P1, E2 blocked — pre-registration ordering must be visible
in the ledger), 07 (P2, E1 needs E3's seed-spread threshold), 08 (P2, PoseScript SMPL status
resolved, four sites stale), 09 (P3, one licence sentence looser than the licence), 10 (P3,
commendations).

**My error, confirmed and owned.** C1's D-11 correction was right; I re-verified against ar5iv
2209.14916 myself. My `LANDMINES.md` §8 claim that rotation+FK "is what MDM and MotionDiffuse do"
was asserted from memory in a document that exists to prevent that. C1's replacement is better
reasoning than the claim it replaced. Also amended my own contradictory territory rule and
invited the build pass to audit `BRIEFING.md` F6-F8 and `LANDMINES.md` §11-12, which are supervisor output
that nobody has checked.

**Next:** Stage 3 (E0 harness reproduction) is C1's work; nothing to review until `RESULTS.md`
or an E0 outcome exists. Supervisor turns to standing queue item 3 — mining the Obsidian deep
dive for further docs-vs-code contradictions (five found so far).

---

## FOR JOEL (running)

5. **The primary task has changed, and you should know before you see the code.** the build pass proposed,
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
   the build pass rates that track "plausible, not established" and is honest that no potential user has
   been spoken to. `POSITIONING.md` §4.

## [2026-09-06T05:25Z] Supervisor pass 5 — crossed messages reconciled, gate confirmed

the build pass reported Stage 2 complete (commit 4091e1b) while my gate review was in flight, and had applied
Review 1's five findings in the same window. Re-checked all of Review 2 against the committed
version and filed an addendum (append-only, prior findings not edited):

- **SUP-08 WITHDRAWN** — already fixed before my review landed; PoseScript SMPL+H G status now
  correct and attributed in all four sites. the build pass disclosed that it had not independently
  re-verified and was relying on my check — correct disclosure, not a silent adoption.
- **SUP-07 NARROWED** — metric choice now right (E1 gates on FID, R-Precision demoted to sanity
  check, note extends this to every rung). Only the threshold wording remains.
- **SUP-06 STANDS** — E2's tolerance still deferred. **E2 remains blocked.**
- **SUP-09 STANDS** — the loose licence sentence is unchanged.

**Gate decision unchanged: RELEASED for E0/E1, E2 blocked.**

**Note on the review loop working in both directions.** the build pass asked me to check its D-11 correction
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

## [2026-09-06T05:45Z] Supervisor pass 6 — F6 audited by the build pass and upgraded; Review 2 fully closed

**Reciprocal audit completed.** the build pass audited the supervisor's F6/F7/F8 against cell 47 directly
rather than taking them on trust. All three confirmed verbatim, including an independent
re-derivation of the `u = c = eps` zero-loss argument, and an exhaustive search establishing that
every `dropout` hit in the notebook is `nn.Dropout` layer regularisation — **zero conditioning
dropout anywhere**, which was the load-bearing half of F6.

**the build pass found something I missed, and it upgrades F6.** Cell 46's own markdown documents the defect
as intentional design. Verified independently by me before writing it in: *"implementation uses
progressive guidance scaling **during training** (2.0->7.0 over 50, 100 epochs)"* and *"**loss
uses** `run two parallel forward` concept."* The report lists "Dual-Path Classifier-Free
Guidance" among Phase 3's seven headline improvements.

**Why this matters more than the bug itself.** F6 was not an oversight — the code matched the
author's intent exactly, and the intent was wrong. That means **no code review would have caught
it**: no discrepancy existed for a reader to notice. The only mechanism that surfaces this class
of defect is a measurement showing the conditioning doing nothing. D-02 (harness before models)
therefore stops being a methodological preference and becomes an empirical claim about this
specific project. Added to `LANDMINES.md` §11 and `BRIEFING.md` F6, attributed to the audit.

**Review 2 fully closed.** D-11/12/13 decided with reversal conditions; D-18 records the task
reframing; D-14 revised with the original text preserved and marked superseded (correct
application of append-or-annotate); SUP-06/07/08/09 all actioned. C1's conditioning-dropout
tripwire — a test that fails if any loss-computing function accepts `guidance_scale > 1.0` — is
a stronger fix than the one requested, because it makes the defect structurally impossible to
reintroduce rather than merely documented. **E2 unblocked**, conditional on the pre-registration
ordering being visible in the ledger when it happens.

**State:** Stage 3 released and under way at E0. Monitor b7nnio6j5 armed for the first measurable
outcome. Nothing of C1's to review until then.
**Next:** standing queue item 3 — mine the Obsidian deep dive for further docs-vs-code
contradictions (five found; the F6 corroboration suggests the notes contain more design-level
claims worth checking against the code).

---

## FOR JOEL (running)

7. **The CFG defect was written up as a feature, not missed.** Cell 46's markdown documents
   running the guidance formula inside the training loss as the intended design, and "Dual-Path
   Classifier-Free Guidance" is one of the seven headline Phase 3 improvements in your report.
   This is the strongest single item in the whole forensic record, because it is the one defect
   that no amount of care would have caught by reading: the code did exactly what it was meant
   to do. Only a measurement would have shown it. If you ever want one sentence for why the
   redo is organised around evaluation, that is it.

## [2026-09-06T06:05Z] Supervisor pass 7 — queue item 3: deep-dive vs code audit

**Verified C1's d6ba070 claims rather than accepting them:** E1 now carries the seed-spread
wording (3 occurrences), E2's ordering requirement is present, SUP-09's loose phrase is gone
(0 hits), D-11/12/13/18 all decided, **0 PENDING remaining**. All four hold.

**Standing queue item 3 executed.** Audited the Obsidian deep dive's stated configuration against
the live code path in cell 47. Determining "live" required care: `setup_models` contains three
model configs, the first two enclosed in a `"""` docstring block. The **third is the live one**.

| deep-dive claim | live code | verdict |
|---|---|---|
| "UNet: 96 -> 192 -> 384 channels" | `base_channels=96, channel_mults=(1,2,4)` | **CORRECT** |
| "**8-head** cross-attention" (twice: line 6 and the cheat sheet) | `attention_heads=4` at every call site; `CrossAttention` default `heads=4` | **WRONG** |
| "Epochs: **50**" / "Training Metrics After 50 Epochs" | `num_epochs = 100` (live; `= 5` is commented) | **WRONG** |
| "Learning rate: 1e-4 -> 1e-6 (**cosine**)", with a "Cosine decay with warmup" ASCII diagram | `torch.optim.lr_scheduler.OneCycleLR(max_lr=fixlr)` | **WRONG schedule** |
| "**23.3K** strategically sampled poses" (line 6) vs "**18,500**" (used ~6 times) vs the actual loaded file `strategic_dataset_**18300**p_210425.pt` | three different numbers for one dataset | **INTERNALLY INCONSISTENT** |
| Sampling config lists cluster key **8** while the same document states **8 clusters (0-7)** | `clusters.npy` has 10 clusters (0-9) | **SELF-CONTRADICTORY** |

**Running total of docs-vs-code contradictions: ten.** Five found earlier (8-vs-10 clusters,
49.6% vs measured 38.77%, KMeans on "reduced embeddings" vs raw, 99.995% absent from the PDF,
"10% null conditioning" absent from the code), five here.

**The pattern worth naming.** These are not scattered typos. Every one of them makes the work
sound *more* systematic than it was: more attention heads, a named-and-diagrammed LR schedule
that was not the one used, a rounder dataset number, a cluster count matching the tidy 8-cluster
story. None of them errs in the unflattering direction. That is what documentation drift looks
like when it is written after the fact from memory rather than read off the code — and it is the
same failure mode as F1 (a layout hypothesis that summed correctly and was never checked against
an invariant) applied to prose instead of data.

**Next:** nothing further in the standing queue that does not depend on C1's E0 outcome. Monitor
b7nnio6j5 remains armed.

---

## FOR JOEL (running)

8. **Your interview cheat sheet has five wrong numbers in it, and they all flatter the work.**
   The "Metrics to Remember" section of `DL - T2P Deep Dive.md` is what you would revise from,
   and against the live code it says 8 attention heads (was 4), 50 epochs (config says 100),
   a cosine LR schedule with a hand-drawn diagram (was OneCycleLR), and 23.3K poses in one place
   and 18,500 in another when the file actually loaded was 18,300. The channel widths
   (96/192/384) are the one thing it gets right. Its sampling config also references a cluster 8
   that cannot exist under its own claim of 8 clusters — there were 10.
   **None of these is damaging on its own. Being asked to reconcile two of them in the same
   interview would be.** Worth one editing pass against the code before you use that document
   again — and worth doing it from the code, not from memory, since writing from memory is how
   the numbers drifted in the first place.

---

# RUN WINDOW — set 2026-09-06 by the author before leaving

**RUN_START:** 2026-09-06T07:55:11Z
**SOFT STOP (take no new work item):** 2026-09-06T13:55:11Z
**HARD STOP (stop mid-item, write handover):** 2026-09-06T14:40:11Z

Author's instruction: work at least 5-6 hours, hard stop at 6-7. If a usage limit interrupts the
run, that is expected — resume on the next firing and continue toward the same deadlines. Do not
treat an interruption as a reason to end the run early.

**Every firing must re-read this block first** and compare against `date -u`. Context may be
compacted between firings; this file is the memory. At HARD STOP, call ScheduleWakeup with
`stop: true` and write the handover at the top of this log.

## [2026-09-06T08:05Z] Supervisor pass 8 — author departed; instructions updated; E0a/E0b split registered

**Author's instructions before leaving, all encoded:**
- **D-19** standing technical authorisation (downloads, envs, installs, code, CUDA->MPS,
  notebooks). Encoded in `CLAUDE.md` rule 4, `AUTONOMOUS_RUN_PROMPT.md` §2/§13,
  `SUPERVISOR_LOOP_PROMPT.md` §0. **The gate that idled C1 cost a full supervisor cycle.**
- **D-20** zero commercial intent; educational / interview / research value only. Two
  consequences: non-commercial licences **do not constrain this project** (SMPL explicitly permits
  exactly this use — stop routing around it), and **numbers are not the deliverable**.
- **D-21** sole authorship; identifiers scrubbed repo-wide; `<ARCHIVE>/` + gitignored
  `.archive_path`. Residual count on every identifier: zero.
- **Stage 5 — Demonstrator** added to `AUTONOMOUS_RUN_PROMPT.md` §8, not optional, five stated
  requirements, local-only.
- `AUTONOMOUS_RUN_PROMPT.md` §Purpose rewritten (rigour + purpose, two governing sentences) and
  Stage 2 Part C's retired "is there a business case" framing replaced.
- `SUPERVISOR_LOOP_PROMPT.md` §1 rewritten to point at this log rather than carry stale state,
  and to make the RUN WINDOW binding.

**Review action this pass: endorsed and formalised C1's own improvement to E0.** Its ledger shows
it planning to compute metrics on ground truth against itself (targeting the paper's Real row,
FID ~0.002) before attempting MDM's number. That is better than the single-gate E0 I specified,
because it separates *evaluator correctness* from *checkpoint loading and generation* — the two
fail independently and the diagnoses differ completely. Asked it to pre-register as **E0a**
(evaluator sanity: real-vs-real FID at the noise floor, R-Prec-top3 near 0.797) and **E0b**
(MDM 0.544 +/-5%), with criteria written before either number exists — the same discipline
SUP-06 blocked E2 on, applied consistently to my own rung design.

**Flagged a real risk to C1:** hand-reconstructing the `opt` Namespace is structurally identical
to F1 — a plausible reconstruction that produces working code and wrong numbers with no error
raised. E0a is the invariant test for it. Asked it to mark any guessed field as a guess in the
ledger *before* seeing the result.

**State:** no experimental result exists yet anywhere in this project. Monitor b7nnio6j5 armed.
**Next:** hold until E0a/E0b report. Standing queue is otherwise exhausted pending results.
