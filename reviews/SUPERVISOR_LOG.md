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

## [2026-09-06T08:20Z] Supervisor pass 9 — E0 reviewed. FIRST MEASUREMENT IN THE PROJECT.

**E0 landed.** Evaluator harness sanity check, real-vs-real, 2 seeds, full test split (2099/2099).
R-Prec-top3 **0.720** (paper's Real row: 0.797), FID **0.029** (paper: 0.002), matching score
3.606. Seed spread negligible; ran on CPU so §7 does not apply. **Correctly labelled PARTIAL and
explicitly NOT the D-03 gate** — no generative model is in the loop yet.

**Accepted.** The entry pre-registers its criterion, reports spread, carries a real "Does NOT
establish" list, and documents its own v1 bug (200-candidate R-Precision pool -> 0.280, vs 0.710
at the protocol's 32) rather than silently fixing it. That is the discipline landing on this
project's own work, not just the original's.

**Finding the build pass missed, from its own data (SUP-11, P1).** It attributed both gaps to one
cause (missing multi-crop averaging). Its own progression table refutes that: v2->v3 (n 1024 ->
2099, nothing else changed) moved **FID 0.173 -> 0.029** (6x, super-linear in n) while
**R-Prec-top3 moved 0.710 -> 0.720** (flat). FID estimates a 512x512 covariance and is biased
upward when n/d is small (2.0 -> 4.1 here); R-Precision is batch-wise retrieval over a fixed
32-candidate pool and is structurally insensitive to n. **Different mechanisms, so one hypothesis
cannot cover both.** The v1 row is the clincher: real-vs-real FID of 0.745 at n=200, where the
true value is ~0 — that number was 100% estimator bias.

**Consequence (SUP-12, P1):** the R-Precision residual is now confounded between crop averaging
and checkpoint provenance (`Tevior/text_mot_match`, a third-party re-upload, architecture- but not
cryptographically verified). Decisive test specified: implement `--repeat_time` averaging; converge
to 0.797 means the checkpoint is fine, plateau at 0.72 means suspect it and obtaining the official
artifact becomes blocking for D-03. Told the build pass to run this **before** E0b, since a wrong
checkpoint would make E0b's number uninterpretable without it knowing.

**Wrote `LANDMINES.md` §13 and §14** — R-Precision's candidate-pool dependence, and FID's
covariance bias with the progression table as evidence. **First LANDMINES entries sourced to this
project's own measurements rather than the original's mistakes.**

**Monitor b7nnio6j5 has exited** (it fired and ended). Re-arm on next relevant wait.
**Next:** await the two diagnostics, then E0b. Run window: SOFT 13:55Z / HARD 14:40Z.

---

## FOR JOEL (running)

9. **The project has its first real number.** The evaluation harness works: matched text/motion
   pairs retrieve each other at 72% top-3 out of 32 candidates against ~9% chance. It is not yet
   the full gate — no generated model is in the loop — but the instrument is real and behaving.
   Two gaps from the published reference remain, and they turn out to have different causes; one
   is largely a statistical artifact of sample size, the other is still open and may come down to
   whether the checkpoint we could obtain is the same one the paper used.

## [2026-09-06T08:35Z] Supervisor pass 10 — crossed messages resolved; D-19a; ownership answered

Both build-pass messages crossed with Review 3. Consolidated reply sent. Substance:

- **`POSITIONING.md` ownership answered: it is the build pass's file, it updates it.** I confirmed
  I am not editing it and there is no concurrent edit to collide with — my only edits this window
  were `LANDMINES.md` §13/§14 and D-19a. Directed the D-20 rewrite: drop the licence-routing frame
  from §1/§4 (zero commercial intent means SMPL's terms already permit this use), demote Track B
  from "the licence-clear exit" to one demonstrator option, and make §6's "smallest demonstrable
  thing" concrete enough for Stage 5 to build from.
- **D-19a encoded** (`CLAUDE.md` rule 4, `docs/DECISIONS.md`): the author's relayed threshold —
  ask only at system-breaking scale ("300GB download", "200GB environment"). Recorded explicitly
  as *relayed, not heard first-hand*, and noted that it adds a guardrail above the bar rather than
  widening the authorisation below it, which is why acting on it is safe. Practical effect: MDM's
  checkpoint gets downloaded without a round-trip.
- **Corrected my own finding's framing.** The build pass *did* identify the covariance problem
  itself — it is why v1 was binned rather than reported. SUP-11 is narrower than "you missed it":
  it had the mechanism and did not carry it forward to separate the two *residual* gaps. Said so
  plainly rather than letting an overbroad finding stand.

**Worth recording as the behaviour that matters most so far.** The build pass wrote, unprompted:
*"I didn't quietly redefine E0 to make it pass; the sanity check and the gate are two different
things."* It had every opportunity to report "E0 passed" — the numbers are good, the signal is
real — and instead put the distinction in the entry's title where a skim cannot miss it. **That
exact substitution, a partial result standing in for the gate it resembles, is what produced the
original project.** It did not happen here, and it did not happen because a rule caught it.

**Monitor bzlil63uc armed** for the next E-series entry.
**Next:** await the `--repeat_time` and n-sweep diagnostics (SUP-11/12), then E0b.
Run window: SOFT 13:55Z / HARD 14:40Z. Currently ~08:35Z, ~5h20m to soft stop.

## [2026-09-06T09:05Z] Supervisor pass 11 — E0b reviewed. P0 raised: D-03 gate status misstated.

**E0b ran and FAILED as pre-registered.** MDM's own released checkpoint through this project's
vendored evaluator, n=128, 1 seed: generated FID **1.0731** vs the pre-registered band
0.5168-0.5712. Reported as FAIL with the actual value, tolerance not widened. Driver crash
(`diversity_times` off-by-one) and hand-assembled provenance both disclosed. **The run's process
discipline was correct throughout.**

**SUP-15 (P0).** E0b's "Next" claimed *"D-03's gate is satisfied by E0a per the director's Stage 2
review."* Checked against what Review 2 actually said: condition 1 read *"MET. E0 vendors... gate
at FID within +/-5% of 0.544"* — that judged the **plan** adequate and named the reproduction as
the gate. E0a's own title says "NOT the D-03 gate itself." D-03's text carries the fallback:
reproduce a published figure *or* **explicitly downgrade every number to internally-comparable-
only**. **D-03 recorded as UNRESOLVED**; downstream E1-E4 must carry the internally-comparable-only
label until it resolves. Raised P0 because the claim would otherwise have been inherited silently
by every later result — the exact class of unearned headline this project exists to remove.

Worth noting the likely cause is an honest misreading of my "condition 1 MET" wording rather than
an escape attempt; the build pass had itself labelled E0a correctly one entry earlier. Said so in
the finding.

**SUP-16/17 (P1) — the diagnosis was sitting in its own results table.**
- Bias subtraction: floor at n=128 is `0.1339 - 0.029 = 0.105`; corrected generated FID
  `~0.968`, still ~1.7x outside the band. **Sample size does not rescue the FAIL.**
- **GT R-Prec-top3 reproduced at 0.7969 against the paper's 0.797 — three decimals.** So n=128 is
  *fine* for R-Precision, which independently validates the evaluator and data pipeline, and makes
  the generated R-Prec of **0.7578 a real measurement** — 0.147 *better* than MDM's published
  0.611, while FID is 2x *worse*.
- **Better text alignment plus worse distributional realism is the signature of stronger
  classifier-free guidance than the reference protocol.** Directed a minutes-long check of the
  effective sampling guidance and diffusion step count before anything else.

**SUP-18 (P2):** blocked the proposed ~5 CPU-hour n~1000 rerun until that check runs. A long run
under a wrong configuration would reproduce the same error at higher precision — the expensive
version of this project's founding mistake.

**Next:** await the guidance/step-count check. Re-arm monitor.
Run window: SOFT 13:55Z / HARD 14:40Z. Now ~09:05Z, ~4h50m to soft stop.

---

## FOR JOEL (running)

10. **The harness validation did not pass, and that is a real (documented) branch, not a
    setback.** Running MDM's own published model through our evaluator did not reproduce its
    published score. Two things make this interesting rather than alarming: the *ground truth*
    numbers reproduced almost exactly, so our instrument is demonstrably correct; and the generated
    results are simultaneously *better* than published on text alignment and *worse* on realism,
    which is the fingerprint of one specific sampling setting being off rather than anything deep.
    Likely a configuration bug on our side, being checked now. Until it resolves, every number this
    project produces is labelled as comparable only to itself — which is what D-03 said to do when
    written, months before it mattered.

## [2026-09-06T09:35Z] Supervisor pass 12 — checkpoint provenance CLOSED; sharper test identified

Build pass's E0b report crossed with Review 4. It repeated the D-03 claim; SUP-15 (P0) is already
in its queue.

**SUP-12 (checkpoint provenance) is now CLOSED, in both directions.**
- **Generator:** MDM's checkpoint zip shipped with the author's own 2022 evaluation log — 20
  replications, FID **0.5443±.0442** against the paper's 0.544±.044. Independent confirmation
  obtained *before* running anything.
- **Evaluator:** E0a's ground-truth R-Precision reproduced at **0.7969** against the published
  0.797 — three decimals. Behavioural validation of the third-party re-upload.

Both instruments are validated. Which makes the generated-side discrepancy more interesting, not
less — it can no longer be attributed to either checkpoint.

**A cheaper and sharper test than the one I gave in SUP-17, identified from the build pass's own
find.** That bundled log is the author's run of this exact checkpoint under the correct protocol.
If it reports R-Precision:
- author's log ~**0.611** -> the published number is reproducible under the right protocol, so the
  0.7578 is *our run's configuration* (guidance / step count). A driver bug, findable in minutes.
- author's log ~**0.75** -> the *paper's* 0.611 is the outlier, and we have a genuine discrepancy
  between MDM's published table and its own released artifacts. **That would be a real finding
  about the field's record.**

Costs a file read, not 40 minutes of CPU. **Also flagged: MDM's published R-Prec of 0.611 is a
conspicuous outlier** in the build pass's own `LANDSCAPE.md` §1.3 table — MotionDiffuse 0.782,
MLD 0.772, T2M-GPT 0.775. Our 0.7578 is the *typical* value for this benchmark. Told it to be
curious about that rather than assume our run is the wrong one.

**Answered its two questions:** do not rerun for the Diversity off-by-one alone (not decisive, 40
min real) but bundle the fix into any rerun the config check forces; do not hold, work
cheapest-first. SUP-18 stands — no ~5-hour n~1000 run until the configuration is confirmed.

**Delegated two write-ups to the build pass** (its material, append-or-annotate covers it):
`LANDMINES.md` §15 on the caption `.split("#")` bug — MDM's loader swallowing per-sample parse
errors in a bare `except` and silently producing an **empty** generated dataset, found by
inspecting `repr()` of the raw field rather than reasoning about it. And a ledger note on its
operational finding that deferred waiting does not advance wall-clock for an already-running
background process; only active tool calls do. That constrains how long jobs get supervised here
and is not written down anywhere yet.

**Next:** await the bundled-log check. Monitor bedyegf9l armed.
Run window: SOFT 13:55Z / HARD 14:40Z. Now ~09:35Z, ~4h20m to soft stop.

## [2026-09-06T10:05Z] Supervisor pass 13 — E0b settled by reading, not by spending

**Build pass proposed a second 40-minute replication to estimate variance. Blocked it and read the
bundled author log instead.** 273 lines, 20 replications, shipped inside the checkpoint zip.

**SUP-20 — every published MDM figure reproduces exactly from released artifacts:**
GT R-Prec-top3 0.7977±0.0022 (paper 0.797), **vald R-Prec-top3 0.6110±0.0067 (paper 0.611)**,
GT FID 0.0016 (paper 0.002), **vald FID 0.5443±0.0442 (paper 0.544±.044)**, matching scores
2.9758 / 5.5659 (paper 2.974 / 5.566). **The field's record is sound. The discrepancy is ours.**
I was wrong to invite curiosity about MDM's 0.611 being an outlier — retracted in the finding.

**SUP-17 WITHDRAWN.** The build pass disproved my driver-misconfiguration hypothesis empirically —
guidance 2.5 matching the log filename, 1000 timesteps unrespaced, `p_sample_loop` not DDIM. It
instantiated the model and printed runtime values rather than trusting a code read. Correct method,
my claim was wrong. **Second finding of mine it has overturned.**

**SUP-21 (P1) — the three-way comparison localises the defect and explains an older gap:**

| | E0a | E0b | author |
|---|---|---|---|
| GT R-Prec-top3 | 0.720 | **0.7969** | 0.7977 |
| GT Matching Score | 3.6057 | — | 2.9758 |
| gen R-Prec-top3 | — | 0.7578 | 0.6110 |
| gen FID | — | 1.0731 | 0.5443 |

- **E0a's R-Precision gap was its hand-built data pipeline, not multi-crop averaging.** E0b used
  MDM's own loader and landed on the reference. Same evaluator, checkpoint and dataset — the only
  difference is the pipeline. **The F1-shaped risk I flagged at the start of E0a did bite**, and
  the build pass's own move to the upstream loader is what exposed it. SUP-11's open half closes:
  FID half = small-n covariance bias, R-Precision half = hand-rolled data path.
- **E0b's GT path is correct, so the defect is in generation.** Generated motions are
  simultaneously easier to text-match (+0.147) and further from the real distribution (~2x FID).
  Real-motion contamination ruled out — that lowers FID.
- Leads handed over, cheapest first: generated motion **length distribution** vs GT and how the
  driver sets per-sample `n_frames`; then unique-caption count and caption-to-motion pairing.

**SUP-22:** the variance question the proposed rerun would have answered is already answered by the
log — CInterval 0.0442, replications spanning 0.5323-0.7114. Our 1.0731 is far outside. **40
minutes of CPU saved by reading a file that was already on disk.**

**SUP-23:** generated motions are not cached anywhere, so every follow-up costs another ~39 min.
Told it to persist them. General lesson: in a CPU-bound loop, cache the expensive intermediate.

**Next:** await the length/caption diagnosis. Re-arm monitor.
Run window: SOFT 13:55Z / HARD 14:40Z. Now ~10:05Z, ~3h50m to soft stop.

---

## FOR JOEL (running)

11. **The harness question is resolved, and the answer is reassuring.** MDM's published numbers
    reproduce *exactly* from its released files — we found the author's own 20-run evaluation log
    bundled inside the checkpoint. So the benchmark's published record is trustworthy, and our
    earlier failure to match it is a bug on our side, now narrowed to the motion-generation step
    specifically (our ground-truth numbers match the reference to three decimals). That is a much
    better position than "the field's numbers don't reproduce" — it is findable.
12. **A hand-written data pipeline was silently slightly wrong, and got caught.** The earlier
    evaluator run scored 0.720 where the reference is 0.797; switching to the upstream project's
    own data loader gave 0.797. Same failure *shape* as the original project's decode bug — a
    plausible reconstruction that runs cleanly and produces wrong numbers. It was caught here in
    hours, by comparison against a reference, rather than in months.

## [2026-09-06T10:35Z] Supervisor pass 14 — DECISION: stop chasing E0b, take D-03's fallback, move to E1

Build pass proposed a third 40-minute run (randomized redraw) to test subset composition.
**Refused, and refuted the hypothesis for free from a number it already had.**

**The refutation.** R-Precision uses the same distractor pool and batching for GT and generated.
An "easier" subset inflates both. Sigma estimated from the author's log (vald CInterval 0.0067
over 20 reps -> per-rep std ~0.015 at ~31 batches -> ~0.043 at 4 batches):
- **GT: 0.06 sigma off** (0.7969 vs 0.7977)
- **generated: ~3.5 sigma high** (0.7578 vs 0.6110)
Subset composition cannot produce that asymmetry. Hypothesis dead, zero compute spent.

**Verified the driver myself against MDM's `eval_humanml.py`**, so the build pass can stop looking:
`ClassifierFreeSampleModel` applied once not doubled; `args.batch_size = 32` correctly overriding
the checkpoint's `batch_size: 64` (it caught MDM's own "This must be 32!" comment — the exact bug
class of `LANDMINES.md` §13, not repeated); `use_ema` defaults match. **Driver is faithful.**
Together with its runtime checks on guidance/timesteps/sampler, every cheap surface is eliminated.

**D-22 recorded — the call, and the reasoning behind it.** Closing E0b costs ~5 CPU-hours and buys
*comparability to the published ladder*, not correctness of our own measurements. Correctness is
already established: GT R-Precision reproduces at 0.06 sigma against a 20-replication reference,
and **that is the property every internal comparison depends on**. Under D-20 the project's value
is educational/interview/research plus a demonstrable artifact. **E1 — what the original's task
framing actually cost — is this project's own contribution and nobody else's.** Five hours matching
someone else's published number is five hours not spent on the only number nobody else can produce.

So: D-03's fallback taken explicitly, every downstream number labelled
**internally-comparable-only**, the generation anomaly recorded as a live open question with its
evidence and the cost of closing it, E0a's superseded explanation corrected, and work moves to E1.

**Design instruction carried into E1:** prefer upstream implementations over reconstructions; where
a reconstruction is unavoidable, build the reference comparison *first*. E0a's hand-built pipeline
was wrong and only surfaced because E0b disagreed with it. **That is the F1 lesson landing twice in
one project — once in the original, once in ours.**

**Next:** E1 design and results. Re-arm monitor.
Run window: SOFT 13:55Z / HARD 14:40Z. Now ~10:35Z, ~3h20m to soft stop.

---

## FOR JOEL (running)

13. **I stopped a line of work, and you should know why.** Reproducing another team's published
    number exactly would have cost ~5 hours of compute and bought comparability to a public
    leaderboard. Our evaluator is already proven correct against ground truth, which is what makes
    *our own* comparisons trustworthy. Given your goals are learning, interview value and something
    demonstrable, I judged that time better spent on E1 — measuring what your original project's
    task framing actually cost — because that is the one number in this project that nobody else
    could produce. The unexplained anomaly is documented, not buried, and reversible if you ever
    want the leaderboard claim.

## [2026-09-06T11:10Z] Supervisor pass 15 — round 2 misread; SUP-16 retracted; R-Prec anomaly isolated

**SUP-25 (P1): round 2 did not test generation variance.** `fixseed(seed=10)` makes generation
deterministic; the added diagnostic code only shifted RNG before `gt_loader`'s shuffle. Converting
to counts makes it unambiguous — **vald 97/128 in both runs, identical to 4 s.f.**, while GT moved
102/128 -> 104/128. Same motions scored twice. Its conclusion "further evidence this is not
single-run noise" is unsupported.

**SUP-26 (P1): I retract SUP-16.** Round 2 accidentally ran a better experiment than intended —
generated set held identical, reference redrawn — and **vald FID moved 1.0731 -> 1.3997, a +30%
swing from the reference redraw alone.** A statistic that unstable cannot adjudicate a 2x
difference against a +/-5% tolerance. My bias-subtraction arithmetic assumed a stability the
estimator does not have at n=128. **The build pass's original "this run cannot distinguish"
framing was right and I pushed back on it too hard.**

**SUP-27: the same design strengthens the R-Precision finding.** Generation pinned, GT batching
noise ~+/-2/128 ~ 0.016; generated excess 0.147 ~ **9x that**. Clean split now on the record:
**FID underpowered at this n, no conclusion; R-Precision anomaly real, ~9x noise, unexplained.**

**SUP-28: the only remaining E0b work, and it is free.** Motions are cached, so fix one
full-scale GT reference (E0a's 4198 embeddings) and re-score the cached 128 against it. Kills the
redraw variance and makes every future FID mutually comparable — which is what
"internally-comparable-only" must mean to be worth anything. Then E1.

**Score so far: two supervisor findings withdrawn (SUP-16, SUP-17) against one build-pass error
(the D-03 misstatement).** Recorded deliberately. That ratio is the arrangement working — the
measurements are deciding, not the seniority.

**Next:** SUP-28's free re-score, then E1. Monitors b6bbwpaim and bfovb8dqn both exited; re-arm.
Run window: SOFT 13:55Z / HARD 14:40Z. Now ~11:10Z, ~2h45m to soft stop.

## [2026-09-06T11:35Z] Supervisor pass 16 — D-22 accepted; E1/Stage 5 criteria pre-registered

**Build pass accepted D-22** (commit d545825: "stop chasing E0b, take D-03's fallback, record open
question, move to E1"). E0b closes as UNRESOLVED with its evidence recorded, not buried. Sent a
short unambiguous directive answering its "randomized redraw or n~1000?" question — **neither**,
do SUP-28's free re-score, then E1 — plus the SUP-25 correction it had repeated a third time
(the two runs were not independent in generation; 97/128 both times).

**Independent work this pass: pre-registered the review criteria for E1 and Stage 5, before either
exists.** Same discipline this queue enforced on the build pass twice (SUP-06 on E2's tolerance,
SUP-15 on the D-03 gate) — now applied to my own judging, so "did it succeed" cannot be decided
after seeing what happened.

**E1's bar:** controlled to one variable; multi-seed with the effect exceeding the spread, or the
words "no effect resolvable at this budget"; both arms labelled internally-comparable-only; F3's
1.43x dispersion figure explicitly connected to the resulting delta; a generalisation-limit
section; and the F6/D-02 framing stated empirically. **The failure mode I will look for hardest:
a large satisfying number that is really measuring training budget rather than task framing — a
frame-0 model and a sequence model are not automatically comparable at equal epochs.**

**Stage 5's bar:** one command from a checkpoint, no GPU; comprehensible in thirty seconds without
narration; **the retrieval baseline visible in the interface, not just the report** (the criterion
I expect to be softened and will hold hardest — "just look it up in the training set" is what a
sceptic is silently thinking); failure cases reachable, not curated away; metrics and seed spread
surfaced with the comparability label; reproducible by someone else. **Failure mode: a polished
interface around unestablished quality, metrics tucked out of sight — the author's "usable
PRODUCT" becoming a veneer, which is the opposite of the point.**

**Next:** E1. Re-arm monitor.
Run window: SOFT 13:55Z / HARD 14:40Z. Now ~11:35Z, ~2h20m to soft stop.

## [2026-09-06T11:55Z] Supervisor pass 17 — E1 as specified was a tautology. Redesigned before the scaffold.

The build pass asked, before committing to a model design, whether the ladder still reflected
intent. **It did not, and the flaw was mine.**

**SUP-30 (P1).** `REBUILD_SPEC.md`'s E1 compares frame-0 static pose against full-sequence
generation, scored by the Guo evaluator — which embeds **motion sequences**. Scoring a frame-0 arm
requires replicating one pose across T frames, and a static repeated pose has catastrophic FID
against real motion **because it does not move**. The result is guaranteed before the run. *"A
model that outputs one frame is worse at generating motion than a model that outputs motion"* is
not a finding.

Note the irony worth recording: this is exactly the E1 failure mode I pre-registered one pass
earlier — "a large satisfying number that is really measuring something other than task framing" —
arriving from a direction I had not anticipated when I wrote it. **The pre-registration caught my
own error, not the build pass's.**

**The redesign: hold output space fixed, vary only the pairing.** The original's defect was a
*conditioning mismatch*, not an output-shape choice, and it decomposes into two separable errors —
caption truncation, and frame selection. All arms generate full sequences in 263-d so the evaluator
applies identically:
- **A (control):** full caption -> full sequence
- **B:** first-action clause only -> full sequence — isolates caption truncation
- **C:** full caption -> frame-0-representative conditioning — isolates frame selection

**A vs B is the minimum viable E1** and still converts F3's 1.43x dispersion into a measured delta.

**Two hard preconditions set before any training starts:**
1. **Measure feasibility, then commit.** §7's estimate was a placeholder; the real datum is that
   MDM *generation* cost ~39 min for 128 samples on this CPU, and training costs far more.
   Seconds-per-step measured, full matrix extrapolated, projection written to the ledger before
   starting. **A reduced experiment stated as reduced is a result; a reduced experiment reported as
   complete is the original project's mistake.**
2. **Use MDM's architecture, not a new denoiser.** Vendored, MIT, runs here, already on the 263-d
   representation D-11 selected, published numbers reproducible. Makes the only difference between
   arms the thing E1 measures. A novel denoiser adds a variable and buys E1 nothing. `src/t2p/`
   wraps rather than reimplements — also the shape most likely to leave Stage 5 reachable.

Asked the build pass to update the ladder to A/B/C before scaffolding and to record in DECISIONS
that E1's original specification was mine and was wrong.

**Next:** E1 redesign + feasibility projection. Monitor b2dwtq9f7 armed.
Run window: SOFT 13:55Z / HARD 14:40Z. Now ~11:55Z, ~2h to soft stop.

---

## FOR JOEL (running)

14. **The headline experiment was designed wrong, and it was my error.** As written, comparing
    "single pose" against "full motion" would have produced a huge, impressive-looking number that
    meant nothing — a model that outputs one frame is obviously worse at generating motion, and
    that has nothing to do with the bug we wanted to measure. Redesigned so every arm generates
    motion and only the *caption-to-target pairing* changes, which is what your original project
    actually got wrong. Caught before any training happened, by criteria I had written down an hour
    earlier for judging someone else's work.

## [2026-09-06T10:20Z ACTUAL] Supervisor pass 18 — SUP-28 oversold; E0b closed for good

**TIMESTAMP CORRECTION:** passes 8-17 carried hand-estimated times that drifted ahead of the clock.
Real time is 10:20Z. **RUN_START 07:55Z, SOFT 13:55Z, HARD 14:40Z — ~3h35m to soft stop.**
Future passes: read `date -u`, do not estimate.

**SUP-31 (P2): my SUP-28 was overstated and the build pass's diagnosis is correct.** The
fixed-reference re-score returned **FID 3.2909** — higher than round 1 (1.0731) and round 2
(1.3997). Fixing the reference does not fix the estimator: the *generated* side is still n=128, and
a 512-dim covariance from 128 samples is rank-deficient (rank <=127). A well-conditioned reference
paired against a rank-deficient test covariance is its own instability. My claim that this would
"make every future FID comparable" was wrong as stated.

**Three FID values from the same bit-identical 128 motions — 1.0731 / 1.3997 / 3.2909 — is the
cleanest demonstration in this project that a statistic can measure its own estimator rather than
the thing under test.** Asked for it in `LANDMINES.md` §14 as a worked example; better than the
progression table already there because generation is held constant and only the reference moves.

**The artifact is still right for E1, and I said so rather than abandoning it.** Absolute value
meaningless, *ordering* valid provided every arm uses the same reference at the same generated n —
which is exactly E1's within-project A-vs-B comparison. `fixed_gt_reference.npz` is what makes
E1's FID column mean anything.

**Noted: the build pass had already annotated D-22 itself**, correctly, with attribution, before I
attempted the same edit — my change was redundant and I dropped it. Append-or-annotate working in
the direction it was designed for.

**Running tally of withdrawn/corrected supervisor findings: SUP-16, SUP-17, SUP-28, plus E1's
original specification (SUP-30 was self-caught).** Against one build-pass error (the D-03
misstatement) and one build-pass misread (round-2 independence). Recorded deliberately.

**Next:** SUP-30 — E1's A/B/C redesign and the feasibility projection. That is what I most want
landed before the hard stop; a well-specified experiment with an honest cost estimate is a better
pause point than a half-trained model.

## [2026-09-06T10:30Z] Supervisor pass 19 — SUP-32: the ladder gates E1 on a metric E1 cannot afford

**E1 redesigned to A/B/C as SUP-30 asked** — output space held identical across arms, only the
caption-to-target pairing varies, evaluator applies without confound. Correct.

**SUP-32 (P1): two of my own findings collide inside the new ladder.**
- SUP-02 said gate on FID, not R-Precision (saturated at the frontier: StableMoFusion 0.841 and
  MoMask 0.807 both exceed the Real row's 0.797).
- SUP-31 established FID needs the *generated* side well above the 512-dim embedding or the
  covariance is rank-deficient — proven by three values from one bit-identical set of 128 motions.

E1B's criterion is currently FID-based. **At any generated sample count this hardware affords, that
is unmeasurable.** From the hard datum (~39 min / 128 samples on CPU): 2 arms x 3 seeds gives
**~3.9 h at n=128, ~15.6 h at n=512, ~31.2 h at n=1024 — generation alone, training on top.**

**Resolution: for E1, invert SUP-02. Gate on R-Precision; report FID as secondary with its
instability stated.** SUP-02's saturation argument is about *the frontier*, not about the metric.
E1's arms will be laptop-scale, nowhere near 0.80, where R-Precision has ample range — and it is
**stable at n=128**, which the build pass proved (GT reproduced to three decimals; batching noise
floor +-2/128 ~ 0.016). It estimates no covariance, so sample count does not wreck it.

Scoped the inversion explicitly to the low-quality regime, and asked for `docs/DECISIONS.md` to
record that **SUP-02's guidance is regime-dependent** — written about the published frontier, does
not transfer unchanged to a laptop-scale rebuild. That nuance was missing from my original finding;
my correction to make, not the build pass's.

**Also required of the feasibility projection: include generation, not just training.** Sampling a
diffusion model at 1000 steps is the dominant recurring cost per arm per seed, and it is the number
most likely to be underestimated.

**Fifth supervisor correction of the run** (SUP-16, SUP-17, SUP-28, E1's original spec, now SUP-02's
scope). Each caught before it cost a run.

**Next:** E1 criteria update + feasibility projection. That is the thing I want landed before hard
stop. Re-arm monitor. Now 10:30Z; SOFT 13:55Z, HARD 14:40Z.

## [2026-09-06T10:40Z] Supervisor pass 20 — feasibility probe accepted; E1 staged to fail cheap

**Probe accepted, and it exceeded the brief.** Real per-step timings (2.252 s/step median, <7%
spread, MDM `trans_enc` defaults, 17.88M params, batch 32), a full extrapolation table, and — the
part worth naming — **the MPS failure reproduced deliberately and reported** rather than silently
falling back to CPU. Recorded as **D-23**: training is CPU-only; MDM's diffusion schedule indexes a
float64 array and MPS refuses float64. D-08's MPS assumption is refuted by measurement. Side
benefit: `LANDMINES.md` §7's non-determinism caveat does not apply to CPU training, so seed
variance there is genuine.

**SUP-33 (P1): declined to sign off on 3,000 steps x 4 runs as a single 7.5h commitment.**
3,000 steps is **0.63% of MDM's published budget**. Both arms will be severely undertrained, and
**if neither learns to use text at all, E1A and E1B look identical — a floor effect that reports as
"no measurable difference" and reads as a finding.** That is the most likely route to a confidently
wrong E1.

**Staged instead, with a pre-registered positive control:** run E1A alone, one seed, ~1.9h, gated on
**R-Precision-top3 exceeding chance = 3/32 = 0.0938**. Above chance -> the comparison has power,
proceed. At chance -> **stop at 1.9h instead of 7.5h**, and treat "E1 is not affordable at a budget
that gives it power on this hardware" as a legitimate measured result about what a laptop-scale
rebuild can establish — which it is.

Also flagged: **SUP-32's metric inversion has not landed yet** — `REBUILD_SPEC.md` §6 still says
FID is decisive and E1B's criterion is still FID-based. Running against a superseded criterion
produces an unusable number. And: two seeds give a **range, not a spread** — report the gap against
the within-arm range with n=2 inline; no standard deviation from two points, no significance
phrasing.

**Handover framing set.** My window ends 14:40Z (~3h15m). A 7.5h E1 will not fit; a staged 1.9h
E1A might. Told the build pass to prefer landing the power check plus its gate, and to write
`LEDGER.md` for a successor with none of this conversation.

**Next:** E1A power check. Re-arm monitor. Now 10:40Z; SOFT 13:55Z, HARD 14:40Z.

## [2026-09-06T10:32Z] Supervisor pass 21 — SUP-34: a zero-training measurement of E1B's effect

**SUP-32 verified as landed.** E1A/E1B/E1C and E2 now carry R-Precision-decisive criteria with FID
secondary and its instability flagged inline; a regime note is recorded as D-25. **The build pass
generalised the principle to E3 as well** ("re-assess whether this rung's own generated n affords
trusting FID before treating it as decisive") — not asked for, and correct.

**SUP-34 (P1): E1B's effect can be measured with no training at all, in minutes.**
The evaluator is a *validated joint text-motion embedding space* — E0b reproduced GT R-Precision at
0.7969 against a 20-replication reference of 0.7977. So re-encode the **same real test motions**
against **truncated captions** (the original's own POS-tag first-action rule) and compare to the
0.7969 baseline already measured. The drop **is** the alignment signal the truncation destroyed,
in E1's own metric space, on real data, with no training confound.

Why it goes first: it is an **upper bound** on E1B (a model cannot exploit information truncation
already removed); a **power check with teeth** (near-zero drop -> drop E1B entirely and F3's caption
half is answered in minutes; large drop -> E1B is worth its 7.5h and the expected effect size is
known in advance); it measures the defect **at source** rather than through an undertrained model
that attenuates it; and it is **immune to SUP-33's floor effect** because nothing is trained.

Scoped honestly in the finding: measures encoder information loss, **not** generation quality, and
does not replace E1B as a generative result. Registered as **E1-pilot with its prediction stated
before running**, which converts E1B from an open-ended run into a test of a specific prediction.

**Asymmetry flagged rather than papered over:** E1C's frame-selection half has no clean free
analogue — a frame-0-replicated static motion in retrieval reintroduces the same "static motions
are unusual" confound SUP-30 removed from generation. Told the build pass to state why, not force a
symmetric pilot.

**Sequencing set for the window:** E1-pilot (minutes), then the E1A power check with the 0.0938
gate. Both landing would end this session with a measured prediction *and* a measured power result
— a stronger handover than a partial training run.

**Next:** E1-pilot result. Re-arm monitor. Now 10:32Z; SOFT 13:55Z, HARD 14:40Z (~4h).

## [2026-09-06T10:36Z] Supervisor pass 22 — answered both design questions; three-stage kill-chain set

Build pass landed SUP-30/31/32 (commit 57d8dd6) and asked two design questions before touching any
run. Both answered.

**SUP-35 (P2) — "is 3,000 steps enough / should there be a 500-step pilot?" No, and here is the
better instrument.** A 500-step pilot cannot answer it because **diffusion training loss is a poor
convergence signal by construction** — it averages over uniformly sampled timesteps, so most of its
variance is which timesteps were drawn, not model quality. The problem is not noise, it is the
**absence of a reference scale**.

**Proposal: measure MDM's converged checkpoint's training loss on our own data.** Same
`training_losses` call, same loader, same batch distribution, few dozen batches, forward passes
only. That yields the loss *this exact architecture and objective* reaches at convergence, on our
data. Then 3,000 steps becomes "reaches L against converged reference L*, ratio R" — a reportable
fact instead of a judgment call. Flagged that both sides must average over the same batch count
with the same timestep-draw seed or the comparison inherits the noise it exists to defeat.

**SUP-36 (P3) — one stale row.** E5 still reads "compared against E1's frame-0-HumanML3D result";
after SUP-30 there is no static-pose output arm anywhere. Restate or drop the cross-reference.
Otherwise the ladder is internally consistent at 57d8dd6.

**Three-stage kill-chain set for the remaining window, each stage able to kill the next:**
1. **E1-pilot (SUP-34)** — minutes, zero training. Near-zero truncation drop -> E1B has nothing to
   find, F3's caption half answered.
2. **Converged-loss reference (SUP-35)** — minutes, forward only. Supplies the scale.
3. **E1A power check (SUP-33)** — ~2.5h, gated on R-Prec-top3 > 0.0938, pre-registered.
Right shape when the full matrix is ~10h and the window is ~4h.

**Noted for the record: the build pass caught SUP-32's training-vs-generation cost omission itself**,
while re-deriving my numbers, in the same pass I was flagging it — and appended the correction
rather than silently fixing it. **Second time re-deriving rather than accepting has caught
something.** Also handled the D-23/D-24 numbering collision correctly (renumber, touch no content,
note why); we had written the same MPS finding independently within minutes.

**Next:** E1-pilot result. Re-arm monitor. Now 10:36Z; SOFT 13:55Z, HARD 14:40Z (~4h).

## [2026-09-06T10:42Z] Supervisor pass 23 — SUP-37: blocked the power check. Train-on-test can pass it spuriously.

Build pass pre-registered E1A-power well — hypothesis and criterion before the run, 0.09375 stated
numerically, both branches including "stop", a real "Does NOT establish". **But it disclosed
train-on-test without following it to its consequence, and for a power check that is disqualifying.**

**The argument.** The check exists solely to decide whether ~10h of matrix is worth spending.
Trained and evaluated on the same captions, **R-Precision can exceed chance through memorisation
alone** — so the gate can pass for a reason that does not support the decision it gates. A power
check that can pass spuriously cannot do its only job.

**And the contamination propagates.** A spurious pass -> matrix runs -> E1B still shows a gap,
because truncated captions carry less distinguishing information and therefore memorise less well.
**That would be a real difference in memorisation capacity, reported as the cost of caption
truncation on text-conditioned generation.** Different claims; the wrong one is the one that would
enter the record. Same shape as the defect this project exists to correct.

**Required:** materialise HumanML3D's train split (the `materialize_humanml3d_test_subset.py`
pattern extended to `train.txt` — data processing, minutes, no approval needed under D-19), train
on train, evaluate on test. If genuinely impossible, re-scope the check's own title and hypothesis
to "can this architecture memorise at this budget" and bar it from gating the 10h decision.

**Re-flagged: SUP-34's E1-pilot is still unactioned and is free** — and it can obviate everything
downstream. Run order set: **E1-pilot -> converged-loss reference -> materialise train split ->
power check.** Three of four are minutes.

**Next:** E1-pilot. Re-arm monitor. Now 10:42Z; SOFT 13:55Z, HARD 14:40Z (~4h).

---

## FOR JOEL (running)

15. **I stopped a training run from starting on a contaminated setup.** The planned check would
    have trained and evaluated on the same data. For its purpose — deciding whether a ten-hour
    experiment is worth running — that is fatal, because a model that simply memorises the answers
    would pass it, and we would then have spent the ten hours measuring memorisation while
    reporting it as something else. Fix is cheap (use the proper training split, minutes of data
    prep). Worth noting the shape: this is the same category as your original project's decode bug
    — a setup that runs cleanly, produces a plausible number, and answers a different question than
    the one asked.

## [2026-09-06T10:50Z] Supervisor pass 24 — E1-pilot: THE PROJECT'S FIRST SUBSTANTIVE FINDING

**Result:** full captions R-Prec-top3 **0.8013** -> first-action-truncated **0.6563**, **drop
0.1450**, ~9x the measured noise floor, independently corroborated by matching score
(2.985 -> 3.895). Zero training, minutes of compute, real motions, validated embedding space.

**Why it counts.** F1 was a verified decode bug; F3 was a moderate dispersion ratio (1.43x).
**This is the first large, clean, model-free number quantifying an original-project defect in the
field's own metric.** And the build pass made it trustworthy rather than merely favourable: the
full-caption arm landed at 0.8013 against E0b's independent 0.7969 — 0.0044 apart, inside the noise
floor — proving the pipeline is the *same instrument*, not a new one that happens to agree.

**Two sharpenings raised (Review 7), both minutes, neither a correction:**
- **SUP-38 — length-matched control.** Truncation cut captions 12.62 -> 8.01 words. Part of the
  0.145 may be *length* rather than *content selection*. Control: truncate to the same per-caption
  word count by a content-neutral rule. Same drop -> the effect is length. Smaller drop -> **the
  first-action rule is specifically destructive**, a sharper claim than the current one.
- **SUP-39 — the headline is diluted.** 44.8% of captions hit the first-sentence fallback, which
  changes little for single-sentence captions, so they contribute ~0 drop and pull the mean down.
  Conditional effect on captions **actually truncated** is ~`0.145/0.552 = 0.263` — **16x noise,
  not 9x**. Branch split already instrumented, so this is a re-slice not a re-run. Report both.

**SUP-40 — E1B's value has changed and should be re-decided.** The pre-registered rule said a large
drop justifies E1B's 7.5h. But F3's caption half is now *answered*, model-free. E1B's remaining
marginal value is narrower — whether the effect survives into generation. Proposed reframing E1B as
a reduced-scope propagation check with **the pilot as the headline finding**, which given the window
is likely the better use of what remains.

**Commended (SUP-41):** pre-registering scope and the asymmetry note; validating against E0b rather
than assuming pipeline identity; instrumenting the branch split. And pre-registering *how to react
to a surprise* — that a larger or opposite-signed E1B result "should be flagged rather than absorbed
quietly" — which is rarer than pre-registering the prediction.

**Next:** SUP-39 re-slice, SUP-38 control, E1B scope decision. Re-arm monitor.
Now 10:50Z; SOFT 13:55Z, HARD 14:40Z (~3h50m).

---

## FOR JOEL (running)

16. **The project has its first real finding, and it is about your original project specifically.**
    Your caption-truncation rule — taking only the first action clause — destroys a large amount of
    the text-to-motion matching signal: R-Precision drops 0.145 on identical real motions, about
    nine times the measurement noise. No model, no training, minutes of compute. And once we
    account for the ~45% of captions the rule left essentially untouched, the effect on captions it
    *actually* truncated is roughly **0.26 — about sixteen times noise**. That is a clean,
    defensible answer to one half of what went wrong, obtained without the ten-hour experiment
    originally planned to get it.

## [2026-09-06T10:58Z] Supervisor pass 25 — SUP-42: loss reference is a floor detector, and F6 is why

**Build pass landed all three fast checks (e8ee6f0):** E1-pilot (reviewed, Review 7), the
converged-loss reference, and the train-on-test kill + materializer extension.

**Converged-loss reference measured:** MDM's 475k-step checkpoint scores **mean 0.0563 / median
0.0522** on the same `training_losses` call, same batch distribution, same seeding. Untrained init
~1.1-1.4. **~21x range, ~3.06 nats.** That is what SUP-35 asked for and it is well done.

**SUP-42 (P1): it must not substitute for the R-Precision gate, and the reason is F6.** The original
project's model drove loss down smoothly under an objective that placed **no requirement on using
text at all**. A diffusion model reduces loss substantially by learning the *unconditional* motion
distribution while ignoring conditioning. **Loss falling is evidence of training, not of text
conditioning.** The two instruments answer different questions and both are required. **A model at
3,000 steps could show a healthy loss drop and still sit at chance on R-Precision — that is not
pathological, it is exactly F6's regime**, and if it happens the gate correctly fails.

Gave a log-space reporting formula so the trace is interpretable rather than eyeballed:
`log(L_init/L_obs)/log(L_init/L_conv)`. Loss 1.0 -> 6%, 0.8 -> 13%, 0.5 -> 29%, 0.3 -> 45%,
0.15 -> 68%, 0.08 -> 89%. Caveated that loss is not linear in sample quality and the fraction bounds
"still at init" only.

Also flagged the **4,000-sample train subset** (17% of HumanML3D's train split, ~24 epochs at 3,000
steps) as a small-data-regime limitation for "Does NOT establish"; and noted train-on-train /
eval-on-test now disposes of SUP-37's memorisation concern outright.

**The build pass's own account of the SUP-37 miss is the best line produced today, either side:**
*"that's the gap between naming a limitation and checking whether the limitation disables the
check."* Asked for it in `LANDMINES.md` as a standalone entry — every existing entry there is a
domain trap; **this is a review-discipline trap**, and the repository has now produced a documented
instance. The subtlety worth preserving: it had flagged the issue *in writing*, which is what made
both of us read the flag as the handling. A disclosed limitation is not a handled one, and
disclosure can manufacture false comfort precisely because it looks like rigour.

**Still open, both minutes:** SUP-39 re-slice (conditional effect ~0.26, 16x noise) and SUP-38
length-matched control. Both cheaper than the train-split stream already in flight.

**Next:** those two, then the real power check once train materialisation completes.
Now 10:58Z; SOFT 13:55Z, HARD 14:40Z (~3h40m).

## [2026-09-06T11:01Z] Supervisor pass 26 — power check launched; Stage 5 design recorded

**Power check is running properly.** 4,000 HumanML3D train sequences materialized (prefixed to
avoid colliding with the existing test files), `e1a_power_check.py` rewired to
`--train-split train --eval-split test`, launched 10:58Z, 3,000 steps, the 0.09375 gate exactly as
pre-registered. ~2.5-3h, so it lands ~13:30-14:00Z — inside the window, but not by much.

Worth noting how the build pass handled the HF streaming process hanging after its writes
completed: **it counted the materialized files on disk and confirmed completeness before killing
it**, so the kill was based on observed state rather than an assumption that "it looked done."
Small, and the right instinct.

**SUP-39 checked and returned to the build pass.** I opened
`e1_pilot_caption_truncation_record.json` myself to see whether the dilution re-slice could be done
from the artifact at zero cost. It cannot — the record has aggregates (44.8% fallback, 12,542
captions, 266 floored to 3 words) but **no per-caption branch labels**. It is a one-line
instrumentation change on the next run of that script, not a dedicated run. Told it so.

**SUP-43 filed: Stage 5 design proposal — demonstrate the finding, not the model.**
Written now because it is the piece most likely to be lost if this run ends before Stage 5 starts,
and D-20 makes it non-optional.

The reasoning: **our model will not be good** (0.63% of MDM's budget on a 17% subset), so a
demonstrator built to showcase our generations would be an interface doing work the model cannot.
But the project's contribution was never a better model — it is the diagnosis, and **the pilot's
0.145 caption-truncation effect is real, verified, and entirely invisible in a motion clip.**

So: type a caption, watch the original's own truncation rule chop it, see both conditions generate
side by side, with a nearest-neighbour retrieval pane and the measured numbers on screen.
Generation via MDM's checkpoint — real quality, MIT, already local, stated in the interface as a
cited component rather than implied as ours. **The property that makes it right under this budget:
it is robust to our research failing.** The diagnosis stands either way; a usable E1A model becomes
a fourth pane, not a redesign.

**Next:** await the gate result. Monitor on the record JSON + long fallback.
Now 11:01Z; SOFT 13:55Z, HARD 14:40Z.
