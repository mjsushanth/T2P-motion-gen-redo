# HANDOVER — supervising session, 2026-09-06

**Written at 13:50Z, before the 14:40Z hard stop. Read this first; the passes below are chronological
detail.** RUN_START 07:55Z. Successor: you have none of the conversation that produced this.

## What this session was

A supervising/reviewing session over a build session ("T2P Sonnet Coder C1"), which did all
execution. This session held gates, reviewed findings, and wrote `reviews/`, `docs/LANDMINES.md`,
`docs/DECISIONS.md` entries and the prompt files. **Territory rule that actually worked: append or
annotate, never silently delete another agent's entry.** `reviews/` and `guidance/` stay hard-walled.

## State at handover

| item | status |
|---|---|
| Stage 1 forensics | **DONE.** F1-F8 verified. |
| Stage 2 landscape / spec / positioning | **DONE**, reviewed, gate released. |
| D-03 harness gate | **UNRESOLVED.** E0a passed (evaluator sane); E0b did not reproduce MDM's FID. **Every downstream number is internally-comparable-only** (D-22). |
| E1-pilot | **DONE — the session's substantive output.** See below. |
| E1A power check | **PASSED.** R-Prec-top3 0.2969 vs chance 0.09375. E1 has power. |
| E1B | **not started.** Scope decided, see "next actions". |
| Stage 5 demonstrator | **designed, not built.** SUP-43. |

## The two findings worth carrying forward

**1. Caption truncation cost (E1-pilot).** The original project truncated captions to the first
action clause. Measured in the validated evaluator's retrieval space, no model, no training:
**R-Precision-top3 drops 0.145-0.157 corpus-wide, ~0.27 conditional on the rule firing.**
Decomposed across three controls: **~93% is volume (how much text is removed), ~7% is position
(which part)** — the position component resolved at 5.9 sigma only after averaging over 8 random
placement draws. **The original's "first-action segmentation," presented as a contribution, was
neither clever nor uniquely harmful — one of many ways to discard 35% of the words.**

**2. A healthy loss curve on a bad model (E1A).** Final training loss ~0.19 against a converged
reference of 0.0563 and init ~1.2 — **~60% of the way to convergence at 0.63% of MDM's step budget
— while FID was 7.209, 13x worse than MDM's 0.544.** This is F6's failure mode reproduced
deliberately: **loss measures training, not quality.** The pair is the project's cleanest teaching
artifact.

## Next actions, in order

0. **FIRST, when E1B lands: run SUP-20260906-49's free control.** Score E1A's *cached* generations
   against **truncated** captions. E1A was evaluated on full captions and E1B will be evaluated on
   truncated ones, so the raw A-vs-B gap confounds *model degradation* with *captions being harder
   to retrieve against* — an effect the pilot measured at 0.145 on real motions. Without the middle
   row, E1B's headline is uninterpretable. Text re-encoding only, minutes.
1. **E1B is already training** (launched ~13:40Z, `scripts/e1_train_arm.py --arm b`). Arm design
   verified correct: truncation applied to **both** train and generation captions (faithful to the
   original, which saw truncated text end-to-end), ground-truth reference left on full captions so
   the instrument stays constant across arms. **Let it finish.**
2. **E1A second seed (~2.65h).** *Prerequisite for any A-vs-B statement.* **With one seed of each
   you have a gap and no spread to judge it against — do not let an A-vs-B conclusion enter
   `EXPERIMENT_LOG.md` before this exists.** A second seed of B is worth more than a third of A. The decision rule is "B worse than A
   by more than the seed spread," and there is currently **no spread measured at all**.
3. **E1B: the original's rule, 2 seeds (~5.3h).** Content-neutral length arm dropped as redundant —
   the pilot showed rule and generic truncation are indistinguishable, so one arm yields both claims.
4. ~~Fix the `diversity_times` off-by-one~~ — **DONE** (ea34bc9). One line. Has now fired twice and forced hand-assembled
   records twice.
5. **Design E1C (frame selection).** F3's other half, entirely unmeasured, no cheap analogue. **The
   most valuable remaining experiment.**
6. **Stage 5 demonstrator** (SUP-43) — show the *finding*, not the model.

## Standing rules a successor will otherwise violate

- **D-19/D-19a:** downloads, environments, installs, code, notebooks all pre-authorised. Ask only at
  system-breaking scale. **Never hold the build session at a technical gate.**
- **D-20:** zero commercial intent. Non-commercial licences do **not** constrain this project.
  **Numbers are not the deliverable** — Stage 5 is not optional.
- **D-21:** sole authorship. No institutional identifiers, no third-party names, no assistant
  attributions in tracked files. Archive path lives in gitignored `.archive_path`, referenced as
  `<ARCHIVE>/`.
- **D-22/D-03:** label every number internally-comparable-only until the harness gate resolves.
- **Two measured noise floors now exist** (0.016 from E0b batching; 0.0117 from the restricted
  subset). **State which one you are using.** SUP-46 exists because one was silently reused.

## What I got wrong, recorded because it is the useful part

Six supervisor findings withdrawn or corrected this run: SUP-16 (FID arithmetic overconfident),
SUP-17 (driver-misconfiguration hypothesis, disproved by measurement), SUP-28 (fixed reference would
not stabilise FID), **E1's original specification (a tautology — comparing a static pose to motion,
caught by criteria I had written an hour earlier for judging someone else)**, SUP-02's scope
(regime-dependent, not universal), and the "re-slice not a re-run" advice (linear undilution was
30x off).

**And front-loading: I called it refuted, then corrected to not-resolvable, then it was confirmed at
5.9 sigma. Two wrong readings in opposite directions before the right one.**

Against that, the build session made two errors I caught: a D-03 gate misstatement, and reading two
runs as independent when generation was seeded deterministically.

**That ratio is the arrangement working.** Measurements decided, not seniority. A successor should
expect to be wrong at a similar rate and should write it down at the same rate.

---

# Supervisor log — chronological passes

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

## [2026-09-06T11:08Z] Supervisor pass 27 — SUP-44: the length-control conclusion outruns its control

**The build pass ran a control that deflated its own three-hour-old headline finding**, called it
*"less flattering to the original narrative than I expected going in,"* and reported it as measured.
Recording that explicitly: it is the single hardest thing on the whole discipline list, and most of
the failure modes in `LANDMINES.md` are what happens when it is not done.

**SUP-39 confirmed by real measurement:** conditional drop **0.2724** on the 2,599 keys where the
rule fires, against my estimate of 0.263.

**SUP-44 (P1): the conclusion "length-driven, not rule-specific" is not yet supported.**

| rule | R-Prec-top3 | keeps |
|---|---|---|
| full (~12.6 w) | 0.8013 | everything |
| original's first-action clause (~8.0 w) | 0.6552 | **a prefix** |
| naive first-N-words (~8 w) | 0.6468 | **a prefix** |

**Both truncation arms are prefixes.** Nothing varies position. So:
- **Established (a genuine deflation):** among prefix-preserving rules of the same length, the cut
  point barely matters — the original's "first-action segmentation" is no smarter than a blind
  prefix.
- **Not established:** that position/content is irrelevant. That claim needs a control keeping a
  *different part* of the caption.

**The missing arm — a random contiguous N-word window — was proposed in SUP-38 and not run, and it
is the interesting one.** ~0.65 means genuinely length-driven and the conclusion stands with
evidence. Materially worse (~0.55) means **keeping the prefix is doing real work**, and the finding
inverts into *HumanML3D captions front-load their motion-relevant content* — a statement about the
**dataset** rather than about one project's truncation rule, and the most generalisable thing this
pilot could produce. Same cost as the control just run.

Told it to phrase the finding as "among prefix rules the cut point does not matter" until that arm
exists.

**On E1B scope:** agreed with its narrowing (testing the original's specific rule in generation is
now the weaker question), **with the addition that the random-window result should feed the arm
design** — if position matters, the informative generation arm is position-varied, not another
length-varied one.

**Next:** power check (~13:30-14:00Z) and, if it fits, the random-window arm. Otherwise both go to
the handover. Now 11:08Z; SOFT 13:55Z, HARD 14:40Z.

## [2026-09-06T11:12Z] Supervisor pass 28 — E1-pilot closed. My front-loading hypothesis refuted.

**Random-window control: 0.6470**, against prefix controls 0.6468 (length-matched) and 0.6552 (the
original's rule). **Position effect 0.00022 — a clean null.** My alternative hypothesis (HumanML3D
captions front-load motion content, so the original's rule accidentally preserved the useful part)
is directly tested and **refuted**. Withdrawn. **Sixth supervisor hypothesis to die by measurement
this run.**

The build pass corrected the EXPERIMENT_LOG entry in place rather than leaving the overstated
"length-driven" version beside the new evidence, and flagged its own scope limit unprompted:
**42.7% of keys had no room for a different window**, so they contribute zero position effect
mechanically rather than empirically.

**SUP-45 (P2):** same dilution shape as SUP-39 — report the conditional on the 57.3% where the
window could actually vary. Undiluted arithmetic is `0.00022/0.573 ~ 0.00038`, **still a null**, so
the conclusion is unaffected; but stated rather than derivable matters, or a reader can ask whether
the null is real or an artifact of half the corpus being unable to express the manipulation.
Re-slice, not re-run.

**The pilot closed better than it opened.** Final form, three mutually consistent controls:
> **Caption truncation costs retrievable text-motion alignment roughly in proportion to how much
> text is removed, and essentially independent of which part is removed.** Corpus-wide 0.145-0.157;
> conditional on the rule firing ~0.27.

**And it is a harsher verdict on the original project than the flattering version would have been.**
"Their heuristic was bad" would imply the heuristic did something. What is now shown: the
first-action segmentation — presented as a contribution, with a POS-tagging algorithm behind it —
**was neither clever nor uniquely harmful. It was one of many ways to discard 35% of the words, and
the entire cost came from the discarding.** Told the build pass to write it in those terms.

**Next:** power check, ~13:30-14:00Z. Monitor biioddqhd armed. Now 11:12Z; HARD 14:40Z.

## [2026-09-06T11:16Z] Supervisor pass 29 — SUP-46: I over-claimed a refutation; the sign was misread both ways

**Two errors this pass, one mine on method and one mine on conclusion. The build pass caught the
first; I caught the second.**

**1. My "re-slice, not a re-run" advice was wrong.** I offered `0.00022/0.573 ~ 0.00038` as a
linear undilution standing in for the real restricted measurement. The build pass ran it properly:
**-0.0117, ~30x my estimate.** Its diagnosis is right and it is a correction to my *methodology*,
not my arithmetic — **linear undilution assumes the excluded stratum differs only by contributing
zero**, but here the strata also differ in caption length, so the assumption fails. Recorded as a
supervisor error.

**2. But the restricted result is directionally CONSISTENT with front-loading, not opposite — and
I had already called my hypothesis "refuted" one pass earlier. That was over-claiming.**

| arm | R-Prec-top3 (2,663 keys) |
|---|---|
| length-matched prefix | **0.5456** |
| random window | 0.5339 |
| gap | **+0.0117 favouring the prefix** |

Front-loading predicts *prefix scores higher* — keeping the front preserves the informative part.
Prefix did score higher. The build pass read that as contradicting the hypothesis; it does not. The
magnitude is below the ~0.016 floor, so the honest statement is **"no position effect resolvable at
this precision; the point estimate leans the way the hypothesis predicted"** — a claim about
**power**, not about absence.

Retracting my own pass-28 language ("refuted", "sixth hypothesis to die by measurement"). **Claiming
refutation from an underpowered null is overclaiming in the opposite direction from the one we have
guarded against all day, and it is the easier one to miss because it looks like scepticism.**

**Also flagged:** 0.016 is E0b's ground-truth batching floor at n=128, borrowed here for a different
n and a different comparison whose absolute scores are much lower (0.54 vs 0.65 — consistent, these
being the longer captions that lose most under an 8-word cap). **A floor measured on one
configuration should not be silently reused for another.** Get one for this comparison or label the
borrowed figure approximate.

**Headline unaffected.** "Truncation cost is proportional to how much text is removed" stands on the
full-set result across three controls. Only the secondary position claim moves, from "no effect" to
"no detectable effect."

**Next:** power check (~13:30-14:00Z), the last substantive item likely to land in-window. Monitor
biioddqhd armed. Now 11:16Z; SOFT 13:55Z, HARD 14:40Z.

## [2026-09-06T11:28Z] Supervisor pass 30 — SUP-47: the two arms are not measured with equal precision

Build pass accepted both SUP-46 corrections in place (sign reading, and measuring the noise floor
locally rather than borrowing E0b's 0.016). **The local measurement produced something better than
a floor:**

| arm | across-seed movement |
|---|---|
| length-matched prefix | 0.0008 |
| random window | **0.0124** |
| the effect under test | 0.0117 |

**SUP-47 (P2): that asymmetry is structural.** The prefix is deterministic given a target length —
reseeding only reshuffles retrieval batches. **The random window re-draws the placement itself, so
each seed is a different *treatment*, not a different sample of one.** Its 0.0124 is therefore
mostly treatment variance. With a single draw, "random ~ prefix" cannot separate *"position does not
matter"* from *"these particular placements happened to score about as well as the prefix"* — one
draw from a distribution whose spread equals the effect.

**Cheap closeout proposed, and the build pass is idle-waiting on the power check:** average the
random arm over 5-10 placement draws, text re-encoding only, minutes. Cuts treatment variance by
~sqrt(N) and upgrades the claim from "one draw showed no resolvable difference" to "the expected
effect of placement is under X."

**Pattern noted for `LANDMINES.md` §14's neighbourhood:** *when one arm of a comparison is itself
stochastic, its across-seed spread is treatment variance and must be averaged down before the arms
are comparable.* **This is the second time the project has met it** — E0a flagged MDM's
`repeat_time` averaging as the likely cause of its own residual gap. Same correction, different
dress.

Priority set below the power check, above idle; explicitly told not to delay the gate result.

**Next:** power check (~13:30-14:00Z), then handover. Now 11:28Z; SOFT 13:55Z, HARD 14:40Z.

## [2026-09-06T11:33Z] Supervisor pass 31 — FRONT-LOADING CONFIRMED. My two prior readings were both wrong, in opposite directions.

**The averaging reversed the conclusion rather than tightening it, and the build pass flagged it
loudly as instructed.**

| quantity | value |
|---|---|
| random-window, 8 placement draws | mean **0.5279**, std 0.0082, SEM 0.0029 |
| length-matched prefix (stable, 0.0008) | **0.5456** |
| gap | **0.0177** |
| gap / combined SE (incl. prefix's own) | **5.9 sigma** |

Verified independently — arithmetic holds; including the prefix arm's uncertainty moves 6.1 to 5.9
and nothing else. **Resolved effect.**

**SUP-48 filed. Two things for the writeup:**
- **Not a forking-paths artifact**, and the reasons must be stated rather than left to reconstruct:
  every refinement was proposed on **a priori methodological grounds before its result was seen**,
  and **the hypothesis direction was on the record in advance** (SUP-38, again SUP-44) before any of
  the three controls ran. Remaining caveat, to be stated: the final comparison form was not itself
  pre-registered.
- **The decomposition is the useful form:** volume ~93% of the conditional truncation cost (0.27),
  **position ~7% (0.0177)** — real, resolved, secondary. Better than either the bare null or the
  rule-specific story the pilot opened with.

**My own sequence, recorded deliberately:** pass 28 **"refuted"** (overclaiming from an underpowered
null) -> pass 29 **"not resolvable"** (correct on the evidence then) -> pass 30 proposed the
averaging -> pass 31 **confirmed**. **Both of my first two readings were wrong, in opposite
directions.** The process produced the answer, not either party's judgement, and that is what makes
the final number trustworthy. It belongs in the writeup as-is.

`LANDMINES.md` §17 (stochastic arm's seed controls the treatment; deterministic arm's controls
incidental noise) now ties E0a's flagged `repeat_time` suspicion to a confirmed instance — two
separate observations collapsed into one transferable rule.

**E1-pilot closed. It is this session's substantive output:** two genuine findings, three controls,
every correction on the record, and the original's "first-action segmentation" shown to be neither
clever nor uniquely harmful.

**Next:** power check, the last open item. Now 11:33Z; SOFT 13:55Z, HARD 14:40Z.

---

## FOR JOEL (running)

17. **Second real finding, and it needed four rounds to get right.** Beyond the volume effect: the
    *position* of the text you keep also matters, slightly — HumanML3D captions carry a little more
    motion-relevant signal at the front. Roughly 93% of the truncation cost is how much you remove,
    ~7% is which part. Worth knowing how it was reached: I first called this refuted, then
    downgraded to "can't tell," then proposed the measurement that resolved it — **two wrong calls
    in opposite directions before the right one.** Every step is in the record. That sequence is a
    better advertisement for the method than a clean result would have been, and it is the sort of
    thing worth being able to walk an interviewer through.

## [2026-09-06T13:40Z] Supervisor pass 32 — E1A POWER CHECK: GATE PASSES

| quantity | value |
|---|---|
| chance (3/32) | 0.09375 |
| **E1A R-Precision-top3** | **0.2969** |
| margin | **0.2031 ~ 13x the 0.016 noise floor** |
| vs chance | 3.17x |

**Pre-registered criterion met by a wide margin. E1 has power; SUP-33's floor-effect concern is
retired. Proceed to E1B.**

**Ground truth reproduced at 0.7950 vs the 20-replication reference 0.7977 — 0.0027 apart. Third
independent confirmation of the evaluator, and the most valuable**, because this run used a freshly
materialised train split and a new data path. The instrument survived a change that could have
broken it.

**The result's most instructive feature is the loss/quality divergence — SUP-42 confirmed
empirically rather than argued.** Final loss ~0.19 against converged 0.0563 and init ~1.2 = **~60%
of the way to MDM's converged loss at 0.63% of its step budget**. **And FID 7.209, 13x worse than
MDM's 0.544.** A loss curve 60% converged sitting on a distributionally poor model. **Had we gated
on the loss trace — as the original project effectively did — this model would have looked far
healthier than it is.** Told the build pass to put both numbers side by side; the pair teaches more
than either alone.

Scoping fact for the writeup, with the internally-comparable-only label: **E1A reaches 49% of MDM's
published R-Precision at 0.63% of its training budget.**

**Housekeeping raised:** the `diversity_times` off-by-one finally bit — crashed the run *after*
R-Precision and FID were computed, so the gate is intact, but the script died before its clean exit.
Deferred twice, now cost something; one-line fix, motions cached. And confirm the 3,397-byte record
JSON is a complete write rather than partial, disclosing any after-the-fact assembly as E0b did.

**Next and last:** write the handover. Told the build pass my window closes 14:40Z and that I will
spend it on the handover rather than another review cycle.
Now 13:40Z; SOFT 13:55Z, HARD 14:40Z.

---

## FOR JOEL (running)

18. **The first model this project trained works, and the gate it was built to pass, passed.**
    Trained from scratch on a corrected pipeline, it matches text to motion at roughly three times
    chance — about half of what the field's reference model achieves, on 0.63% of that model's
    training budget. That was the question the whole day was building toward: is a laptop-scale
    rebuild capable of showing the effect we want to measure? Yes.
19. **And it produced the cleanest teaching example of the project's central lesson.** Its training
    loss got ~60% of the way to a fully-converged reference, while its actual output quality
    remained poor — 13x worse than the reference model on the distributional metric. **A healthy
    loss curve on a bad model.** That is precisely the trap your original project fell into, now
    reproduced deliberately, measured, and documented as a demonstration rather than an accident.

## [2026-09-06T13:45Z] Supervisor pass 33 — handover written; orientation docs made true

**Handover written at the top of this file.** A successor reads it first; the chronological passes
are detail. Covers state, the two findings, next actions in order, the standing rules a successor
will otherwise violate, and the six supervisor findings withdrawn this run.

**Then fixed the stalest thing in the repository.** `docs/00_START_HERE.md` §9 still read *"nothing
has been measured in this repository. There is no result to report, positive or negative."* That is
the first substantive section a new reader reaches, and it had been false for hours. Rewritten with
both findings, the unresolved-gate caveat, and the internally-comparable-only label. `README.md`'s
status table likewise still showed Stages 3-5 as "not started."

Both now carry the two findings, the D-03 caveat, and — deliberately — a paragraph on the process:
six supervisor findings withdrawn by measurement in a day, including the headline experiment's
original specification being a tautology caught before it burned ~10h, and the front-loading result
going refuted -> unresolvable -> confirmed. **Kept because the sequence is what makes the surviving
numbers worth believing**, not despite being unflattering.

**Reached SOFT STOP (13:55Z) with this pass. Taking no new work items.** Remaining time to HARD STOP
14:40Z is for anything the build session sends back and a clean close.

## [2026-09-06T13:47Z] Supervisor pass 34 — SOFT STOP. Final review; loop closing at hard stop.

**E1B launched before my sequencing message reached the build session.** Not a problem — both arms
are needed regardless, so the order costs nothing. **Verified the arm design independently and it is
right:** truncation applied to *both* the training captions and the generation-conditioning captions
(faithful — the original's model saw truncated text end-to-end, not at one stage), with the
ground-truth reference left on full captions so the instrument stays constant across arms. It got
both of those without being told.

**Sent the one constraint that actually binds:** do not let an A-vs-B conclusion enter
`EXPERIMENT_LOG.md` before E1A seed 2 exists. One seed of each gives a gap with no spread to judge
it against. Handover's next-actions list updated to lead with this.

**Session closed on the supervising side.** Handover at the top of this file; `docs/00_START_HERE.md`
§9 and `README.md` now carry both findings and the comparability caveat instead of claiming nothing
had been measured.

**Final tally, recorded because it is the point rather than an embarrassment:** six supervisor
findings withdrawn or corrected by measurement in one day — SUP-16, SUP-17, SUP-28, SUP-02's scope,
the "re-slice not a re-run" advice, and E1's original specification (a tautology, caught by criteria
I had written an hour earlier to judge someone else's work). Front-loading went refuted ->
unresolvable -> confirmed at 5.9 sigma. Against two build-session errors caught: the D-03 gate
misstatement and the deterministic-seed independence misread.

**Neither side's judgement produced the surviving numbers. The exchange did.** A successor should
expect to be wrong at a similar rate, and should write it down at the same rate.

## [2026-09-06T13:47Z] Supervisor pass 35 — FINAL. SUP-49 filed; handover updated; loop closing.

**Answered the build session's last design question: the arm design is right, I would not scope it
differently.** But answering it surfaced a confound worth catching before E1B's entry is written.

**SUP-49 (P1).** E1A is evaluated on **full** captions; E1B will be evaluated on **truncated** ones.
So the raw A-vs-B gap confounds **model degradation** with **captions being intrinsically harder to
retrieve against** — an effect the pilot already measured at **0.145 on real motions**. Undecomposed,
E1B could show a pilot-sized gap and be unable to say whether the model degraded at all.

**Free control:** score E1A's *cached* generations against truncated captions. Same motions, same
model, only the retrieval text changes — text re-encoding, minutes. Three-way decomposition, and
**E1B minus the middle row is the model effect.** Critically, the pilot's 0.145 cannot simply be
subtracted: it was measured at R-Precision ~0.80 on real motions, while E1A operates at 0.297, near
the 0.094 floor. **It has to be measured in this model's own range.**

Same failure shape as the pilot's corpus-wide number before SUP-39's re-slice — which the build
session fixed itself, so it already knows the pattern.

**Handover updated: SUP-49 is now item 0 in next actions.** `diversity_times` marked DONE (ea34bc9);
record JSON confirmed complete by parse rather than byte count.

**Loop closing.** Soft stop passed at 13:55Z, hard stop 14:40Z. E1B lands after my window; the build
session carries it forward, as I carried the power check forward for it.

**Closing note for whoever reads this next.** The value of this arrangement was not that the
supervising session was right. **Six of its findings died by measurement in one day**, including the
headline experiment's original specification. The value was that two sessions with different context
audited each other against fetched sources and cached artifacts, and wrote down every reversal.
**Expect to be wrong at a similar rate. Write it down at the same rate.**

---

# RUN WINDOW 2 — set 2026-09-06T15:46:45Z

**RUN_START:** 2026-09-06T15:46:45Z
**SOFT STOP (take no new work item):** 2026-09-07T01:46:45Z
**HARD STOP (stop mid-item, write handover):** 2026-09-07T03:16:45Z

Author authorised 10-12 hours. If a usage limit interrupts, resume on the next firing toward these
same deadlines. **Re-read this block every firing and compare against `date -u`.**

## D-26 — the E1 ladder stops after E1B + seed 2. Then Stage 5. · JUDGEMENT (review pass)

The author pushed back on continuing caption-truncation archaeology after the task reframe, and the
push is substantially right. The reframe eliminated F3's *frame-selection* half by construction;
only the *caption* half survived, and **the pilot already answered it model-free, in minutes.**

**So the ladder stops when the A-vs-B comparison is interpretable, not when it is exhausted:**
1. E1B completes (running).
2. **SUP-49 decomposition control** — score E1A's cached generations against truncated captions.
   Free, and without it the A-vs-B gap confounds model degradation with caption retrievability.
3. **E1A seed 2** — required; one seed of each gives a gap with no spread to judge it against.
4. **Write E1 up. Then STOP the ladder and start Stage 5.**

**Deferred, deliberately: E1C, third seeds, the n~1000 FID rerun, and E0b closure.** All are
legitimate and none is worth the wall-clock against a demonstrator the author has asked for twice.
**Would reverse if:** E1B's result is anomalous in a way that a further arm would explain, or the
author says otherwise.

**Stage 5 is the priority once E1 is written up.** Design is SUP-43: demonstrate the *finding*, not
the model. Judge it against the pre-registered Stage 5 criteria already in `reviews/REVIEW_QUEUE.md`
— especially the retrieval-baseline-visible-in-the-interface one, which I said I would hold hardest.


## [2026-09-06T16:05Z] Supervisor pass 36 — RUN WINDOW 2 opens. SUP-50: the scrub plan defeats itself.

**Window checked:** now 16:05Z, RUN_START 15:46Z, SOFT 01:46Z, HARD 03:16Z. ~9h40m to soft stop.

**E1B is mid-generation** — 2 of ~5 batches at ~9.5 min each, so ~20-30 min out. Nothing to review
from it yet. Reviewed the five commits that landed since my last pass instead.

**SUP-50 (P1) — verified two things independently:**

1. **The history rewrite has NOT run.** `89d31d3` survives at its original hash; a `filter-repo`
   rewrite changes every downstream hash. Authorisation is recorded (`8b781cb`), execution pending.
   Sequencing it after E1B is fine — flagged only so the record is not mistaken for the deed.
2. **The trap.** `LEDGER.md` ~1608-1632 now holds the replacement rules verbatim — the course-code
   mapping and the rest. **The document describing the scrub is now the largest concentration of the
   strings the scrub exists to remove**, tracked, inside the repo the rewrite will run over.
   - Rewrite everything -> the entry becomes `the course==>the course`; **the record of the
     operation is destroyed.**
   - Exclude `LEDGER.md` -> the strings survive and the scrub fails its purpose.

   **You cannot document a literal-string scrub inside the repository being scrubbed, using the
   literals.** Fix given: keep the expressions file gitignored (same pattern as `.archive_path`),
   and describe the operation in prose without quoting the literals. **Verify two properties after,
   not one** — every `git log -S` empty *and* the ledger entry still legible. Right now both checks
   fail solely because of this entry, so a naive rewrite could pass the absence check while silently
   mangling the record.

**Also verified, no action:** `f74a3a9` builds SUP-49's caption-retrievability control *into* the
E1A seed-2 run rather than bolting it on afterwards — better than what I asked for, the
decomposition now arrives with the seed instead of needing a third pass. `90b4917` corrects the
18x-vs-13x noise-floor discrepancy.

**Monitor blh8kjpfk armed** on E1B completion.
**Next:** E1B review, then the decomposition + seed 2, then per D-26 stop the ladder and start
Stage 5.

## [2026-09-06T16:12Z] Supervisor pass 37 — I committed the exact error SUP-50 describes. Fixed.

**The build session caught me, and it was right.** SUP-50 warned that a literal-string scrub cannot
be documented using the literals — **and quoted the literals verbatim while doing so**, four times in
`reviews/REVIEW_QUEUE.md` and once in this log, into a public repository, in the same paragraph as
the warning. The rewrite it warned about had *already run*, so my finding re-leaked into current
HEAD everything the rewrite had just removed.

**Fixed:** literals replaced with structural placeholders; the real expressions stay in the
gitignored file. Verified after — `git grep` returns **0 tracked files**, working-tree markdown sweep
clean. **Uncommitted; the build session pushes it, since I do not commit.**

**Method note, and the build session reasoned it out first.** I edited **in place** rather than
appending a correction, breaking this project's standing append-or-annotate rule. Its justification
is the right one: **preserving the original text would preserve exactly the data the correction
exists to remove.** When the content *is* the defect, annotation cannot fix it.

**New general rule, arrived at independently by both sessions within an hour — the strongest kind of
evidence a rule is real:** *a redaction cannot be documented by quotation. Describe the shape of what
was removed, keep the literals outside version control, and accept that this one class of correction
must overwrite rather than annotate.* Appended to SUP-50; suggested to the build session for
`LANDMINES.md` next to §16's review-discipline entry, its file and its call.

**Residual flagged, not actionable:** anyone who cloned before the force-push holds the old objects,
and GitHub may retain unreferenced objects server-side for a period. Close to theoretical for a repo
this new and quiet, but it should be a *known* limitation rather than an assumed-complete cleanup.

**Seventh supervisor error of the project**, and the first that was actively harmful rather than
merely wrong — the others cost compute or precision; this one re-published the data it was written to
protect.

**Monitor re-arming on E1B.** Now 16:12Z; SOFT 01:46Z, HARD 03:16Z.

## [2026-09-06T16:25Z] Supervisor pass 38 — E1B: 0.80 sigma. Ladder stops a rung early. Stage 5 begins.

**E1B landed.** R-Prec-top3 **0.34375 (44/128)** vs E1A's **0.2969 (38/128)** — nominally *higher*,
the opposite of the pre-registered direction. FID went the other way (7.209 -> 8.340) but is
secondary and unreliable at this n.

**SUP-51 (P0): the gap is six samples, and it is 0.80 sigma.**
```
E1A 0.2969 +/- 0.0404 | E1B 0.3438 +/- 0.0420 | gap +0.0469 +/- 0.0583 -> 0.80 sigma
```
Binomial noise **alone**, before training-seed variance. **No directional statement is available.**
Told the build session plainly: writing this up as "truncation helps" would be this project's
founding error in a new costume. And it is *not supported*, not *refuted* — SUP-46's distinction,
applied a second time.

**SUP-52 (P1): a second seed cannot rescue it, so do not spend 2.65h on one.** D-26 required seed 2
so the gap would have a spread to be judged against; **that is superseded by a cheaper, stronger
argument — binomial noise (0.058) already exceeds the gap (0.047)**, and seed variance can only
enlarge it. Cost to resolve at 3 sigma: **~1,780 samples/arm ~ 9.0h of generation per arm per seed.**
Not affordable. Redirected to SUP-49's free caption-retrievability control instead (already built
into the seed-2 script, minutes, cached generations).

**SUP-53 (P1): this is a legitimate result and it was pre-registered as one.** SUP-33, before any of
it ran: *"'E1 is not affordable at a budget that gives it power on this hardware' — itself a
legitimate, honestly-labeled finding."* **That branch is now taken.** The result is a **bound on what
this setup can detect, not a bound on the effect.**

**Not a failed experiment.** E1A passed its gate at 3.2x chance; ground truth reproduced 0.7950 vs
0.7977; costs matched projections to within minutes. **Everything worked except the affordability of
the question** — and the caption question itself is already answered by the pilot, model-free, at
9-17x its noise floor. E1B only ever tested propagation.

**D-26 amended: the ladder stops one rung earlier than planned.** SUP-49's free control -> write E1
up including the power result -> **Stage 5.** E1C, seed 2, third seeds: legitimate, unaffordable,
documented as such.

**Stage 5 is now the priority with ~9h of window left.** Design SUP-43, criteria already
pre-registered. Holding hardest on: **the retrieval baseline visible in the interface, not just
described in the report.**

Now 16:25Z; SOFT 01:46Z, HARD 03:16Z.

---

## FOR JOEL (running)

20. **The generation experiment came back as noise, and that is a real answer.** The two arms
    differed by six samples out of 128 — 0.8 sigma, statistically indistinguishable. Resolving an
    effect that small would need about nine hours of generation per arm, which this laptop cannot
    afford. So the honest finding is a **bound on what this setup can detect**, not a claim about
    truncation. Worth noting the pipeline itself worked perfectly throughout — the model trained,
    the gate passed, the costs matched predictions. The only thing that failed was the affordability
    of the question, and that was pre-registered this morning as a legitimate outcome.
21. **Which means the ladder stops and the demonstrator starts**, earlier than planned and closer to
    what you asked for twice. The caption finding was already answered by the cheap experiment; the
    expensive one was only ever testing whether it propagated.

## [2026-09-06T16:38Z] Supervisor pass 39 — SUP-54: partially reversing SUP-52; a mechanism pre-registered

**Build session launched E1A seed 2 before Review 9 arrived. On reflection its version is better
than my instruction and I let it stand.** I said run SUP-49's control standalone and skip seed 2; it
had already folded the control *into* the seed-2 script, so one run delivers both, and the CPU was
idle the moment E1B exited. Killing it would forfeit the control to save nothing.

**SUP-52's arithmetic stands, its conclusion narrows:** binomial noise (0.058) still exceeds the gap
(0.047), so seed 2 **cannot resolve A-vs-B** — but it **decomposes** it, which is the useful part:
- control ~0.34 -> whole gap is caption-side, models indistinguishable
- control ~0.30 -> caption side neutral, gap is model-side (still 0.80 sigma)
- control <0.30 -> truncation hurts retrieval *and* E1B beat it anyway -> genuinely odd
Plus a **measured** seed spread, which beats my theoretical binomial bound in the writeup.

**Pre-registered a mechanism for the build session's own live question, before the number lands.**
It asked why the control landing near 0.34 would contradict the pilot's 0.145 truncation cost.
**Proposed resolution: caption specificity is an asset only when the motion is good enough to match
it.** At R-Prec 0.30 the generated motion is vague; a short generic caption may match a vague motion
*better* than a long specific one, because the specific caption's extra content has nothing to
attach to and acts as embedding noise. If so the two results are not in conflict — **truncation
destroys information that helps at high generator quality (real motion, 0.80) and is
neutral-to-helpful at low quality (our model, 0.30).** That is a claim about the *interaction between
conditioning specificity and generator quality*, more interesting than either arm, and testable at
higher quality later. **Registered as a prediction now so it cannot become a post-hoc story.**

**Corrected the build session's self-assessment downward.** It flagged E1B as having run without
pre-registration. **Separate two things:** its hypothesis and success criterion *were* registered in
`REBUILD_SPEC.md` §6, written before the run; what was missing is the per-run record table. **That is
a documentation-consistency gap, not a pre-registration failure**, and grading it as the latter
over-penalises. Told it to record which it was.

**Eighth supervisor correction**, and the second of my own instructions overturned by the build
session having already done something better.

**Next:** decomposition + seed spread (~2.5h), E1 writeup, then Stage 5 with ~10h of window.
Now 16:38Z; SOFT 01:46Z, HARD 03:16Z.

## [2026-09-06T16:48Z] Supervisor pass 40 — SUP-55: pre-registration was missing a power calculation, and that was my miss

**E1 formally closed by the build session** (12b628c): my arithmetic re-derived independently and
matched exactly (z=0.805, ~1,779 samples/arm/seed for 3 sigma), D-26 written, E1B's entry moved
PARTIAL -> RESOLVED, E1C designed-not-built, extra seeds documented as unaffordable rather than
silently dropped.

**SUP-55 (P1) — the build session's process note names a specific miss of mine.** It observed that
E1B's pre-registration carried a hypothesis and a criterion but **no power calculation**, and that
this is why the ceiling went unnoticed until 2.5h had been spent.

**I gated on the wrong power question.** SUP-33 asked *"can the model learn anything at all?"* and
answered it correctly with the chance gate. **It never asked "given the effect we expect, what n
resolves it?"** Only the second question bears on whether a comparison is worth running.

**The calculation that was available beforehand and would have stopped the run:** at R-Prec ~0.30,
n=128, SE of the gap is **0.0573**, so the **minimum detectable effect at 3 sigma is 0.172**. Against
the pilot's own predicted effects — 0.145 unattenuated gives **2.5 sigma**; a plausibly attenuated
0.05 gives **0.9 sigma**. **Even the full undiminished pilot effect could not have reached 3 sigma at
that n**, and every reason we had said it would be attenuated. **One minute of arithmetic, from
numbers already in hand, before ~5 CPU-hours across two arms.**

**Standing rule proposed: pre-registration is hypothesis + criterion + minimum detectable effect.**
This project ran seven rungs on two of three. Proposed a template block carrying expected effect,
SE at planned n, MDE at 3 sigma, and a powered/underpowered verdict — **with an explicit "why we run
it anyway" line**, because underpowered runs are sometimes right and running one *unknowingly* is
what turns noise into a finding. Proposed for `EXPERIMENT_LOG.md`'s template (build session's file)
and `LANDMINES.md` beside §16/§17 as a review-discipline trap.

**Second standing rule this project has gained from the build session interrogating its own process
rather than mine** — the first being "naming a limitation versus checking whether it disables the
check." **Both are more transferable than any domain finding either of us produced.**

**Next: Stage 5.** ~9h of window. Design SUP-43, criteria pre-registered.
Now 16:48Z; SOFT 01:46Z, HARD 03:16Z.

## [2026-09-06T16:32Z] Supervisor pass 41 — Stage 5 started; reviewed the modules before the interface exists

**`demo/` created:** `truncate.py`, `retrieval.py`, `generate_wrapper.py`, `render_real_motion.py`.
**Reviewed early on purpose** — steering a design is free now and expensive after an interface is
built on it.

**Shape is right:** wraps MDM's own `generate.py` rather than reimplementing; uses the released
checkpoint so quality is real; docstring states plainly this is *not our model* and the demo is
about conditioning; renders the retrieval baseline as an actual moving skeleton, not a caption;
reuses F1's corrected decode. All four match SUP-43.

**SUP-56 (P1) — as built it fails its own 30-second criterion.** Generation is CPU-only, 1000
timesteps, several minutes per call. The pre-registered criterion is comprehension in thirty seconds
without narration; **a five-minute wait per interaction fails the one criterion I said I would hold
hardest.** Fix: **ship pre-generated examples** so the demo opens on a working side-by-side, *then*
offer free text with an honest progress state. Both criteria are satisfiable together — instant
comprehension from the shipped set, uncurated failure from the free-text path — and neither alone
does it. Told it to choose the shipped captions **before** seeing their outputs and to include one
where the model does badly.

**SUP-57 (P1) — the demo's truncation may not be the truncation we measured.** `truncate.py` runs the
original regex over **spaCy** tags because free text has none; the pilot ran it over **HumanML3D's
own** tags. The docstring is honest that this is "close enough for the same regex to fire the same
way" — **but that is a load-bearing assumption**, since the demo exists to show a 0.145 effect
measured under the other tagging. Validation is cheap: run both paths over the same HumanML3D
captions and report the agreement rate.

**SUP-58 (P2) — a weak baseline flatters the model, which is the failure SUP-43 was written to
prevent.** TF-IDF was chosen for transparency; the instinct is right, the consequence is not. **The
strongest form of "just look it up" is embedding retrieval, and we already have it validated in the
evaluator's text encoder.** Beat TF-IDF and the sceptic says "weak lookup"; beat the evaluator's own
encoder and there is no reply. Also told it: **if the baseline sometimes wins, show that** — a demo
where the baseline occasionally beats the model is far more credible than one where it never does.

**SUP-59 (P3):** spaCy/`en_core_web_sm` must reach pinned demo requirements + README download step or
the reproducibility criterion fails at the first hurdle; `demo/README.md` is easier written alongside
the code than reconstructed later.

**Seed 2 still running.** Now 16:32Z; SOFT 01:46Z, HARD 03:16Z.

## [2026-09-06T16:40Z] Supervisor pass 42 — Review 11: the demo's visual payload does not match its finding

**Stage 5 app built** (`demo/app.py`, Gradio). Against the pre-registered criteria: one command ✓;
**retrieval baseline visible in the interface ✓ and done well** (own always-on pane, labelled
`"Just look it up" baseline` — the criterion I said I would hold hardest, built without softening);
failure cases reachable ✓; metrics + comparability label on the page ✓; 30-second comprehension ✗.

**Caveat writing is better than most people manage** — *"the R-Precision difference between them, if
any, cannot be claimed to be caused by truncation at this scale"* puts D-26 **in the interface**.

**SUP-60 (P1) — the mismatch I did not see until reading `app.py`:**
- **Shown, large:** two generated videos, full vs truncated caption.
- **Established:** that comparison is **not resolvable** (0.80 sigma, D-26).
- **Measured:** a **0.145 retrieval-space drop** — present on the page as *a table*.

**The visual payload is the thing we could not measure; the measured thing is text.** Viewers watch
videos and skim prose, so the page's most salient element invites exactly the inference its caveat
forbids. **A caveat contradicting the page's own dominant element loses**, and no amount of rewording
fixes it.

**Fix proposed: visualise the measured effect instead.** Truncation changes *what a caption
retrieves* — that **is** the 0.145. Full caption -> nearest real motion; truncated caption -> nearest
real motion; both panes real, so generator quality is not a confound, and the difference between
them is the destroyed information. A non-specialist reads it instantly.

**This resolves three things at once and supersedes SUP-56:** retrieval is instant, so the
30-second criterion is met **without** pre-generated examples — the fast path *is* the finding.
Payload matches finding. Over-claim risk removed. Generation panes stay, demoted below the fold and
framed as illustrative rather than evidential.

**SUP-58 upgraded in consequence:** if retrieval becomes the headline rather than a baseline,
TF-IDF-vs-encoder stops being a fairness question and becomes a **fidelity** one — the 0.145 lives in
the evaluator's embedding space, so TF-IDF would visualise a different quantity than the table above
it. Use the encoder.

**Noted approvingly:** the build session deferred the generate-path test rather than contend with the
seed-2 job, and refuses to call Stage 5 core-path-verified until it has driven it through a browser.
Correct standard.

Now 16:40Z; SOFT 01:46Z, HARD 03:16Z.

## [2026-09-06T16:48Z] Supervisor pass 43 — SUP-61: validated my own design proposal before the build session spent time on it

**SUP-58 landed** (embedding retriever added alongside TF-IDF, pane relabelled). **SUP-60's layout
change has not** — Review 11 crossed with that edit.

**Rather than repeat myself, I tested the premise of my own proposal.** SUP-60 assumed truncation
visibly changes what a caption retrieves. **That was an assumption I had not checked**, and if it
were weak the demo would be unimpressive after the build session had spent hours on it.

Method: 6,000 HumanML3D captions, 300 sampled, the original's truncation rule applied via
**HumanML3D's own tags** (sidestepping SUP-57's spaCy question), TF-IDF nearest-neighbour retrieval.

| measurement | value |
|---|---|
| rule actually shortens the caption | 62.3% |
| shortened captions retrieving a **different** motion | **46.0%** |
| caption retrieves its **own** motion — full | **78.3%** |
| caption retrieves its **own** motion — truncated | **55.0%** |
| **drop** | **−23.3 points** |

**Premise holds.** And the last row is a better headline than the table currently on the page:
**"a full caption finds its own motion 78% of the time; truncated, 55%."** Same phenomenon as the
0.145 R-Precision drop, stated so a non-specialist needs no explanation at all.

**The 46% figure also solves the first-impression problem instead of creating one.** Half of user
inputs will show no visible difference. **Print the aggregate beside the live example** — "truncation
changed the retrieved motion 46% of the time; here is your caption" — and a null becomes
*informative*, one draw from a stated distribution. **This is strictly better than curating shipped
examples, so I withdrew SUP-56 in its favour.** Retrieval is instant, the distribution is stated, and
every sample is honest including the negatives.

**Two caveats I attached to my own numbers:** they are TF-IDF, not the evaluator's encoder, so the
demo must **re-measure with the encoder and print those** — mine establish the premise, not the
display values. And I used HumanML3D's tags, so SUP-57's spaCy-fidelity question is untouched.

**Ninth supervisor correction, but a different kind:** the first four were errors caught after the
fact; this one was a proposal checked *before* it cost anyone anything. **That is the cheaper
version of the same discipline.**

Now 16:48Z; SOFT 01:46Z, HARD 03:16Z.
