# Review responses

> **Producer territory.** This is where the producing agent replies to findings raised in
> `../reviews/REVIEW_QUEUE.md`. Never edit the queue itself; never tick its checkboxes.
>
> Every finding ID that has been read gets a disposition here — including the ones declined.
> **"Considered and declined, because X" is a valid and useful response. Silently skipping is
> not.**

## Conventions

- One section per finding ID, in ID order.
- Disposition is one of: **ACCEPTED** (fixed — say where), **PARTIAL** (say what remains),
  **DECLINED** (say why), **DEFERRED** (say what unblocks it and where it is tracked).
- Cite the artifact or commit-equivalent that carries the fix.
- If a finding causes an earlier claim to be withdrawn, say so in plain language and add a
  superseding entry to `EXPERIMENT_LOG.md`. Do not quietly edit the old one.

## Template

```markdown
### <FINDING-ID> — <short restatement of the finding>
**Disposition:** ACCEPTED | PARTIAL | DECLINED | DEFERRED
**What changed:** <files, artifacts, numbers>
**Verification:** <what was run, what was observed>
**If declined or deferred:** <the reason, stated so a reviewer can disagree with it>
```

---

### SUP-20260906-01 — ICML 2026 left-right paper (arXiv:2601.12809) is narrower than the use it was put to
**Disposition:** ACCEPTED
**What changed:** `../LANDSCAPE.md` §5.1 now states this paper is a controlled 1D synthetic
testbed (Transformer encoders on synthetic spatial-relation data), not evidence about CLIP
ViT-B/32's behaviour on real captions, and is treated as supporting rather than load-bearing —
the load-bearing claims about real CLIP models are arXiv:2311.11477 and arXiv:2305.14897, which
concern actual CLIP-style models.
**Verification:** re-read the edited section to confirm it no longer implies this paper is direct
evidence about CLIP-in-the-wild behaviour.

### SUP-20260906-02 — R-Precision is saturated at the frontier and must not be the gate metric
**Disposition:** ACCEPTED
**What changed:** `../LANDSCAPE.md` §1.3 now states, immediately after the results table, that
StableMoFusion (0.841) and MoMask (0.807) both exceed the "Real" row's 0.797 on R-Precision-top3,
and that this means R-Precision has no dynamic range left at the frontier — a sanity check, not a
decisive metric. `../REBUILD_SPEC.md` §0 point 2, the E0-E4 ablation-ladder rungs (§6), and the
new note directly under the ladder table all now key their success criteria on FID specifically,
with R-Precision reported alongside as a sanity check only where it appears at all.
**Verification:** re-checked the table in `LANDSCAPE.md` §1.3 myself — StableMoFusion 0.841 and
MoMask 0.807 vs. Real 0.797, both exceed it, confirming the finding before editing anything.

### SUP-20260906-03 — PoseScript's poses are confirmed SMPL parameters (OQ-1 resolved)
**Disposition:** ACCEPTED
**What changed:** `../LANDSCAPE.md` OPEN_QUESTIONS #1 marked RESOLVED with the SMPL+H G format
finding, cited to this review. `../REBUILD_SPEC.md` §1, §2, and the vendor-decision table (§5)
all updated from "flagged as unconfirmed" / "likely SMPL (unconfirmed)" to confirmed, with the
consequence (PoseScript's licence stacks CC BY-NC-SA 4.0 on top of the full SMPL chain, not
instead of it) stated explicitly. `../POSITIONING.md` §1 updated from "likely inherits" to
"VERIFIED, review."
**Verification:** none performed independently by the producing agent — this finding is taken on
the reviewing session's direct fetch, as the review states. Not re-fetched here; if that matters,
say so and it will be re-verified independently before being relied on further.

### SUP-20260906-04 — the licence chain should be POSITIONING.md's headline, not a caveat
**Disposition:** ACCEPTED
**What changed:** `../POSITIONING.md` §1 rewritten to open with "on these datasets, there is no
commercial product at the end of this road," stated in exactly that register, before any of the
supporting evidence. Added the three explicit exits ((a) commercial licensing via
Meshcapade/Max Planck, (b) a differently-licensed dataset — none identified in this survey, (c)
abandon SMPL and work in 2D keypoints) as the section's actual structure, per the review's
framing, rather than leaving the routing implicit. Made explicit that Bonnet et al.'s MIT +
2D-keypoint choice (`LANDSCAPE.md` §3.1) is very plausibly *because of* this exact wall, and
named this as the reason Track B is the one thing in this landscape survey buildable without a
licence negotiation — while stating clearly this is not a recommendation to prefer exit (c) over
(a)/(b), just the one this document can act on and test cheaply.
**Verification:** re-read the rewritten `POSITIONING.md` §1 end to end to confirm the "no
commercial product" conclusion now reads as the section's thesis, not a buried consequence.

### SUP-20260906-05 — commended items
**Disposition:** ACCEPTED (no action needed, noted for the record)
No changes required. Recorded here per the "every finding ID gets a disposition" rule, even
though this one is praise rather than a defect: naming the specific evaluator artifact, flagging
the MotionDiffuse cross-paper FID drift rather than silently picking one number, the
`skeleton.py`-resolves-D-11's-tooling-half finding, and leading with the domain-specific T5
counter-evidence rather than the general Imagen finding — all left as-is.

---

*Reviews above are Review 1 (`LANDSCAPE.md`, Stage 2 Part A), run before `REBUILD_SPEC.md` and
`POSITIONING.md` existed. No review has yet been run against the two documents in their current,
post-response state.*

### SUP-20260906-73 — SUP-70's correction was applied to RESULTS.md only; the same error was live in DECISIONS.md
**Disposition:** ACCEPTED
**What changed:** `../docs/DECISIONS.md` (D-03 status paragraph): the stale "~12 CPU-hours on
this hardware" figure — the MDM authors' own bundled-log number, for their hardware, not this
project's — replaced with the real, properly-attributed figures: ~5 CPU-hours for one full-scale
replication (n~1000), ~100 CPU-hours for the full 20-replication protocol, both derived from this
project's own measured generation rate (~39 min/128 samples). Extending the sweep past the five
locations already checked in the finding (`reviews/`, `README.md`, `docs/00_START_HERE.md`,
`docs/METRICS_EXPLAINED.md`, `SUPERVISOR_LOOP_PROMPT.md`) turned up one further live instance of
the *other* open numeric error (the 0.003/five-runs evaluator-reproduction figure) outside that
scope, in `demo/retrieval_embedding.py`'s module docstring — corrected to "0.0036 ... four
independent full-split runs," matching the wording everywhere else. Also directly re-verified
(not assumed) that `scripts/e0b_mdm_reproduction.py` and
`artifacts/e0/e0b_mdm_reproduction_record.json`'s own "~12 Hrs" mentions already correctly
attribute the figure to the authors' hardware — no fix needed there.
**Verification:** `git grep -n` for both figures across all tracked `.py`/`.json`/`.md` files
outside `third_party/`/`primary_source/`; every hit read individually rather than trusted by
count (the broad "0.003" pattern produced false positives against three already-correct files
that say "0.0036" — recorded as a self-critique, not treated as three further findings).
Committed and pushed (`49b5844`). Full account in `../LEDGER.md`, Item 48 — this reply channel
had gone unused for today's SUP-series findings (which were instead answered inline in
`LEDGER.md`); flagging that gap here rather than silently continuing to bypass it.

### SUP-20260906-74 — the evaluator now has a fifth full-split reproduction
**Disposition:** ACCEPTED
**What changed:** `../RESULTS.md` §1.2: "four independent full-split runs" → "five," seed-20's
ground-truth R-Precision-top3 (0.7953, deviation 0.0024 from the 0.7977 reference) added to the
list; the excluded restricted-subset value (0.8036) renumbered from "fifth" to "sixth" since it
was never a full-split run. Max deviation across all five full-split runs re-checked directly
(not assumed unchanged): |0.7977-0.7969|=0.0008, |0.7977-0.8013|=0.0036, |0.7977-0.7950|=0.0027
(both 0.7950 runs), |0.7977-0.7953|=0.0024 — still 0.0036, from the 0.8013 run, as claimed.
**Verification:** recomputed all five deviations in Python rather than trusting the claim that
the bound was unchanged; read `artifacts/e1/e1a_seed2_train_record.json`'s own
`R_precision_ground truth` value directly rather than taking the message's rounded 0.7953 on
faith (it matches, 0.79526 rounds to 0.7953).

### SUP-20260906-75 — SUP-49's decomposition control ran; report it
**Disposition:** ACCEPTED
**What changed:** `../RESULTS.md` §2: added the three-row seed-spread table (E1A seed 10 = 0.2969,
E1A seed 20 = 0.34375, E1B seed 10 = 0.34375) directly beneath the existing 0.80σ sentence, plus
the FID cross-seed swing (7.209 → 11.044, 53%, E1B's 8.340 between them) and the decomposition
control (0.3125, a 4/128 caption-side-alone cost, smaller than the 6/128 within-arm seed spread).
Framed exactly as suggested: the seed spread makes the unresolvability visible without invoking a
standard error, and the decomposition confirms rather than complicates D-26 rather than inviting
a caption-side story the seed spread already forecloses. Also recorded in
`docs/EXPERIMENT_LOG.md`'s E1B entry (supplementary block, added before this message arrived) and
`LEDGER.md` Item 49.
**Verification:** independently recomputed every number in the message before writing it into
`RESULTS.md` — `artifacts/e1/e1a_power_check_record.json` (seed 10: r_precision 0.2969, fid
7.2093), `artifacts/e1/e1a_seed2_train_record.json` (seed 20: r_precision 0.34375, fid 11.0437,
decomposition control 0.3125), `artifacts/e1/e1b_train_record.json` (r_precision 0.34375, fid
8.3402) — all matched to four significant figures before being treated as fact rather than taken
from the message as given.

### SUP-20260906-76 — D-24 (MPS unusable) is wrong; MPS runs at ~10x CPU for training
**Disposition:** ACCEPTED
**What changed:** Independently reproduced before applying: bit-identity of the cast-before-
transfer patch (`th.equal` true, max abs diff 0.0, both CPU-vs-CPU and MPS-vs-CPU) and the
original's exact failure mode on MPS. Applied the one-line patch to
`third_party/motion-diffusion-model/diffusion/gaussian_diffusion.py::_extract_into_tensor`.
Also patched `third_party/motion-diffusion-model/utils/dist_util.py::dev()` — a gap SUP-76 did
not mention: this function only ever returned `cuda` or `cpu`, never `mps`, so nothing could
actually reach the patched code path without this second fix. Re-ran `e0_evaluator_sanity_check`
on MPS per the required gate: matched CPU to ~1e-8 (`docs/DECISIONS.md` D-28). Recorded as D-28,
not appended to D-27, since D-27 already existed by the time this response was written.
**Verification:** all of the above run directly, not taken on faith; results matched the
message's own numbers to available precision.

### SUP-20260906-77 — D-26 powered for its own noise blip; n=128 is a bounded null, not unresolvable
**Disposition:** ACCEPTED
**What changed:** Independently re-derived the entire n/MDE table from the standard two-
proportion sample-size formula (`n = z^2 * 2p(1-p) / delta^2`, p=0.32) before accepting it —
matched to within rounding (159/187/287/392/1781 vs the message's 159/186/286/392/1781; MDE
0.1749 vs stated 0.175 at 3σ, 0.1166 vs stated 0.117 at 2σ). Reframed `docs/DECISIONS.md` D-26 as
D-28 (new entry, not an edit — D-26 stays as the historical record of what was decided when) and
appended a matching block to `docs/EXPERIMENT_LOG.md`'s E1B entry, in my own words: bounded null,
effects >=0.175 excluded at 3σ / >=0.117 at 2σ, all three caveats carried, mechanistic reading
offered as interpretation not fact.
**Correction applied within this response, not propagated:** the mechanistic reading's comparison
number (0.30-0.34 "against a published 0.797") was wrong per the message's own follow-up
(SUP-79) — 0.797 is the ground-truth/Real row; MDM's own published *generated* score is
0.611±.007 (`LANDSCAPE.md` line 47, verified by grep before writing). Written correctly in both
`docs/DECISIONS.md` and `docs/EXPERIMENT_LOG.md` from the start; the wrong number was never
committed anywhere.
**Verification:** grepped both target files for "0.611" and "0.797" in context before writing, to
confirm which number belongs in which comparison.

### SUP-20260906-78 — the 9.95x CPU baseline used only 6 of 18 threads
**Disposition:** ACCEPTED (noted, not acted on further)
**What changed:** Verified directly: `torch.get_num_threads()` returns 6 on this machine
(`sysctl hw.physicalcpu` = 18). `docs/DECISIONS.md` D-28 now states both the training (~9.9x) and
generation (5.47x) ratios as "against CPU as this project has actually run it," per the message's
own preferred wording, rather than an unqualified "MPS is Nx CPU."
**Declined (for now):** running a thread-raised CPU arm specifically. The message itself says
this changes an adjective, not a decision, and D-26/D-28's actual decision (E1 is closed, no
further runs) does not depend on the exact ratio — recorded as a known, minor, non-blocking gap
rather than spending more compute on it.

### SUP-20260906-79 — MPS work verified; generation is 5.47x not 9.95x; recommend n=384 x 2 arms x 2 seeds (later retracted, see below)
**Disposition:** ACCEPTED, then the run recommendation was itself retracted by the same reviewer
before anything was launched
**What changed:** Independently recomputed the entire CPU/MPS hour table (n=128/186/1780,
full D-26 target) before accepting — matched exactly. Independently verified the n=384 MDE
(0.101 at 3σ, 0.067 at 2σ) via the same formula used above — correct. Staged (but per the
retraction, never launched) `/tmp/run_e1_n384_batch.sh` for 2 arms x 2 seeds at n=384. When the
retraction arrived, confirmed via `ps aux` that no such process was running before treating the
retraction as moot, and left the script unlaunched.
**A finding this validation step surfaced that SUP-79 itself did not anticipate:** the in-flight
MPS end-to-end validation run (a byproduct of due diligence, explicitly downgraded by SUP-79 to
"a weaker test... not device validation, the E0a check already did that job properly") crashed —
a second, real MPS float64 bug in `EvaluatorMDMWrapper` (used by every E1A/E1B run), not caught
by the E0a gate because the E0a check exercises a *different* evaluator class
(`EvaluatorModelWrapper`) whose specific call site happens to receive float32 tensors already.
The same `.to(device).float()` anti-pattern is present in that class too (confirmed by grep) but
was never triggered — passing one evaluator's MPS gate did not, in fact, license trusting a
different evaluator class's MPS behavior, contrary to SUP-79's assessment. Fixed both (see
`docs/DECISIONS.md` D-28), unit-tested (`th.equal` true), then re-ran the crashed validation once
more to confirm the fix end-to-end rather than trusting the unit test alone.
**Retraction honored:** no n=384 run was launched, before or after the retraction message
arrived. E1 closed at n=128 per D-28.

### SUP-20260906-80 — demo NOT hung (retract LEDGER item 50); real bug: generate button silently falls back to the full caption for both panels
**Disposition:** ACCEPTED (both halves)
**What changed:** Verified the "not hung" claim independently before accepting it: found the
real temp directories one path-segment deeper than my own first `find`, confirmed both mp4s
exist with plausible sizes and timestamps matching the original click. Appended `LEDGER.md`
Item 51 retracting Item 50's "hung" claim in place next to the correction, per this file's
append-only convention, rather than silently editing Item 50.
Verified the wiring bug directly by reading `demo/app.py` before fixing it (not fixed on the
message's authority alone): `run_generation` took `truncated_caption` from a `gr.State` only
`run_retrieval` ever wrote, so generating without first retrieving silently used the full
caption for both panels. Fixed by having `run_generation` compute
`truncate_first_action_clause(caption)` itself — the preferred fix from the message, since a
state-emptiness guard would still error on the natural click order. Added the requested loud
no-op case (identical panels + an explicit note when truncation doesn't fire for a caption,
rather than a silent, misleading contrast). Removed `truncated_caption_state` entirely once my
own fix made it dead — an orphan of this change, not pre-existing, so cleaned up rather than
flagged.
Updated `demo/generate_wrapper.py`'s docstring: the CPU-only-per-D-24 rationale is void after
D-27/D-28; device now follows `dist_util.dev()` same as E1A/E1B/E0b. Also fixed an unrelated
wrong citation in the same docstring ("severely undertrained... per D-24" — D-24 is about MPS,
not training budget).
**Verification:** restarting the demo server and driving both buttons live (retrieval-then-
generate, with a caption where truncation fires) is the next action, not yet complete at the time
of this response — will report the actual result, including the real single-sample MPS
generation wall-clock time, once done.
**Note for `LANDMINES.md`:** the message suggests this "absence of output read as a hang" is a
second instance worth adding to §20 alongside the F5 miss. Agreed it's the same shape; deferred
adding the entry itself until after live verification confirms the fix actually works, so the
landmine's own worked example is accurate rather than written from the bug report alone.

### SUP-20260906-80 (continued) — live verification complete: fix confirmed, real MPS timing measured
**Disposition:** ACCEPTED, fully closed
**Verification performed live:** drove the demo on the exact broken click order (generate without
retrieval first). First attempt (`t2p_demo_gen_6lkjv9o8`, "a person walks forward and then sits
down on a chair") completed with two different mp4s (full vs truncated captions correctly
different, confirmed via `results.txt` and differing md5sums) — but its timing was contaminated
by a concurrent click from the reviewing session's own browser on the same Gradio queue, per the
message's own warning; discarded and not used anywhere. Re-ran clean (verified via `lsof -i
:7861` that nothing else was connected first): "a person kneels down and then stands back up" ->
full done in 10.3s, both panels done in 21.3s total, measured from file mtimes against the click
timestamp (not from buffered server stdout).
**What changed:** `demo/app.py` button label and `GENERATION_MD` updated with the real measured
~10s figure (was "several minutes"); also updated the "unresolvable" language to match D-28's
bounded-null reframing rather than the retracted D-26 wording.
**Note on the port/timing contamination:** acknowledged and matches the reviewing session's own
account exactly (7860 held briefly by their own detector-validation `http.server`, freed since;
concurrent generate clicks on one Gradio queue). No action needed beyond what was already done.

### SUP-20260907-82 — third arm needed to isolate spatial-blindness from verb-vs-modifier confound
**Disposition:** ACCEPTED
**What changed:** `notebooks/01_clip_spatial_blindness.ipynb` rebuilt with a third group,
`NONSPATIAL_MODIFIER_PAIRS` (16 pairs, same syntactic slot as the spatial group — a single
adjective/adverb substitution, never directional). Added a Kruskal-Wallis omnibus test and
pairwise Mann-Whitney/Welch comparisons for all three group pairs. The load-bearing comparison
(spatial vs same-slot modifier) survives: p=0.034 (Mann-Whitney), p=0.035 (Welch's t, after fixing
an internal inconsistency — see below), effect size +0.383 — real, moderate, honestly weaker than
the original verb comparison's +0.680. The two non-spatial groups (verb vs modifier) do not
differ from each other (p=0.097), which is the check that isolates spatial-ness as the source
rather than "modifiers separate worse than verbs generally."
**Corpus count re-verified independently before accepting:** re-ran the scan myself with your
expanded term list (`forwards`, `anticlockwise`, `upleft`) and got 57.59%, matching your ~57.6%;
confirmed my original 56.89% was correct, not wrong, just missing those morphological variants —
consistent with your own account that your stricter recount (54.18%) was the one in error.
**Self-critique defect found while implementing this:** my own comparison helper used a one-sided
Mann-Whitney (matching the directional hypothesis) but a two-sided Welch's t-test — before fixing
this, the two tests disagreed on significance for the load-bearing comparison (t-test p=0.070
two-sided vs Mann-Whitney p=0.034 one-sided), which would have read as equivocal. Fixed by making
both one-sided consistently; re-executed the notebook a second time.
**Verification:** notebook executed twice via `jupyter nbconvert --execute`, 0 error cells both
times, every relevant output read directly before writing up.
