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
