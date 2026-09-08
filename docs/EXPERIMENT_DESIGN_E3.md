# EXPERIMENT_DESIGN_E3 — does CLIP's spatial deficit actually reach generated motion?

**Status: DESIGN ONLY as of this section. Nothing has been run yet at the time this is written.**

## 1. The question

Notebooks 01 and 01b established that MDM's frozen CLIP text encoder under-separates spatial
language (left/right, forward/backward) from comparable non-spatial language, in embedding
space, on hand-written minimal pairs (n=40/group, Cohen's d=+0.995). **Whether that deficit
survives into the actual generated motion — on real HumanML3D captions, at the retrieval metric
this project's whole evaluation harness depends on — has never been measured.** E3 measures it
directly: generate every test-split caption once with the released checkpoint, no training, and
compare R-Precision on the spatial subset against the non-spatial subset.

## 2. Why this comes before E2

`docs/EXPERIMENT_DESIGN_E2.md` asks whether *repairing* CLIP's spatial encoding (a trained
adapter) improves generation on spatial captions specifically. That question is only
interpretable if E3 has already shown there is a real gap to close. If E2 ran first and returned
a null result, the null would be ambiguous on its face: did the adapter fail to help, or was
there never a generation-level gap for it to help with? **E3 removes that ambiguity, and it is
strictly cheaper and lower-risk than E2** — no training, no adapter, just generation and scoring
with infrastructure this project has already built and validated. Running it first is not
optional caution; it is what makes E2's own result interpretable either way it lands.

## 3. Design

**Generate:** every entry in HumanML3D's standard test-split evaluation pool, once each, with
the released checkpoint (`checkpoints/mdm/humanml-encoder-512/humanml_trans_enc_512/
model000475000.pt`), fixed seed, guidance scale 2.5, 1,000 diffusion steps (no respacing) — the
exact same generation convention every other number in this project has used (E0b, E1A, E1B). No
training, no new model, no adapter.

**A population correction made before running, not after:** the design originally assumed 4,198
samples, matching notebook 05's direct read of `test.txt` (one entry per unique sequence). The
actual evaluation-mode data loader — the same one E0b, E1A, and E1B already draw their own
R-Precision numbers from, just truncated to n=128 there — draws from a cached pool of **4,648**
entries: the same 4,198 base sequences plus 450 extra entries for sequences that have more than
one annotated caption (each extra entry letter-prefixed to stay a distinct dataset key, standard
HumanML3D convention). Restricting generation to an arbitrary 4,198-entry prefix of this pool
would be a new, one-off population no other number in this project uses. Generating the full,
standard 4,648-entry pool instead keeps this measurement on the exact same population every
other evaluation number here already stands on. **The spatial/non-spatial split is therefore
computed fresh from whichever captions actually get generated (§3 below), not assumed to
reproduce notebook 05's exact 2,448/1,750 counts** — those counts remain correct for their own
4,198-sequence population; this experiment's own realized counts are reported as measured.

**Score with both evaluators this project has working and cross-checked (notebook 02):** Guo's
R-Precision (euclidean ranking, its own published convention) and TMR's (cosine ranking, its
own). Batches of 32 for both, matching the field's own protocol.

**Report separately, not pooled:**
- the spatial subset (~2,700 captions expected, scaling notebook 05's 58.3% test-split rate to
  4,648 — the realized count is reported directly, not assumed)
- the non-spatial subset (~1,940 captions expected, same scaling)
- both subsets further split into caption-length terciles, computed once from each subset's own
  word-count distribution (same convention as `docs/EXPERIMENT_DESIGN_E2.md` §3) — because
  notebook 05 already found spatial captions run 41% longer on average (14.38 vs 10.19 words,
  Cohen's d=+0.592, not a large-n artifact), and any spatial-vs-non-spatial comparison that
  doesn't hold this out is not measuring what it claims to.
- FID per subset, as a secondary check (see §5).

## 4. Power — stated before any result exists

Using this project's own validated binomial-floor formula (`docs/LANDMINES.md` §18,
`n = z²·2p(1-p)/delta²`, p=0.35 matching this checkpoint's own severely-undertrained operating
range, z=3 for a 3σ threshold):

| n | MDE (3σ) |
|---:|---:|
| 128 (every prior generation run this project has done) | 0.179 |
| ~2,700 (expected spatial subset) | 0.039 |
| ~1,940 (expected non-spatial subset) | 0.046 |
| ~900 (one length tercile within the spatial subset, ~2,700/3) | 0.068 |

(Recomputed once the realized subset sizes are known, in the results themselves — these are
the pre-run estimates the design is committing to, not the final reported numbers.)

A **~4.4x** improvement in resolving power over every generation-side comparison this project
has run so far (E1A vs E1B included). Effects that were invisible at n=128 all session become
measurable at these subset sizes, without spending a single additional sample beyond what
generating the full test split already produces.

## 5. FID becomes trustworthy here for the first time

Notebook 04 measured FID's own real-vs-real estimator-bias floor directly: **1.577 at n=128**,
falling to **0.109 at n=2,000**. Every FID this project has computed so far (E0b, E1A, E1B) was
computed at n=128 and is dominated by that same bias, not by anything about model quality. At
n=1,750 and n=2,448, FID is close to or above the range notebook 04 showed is estimator-noise-free.
**This is the first FID this project reports at a sample size where the metric is not known to be
broken by construction** — reported per subset, cross-checked directly against notebook 04's own
floor curve rather than trusted on the strength of a bigger n alone.

## 6. Pre-registered interpretation — written before the run, not after

- **Spatial materially worse than non-spatial, beyond the MDE:** the embedding-space deficit
  propagates into generation. E2 becomes well-motivated, and this experiment states the actual
  gap size an adapter would need to close. **The caption-length confound makes this direction
  conservative, not inflated**: longer captions carry more retrievable signal (a more distinctive
  query is less confusable with other candidates in the same retrieval batch), and spatial
  captions are the longer group — so the confound should push the spatial subset's R-Precision
  *up*, working against this outcome. A deficit found despite that push is a real, understated
  one, not an artifact of the length difference.
- **No difference, or spatial better, beyond the MDE:** the embedding-space deficit does **not**
  propagate to generation at this checkpoint's operating range. This is the more surprising
  outcome of the two, and arguably the more valuable one: it would mean a text-encoder repair is
  aimed at a gap that doesn't reach the actual output, and it would save the cost of building and
  training an adapter for a problem that isn't there. **This outcome is reported with exactly the
  same prominence as the first**, not buried as a null result.
- **Effect inside the MDE at every subset size tested:** reported as indistinguishable from
  noise, exactly as this project's own prior stopping decisions (`docs/DECISIONS.md`
  D-24/D-27/D-28) already established it will do, rather than over-read a favorable- or
  unfavorable-looking number that the power calculation in §4 says this design cannot resolve.

## 7. Compute

**~4.07 hours** at this project's own directly measured local rate (3.15 s/sample, batch 32,
1,000 diffusion steps, `trans_enc` 17.9M-param architecture — the same rate `docs/
EXPERIMENT_DESIGN_E2.md` §7 derived and used), revised up from the original 3.67h estimate to
match the corrected 4,648-sample population (§3): 4,648 samples × 3.15 s/sample ≈ 14,641s ≈
4.07h. **This machine, not Kaggle** — same reasoning as E2 §7: this machine has already closed
every silent-failure surface on the MPS path that a fresh Kaggle environment would reopen, for a
hypothetical speedup nobody has measured there.

All 4,648 generated motions held in memory at once are small: (4648, 196, 263) float32 ≈ 0.96 GB
— comfortably resident, no disk round-trip needed during scoring. **Generation itself is
checkpointed incrementally to disk, one batch of 32 at a time**, independent of the in-memory
scoring step: an interrupted run resumes from the last completed batch rather than losing the
whole ~4-hour run, since the generation loop and the batch-caching are the same operation, not
an afterthought bolted on top. A 32-batch smoke test (n=64, two batches) was run first and timed
at 3.22 s/sample -- within noise of the 3.15 s/sample this estimate assumes.

## 8. What this does not claim

One checkpoint (this project's own severely-undertrained model, 3,000 of MDM's 475,000 published
steps), one seed, one dataset. This measures whether *this* model on *this* benchmark shows a
generation-level spatial gap — not a general property of text-to-motion systems, and not a claim
about what a fully-trained model would show. It is also not a claim that this checkpoint's
absolute quality is representative of the field (it isn't — see `RESULTS.md` §1.4); it is a
same-model, same-budget, within-checkpoint comparison, which is exactly what the spatial-vs-
non-spatial split needs and no more.

**Would reverse if:** the honest MDE at the subset sizes actually measured (§4) turns out larger
than any effect worth caring about once real variance is accounted for (e.g. if per-sample
retrieval outcomes turn out far more correlated within a batch than the independent-Bernoulli
assumption behind §4's formula assumes) — in which case this design's own numbers, not a
qualitative impression, are what say so.
