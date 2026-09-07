# Notebook style guide — reviewer-authored, binding on `notebooks/`

**Reviewer output. Read-only for the producing session.** Written 2026-09-07 after the author
directed that notebooks carry visual intuition and approachable explanation rather than dense
compressed text, and that `notebooks/01_clip_spatial_blindness.ipynb` serve as the reference
implementation.

**Read notebook 01 end to end before applying this.** This file names the patterns; the notebook
shows them working. Where the two disagree, the notebook wins and this file is wrong — tell me.

---

## The governing idea

A ledger entry is written for someone who already knows the problem. **A notebook is written for
someone who does not, and who will stop reading if the first screen does not tell them why to
care.** Every rule below follows from that one difference.

Density is not rigour. The rigour lives in the measurement section and must not be reduced. What
gets cut is *provenance in the lede*, *unexplained numbers*, and *prose doing a diagram's job*.

## The six layers, in order

1. **The question in one sentence.** Plain language, no jargon, no citations. A reader should know
   what is at stake before the first import.
2. **The intuition** — an analogy and a diagram, before any math or code. `04`'s "photos of a cloud"
   passage is the standard to match: it makes rank deficiency obvious without a single symbol.
3. **The setup** — what is being measured, and how.
4. **The measurement** — full rigour, undiluted. This is where density belongs.
5. **What it means** — plain language again.
6. **What would change my mind** — concrete falsifiers, ranked by cheapness.

Provenance — review history, finding IDs, which round changed what — goes in an **appendix at the
end**. It is audit material, not the lede. Notebook 01's opening was 471 words of design history
before a reader learned what a minimal pair was; that is the specific failure this rule prevents.

## Patterns worth copying, with the reason each exists

**1. Calibrate before you report.** Never present a number a reader has no scale for. Notebook 01
prints reference anchors first: identical sentences 1.0000, an unrelated topic 0.7190. Only then
does 0.9707-vs-0.9442 mean anything — the reader can see the *usable range* is narrow, so a 0.03
gap is large rather than negligible. **Ask of every headline number: does the reader know what
"big" would be?** If not, anchor it.

**2. When a display contradicts the finding, say so louder.** Notebook 01's hand-picked slow/fast
anchor scores *higher* than its left/right anchor — the reverse of the group result. Rather than
swap the example, the notebook prints the group means directly beneath and states the reversal
outright, making it the argument for why 40 pairs beat anecdotes. **A contradiction you confront
teaches; one you quietly remove is the §19 defect.** I nearly shipped that error myself.

**3. Every figure needs a reading guide.** A markdown cell after each figure saying what to look at
and what it means. An unexplained figure is decoration.

**4. Axis labels should carry the direction's meaning**, not just the quantity:
`"cosine similarity within minimal pair\n(further RIGHT = the encoder sees LESS difference)"`.
The reader should not have to reconstruct which way is bad.

**5. ASCII diagrams for flow, matplotlib for structure.** Data flow (caption -> encoder -> vector ->
generator) reads better as a fenced text diagram — it renders everywhere and costs nothing.
Experimental structure and anything quantitative gets drawn.

**6. Build to the hard case through a toy.** Notebook 04 does this correctly: 2-D ellipse, then 3-D
ellipsoid, then 512-D. The toy is what makes the real case intuitive. Do not open at full
dimensionality.

## The one thing to change in notebooks 03 and 04

Both end their "what it means" section with a **`print()` of a long f-string**. The instinct behind
it is right and I do not want it lost: the numbers are computed, so they cannot drift from the
result — that is this project's own rule that every number comes from a cell.

But it renders as a wall of monospace with no emphasis, no structure, and no line-break control —
the exact density problem, reintroduced at the most important moment in the notebook.

**Use rendered markdown with live values instead.** Verified available here (IPython 9.15.0):

```python
from IPython.display import Markdown, display

display(Markdown(f"""
**The finding.** At n=128 per side, real data compared against *equally real* data scores a mean
FID of **{fids_at_128.mean():.3f}** — not 0 — purely because 128 points cannot fill out a
512-dimensional shape.

| samples per side | mean FID |
|---|---|
| 128 | {fids_at_128.mean():.3f} |
| 1024 | {fids_at_1024.mean():.3f} |
| 2000 | {fids_at_2000.mean():.3f} |

The number shrinks as `n` grows, exactly as rank deficiency predicts — nothing about the data changed.
"""))
```

This keeps every number live and computed while giving headings, **bold**, and real tables. Apply it
to the closing section of 03 and 04, and anywhere else a `print()` is carrying more than about
thirty words of prose.

## Hard rules

- Every number in the notebook is produced by a cell in that notebook. No pasting from the ledger.
- Notebooks are committed **executed**, with outputs. An unexecuted notebook is not a deliverable.
- Report negative and weakened results as findings. A notebook proving a limitation is worth as
  much as one proving a capability.
- Do not over-claim at small n. Three landmines and several of my own errors are precedent.
- **Read the executed output before writing the verdict.** Do not assume a cell did what you
  intended — that is how the calibration reversal in 01 was caught, and how it would have shipped.

---

## Figure rules (added 2026-09-07 after SUP-20260907-85)

**1. Comparison panels share an axis.** When two panels contrast a good case against a bad case,
they use the same x-limits unless there is a stated reason not to. Matplotlib auto-scales each
subplot independently by default, and that default **silently destroys the comparison the figure
exists to make** — notebook 03's bone-length figure rendered a distribution constant to seven
decimal places as a wide bell curve, so the *correct* decode looked noisier than the broken one.
If a shared scale makes the good case invisibly narrow, that is the finding: annotate the spike,
do not zoom until its noise fills the frame. A zoomed inset is fine as a secondary panel, never as
the primary comparison.

**2. Look at the rendered image before shipping it.** Not the code that produced it — the PNG.
Overlapping annotations, labels overflowing the axes, and clipped titles are invisible in source
and obvious on sight. Three such defects shipped in one figure in notebook 03.

**3. Narrow segments get outside labels.** Text inside a bar that is 4 units wide out of 263 will
overflow into its neighbour. Use leader lines.

**4. Leave margin.** Extend limits past the data so edge labels are not clipped, and use
`constrained_layout=True` so suptitles are not cut off.

**5. A figure should carry its own argument.** A reader who sees only the image, with no
surrounding prose, should reach the right conclusion. Annotate regions directly on the plot —
`04_eigenvalue_spectrum.png` would be complete with "127 directions with measurable spread" and
"385 directions at numerical zero, never sampled" written on the two sides of its cliff.
