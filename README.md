# T2P-Reboot — text-to-pose generation, rebuilt around measurement

A 2026 course project spent three months on CLIP-conditioned diffusion for text-to-pose
generation and produced no measurement — every number in its report is a training loss.

**This is a full redo, not a repair.** The architecture, the dataset, the problem framing and
the reason this work should exist at all are open questions here, not inherited ones. Stage 1
established that the original was compensating for a data-decode bug, which means very little
of its design carries forward on merit.

Measurement is the **discipline**, not the agenda. Nothing counts as a result until it comes
off a validated harness on a held-out split — but that is how the redo is kept honest, not what
the redo is for.

**Start at [`docs/00_START_HERE.md`](docs/00_START_HERE.md).**

---

## The task

Given `"a person raising their right arm"`, generate a single static 3D human pose — 22 joints
in space — that looks like a human, matches the text, and is not a memorised training example.

## Status

| stage | output | status |
|---|---|---|
| 1 — Forensics: which suspected defects are real? | `FORENSICS.md` | **done** — F1-F8 verified |
| 2 — Landscape + rebuild spec + positioning | `LANDSCAPE.md`, `REBUILD_SPEC.md`, `POSITIONING.md` | **done**, reviewed |
| 3 — Evaluation harness + measured baseline | `docs/EXPERIMENT_LOG.md` E0a/E0b | **partial** — evaluator validated; published-number gate **unresolved** |
| 4 — Experiments | `docs/EXPERIMENT_LOG.md` E1-pilot, E1A | **in progress** — two findings, see below |
| 5 — Local demonstrator | `demo/` | designed (`reviews/` SUP-43), not built |

**Findings so far.** Caption truncation costs retrievable text-motion alignment **in proportion to
how much text is removed** (~0.145-0.157 corpus-wide, ~0.27 conditional), essentially independent
of which part is removed — ~93% volume, ~7% position. And the first model trained here reached
~60% of a converged reference loss at 0.63% of its budget while scoring 13x worse on FID: **a
healthy loss curve on a bad model.**

**Every number here is internally-comparable-only.** The harness reproduces ground truth to within
0.0036 of the published reference across four independent full-split runs, but reproducing a
published generated-model figure did not
succeed at affordable sample sizes, so comparability to published results is **not** claimed.

**Nothing has been trained or measured here yet.** There is no result to report, positive or
negative. When that changes, it goes in `docs/EXPERIMENT_LOG.md` with the split, the sample
count, the seeds and the spread.

## How the record works

- Every claim is labelled **VERIFIED** (ran or fetched here, output saved) or **UNVERIFIED**.
- Every measurement lives in `artifacts/*_record.json`. Nothing is re-derived from memory.
- `LEDGER.md` is append-only and is the memory across interrupted sessions.
- `reviews/` and `guidance/` belong to other agents; this side never writes there.

## Layout

```
docs/            the research record and the rules      LEDGER.md      append-only run log
src/t2p/         the machinery (Stage 3)                BRIEFING.md    forensic read of the original
tests/           silent-failure tests                   CLAUDE.md      agent instructions
notebooks/       narrative + experiments                AUTONOMOUS_RUN_PROMPT.md
artifacts/       *_record.json, figures                 primary_source/  official HumanML3D files
reviews/         reviewer territory (read-only)         guidance/      collaborator territory (read-only)
```

## Running an unattended session

Open a session **in this directory**, set permissions to accept-edits, then:

```
/loop Follow AUTONOMOUS_RUN_PROMPT.md in this repo. Read LEDGER.md first and resume from the last incomplete item.
```

## Hardware

Apple Silicon. **MPS, not CUDA.** MPS is non-deterministic — every comparison needs multiple
seeds and a reported spread (`docs/LANDMINES.md` §7).

## Source material

The original project lives at `<ARCHIVE>/` and is **read-only**.
