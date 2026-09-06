# T2P-Reboot — agent instructions

Text-to-pose generation, rebuilt from a failed 2026 course project. **Forensics first,
evaluation second, models third.** Read `docs/00_START_HERE.md` before doing anything.

## The five rules that matter most here

1. **`/Users/joel/MJS_ROOT/MJS_STUDY/<ARCHIVE>/` is READ-ONLY.** It is the archived
   original. Never edit, move, rename or delete anything under it. Copy out what you need.
2. **Label every factual claim VERIFIED or UNVERIFIED.** VERIFIED means you ran it, fetched
   it, or observed it *here*, with the output saved. Never quote a benchmark number, paper
   result, dataset size, licence or API signature you have not personally checked. An
   admitted gap beats a confident guess.
3. **A training loss is not a result.** The original project died of this. No claim of
   quality is admissible unless it comes from the evaluation harness on a held-out split.
   See `docs/LANDMINES.md` §4.
4. **Ask before installing.** No `conda create`, `pip install`, `uv pip install` or
   `brew install` without Joel's approval. Write the env file, print the command, stop.
   (Exception: using an already-existing conda env requires no approval.)
5. **No git commits.** Ever, without Joel.

## Where things go

| you produce | it goes in |
|---|---|
| append-only run log, one entry per work item | `LEDGER.md` (canonical; **the** ledger) |
| an experiment result | `artifacts/<NN>_<name>_record.json` **and** an entry in `docs/EXPERIMENT_LOG.md` |
| a design choice with a reason | `docs/DECISIONS.md` (D-nn) |
| a trap that gives a wrong answer with no error | `docs/LANDMINES.md` |
| a term or acronym, first time it appears | `docs/GLOSSARY.md` |
| a reply to a reviewer finding | `docs/REVIEW_RESPONSES.md`, citing the finding ID |
| anything needing Joel's judgement | `OPEN_QUESTIONS` at the bottom of `LEDGER.md` |

**Territory.** `reviews/` belongs to reviewer agents and `guidance/` to collaborator agents:
read them, never write in them, never tick their checkboxes. Reply in your own files.

## Ledger entry format (append-only, newest above OPEN_QUESTIONS)

```markdown
## [ISO timestamp] Item N — <name>
**Status:** complete | partial | blocked
**Acceptance criteria:** (written BEFORE producing)
**Files changed:**
**Environment changes:** (or none)
**Self-critique defects found:**
**Revisions made:**
**Verification performed:** (what you actually ran, and what you observed)
**Next:**
```

## Code standards

- Absolute imports only. **No `sys.path` manipulation, no `parent.parent` chains, ever.**
  The fix for a failed import is `uv pip install -e .` / `pip install -e .`, not a path hack.
- Python 4-space indent, type hints on every function signature, no emojis in code.
- Notebooks are narrative; `src/t2p/` is machinery. If logic would be pasted twice, it
  moves to a module and gets a test.
- Surgical changes. Do not "improve" adjacent code.
- Hardware: Apple Silicon, **MPS not CUDA**. Never write CUDA-assuming code. MPS is
  non-deterministic — see `docs/LANDMINES.md` §7.

## Deferred on purpose

The colour-zoned Excel knowledge tier (`Knowledge_Docs/*.xlsx`) is built **last**, after the
research record is complete. Do not start it early.
