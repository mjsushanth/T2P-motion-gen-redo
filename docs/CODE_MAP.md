# CODE MAP — what every file is for, and why it exists

> **Purpose.** An agent's hardest problem is not syntax, it is *intent*: knowing which file to
> touch and what a function is supposed to guarantee. This file answers "what is this for?"
> for everything in the repository.
>
> **The governing rule of this codebase:** **notebooks are narrative, modules are machinery.**
> If logic would be pasted into a second notebook, it belongs in `src/t2p/` and gets a test.
> The predecessor project violated this so completely that its entire system lived in four
> notebook cells of 44k, 46k, 52k and 101k characters, with three near-duplicate copies of the
> same classes in one file. See `DECISIONS.md` D-15.

---

## Status: PLANNED, NOT BUILT

**No `src/`, `tests/` or `notebooks/` exists yet.** This section is the target layout from
`REBUILD_SPEC.md`, recorded here so that when Stage 3 starts there is one agreed shape. It is
provisional until Stage 2 finalises the dataset and representation decisions (D-11, D-12, D-13).

```
T2P-Reboot/
│
├── docs/00_START_HERE.md    READ FIRST
│
├── src/t2p/                 the machinery. importable, tested, no notebook dependencies.
│   ├── __init__.py          package docstring + the import contract
│   ├── config.py            typed run configuration; every experiment is a config object
│   ├── data/                dataset loading and the pose representation
│   │   ├── humanml3d.py     correct 263-dim decode; recover_from_ric; frame selection
│   │   ├── posescript.py    static-pose dataset (if D-12 selects it)
│   │   └── skeleton.py      the kinematic tree, joint names, bone list. ONE source of truth.
│   ├── repr.py              position <-> rotation conversions; differentiable FK layer
│   ├── models/              text encoder, denoiser, conditioning
│   ├── diffusion.py         noise schedule, training objective, sampling, CFG
│   ├── eval/                THE HARNESS. built and validated before any model. D-02, D-03.
│   │   ├── metrics.py       R-Precision, FID, MM-Dist, Diversity, MultiModality, MPJPE
│   │   ├── extractors.py    the feature extractors the protocol requires
│   │   └── protocol.py      the exact published evaluation procedure, cited
│   └── viz.py               skeleton rendering, with the coordinate frame stated explicitly
│
├── tests/                   pytest. Emphasis on SILENT failure modes:
│                            decode round-trips, bone-length invariants,
│                            normalise/denormalise inverses, skeleton connectivity,
│                            metric implementations against known fixtures.
│
├── notebooks/               narrative + experiments. executed, outputs saved.
├── scripts/                 one runnable script per experiment. named for its E-number.
├── artifacts/               *_record.json (never re-derive), figures, small weights
├── configs/                 experiment configs, one file per ablation rung
└── Knowledge_Docs/          the Excel knowledge tier. BUILT LAST. D-16.
```

---

## Dependency direction

```
   skeleton.py  <-  repr.py  <-  data/  <-  models/  <-  diffusion.py
        ^              ^                                      |
        |              |                                      v
        +--------------+------------------  eval/  <----------+
                                              ^
                                              |
                        notebooks/ and scripts/ import everything; nothing imports them
```

`skeleton.py` is the single source of truth for the kinematic tree and bone list. If two files
disagree about which joint index is the left elbow, that is the bug from `LANDMINES.md` §1
reappearing in a new costume.

---

## Files that exist now

| path | what it is |
|---|---|
| `../BRIEFING.md` | the review pass's forensic read of the original project. Findings F1-F5 with test recipes. |
| `../LEDGER.md` | append-only run log. **The** ledger. |
| `../CLAUDE.md` | agent instructions loaded automatically in this directory |
| `../AUTONOMOUS_RUN_PROMPT.md` | the self-contained instruction set for unattended runs |
| `00_START_HERE.md` | orientation |
| `DOCUMENTATION_INDEX.md` | the router — every document, by what you want |
| `LANDMINES.md` | 10 traps that give wrong answers with no error |
| `DECISIONS.md` | D-01..D-16, with reversal conditions |
| `GLOSSARY.md` | every acronym and term |
| `EXPERIMENT_LOG.md` | the research record. empty; E000 reserved. |
| `REVIEW_RESPONSES.md` | producer's replies to reviewer findings |
| `../reviews/REVIEW_QUEUE.md` | reviewer territory. read only. |
| `../guidance/` | collaborator territory. read only. |
