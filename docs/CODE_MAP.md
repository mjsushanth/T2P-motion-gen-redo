# CODE MAP — what every part of this repository is for

> **The governing rule of this codebase:** **notebooks are narrative, scripts are machinery.**
> The predecessor project violated this so completely that its entire system lived in four
> notebook cells of 44k, 46k, 52k and 101k characters, with three near-duplicate copies of the
> same classes in one file.

An earlier plan for this repository proposed a conventional `src/`+`tests/`+`configs/` package
layout. That plan was superseded: the deliverable format that actually shipped is executed Jupyter
notebooks, each one self-contained and re-derivable from its own cells. What follows describes
what is actually here.

```
T2P-Reboot/
│
├── README.md                top-level entry point
├── RESULTS.md                what this project found, for a reader who has read nothing else
├── FORENSICS.md              what the original implementation got wrong, and how it's known
├── LANDSCAPE.md              published text-to-motion results this project measures itself against
├── POSITIONING.md            what this project is worth, licence question included
│
├── notebooks/                the research record. nine executed notebooks, each answering one
│                             question end to end; see notebooks/README.md for the index.
│                             Every number in every notebook comes from a cell in that notebook.
│
├── docs/
│   ├── LANDMINES.md          traps that produce plausible wrong answers with no error raised
│   ├── DECISIONS.md          every non-obvious technical decision, with the reasoning
│   ├── GLOSSARY.md           every acronym and term, defined once
│   ├── METRICS_EXPLAINED.md  what R-Precision and FID actually measure, and why
│   ├── EXPERIMENT_DESIGN_E2.md   the next experiment, fully specified before running it
│   └── CODE_MAP.md           this file
│
├── demo/                     the interactive demonstrator (`demo/README.md` for setup): types a
│                             caption, shows what a real system retrieves and generates from it
│
├── scripts/                  one runnable script per experiment, named for what it measures
│                             (e0b_*, e1_*, e1a_*, materialize_*)
│
├── artifacts/                *_record.json (never re-derived by hand) and rendered figures,
│                             organized by experiment (e0/, e1/, forensics/, demo/)
│
├── primary_source/           HumanML3D's own official processing code, fetched read-only for
│                             forensic comparison against the archived original implementation
│
├── checkpoints/              downloaded model weights (MDM's released checkpoint, TMR, the
│                             text-to-motion evaluator) — gitignored, re-fetched via a documented
│                             download step, never committed
│
└── third_party/              vendored code this project reuses rather than reimplements: MDM
                              (the diffusion model), TMR (a second, independent evaluator), and
                              the official text-to-motion evaluation harness
```

---

## Dependency direction

Notebooks and `demo/` import from `scripts/` and `third_party/`'s vendored code; nothing in
`scripts/` or `third_party/` imports from a notebook. `third_party/`'s own kinematic-tree and
bone-list code is the single source of truth for joint indices and connectivity — if two files
disagree about which joint index is the left elbow, that is the bug from `docs/LANDMINES.md` §1
reappearing in a new costume.
