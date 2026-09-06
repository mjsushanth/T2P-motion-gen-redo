# Documentation index — the complete resource directory

> Every document in this repository, routed by what you are trying to do. If it is not here,
> it does not exist. **Keep this file current — a stale router is worse than none.**

---

## Getting started — in this order

| # | resource | why | time |
|---|---|---|---|
| 1 | [00_START_HERE.md](00_START_HERE.md) | orientation. what this is, what is real, the 7 traps | 5 min |
| 2 | [LANDMINES.md](LANDMINES.md) | 10 traps that give plausible wrong answers with no error. **Read before writing code.** | 10 min |
| 3 | [../BRIEFING.md](../BRIEFING.md) | the forensic read of the original project. findings F1-F5 with test recipes | 10 min |
| 4 | [../FORENSICS.md](../FORENSICS.md) | which of those findings survived contact with the data *(Stage 1 output)* | — |
| 5 | [DECISIONS.md](DECISIONS.md) | why the project is shaped this way, and what would reverse each choice | 10 min |
| 6 | [METRICS_EXPLAINED.md](METRICS_EXPLAINED.md) | **what FID / R-Precision / spread actually are**, why these and not others, and where each breaks. Every number in it was measured here. | 25 min |

---

## By what you are doing

| I am... | read |
|---|---|
| orienting from cold | [00_START_HERE.md](00_START_HERE.md) |
| about to write code | [LANDMINES.md](LANDMINES.md) -> [CODE_MAP.md](CODE_MAP.md) -> [DECISIONS.md](DECISIONS.md) |
| about to change a design | [DECISIONS.md](DECISIONS.md) — it was probably already considered |
| looking up a term | [GLOSSARY.md](GLOSSARY.md) |
| not sure what a metric *means* | [METRICS_EXPLAINED.md](METRICS_EXPLAINED.md) — the explanative version of the glossary |
| checking what has been measured | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) |
| resuming an interrupted run | [../LEDGER.md](../LEDGER.md), last incomplete item |
| running an unattended session | [../AUTONOMOUS_RUN_PROMPT.md](../AUTONOMOUS_RUN_PROMPT.md) |
| responding to a review | [REVIEW_RESPONSES.md](REVIEW_RESPONSES.md), citing finding IDs |
| judging whether a number is trustworthy | look for VERIFIED / UNVERIFIED. If unlabelled, treat as unverified. |

---

## Engineering documentation

| category | resource | description |
|---|---|---|
| **Agent rules** | [../CLAUDE.md](../CLAUDE.md) | loaded automatically. the five rules, file routing, ledger format, code standards |
| **Code** | [CODE_MAP.md](CODE_MAP.md) | target layout, dependency direction, what every file guarantees. **Planned, not built.** |
| **Decisions** | [DECISIONS.md](DECISIONS.md) | D-01..D-16, each with rejected alternatives and reversal conditions |
| **Warnings** | [LANDMINES.md](LANDMINES.md) | the traps, with evidence and the fix |
| **Terms** | [GLOSSARY.md](GLOSSARY.md) | every acronym expanded |

---

## Research documentation

| category | resource | description |
|---|---|---|
| **Results** | [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) | the primary research record. E-series, chronological, every number sourced. **Currently empty.** |
| **Diagnosis** | [../BRIEFING.md](../BRIEFING.md) | what the original project did and what is wrong with it |
| | [../FORENSICS.md](../FORENSICS.md) | Stage 1 verdicts *(pending)* |
| **Plan** | [../LANDSCAPE.md](../LANDSCAPE.md) | 2024-2026 field survey, every claim sourced *(Stage 2, pending)* |
| | [../REBUILD_SPEC.md](../REBUILD_SPEC.md) | the evaluation-first build plan and ablation ladder *(Stage 2, pending)* |
| **Process** | [../LEDGER.md](../LEDGER.md) | append-only run log + `OPEN_QUESTIONS` |
| | [../SESSION_REPORT.md](../SESSION_REPORT.md) | written at the end of each autonomous run *(pending)* |

---

## Territory — who may write where

| directory | owner | everyone else |
|---|---|---|
| [../reviews/](../reviews/) | reviewer agents | **read only** |
| [../guidance/](../guidance/) | collaborator agents | **read only** |
| `docs/` `src/` `tests/` `notebooks/` `scripts/` `artifacts/`, `LEDGER.md`, `README.md` | the producing agent | read only |

A reviewer that edits the work destroys the record that a defect ever existed. That record is
the signal worth keeping. There is no exception, not even for a typo.

---

## Reference material

| resource | what it is |
|---|---|
| [../primary_source/](../primary_source/) | files pulled from the official HumanML3D repo. **The authority on the 263-dim layout.** Never paraphrase it from memory — read it. |
| `<ARCHIVE>/` | the original project. **READ-ONLY.** 2 notebooks, the PDF report, 3 READMEs, the win-64 env yml, EDA artifacts |
| `Study_Notes_Obsd/091 AI Coursework - Project Deepdives/DL - T2P Deep Dive.md` | 86 KB of the original author's own notes: mental models, phase-by-phase narrative, troubleshooting guide |
| [../Knowledge_Docs/](../Knowledge_Docs/) | the colour-zoned Excel knowledge tier. **Built last** (D-16). Empty by design. |

---

## Answers to specific questions

| question | where |
|---|---|
| What is HumanML3D's 263-dim layout, exactly? | [LANDMINES.md](LANDMINES.md) §1, confirmed against [../primary_source/](../primary_source/) |
| Why did the original project fail? | [00_START_HERE.md](00_START_HERE.md) §1 |
| Was the 66-dim slice really wrong? | [../FORENSICS.md](../FORENSICS.md) F1 |
| Why not just fix the old notebooks? | [DECISIONS.md](DECISIONS.md) D-01 |
| Why build metrics before models? | [DECISIONS.md](DECISIONS.md) D-02, D-03 |
| Why is a loss curve not a result? | [LANDMINES.md](LANDMINES.md) §4 |
| Should we predict positions or rotations? | [DECISIONS.md](DECISIONS.md) D-11 — pending, strongly favoured |
| HumanML3D or PoseScript? | [DECISIONS.md](DECISIONS.md) D-12 — pending, decided in Stage 2 |
| Why does the model confuse left and right? | [LANDMINES.md](LANDMINES.md) §6 |
| Can I trust this number? | Check for VERIFIED / UNVERIFIED. If unlabelled, no. |
| What should I do next? | [../LEDGER.md](../LEDGER.md), last incomplete item, then the work queue in [../AUTONOMOUS_RUN_PROMPT.md](../AUTONOMOUS_RUN_PROMPT.md) §8 |
