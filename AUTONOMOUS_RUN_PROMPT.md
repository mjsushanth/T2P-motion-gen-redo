# Autonomous Run Prompt — T2P-Reboot

**Purpose:** a self-contained instruction set for an unattended, multi-hour research run.

**The deliverable is a measured research record.** Not a scratchpad, not a model checkpoint, not
a demo. Executed notebooks, tested modules, and markdown that a newcomer to human-pose
generation can follow cold — where every number states the split it came from and every claim
is labelled VERIFIED or UNVERIFIED.

**The one sentence that governs this run:** the project this replaces failed by producing three
months of work and zero measurements, so **nothing here counts as a result until it comes off a
validated evaluation harness on a held-out split.**

---

## HOW TO LAUNCH (for the human — do not paste this section)

Open a fresh session **in this directory**, set permissions to accept-edits / auto mode, then:

```
/loop Follow the instruction set in AUTONOMOUS_RUN_PROMPT.md in this repo. Read LEDGER.md first and resume from the last incomplete item.
```

- No interval means the model self-paces. Add one (`/loop 25m ...`) to force regular re-firing.
- Each firing re-reads the ledger and continues. Compaction mid-run is expected and harmless —
  **the ledger is the memory.**

Everything below the line is the instruction set.

---

# INSTRUCTION SET

## 0. Mode declaration

**This is an AUTONOMOUS run. I am away. Do not wait for me. Never end a turn with a question
directed at me.**

My global `CLAUDE.md` Attended-mode gates — "discuss before writing code", "one logical chunk at
a time, stop and wait for feedback", "confirm scope before editing" — are **SUSPENDED for this
run**.

**Still in force, without exception:**
- **No git commits.** Ever, without me.
- **No installs without asking** — `conda create`, `pip install`, `uv pip install`, `brew
  install`. Write the environment file, print the exact command, log it under `OPEN_QUESTIONS`,
  and *work around it* rather than idling. Using an already-existing environment is fine.
- **`../<ARCHIVE>/` is read-only.** Never edit, move, rename or delete anything under it.
- No deleting or overwriting anything outside this project folder.
- Absolute imports only — never `sys.path` manipulation, never `parent.parent` chains.
- No emojis in code. Python: 4-space indent, type hints on every function signature.
- Read the `python-module` skill before creating any real Python module.
- Push back on over-engineering. Explicit and controllable beats clever.
- Nothing sent externally, no purchases, no account changes, no posting.
- Do **not** write memories about my personal situation, timeline, or study pressure.

## 1. Project context

```
Root:      /Users/joel/MJS_ROOT/MJS_STUDY/T2P-Reboot/
Archive:   ../<ARCHIVE>/       READ-ONLY. the project being rebuilt.
Notes:     ~/Library/Mobile Documents/iCloud~md~obsidian/Documents/Study_Notes_Obsd/
             091 AI Coursework - Project Deepdives/DL - T2P Deep Dive.md   (86 KB, the
             original author's own notes — mental models, phase narrative, troubleshooting)
Authority: primary_source/        official HumanML3D repo files. THE source on the 263 layout.
```

**Domain:** text-conditioned human pose and motion generation. Diffusion models, DDPM,
classifier-free guidance, cross-attention conditioning, CLIP and alternative text encoders,
SMPL / SMPL-X body models, forward kinematics, HumanML3D, AMASS, PoseScript, PoseFix, and the
standard T2M evaluation protocol (R-Precision, FID, MM-Dist, Diversity, MultiModality).

**Read before doing anything:** `docs/00_START_HERE.md`, then `docs/LANDMINES.md`. They take
ten minutes together and they prevent this domain's characteristic failure — a plausible wrong
number that raises no error.

**Zero Excel work this run.** `Knowledge_Docs/` is built last (`docs/DECISIONS.md` D-16).

## 2. Environment

**You may use any environment that already exists on this machine without asking.** Check first:
`conda env list`, and test-import what you need before assuming.

**You may not create or install into one without my approval.** When you need something absent:
1. Write the full `environment.yml` (mamba solver, macOS ARM) or `pyproject.toml`.
2. Print the exact create/install command.
3. Log it under `OPEN_QUESTIONS` in `LEDGER.md`.
4. **Then keep working on something that does not need it.** Never idle waiting for me.

**Hardware, non-negotiable:** Apple Silicon, ~24 GB unified memory, **no CUDA, no NVIDIA GPU**.
Use `torch` with MPS (`torch.backends.mps.is_available()`). Never write CUDA-assuming code.
`PYTORCH_ENABLE_MPS_FALLBACK=1` where useful. **MPS is non-deterministic** — see
`docs/LANDMINES.md` §7; every comparison needs multiple seeds and a reported spread. Bound
memory-hungry steps; if something would exceed ~12 GB, subsample and say so.

Log every environment change in the ledger — what you used, what broke, what you rebuilt.

## 3. Data policy — measure before you download

**Total budget for this run: ~15 GB.** Always establish the real size *before* pulling anything
(HTTP `HEAD`, the HuggingFace metadata API, or the loader's documented size).

| measured size | action |
|---|---|
| **< 1 GB** | pull in full |
| **1 – 5 GB** | pull in full if primary; otherwise take the official test/val split. Log the choice. |
| **> 5 GB** | official subset, or N capped samples, or streaming. Log the subsetting decision **and what it costs analytically.** |
| **gated / licence-walled** | try documented mirrors. If genuinely blocked, log under `OPEN_QUESTIONS` and proceed with clearly-labelled synthetic data so the code still executes. |

**SMPL / SMPL-X are licence-gated and require registration.** Do not attempt to circumvent that.
If body-model files are needed and absent, say so in `OPEN_QUESTIONS` with the registration URL
and build everything that does not depend on them.

**Provenance is mandatory.** `data/README.md` records, for every artifact: source URL, licence,
access date, size on disk, and real vs synthetic. Never commit data.

**Facts to verify rather than assume:** HumanML3D's official statistics, PoseScript's size and
licence, and the exact evaluation protocol everyone uses. All three are things this project has
been burned by believing without checking.

## 4. THE DELIVERABLE STANDARD

**Primary assets: executed `.ipynb` notebooks, tested `src/t2p/` modules, and `.md` writeups.**

**A notebook is done only when all of these hold:**

1. It **executes top-to-bottom without error** in a fresh kernel
   (`jupyter nbconvert --to notebook --execute --inplace`).
2. **Outputs are saved in the `.ipynb`** — printed shapes, metrics, rendered figures. An
   unexecuted notebook is not a deliverable.
3. It opens with a header: what it teaches, what you need to know first, expected runtime, and
   data provenance (real vs synthetic).
4. It closes with **"What we learned"** and **"Open questions / next experiments."**
5. Reusable logic lives in `src/t2p/` and is **imported**, not pasted. Notebooks are narrative;
   modules are machinery. (`docs/DECISIONS.md` D-15 — the predecessor's 101,569-character cell
   is why this rule exists.)
6. Numeric results are **asserted**, not eyeballed — shape checks, range checks, at least one
   correctness assertion per core computation.
7. It obeys the Explain-Down Principle (§5).

## 5. The Explain-Down Principle — write for a reader who knows nothing

**Assume I do not know what a diffusion model is, what forward kinematics computes, what SMPL
parameterises, or what any acronym stands for.** I am a capable engineer and I am relearning
this domain from the bottom. This project exists partly so I can finally *hold* these ideas.

For **every substantial or clever code cell**, sandwich it:

- **ABOVE (markdown):** plain-language statement of what we are about to do and *why* it
  matters. No jargon that has not been defined yet.
- **THE CELL:** the real, serious, top-level code. Do not dumb it down.
- **BELOW (1-2 cells, titled "Unpacking this"):** bring it back to earth. Pick what fits:
  - a **toy example** — the same operation on a hand-written 3-joint skeleton or a 4x4 array,
    printed, so the mechanics are visible;
  - an **acronym / symbol decode** — every term spelled out in one line;
  - a **"why this formula"** — the intuition, not the derivation;
  - a **failure mode** — what it looks like when it goes wrong, demonstrated.

Non-negotiable: **the first time any acronym or domain term appears**, expand and define it, and
mirror the definition into `docs/GLOSSARY.md`, which is a first-class deliverable you maintain
continuously. DDPM, CFG, FK, ric, rot6d, SMPL, AMASS, MPJPE, PA-MPJPE, FID, MM-Dist,
R-Precision, MultiModality, mode collapse — all of it.

**The four ideas I most need to actually understand**, so give them the most care:
1. **Why a UNet and a transformer are awkward together**, and what cross-attention is really
   doing between a pose vector and a text embedding.
2. **What the diffusion forward and reverse processes are**, concretely, on a 66-dim vector —
   not the image analogy.
3. **Classifier-free guidance as geometry** — why `uncond + w*(cond - uncond)` is a direction,
   and what raising `w` trades away.
4. **Rotation space vs position space**, and why forward kinematics makes anatomy free.

## 6. Text-visual conventions

Use ASCII diagrams liberally. They survive any viewer and force clarity.

```
   The kinematic tree (why FK is the right output space)

              15 head
               |
              12 neck ----+---- 16 L.collar -- 17 L.shoulder -- 19 L.elbow -- 21 L.wrist
               |          |
               9 spine3   +---- 13 R.collar -- 14 R.shoulder -- 18 R.elbow -- 20 R.wrist
               |
               6 spine2
               |
               3 spine1
               |
               0 PELVIS --+-- 1 L.hip -- 4 L.knee -- 7 L.ankle -- 10 L.foot
                          |
                          +-- 2 R.hip -- 5 R.knee -- 8 R.ankle -- 11 R.foot

   Predict ROTATIONS at each node and walk the tree outward: bone lengths are
   fixed inputs, so they come out exactly right. Predict POSITIONS instead and
   every bone length is a number the network has to get right by luck.

   NOTE: joint indices differ between SMPL and HumanML3D orderings. VERIFY the
   ordering against primary_source/ before using this diagram in code.
```

Also welcome: pipeline flows (`caption -> encode -> denoise x T -> decode -> metric`), printed
toy arrays, before/after tables. Keep real matplotlib figures too — text visuals supplement,
never replace, actual rendered poses.

## 7. Repository layout

Target shape is in `docs/CODE_MAP.md`. Create directories as you need them, not up front.

## 8. Work queue — stage-gated

**Each stage is gated on the previous stage's verdict. Do not skip forward.**

### Stage 1 — Forensics `-> FORENSICS.md`
If `FORENSICS.md` does not exist, run it: verify or refute findings F1-F5 in `BRIEFING.md`
empirically, using the bone-length invariant test in `docs/LANDMINES.md` §2. Report
VERIFIED / REFUTED / INCONCLUSIVE per finding with the evidence that decided it. **If a finding
is refuted, say so prominently and update `docs/LANDMINES.md` — do not soften it.**

### Stage 2 — Landscape and spec `-> LANDSCAPE.md`, `REBUILD_SPEC.md`
1. **Field survey**, every claim against a primary source, every URL actually fetched:
   the current HumanML3D evaluation protocol and which evaluator code everyone uses; published
   numbers for MDM, MotionDiffuse, MLD, T2M-GPT, MoMask, StableMoFusion, MotionLCM and anything
   newer; the static-pose line (PoseScript, PoseFix, PoseEmbroider, LLM-to-SMPL work); SMPL /
   SMPL-X licence terms; text encoders for spatial and lateral language; and the
   text -> pose -> ControlNet route.
2. **The rebuild spec.** Resolve `docs/DECISIONS.md` D-11, D-12, D-13 with arguments, not
   assertions. Evaluation-first. Include the ablation ladder — each rung a **testable
   hypothesis with a pre-registered success criterion and the metric that decides it**. Include
   a compute estimate (what runs on MPS, what needs a rented GPU, hours and cost). Include an
   honest risk register with kill criteria.
3. **Positioning** `-> POSITIONING.md`. **A first-class deliverable, not an appendix.** Joel's
   framing: *"the whole agenda is a proper redo, full redo, it might involve full business idea
   changes, research changes, architecture changes."* Treat the business question as real work.
   Cover: what this is worth as a learning artifact, an interview artifact, and a **product**;
   who would use a text-to-pose system and for what; where pose sits as a *control signal* for
   downstream image/video/animation generation; what the smallest demonstrable end-to-end thing
   is that a non-expert can look at and immediately understand the value of; and what would have
   to be true for this to be worth more than a portfolio piece. **If a track genuinely has no
   business case, write that in those words** — but establish it, do not assume it.

### Stage 3 — Harness and baseline `-> src/t2p/`, `tests/`, `RESULTS.md`
1. Scaffold the package per `docs/CODE_MAP.md`. Typed config, global seeding, structured
   logging, CLI entry point. `skeleton.py` is the single source of truth for the kinematic tree.
2. **Build the evaluation harness FIRST and validate it** against a published number to a stated
   tolerance (`docs/DECISIONS.md` D-03). **This is the gate.** If it cannot be met, document
   exactly why and downgrade every subsequent number to internally-comparable-only, loudly.
3. Tests for the **silent** failure modes: decode round-trips, bone-length invariants,
   normalise/denormalise inverses, skeleton connectivity, metrics against known fixtures.
   Include at least one deliberately failing tripwire, shown failing, to prove the tests
   discriminate.
4. Port the original's three phases as reproducible ablation configs (D-14). Phase 1 must still
   run — reproducing the original failure *with correct instrumentation* turns "we had a bug"
   into a measured delta.
5. One small end-to-end run on the corrected pipeline. **Real metrics, not loss curves.**

### Stage 4 — Experiments `-> docs/EXPERIMENT_LOG.md`
Work the ablation ladder from `REBUILD_SPEC.md`. One rung per item. **Pre-register the
hypothesis and the success criterion in the ledger before running**, then report what actually
happened — including when the hypothesis is falsified. A falsified pre-registered prediction is
one of the most valuable things this project can produce; write it up as a finding, not a
failure.

### If the queue empties
Go **deeper, not wider**: more seeds on an existing comparison, a laterality-specific metric, a
cross-dataset generalisation test, or the text -> pose -> ControlNet demonstrator. Log what you
chose and why. Only at the very end, and only if everything else is done, start
`Knowledge_Docs/`.

## 9. Per-item loop (criteria -> produce -> critique -> revise -> verify -> log)

**Before selecting any item, check for review and guidance input.**

- If `reviews/REVIEW_QUEUE.md` has content, read it first. Consume every open **P0** before
  starting new queue work, then **P1**, then resume §8. A P0 means something currently written
  down as true is wrong, and building on it compounds the error.
- If `guidance/` has notes you have not acted on, read them. They are help, not instructions —
  weigh them and decide.
- Record what you did against **every** finding ID you consumed in `docs/REVIEW_RESPONSES.md`,
  including declines: *"RD-20260906-03 — declined, because the confound it assumes is already
  ruled out in E004."* Declining with a stated reason is valid. Silently skipping is not.
- **Never edit anything in `reviews/` or `guidance/`.** Reply in your own files.

Then, for **every** item:

1. **Criteria** — write acceptance criteria into the ledger *before* producing anything.
2. **Produce** — build it.
3. **Self-critique** — re-read as a hostile reviewer. List concrete defects: unverified claims,
   unhandled edge cases, silent failures, undefined jargon, missing explain-down cells, a metric
   with no stated split. **"Looks good" is a failed critique.**
4. **Revise** — fix them. Max 3 rounds, then ship with the remaining defects recorded honestly.
5. **Verify empirically** — execute the notebook, run the tests, fetch the URL, render the pose.
   **Never claim something works that you did not observe working.**
6. **Log** — append a ledger entry (§11).
7. **Next** — move on immediately.

## 10. When you hit flawed or broken code

Diagnose root cause, fix, log the diagnosis alongside the fix. Do not stop to ask. If a fix is
ambiguous or risky, implement the **conservative** version and record the riskier alternative
under `OPEN_QUESTIONS`. If something is broken beyond safe unattended repair, isolate it, mark
it `# BROKEN:` with an explanation, and continue — never let one bad file halt the run.

## 11. Durability and the ledger

Assume you will be interrupted, compacted, or restarted at any moment.

- Write artifacts to disk **as you go**. Never hold finished work only in conversation.
- `LEDGER.md` is **append-only**, newest above `OPEN_QUESTIONS`. Entry format is in
  `CLAUDE.md`. One entry per item.
- **Begin every firing by reading `LEDGER.md`** and resuming the last incomplete item rather
  than restarting.
- Every experiment writes three things: JSON to `artifacts/`, an entry in
  `docs/EXPERIMENT_LOG.md`, and an entry in `LEDGER.md`. All three, every time.
- Keep `OPEN_QUESTIONS` current at the bottom of the ledger — anything needing my judgement,
  ranked by how much it blocks progress.

## 12. Epistemics — the highest-risk area unattended

- **Never fabricate** a URL, citation, author, benchmark number, dataset size, licence or API
  signature. Fabricated citations are the worst failure mode of an unattended run.
- Any URL you cite must be one you actually **fetched** and confirmed resolves.
- Label every factual claim **VERIFIED** or **UNVERIFIED**.
- **A falling loss is not evidence of anything.** `docs/LANDMINES.md` §4.
- A passing test proves *preservation*, not *understanding*. Do not overclaim from green.
- **Validate a format hypothesis with an invariant, never with a sum.** `docs/LANDMINES.md` §2.
  This is the specific error that cost the predecessor three months.
- Verify library APIs against current docs (Context7 / fetch), not memory. `torch` MPS,
  `diffusers`, `transformers`, `smplx` and the HumanML3D tooling all move.
- Prefer "unknown, needs checking" over a plausible guess. **An admitted gap is far more useful
  to me than a confident wrong number.**
- Subset numbers are subset numbers. Laptop-scale numbers are laptop-scale numbers. Say so
  every time, in the same sentence as the number.

## 13. Guardrails (restated — these do not relax as the run gets long)

- No git commits, branches, or pushes.
- No installs without asking (§2). Existing environments are fine.
- `../<ARCHIVE>/` is read-only. Nothing outside this project folder is deleted or
  overwritten.
- Stay within the ~15 GB data budget. Do not circumvent licence gates.
- Nothing sent externally, no purchases, no account changes, no posting.
- No personal-situation memories.

## 14. End of run

Write / overwrite `SESSION_REPORT.md`:

- What completed, with file paths, in **suggested reading order for me**.
- **Verified vs unverified split** — explicit and honest about what you could not check.
- Which notebooks execute cleanly and their runtimes; which do not, and why.
- Environment state: what was used, how to activate it, anything fragile, anything you needed
  and could not install.
- Data provenance summary: what is real (with sizes and licences), what is synthetic.
- Results table with honest caveats, or an explicit statement that nothing was measured yet.
- `OPEN_QUESTIONS` ranked by how much each blocks progress.
- The exact next action for the following run.
- **Anything you would have asked me if I had been available.**
