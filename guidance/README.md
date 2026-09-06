> **Collaborator territory.** Written by research-collaborator agents. Read-only to the
> producing agent and to reviewers.

# Guidance

This directory holds *help*, not findings and not instructions: domain explainers, derivations,
literature synthesis on a narrow question, proposed experimental designs with confounds
pre-controlled, and worked answers to specific stuck points.

**The distinction that matters:** a reviewer judges what exists (`../reviews/`); a collaborator
supplies what is missing (here). A collaborator never appends to the review queue.

**How it gets used.** An autonomous run reads this directory before selecting work, weighs what
it finds, and decides. Guidance is advice — the producing agent may decline it, and says so in
`../docs/REVIEW_RESPONSES.md` if it was substantive.

**How it gets produced.** Autonomous runs write blocked items to `OPEN_QUESTIONS` at the bottom
of `../LEDGER.md`. Those are the natural input to a collaborator session.

---

*Empty. The first useful request will likely be the one named in `DECISIONS.md` D-11: an
argued case for rotation-space output with differentiable forward kinematics versus direct
position prediction, for a single static pose rather than a motion sequence.*
