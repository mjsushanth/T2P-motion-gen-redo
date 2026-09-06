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

*No reviews have been run yet. `../reviews/REVIEW_QUEUE.md` is empty.*
