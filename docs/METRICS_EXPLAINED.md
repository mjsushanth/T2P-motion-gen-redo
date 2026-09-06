# METRICS EXPLAINED — what these numbers are, why we use them, and where they break

> Written for someone who is capable but new to this corner of the field. Every number in this
> document is one this project actually measured. Read `00_START_HERE.md` first for orientation;
> this file is the "what does that number even mean" companion to `EXPERIMENT_LOG.md`.
>
> Companion files: `GLOSSARY.md` (one-line lookups), `LANDMINES.md` (the traps, with evidence).

---

## 0. The one idea everything else hangs on: a shared embedding space

Both headline metrics work the same way underneath. There is a **trained pair of encoders**:

```
   a caption  ──► [text encoder]   ──►  a 512-number vector
   a motion   ──► [motion encoder] ──►  a 512-number vector
```

They were trained together so that **a motion and its own caption land close together**, and
mismatched pairs land far apart. That is the whole trick. Once you have it:

- **"Does this motion match its text?"** becomes *are their two vectors close?* → **R-Precision**
- **"Do these motions look like real ones?"** becomes *do their vectors have the same
  distribution as real motions' vectors?* → **FID**

**Neither metric looks at motion directly.** They look at the encoder's opinion of it. That matters
enormously and we will come back to it.

**Which encoders?** Guo et al.'s, from `text-to-motion` — the same weights every paper in this
space uses. That is deliberate: a shared instrument is what makes numbers comparable between
papers. It is also why validating our copy of it mattered so much (§7).

---

## 1. The 263 numbers — and the arithmetic trap that cost the original project three months

HumanML3D stores each frame of motion as **263 floats**. Not arbitrary — a concatenation:

```
index        contents                                             size
---------    -------------------------------------------------    ----
[0]          root angular velocity (turning speed)                   1
[1:3]        root linear velocity (x, z) — how fast it travels       2
[3]          root height (y)                                         1
[4:67]       ric_data — local positions of the 21 NON-root joints   63   (21 x 3)
[67:193]     rot_data — 6D rotations for 21 joints                 126   (21 x 6)
[193:259]    local velocities, all 22 joints                        66   (22 x 3)
[259:263]    foot-contact flags (is each foot planted?)              4
                                                          total =  263
```

**Why "21" and not 22 for positions.** The root joint (pelvis) is the origin of the local frame —
its position relative to itself is always zero, so storing it would waste 3 numbers. The root's
information lives in the 4 root scalars instead. **This is the detail the whole disaster turned on.**

**The trap.** The original project assumed `[0:66]` was "22 joints × 3 coordinates." A natural guess.
It checked its layout hypothesis by summing:

```
   its guess:     66 positions + 66 velocities + 126 rotations + 5 other  = 263  ✓
   the truth:      4 root      + 63 ric        + 126 rot       + 66 vel + 4 feet = 263  ✓
```

**Both sum to 263.** The arithmetic agreed, so the guess was never questioned. But slicing `[0:66]`
grabs 4 root scalars plus 62 of the 63 position values — so "joint 0" is actually
`[turning speed, x-velocity, z-velocity]`, a velocity triple masquerading as a position, and every
joint after it is shifted by 4 numbers, splicing X/Y/Z across joint boundaries.

**How you catch this in ten minutes: use an invariant, never a sum.** A skeleton's bone lengths are
constant. Decode, measure the 21 bone lengths, look at their variation:

| decode | bone-length coefficient of variation |
|---|---|
| correct `ric` decode | **0.00008%** — constant to the floating-point noise floor |
| the original's `[:66]` | **25.8% average, up to 81.7%** |

The correct decode is *exactly* constant because HumanML3D retargets every subject onto one
canonical skeleton before encoding. **Bone lengths are a constant of the dataset** — which means
the original project's entire "anatomy loss" apparatus was machinery to recover a number the data
hands you for free.

> **The transferable rule:** every packed format has an invariant. Find it and test against it. A
> dimension count that adds up is not verification.

---

## 2. R-Precision — "can you find the right caption?"

### The mechanic

Take one generated motion. Encode it. Now take **32 candidate captions**: its own, plus **31
decoys** drawn from other samples. Encode all 32. Rank them by distance to the motion's vector.

- **top-1**: was the correct caption ranked #1?
- **top-3**: was it in the top 3?

Average over many motions. **top-3 is the one everyone reports.**

### Why 32, and why it is load-bearing

Chance performance is `k / N`:

```
   top-3 out of 32 candidates  =  3/32  =  0.0938   ← the number our power gate used
   top-3 out of 200 candidates =  3/200 =  0.015
```

**Change the pool size and you change the number, with no error raised.** This project made exactly
that mistake: an early run pooled 200 candidates instead of 32 and scored **0.280**. Re-batched to
32 — same data, same model, same code — it scored **0.710**. A 2.5× swing from a batching detail.

MDM's own evaluation script carries the line:
`args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!`

> **Rule: an R-Precision number without a stated pool size is meaningless.** Always quote chance
> alongside it. "0.2969, against a chance of 0.0938" is a sentence; "0.2969" is not.

### The reference points you should memorise

| what | top-3 | reading |
|---|---|---|
| chance (32 candidates) | **0.094** | a model that has learned nothing |
| our model, 3,000 steps | **0.297** | 3.2× chance — it learned *something* |
| MDM (published) | **0.611** | a properly trained model |
| real human motion | **0.797** | the ceiling; even real data isn't 1.0 |

**Why real data doesn't score 1.0.** Captions are ambiguous. "A person walks forward" describes
thousands of motions in the dataset. Some decoys genuinely match the motion as well as its own
caption does. **This ceiling is a property of language, not a flaw in the metric.**

### Where it breaks

At the frontier, it **saturates**. StableMoFusion scores 0.841 and MoMask 0.807 — both *above* real
data's 0.797. When generated beats real, the metric has no headroom left and cannot rank the top of
the field. It is excellent at 0.30 and useless at 0.80.

---

## 3. FID — "does this look like real motion?"

### What a Fréchet distance actually is

Take all your generated motions, encode them, and you have a cloud of points in 512 dimensions. Do
the same for real motions. **FID asks: how far apart are these two clouds?**

It answers by fitting a Gaussian (a bell curve, generalised to 512 dimensions) to each cloud, which
means summarising each with two things:

- **μ (mu)** — the mean. Where the cloud's centre sits. A 512-number vector.
- **Σ (Sigma)** — the covariance. The cloud's *shape*: how spread out, in which directions,
  and which dimensions move together. **A 512 × 512 matrix — 262,144 numbers.**

Then:

```
   FID  =  |μ₁ − μ₂|²   +   Tr( Σ₁ + Σ₂ − 2(Σ₁Σ₂)^½ )
            ^^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^^
            centres apart?     shapes different?
```

Lower is better. Identical distributions → 0.

**The critical asymmetry:** the mean is 512 numbers and easy to estimate. **The covariance is
262,144 numbers**, and that is where everything goes wrong.

### Why "FID between two disjoint halves of real data" is the test that exposes it

This is your question and it is the sharpest diagnostic in the whole document.

Split real data into two halves that share no samples — **disjoint**. Both halves are real motion,
drawn from the same distribution. So the true FID between them is **zero**.

Measure it and you learn *how much FID your measurement procedure invents when the answer is
known to be nothing.* It is a calibration test, and it is free.

Our results:

| samples per half | FID (true answer: ~0) |
|---|---|
| 200 | **0.745** |
| 1,024 | 0.173 |
| 2,099 | 0.029 |

**At n=200, real-vs-real scored 0.745 — larger than MDM's published FID of 0.544.** That entire
number was fabricated by the estimator. Nothing was wrong with the data.

### The mechanism: rank deficiency

Estimating a 512×512 covariance from n samples, the sample covariance has **rank at most n−1**.

```
   n = 128  →  rank ≤ 127  in a 512-dimensional space
```

**385 of the 512 directions have literally no information in them.** The matrix is not merely noisy;
it is degenerate. The `(Σ₁Σ₂)^½` term is then computed from a matrix that cannot represent the
space it claims to describe, and the result inflates.

**Rule of thumb: you want n comfortably larger than d. n ≈ d is severely under-sampled;
n < d is broken.**

### The demonstration that settles it

Hold the generated motions **bit-identical** — same 128 samples, byte for byte — and change only
what you compare them against:

| reference | FID |
|---|---|
| 128 real samples, draw A | **1.0731** |
| 128 real samples, draw B | **1.3997** |
| 4,640 real samples (full scale) | **3.2909** |

**Three values. Same motions. The model never changed.** If a number moves 3× while the thing it
measures is frozen, it is measuring the measurement.

Note the last row is *worse*, which is counter-intuitive: fixing the reference doesn't help, because
the **generated** side is still 128 and still rank-deficient. Pairing a well-conditioned reference
against a degenerate test covariance is its own instability. **Both sides need adequate n.**

### The diagnostic that separates bias from signal

R-Precision **does not estimate a covariance** — it is batch-wise retrieval over a fixed pool. So it
is insensitive to total sample count. That gives you a free test:

```
   sweep n.   whatever moves is estimator bias.
              whatever stays put is a real difference.
```

Over the same sweep, FID moved 6× while R-Precision moved 0.710 → 0.720. **That divergence proved
two apparent gaps had different causes** — one was a statistical artifact, the other was real.

---

## 4. Spread, noise floors, and why "the effect exceeds the spread" is the only honest criterion

### Where variation comes from

Run the identical configuration twice and you get different numbers, from:

- **which samples landed in which retrieval batch** (shuffling)
- **stochastic generation** (diffusion sampling from different noise)
- **training randomness** (initialisation, data order)
- **hardware non-determinism** (GPU float ordering; not an issue for us — we train on CPU)

**A "noise floor" is how much the number moves when nothing meaningful changed.** You measure it by
re-running with a different seed and watching the wobble.

### Why this is the whole ballgame

An effect smaller than the noise floor is not an effect. Our measured floors:

| comparison | floor |
|---|---|
| ground-truth R-Precision batching, n=128 | **±0.016** |
| the restricted-subset comparison | **±0.0117** |

Both real, both measured, **different because they came from different configurations.** We nearly
made the mistake of quoting one for the other — a floor measured on one setup does not transfer to
another. **State which floor you used.**

Now the numbers become readable:

| finding | effect | floor | verdict |
|---|---|---|---|
| caption truncation cost | 0.145 | 0.016 | **9× floor — real** |
| ...on captions actually altered | 0.272 | 0.016 | **17× floor — very real** |
| position (single draw) | 0.0117 | 0.0117 | **1× floor — cannot tell** |
| position (8 draws averaged) | 0.0177 | 0.0030 SEM | **5.9× — real** |

### The subtlety that reversed a conclusion: treatment variance vs sampling variance

Two arms, re-run with a new seed:

| arm | movement |
|---|---|
| length-matched prefix | **0.0008** |
| random window | **0.0124** |

Why the 15× difference? **The prefix is deterministic** — reseeding only reshuffles batches. **The
random window redraws the window placement**, so each seed is *a different experimental treatment*,
not a different sample of the same one.

Its 0.0124 wasn't measurement noise. It was the treatment itself varying. So a single draw couldn't
distinguish *"position doesn't matter"* from *"this particular set of placements happened to score
about as well."*

Averaging over 8 placements cut it by √8 and **reversed the finding** — from "no detectable effect"
to a confirmed 5.9σ result.

> **Rule: when one arm of a comparison is itself stochastic, its across-seed spread is treatment
> variance and must be averaged down before the arms are comparable.**

### Reading σ (sigma)

`gap ÷ standard error`. Roughly: 1σ is noise, 2σ is suggestive, 3σ+ is solid, 5σ+ is
"physicists call this a discovery." Our position effect at 5.9σ is genuinely resolved — but note it
is 5.9σ on an effect of **0.018**, versus a volume effect of **0.27**. **Statistical significance
and practical importance are different questions.** Position is certainly real and mostly
unimportant.

---

## 5. The other three metrics, briefly

| metric | question | direction |
|---|---|---|
| **MM-Dist** (matching score) | average distance between a motion and its own caption | lower better |
| **Diversity** | spread across motions from *different* prompts | match real data |
| **MultiModality** | spread across motions from the *same* prompt | higher = the model represents a distribution, not one answer |

**Diversity is not "higher is better."** Real motion has a specific diversity (~9.5 here). Scoring
*above* it means generating implausible variety — noise reads as diverse.

**MultiModality catches mode collapse.** Ask for "a person walks" ten times; if you get the same
motion each time, the model learned one answer to a question with many.

**MM-Dist is R-Precision's continuous cousin** — same embedding space, but the actual distance
rather than a rank. Useful because it moves when R-Precision is saturated or coarse. Ours
corroborated the truncation finding independently: 2.985 → 3.895.

---

## 6. Why *these* metrics and not something obvious

**Why not "does it look right"?** Doesn't scale, isn't reproducible, and the failures that matter
are subtle. But note: **the demonstrator we designed is precisely a return to human judgement** —
because for *communicating* a result, "watch this person move" beats any table.

**Why not per-joint position error (MPJPE)?** It needs a single correct answer to compare against.
"A person walks forward" has thousands of correct answers. **MPJPE is right for pose *estimation*
(recover the pose in this video — one truth) and wrong for pose *generation* (invent a plausible
one — many truths).** Understanding why is worth more than either metric.

**Why not training loss?** This is the one that killed your original project, so it gets its own
section.

---

## 7. Why training loss is not a quality metric — with today's proof

Diffusion training loss measures *"can the model predict the noise added at a randomly chosen
timestep?"* Useful for optimisation, and it does track learning.

**It does not track output quality**, and this project produced an unusually clean demonstration.

Our first trained model:

| measurement | value | reading |
|---|---|---|
| final training loss | 0.19 | vs converged 0.056, init ~1.2 → **~60% of the way to convergence** |
| FID | **7.209** | **13× worse than MDM's 0.544** |

**A loss curve 60% of the way to convergence, on a model that is distributionally poor** — reached
at 0.63% of MDM's training budget.

There is a second, worse mechanism. A diffusion model can drive loss down substantially by learning
the **unconditional** distribution — how bodies move in general — while ignoring the text entirely.
**Loss falling is evidence of training. It is not evidence of conditioning.**

That is exactly what happened in the original project: its guidance formula was folded into the
training objective, creating a loss that could be driven to zero by a model that never read the
caption. The loss fell beautifully. That was the problem.

> **The rule this project is organised around:** a number is a result only when it comes off a
> validated evaluation harness on held-out data. Everything else is a diagnostic.

---

## 8. The limitation that qualifies every number here

**All of this measures the encoder's opinion.** R-Precision and FID both operate on 512-number
summaries produced by a model trained on this dataset. So:

- A motion that fools the encoder scores well whether or not a human would accept it.
- Failure modes the encoder is blind to are invisible to both metrics.
- Comparisons are only valid **between systems evaluated with the same encoder weights** — which is
  why the field standardised on one set, and why we vendored exactly that set.

And specific to this project: **our harness reproduces ground truth to within 0.003 across five
independent runs, but did not reproduce a published *generated-model* number at affordable sample
sizes.** So every number here carries the label **internally-comparable-only** — trustworthy
against our own other numbers, not against the published leaderboard.

Saying so is not a weakness in the work. Not saying so would be.
