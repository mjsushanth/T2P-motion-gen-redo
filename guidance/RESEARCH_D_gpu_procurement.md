# Research D: GPU Procurement for a $10-30 Budget

**Question under test:** for a 17.9M-param transformer (+ frozen CLIP ViT-B/32) trained on
HumanML3D (4.36 GB), batch 32-64, seq len 196 — where measured on the researcher's own Apple
M5 Pro at 0.231 s/training-step and 3.15 s/generated-sample (MPS) — what is the cheapest
reliable way to rent GPU time in September 2026, and is renting even necessary?

**Epistemic key:** VERIFIED = I fetched the page directly and am reporting what it contained.
UNVERIFIED = surfaced only via a web-search snippet/aggregator I did not independently fetch.
Every price below carries one of these tags plus a URL and the date I observed it. Aggregator
sites (getdeploying, gpuperhour, gpus.io, synpixcloud, computeprices, etc.) scrape marketplace
listings and go stale fast — flagged explicitly where a number looked aggregator-derived rather
than pulled from the provider's own page.

---

## 1. Is Vast.ai actually cheapest in 2026?

**Price table, per-GPU-hour, September 2026:**

| Provider | GPU | On-demand | Spot/interruptible | Source | Status |
|---|---|---|---|---|---|
| Vast.ai | RTX 4090 | $0.30/hr | $0.08/hr | [getdeploying.com/gpus/nvidia-rtx-4090](https://getdeploying.com/gpus/nvidia-rtx-4090) (dated "Sep 7, 2026" on page) | VERIFIED (fetched, aggregator) |
| Vast.ai | RTX 3090 | listed "from $0.07/hr" on GPU-specific page, but the number did not render in my fetch | — | [vast.ai/pricing/gpu/RTX-3090](https://vast.ai/pricing/gpu/RTX-3090) | UNVERIFIED (page fetch returned no live figure — live prices load via JS) |
| RunPod | RTX 4090 | $0.34/hr (Community) / $0.74/hr (Secure) | ~$0.29-0.44/hr (aggregator estimate, RunPod does not publish a spot rate) | [runpod.io/pricing](https://www.runpod.io/pricing) | VERIFIED (fetched directly, official page) for on-demand; spot is UNVERIFIED |
| RunPod | RTX 3090 | $0.22/hr (Community) / $0.50/hr (Secure) | — | same | VERIFIED |
| RunPod | L4 | $0.44/hr (Community) / $0.49/hr (Secure) | — | same | VERIFIED |
| Lambda Labs | cheapest GPU (RTX 6000 Ada) | $0.69/hr | none — Lambda has no spot tier | [Lambda pricing coverage](https://gpuperhour.com/providers/lambda-labs) | UNVERIFIED (aggregator); Lambda does not list consumer cards (3090/4090/T4) at all — their floor is enterprise-class |
| Paperspace | A4000 | ~$0.76/hr | none advertised | [Spheron summary of Paperspace pricing](https://www.spheron.network/blog/paperspace-pricing-2026/) | UNVERIFIED (aggregator) |
| TensorDock | RTX 4090 | "from $0.35/hr" | not shown | [tensordock.com](https://www.tensordock.com/) | VERIFIED (fetched directly) — note the site's own marketplace listings can go lower (aggregators cite $0.12-0.46/hr range for TensorDock's full listing spread) |
| Hyperstack | RTX A6000 | $0.50/hr | $0.40/hr spot | [hyperstack.cloud/gpu-pricing](https://www.hyperstack.cloud/gpu-pricing) | VERIFIED (fetched directly) — no consumer cards (3090/4090/T4/L4) listed at all |
| Together AI | H100 (cheapest listed tier) | $2.55-5.49/hr | none consumer-relevant | [cloudzero.com Together AI pricing](https://www.cloudzero.com/blog/together-ai-pricing/) | UNVERIFIED (aggregator) — no GPU below H100-class; irrelevant to this budget |
| Modal | T4 | ~$0.164/hr ($0.000164/sec) | n/a (serverless, no spot concept) | [Spheron summary of Modal pricing](https://www.spheron.network/blog/modal-gpu-pricing-2026-per-second-billing/) | UNVERIFIED (aggregator); Modal gives **$30/month free credit** on every account per the same source |
| Google Colab Pro+ | A100 40GB (via compute units) | $49.99/mo for 500 CU; A100 burns ~5.4 CU/hr -> ~92 GPU-hrs/mo effective | n/a | [aicoolies.com Colab pricing](https://aicoolies.com/pricing/google-colab) | UNVERIFIED (aggregator) |
| Kaggle | P100 16GB / T4x2 | **free**, 30 GPU-hours/week | n/a | [Kaggle weekly usage thread](https://www.kaggle.com/general/108481) | UNVERIFIED (community post, but consistent across multiple sources) |

**Answer: no, not cleanly.** Vast.ai's headline on-demand number ($0.30/hr for a 4090) is
*not* the market floor once you count RunPod Community ($0.34/hr, close behind) and
TensorDock ("from $0.35/hr"). Vast.ai's real edge is its **interruptible/spot tier**
($0.08/hr for a 4090 per the getdeploying table above) — nobody else consistently
undercuts that for consumer cards. But that number comes from a scraped aggregator, not
Vast.ai's own JS-rendered pricing page (which I could not extract live figures from — see
row 2). Treat the $0.08/hr figure as directionally right (Vast.ai's marketplace model with
unverified hobbyist hosts genuinely produces sub-$0.10 listings) but not load-bearing to
the cent.

**The free tiers are the actual headline finding.** Kaggle's 30 free GPU-hours/week
(P100 or dual T4) and Modal's $30/month free credit are both large relative to a
$10-30 budget for a model this small. See §6/recommendation.

---

## 2. Which GPU is right for THIS workload?

**Core finding: for a 17.9M-param model at batch 32-64, seq-len 196, the bottleneck is
almost certainly kernel-launch/Python-loop overhead, not GPU memory bandwidth or raw FLOPs.**
Multiple sources on small-transformer training corroborate this: "kernel launch overhead
becomes increasingly problematic with smaller batch sizes and smaller model sizes,"
and per-kernel launch overhead dominates specifically when "there is limited work per
kernel" (UNVERIFIED, WebSearch synthesis over ["LLM GPU & Training Terminology" (2026)](https://nandigamharikrishna.substack.com/p/llm-gpu-and-training-terminology)
and related sources — no single primary paper isolated the exact crossover point for a
model this size, so treat the mechanism as directionally correct rather than numerically
proven).

Practical implication: an RTX 4090 will very likely **not** run this model meaningfully
faster than an RTX 3090, A4000, L4, or even a T4 — the GPU spends most of its time idle
between tiny kernel launches rather than saturated on compute. This matches the plain
economics: **a 4090 is overkill.** The right frame is $/GPU-hour x hours-to-converge, and
for a model this small the "hours-to-converge" term is nearly constant across these GPU
tiers (all have more compute and bandwidth than 17.9M params can use at batch 64/seq 196),
so the cheaper GPU wins almost by default.

- **3090** ($0.07-0.22/hr depending on source/tier) is the best-supported choice: 24GB VRAM
  (more than this model will ever touch), CUDA compute capability sufficient for any
  packages the model needs, and consistently the cheapest "real" consumer card across
  every provider's table above.
- **A4000** (16GB, ~$0.06-0.25/hr per [getdeploying.com/gpus/nvidia-a4000](https://getdeploying.com/gpus/nvidia-a4000), UNVERIFIED aggregator) is a fine second choice, slightly cheaper on some marketplaces, less VRAM headroom but still far more than 17.9M params + CLIP ViT-B/32 needs.
- **L4** ($0.44-0.80/hr) is priced for its inference-serving efficiency (int8/fp8, media
  encode/decode) that this training workload does not use — worse value here than 3090/A4000.
- **T4** (~$0.16-0.59/hr depending on provider, plus free on Kaggle/Colab) is older and
  slower per-core, but since the workload is overhead-bound rather than compute-bound, a
  free T4 (Kaggle) likely finishes a training run in not-much-more wall-clock time than a
  paid 4090, at zero dollar cost.

**Would a cheaper GPU with more hours beat a fast one with fewer?** For this specific
workload: yes, very likely, because the "fast" GPU's advantage is mostly wasted on a model
this small. The exception is if per-step time scales with host/PCIe/driver overhead that
varies more by *provider quality* than by *GPU tier* — which is the actual axis worth
optimizing (see gotchas, §5), not GPU-tier compute.

---

## 3. Spot/interruptible vs on-demand

**Interruption rates (VERIFIED, fetched directly), September 2026, from
[thundercompute.com/blog/cloud-gpu-spot-instance-availability](https://www.thundercompute.com/blog/cloud-gpu-spot-instance-availability)
(published Sep 1, 2026, trailing-30-day data):**

| Provider/GPU class | Interruption rate | Warning window |
|---|---|---|
| AWS H100 (p5) | <5% | 2 minutes |
| AWS A100 (p4d) | 15-20% | 2 minutes |
| AWS V100 (p3) | >20% | 2 minutes |
| GCP A100 | ~2.3%/hr | 30 seconds |
| GCP H100 | ~4.1%/hr | 30 seconds |
| RunPod Community Cloud (spot) | not quantified as a % — "<5 min notice" | 30 sec-5 min |
| Vast.ai | "variable by host," no % given | 15-second interruption notice per other sources (UNVERIFIED) |

RunPod specifically: **5-second SIGTERM before SIGKILL** on spot pod termination
(VERIFIED-adjacent — consistently reported across multiple RunPod-authored and
third-party sources, e.g. [runpod.io/blog/spot-vs-on-demand-instances-runpod](https://www.runpod.io/blog/spot-vs-on-demand-instances-runpod), UNVERIFIED as I did not fetch this specific page directly).

**Is checkpoint-resume worth it at these prices?** Given the on-demand/spot price gap is
roughly 2-4x (e.g. RunPod 4090 $0.34 vs ~$0.30 spot estimate is *not* a good example — the
real gap shows up on Vast.ai, $0.30 on-demand vs $0.08 spot per the table in §1, a ~3.75x
difference) and the *dollar amounts at this project's scale are tiny* (a few dollars total),
the arithmetic favors spot **only if checkpointing is already good practice regardless**
(which it should be, interruption or not — power blips, laptop-side crashes, etc. are also
risks). One important asymmetry: the interruption/re-provision cycle costs *wall-clock time*
(re-schedule, re-pull image, re-download dataset) that a solo researcher on a tight personal
schedule may value more than the few dollars saved. Recommendation: use spot, but only if
checkpointing every few minutes to a network volume that survives pod destruction (see §6)
— the dollar savings are real but small; the time cost of a bad interruption is the bigger
risk for a project this size.

---

## 4. The hidden costs — where small budgets leak

- **Storage while stopped**: on both Vast.ai and RunPod, **disk/storage billing continues
  after you stop an instance** — only *destroying* it (not just stopping/pausing) ends the
  charge. RunPod: Volume Disk (idle) **$0.20/GB/month**; Network Storage **$0.07/GB/GB/month
  under 1TB** (VERIFIED, fetched directly from [runpod.io/pricing](https://www.runpod.io/pricing)).
  Vast.ai: reported **$0.10-0.15/GB/month** for stopped-instance disk (UNVERIFIED,
  aggregator synthesis, not confirmed on Vast.ai's own JS-rendered page). For a 4.36 GB
  dataset plus checkpoints, this is cents/week, not a real risk at this project's scale —
  but the mechanism ("stopped is not free, destroyed is free") is the actual trap: leaving
  a stopped instance around for a week between "let me pick this up later" sessions is
  the classic way a $10 budget becomes $15.
- **Egress fees**: near-zero risk here. RunPod charges nothing for data egress (VERIFIED,
  fetched directly). Vast.ai/TensorDock (marketplace models) generally don't charge egress
  either (UNVERIFIED, consistent across aggregator sources); the classic egress trap
  ($0.08-0.12/GB on AWS/GCP/Azure) simply does not apply to the neocloud providers this
  project should be using.
- **Minimum billing increments**: Vast.ai and RunPod both advertise **per-second billing,
  no minimum hours** (VERIFIED for Vast.ai's stated policy on its own pricing page,
  [vast.ai/pricing](https://vast.ai/pricing); VERIFIED for RunPod, "bills per second on
  every Pod tier" per multiple sources). This matters for a workload this small — a 20-minute
  smoke test costs 20 minutes, not a rounded-up hour.
- **Setup/image-pull time billed**: **yes, you pay for it.** Time spent pulling a large
  Docker image, waiting for CUDA/PyTorch container extraction, and downloading the 4.36 GB
  HumanML3D dataset from HuggingFace all bills as wall-clock instance time on both
  providers. Vast.ai mitigates this somewhat via host-side Docker layer caching for popular
  base images (nvidia/cuda, common PyTorch images) — VERIFIED via
  [docs.vast.ai/documentation/instances/templates/docker-environment](https://docs.vast.ai/documentation/instances/templates/docker-environment)
  concept, not independently timed. Practical mitigation: pick a small, popular, pre-cached
  image (see §6) rather than a custom multi-GB image built fresh each time.
- **The single biggest realistic leak for THIS project**: forgetting a stopped-but-not-destroyed
  instance across a multi-day gap between work sessions. At this project's dataset/checkpoint
  size (single-digit GB), the dollar exposure is small (cents to low dollars/week) but it is
  the one mechanism that actually erodes a $10-30 budget without any training happening.

---

## 5. Practical gotchas people report

- **Vast.ai unverified-host reliability**: unverified hosts have "unstable network peering,"
  which inflates checkpoint-upload and dataset-download time; when a host's software detects
  an error, the machine is auto-marked "Unverified" until fixed (VERIFIED via
  [docs.vast.ai/host/verification-stages](https://docs.vast.ai/host/verification-stages)).
  Practical rule: filter to Vast.ai's verified-host tier when renting, even though it costs
  slightly more — the reliability delta is reported as comparable to RunPod's paid tier
  (UNVERIFIED, aggregator comparison). Also reported: stopped instances sometimes get stuck
  in scheduling limbo on restart rather than relaunching cleanly (UNVERIFIED, single
  aggregator source, not corroborated elsewhere).
- **Disk running out mid-run**: not specifically documented for this dataset size in my
  searches, but the generic pattern (base image + PyTorch + CUDA + HF cache + checkpoints
  routinely exceeding a stingy default disk allocation) is well known enough that the
  practical fix is to explicitly request 30-50GB disk on instance creation rather than
  accepting a provider default, even though HumanML3D itself is only 4.36 GB (HF's local
  cache, pip package installs, and CUDA toolkill add up fast).
- **CUDA/driver mismatches with recent PyTorch**: actively reported in 2026. RunPod's own
  cu128 PyTorch 2.8.0 template was found to silently install PyTorch 2.4.1 instead, because
  no cu128 wheel existed yet for that PyTorch version — pip silently fell back
  (VERIFIED-adjacent, per [github.com/runpod/containers issue #114](https://github.com/runpod/containers/issues/114),
  found via search, not independently opened). Separately, RunPod has no way to pin a
  minimum host CUDA driver version at pod-creation time, so a pod can land on hardware whose
  driver is older than the container expects, producing "unsatisfied condition: cuda>=12.6"
  failures (UNVERIFIED, aggregator search synthesis). **Mitigation: pick a stable,
  well-aged image (e.g., an official `pytorch/pytorch` tag with a CUDA version 1-2 minor
  versions behind bleeding-edge) rather than the newest template**, and verify
  `torch.cuda.is_available()` / `nvidia-smi` immediately after boot before doing anything else.
- **Containers lacking build tools**: base runtime CUDA images (non-`devel` tags) omit
  compilers/headers needed if any pip package needs to compile a CUDA extension at install
  time. Use a `-devel` tagged image (e.g. `nvidia/cuda:12.8.0-cudnn-devel-ubuntu22.04` or
  `runpod/pytorch:*-devel-*`) if any dependency might need `ninja`/`nvcc` at install time
  (UNVERIFIED specifics, but this is a well-known, widely corroborated pattern).
- **HumanML3D-specific gotcha not in the original prompt's framing, but important**: the
  *official* HumanML3D dataset cannot legally be redistributed pre-processed, because it is
  derived from AMASS, whose license forbids redistribution — the official repo
  ([github.com/EricGuo5513/HumanML3D](https://github.com/EricGuo5513/HumanML3D)) ships only
  *scripts* to regenerate the data from a separately-obtained AMASS license (VERIFIED via
  search snippet of the repo's own README language, not independently opened). The
  `TeoGchx/HumanML3D` HuggingFace mirror this project plans to use is therefore an
  **unofficial third-party redistribution** — convenient and apparently matches the
  expected size/row counts (4.36 GB, 29,228 rows across train/val/test per
  [huggingface.co/datasets/TeoGchx/HumanML3D](https://huggingface.co/datasets/TeoGchx/HumanML3D),
  VERIFIED fetched directly), but treat it as a **quality-unverified, license-uncertain
  mirror**: worth a spot-check of a few samples against known HumanML3D statistics (or
  against another mirror) before trusting it as ground truth, and worth having a fallback
  plan (the official extraction pipeline against a personally-obtained AMASS license) if the
  mirror turns out corrupted or disappears.

---

## 6. Fastest path from zero to training

1. **Provider/image**: on RunPod or Vast.ai, start from an official pre-built PyTorch+CUDA
   image with build tools included, e.g. `runpod/pytorch:*-devel-*` (RunPod's own template,
   includes Jupyter+SSH out of the box per
   [docs.runpod.io](https://docs.runpod.io) — VERIFIED via search synthesis of RunPod's own
   docs, not independently opened) — this avoids both the CUDA/PyTorch mismatch gotcha (§5)
   and the missing-build-tools gotcha, and benefits from host-side layer caching since it's
   a popular image.
2. **Request disk generously up front**: 30-50GB, not the default — cheap relative to
   compute cost, and avoids a disk-full failure mid-run (§5).
3. **Pull the dataset via `huggingface_hub`/`hf` CLI, not raw `git clone`**: `pip install -U
   huggingface_hub` then `hf download TeoGchx/HumanML3D --repo-type dataset --local-dir
   ./data` (or Python `snapshot_download`) — avoids git-lfs/git-xet setup friction reported
   in generic HF-download gotchas (UNVERIFIED specifics for this exact dataset, but this is
   the documented standard path per
   [huggingface.co/docs/hub/datasets-downloading](https://huggingface.co/docs/hub/datasets-downloading)).
4. **Persist checkpoints against interruption**: on RunPod, attach a **Network Volume**
   (persists independently of the pod, survives pod destruction, can be reattached to a new
   pod) rather than writing only to the pod's local/container disk (VERIFIED concept via
   [docs.runpod.io/tutorials/introduction/containers/persist-data](https://docs.runpod.io/tutorials/introduction/containers/persist-data)).
   On Vast.ai, use the persistent-storage option and checkpoint every few minutes so a
   spot-interruption at 15-second notice never loses more than a few minutes of progress.
   Given the RunPod-reported 5-second SIGTERM-before-SIGKILL window on spot pods, the
   training loop should checkpoint frequently on a fixed step interval rather than relying
   on catching the interrupt signal to save just-in-time.
5. **Zero-cost dry run first**: because Kaggle gives 30 free GPU-hours/week and this
   workload is almost certainly overhead-bound rather than compute-bound (§2), do the first
   smoke test and possibly the entire first full training run on Kaggle's free T4/P100
   before spending anything, reserving paid rented time for parallel/repeated experiment
   arms that would exceed Kaggle's weekly cap or 9-hour session limit.

---

## Recommendation

**Winner: Vast.ai, RTX 3090, spot/interruptible, with checkpointing to persistent storage
every few minutes; verified hosts only.** Rationale: this workload is overhead-bound, not
compute-bound (§2), so the 4090's extra throughput is largely wasted; the 3090 is the
cheapest "real" GPU across every table in §1; spot pricing (~$0.07-0.22/hr depending on
listing) makes even a multi-hour run cost pennies to a few dollars; checkpointing is cheap
insurance against both spot interruption and this project's own tight budget.

- **(a) One full MDM-scale training run** (using the ~20-hour reference from the original
  MDM paper on a 2022-era GPU as the workload's rough scale, and assuming a similar consumer
  GPU produces comparable-or-better wall-clock given this project's overhead-bound regime):
  roughly **20 GPU-hours x ~$0.10-0.20/hr (Vast.ai 3090 spot) = an estimated $2-5**, plus
  a few cents of storage. This is a rough scaling estimate, not independently timed on this
  exact model — UNVERIFIED beyond the source prices themselves.
- **(b) A 4-arm experiment** (4 independent training runs, assuming similar per-run cost and
  no special savings from shared setup): **roughly $8-20**, comfortably inside the $10-30
  budget, though realistically expect some spend on false starts, debugging runs, and
  interruption-driven re-runs to push this toward the top of that range.
- **Cheapest viable fallback at $0 budget**: **Kaggle's free tier** (30 GPU-hours/week,
  P100 16GB or dual T4) is very likely sufficient outright for this model size, given the
  overhead-bound workload means Kaggle's older/free GPUs should not be dramatically slower
  than a paid consumer card for this specific model. Google Colab's free T4 (session limit
  12 hours, 90-minute idle timeout, must checkpoint to Drive) is the second-best zero-cost
  option if Kaggle's 9-hour session cap or weekly quota becomes binding.
