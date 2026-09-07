"""Is spatial information present in CLIP's per-token features but lost in the pooled vector
MDM actually conditions on?  No training; pure probe."""
import sys, numpy as np, torch, clip
sys.path.insert(0, ".")

model, _ = clip.load("ViT-B/32", device="cpu", jit=False)
clip.model.convert_weights(model); model.eval()
for p in model.parameters(): p.requires_grad = False

CTX = 22  # MDM uses max_text_len 20 + sos/eos

def tokenize(t):
    toks = clip.tokenize([t], context_length=CTX, truncate=True)
    pad = torch.zeros([1, 77 - CTX], dtype=toks.dtype)
    return torch.cat([toks, pad], dim=1)

@torch.no_grad()
def reps(text):
    toks = tokenize(text)
    x = model.token_embedding(toks).type(model.dtype)
    x = x + model.positional_embedding.type(model.dtype)
    x = x.permute(1, 0, 2)
    x = model.transformer(x)
    x = x.permute(1, 0, 2)
    x = model.ln_final(x).type(model.dtype)          # [1, 77, 512] per-token
    eot = toks.argmax(dim=-1)
    pooled = x[torch.arange(1), eot] @ model.text_projection   # what MDM uses
    n = int(eot.item()) + 1
    return pooled[0].float().numpy(), x[0, :n].float().numpy(), n

def cos(a, b):
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))

SPATIAL = [("a person raises their left arm","a person raises their right arm"),
           ("a person turns left","a person turns right"),
           ("a person walks forward","a person walks backward"),
           ("a person kicks with their left leg","a person kicks with their right leg"),
           ("a person leans forward","a person leans backward"),
           ("a person walks to the left","a person walks to the right"),
           ("a person spins clockwise","a person spins counterclockwise"),
           ("a person bends their left knee","a person bends their right knee")]
CONTROL = [("a person raises their arm slowly","a person raises their arm quickly"),
           ("a person walks slowly","a person walks quickly"),
           ("a person kicks the red ball","a person kicks the blue ball"),
           ("a person waves their small flag","a person waves their large flag"),
           ("a person lifts a heavy box","a person lifts a light box"),
           ("a person opens the old door","a person opens the new door"),
           ("a person holds a wet cloth","a person holds a dry cloth"),
           ("a person climbs a tall ladder","a person climbs a short ladder")]

def run(pairs, name):
    pooled_s, meantok_s, maxdiff = [], [], []
    for a, b in pairs:
        pa, ta, na = reps(a); pb, tb, nb = reps(b)
        pooled_s.append(cos(pa, pb))
        m = min(na, nb)
        meantok_s.append(cos(ta[:m].mean(0), tb[:m].mean(0)))
        # per-token: the single most-divergent aligned position
        per = [cos(ta[i], tb[i]) for i in range(m)]
        maxdiff.append(min(per))
    print(f"{name:>10}  pooled(MDM)={np.mean(pooled_s):.4f}   mean-token={np.mean(meantok_s):.4f}   "
          f"most-divergent-token={np.mean(maxdiff):.4f}")
    return np.array(pooled_s), np.array(meantok_s), np.array(maxdiff)

ps, ms, xs = run(SPATIAL, "SPATIAL")
pc, mc, xc = run(CONTROL, "CONTROL")
print()
from scipy import stats
for lbl, s, c in (("pooled (what MDM uses)", ps, pc),
                  ("mean over tokens", ms, mc),
                  ("most-divergent token", xs, xc)):
    gap = s.mean() - c.mean()
    d = gap / np.sqrt(((len(s)-1)*s.std(ddof=1)**2 + (len(c)-1)*c.std(ddof=1)**2)/(len(s)+len(c)-2))
    p = stats.mannwhitneyu(s, c, alternative="greater").pvalue
    print(f"  {lbl:<24} spatial-minus-control gap = {gap:+.4f}   d={d:+.3f}   p={p:.4f}")
print("\n(gap > 0 means spatial pairs are LESS separated than controls = blindness)")
