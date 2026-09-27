"""RL-30 verifier VA (ALREADY CAPTURED lens), step 3 for 'S1_side_token_drop_rate'. READ-ONLY: writes VA_drop_3_content.json.
Bootstrap CIs (Poisson over V1 S1) for the CONTENT-token cases, where the hypothesis should bite hardest:
 (i) pure single-token drops of a non-legal/non-honorific token, 0.02<=p<0.99;
 (ii) any band pair whose S1-only tokens (after typo pairing) are all content tokens.
Feature = sum log(dr/prior) (label-free test pseudo-positive table, a=20). Strata: p20, p20 x model-IDF quintile (LF col 73)."""
import os, sys, json, time, re
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
import e023_stage2 as S2
from rapidfuzz.distance import Levenshtein as LV
from scipy.stats import rankdata
T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")
def ntoks(s):
    s = fold(s if isinstance(s, str) else ""); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))
def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)
def diff(A, B):
    pa, pb = set(A - B), set(B - A)
    if pa and pb:
        for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
            if x in pa and z in pb: pa.discard(x); pb.discard(z)
    return pa, pb
DR, PRIOR = {}, {}
for C in ("US", "India"):
    t = pd.read_csv(os.path.join(OUT, f"A_p2_noise_{C}_testPP.csv"), keep_default_na=False, na_values=[""]); t = t[t.s1has > 0]
    PRIOR[C] = float(t.pdrop.sum() / t.s1has.sum()); DR[C] = dict(zip(t.tok, (t.pdrop + 20 * PRIOR[C]) / (t.s1has + 20)))
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
V = S2.load_set("V1", "a50n10d10a")
p = np.load(PATHS["v1_p_new"]).astype(np.float64); y = V["y"].astype(int)
si = V["s1idx"]; s1ids = V["s1_ids"][si]; cand = V["cand"]; ctry = V["country"][si]
idf1 = np.asarray(V["LF"][:, 73]).astype(np.float64)
work = np.flatnonzero((p >= 0.02) & (p < 0.99))
s1n = S1t.name.reindex(s1ids[work]).values; rn = RECt.name.reindex(cand[work]).values
rows = []
for q, k in enumerate(work):
    pa, pb = diff(ntoks(s1n[q]), ntoks(rn[q]))
    if not pa or any(t in LEGAL or t in HONOR for t in pa): continue
    C = ctry[k]
    rows.append((k, len(pa) == 1 and not pb, sum(np.log(DR[C].get(t, PRIOR[C]) / PRIOR[C]) for t in pa)))
K = np.array([r[0] for r in rows]); single = np.array([r[1] for r in rows]); lr = np.array([r[2] for r in rows])
lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
def auc(f, yy):
    npos = (yy == 1).sum(); nneg = len(yy) - npos
    if npos == 0 or nneg == 0: return np.nan, 0
    r = rankdata(f); return (r[yy == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg), npos * nneg
def cauc(f, st, yy):
    tot = ws = 0.0
    for s in np.unique(st):
        m = st == s; a, w = auc(f[m], yy[m])
        if w: tot += a * w; ws += w
    return tot / ws if ws else np.nan
rng = np.random.default_rng(5); res = {}
for nm, m in (("pure_single_content_drop", single), ("all_S1only_content", np.ones(len(K), bool))):
    k = K[m]; f = lr[m]; yy = y[k]
    s20 = np.digitize(lp[k], np.quantile(lp[k], np.linspace(0, 1, 21)[1:-1]))
    sI = s20 * 10 + np.digitize(idf1[k], np.quantile(idf1[k], [0.2, 0.4, 0.6, 0.8]))
    out = dict(n=int(len(k)), pos=int(yy.sum()), FN=int(((p[k] < NEW_TH) & (yy == 1)).sum()), FP=int(((p[k] >= NEW_TH) & (yy == 0)).sum()))
    for sn, st in (("p20", s20), ("p20_x_IDF5", sI)):
        a0 = cauc(f, st, yy); bs = []; rs = []
        for _ in range(200):
            w = rng.poisson(1.0, len(V["s1_ids"]))[si[k]]; idx = np.repeat(np.arange(len(k)), w)
            bs.append(cauc(f[idx], st[idx], yy[idx]))
        out[sn] = dict(auc=round(float(a0), 4), ci90=[round(float(np.nanquantile(bs, 0.05)), 4), round(float(np.nanquantile(bs, 0.95)), 4)],
                       auc_random=round(float(cauc(rng.random(len(k)), st, yy)), 4))
    res[nm] = out; print(nm, json.dumps(out), f"{time.time() - T0:.0f}s", flush=True)
json.dump(res, open(os.path.join(OUT, "VA_drop_3_content.json"), "w"), indent=1)
