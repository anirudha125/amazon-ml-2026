"""RL-30 verifier VA, step 3 (V1, labelled). A's mechanism claim: the TOK16 IDF gives French fillers (associes, fils, cie ...)
HIGH distinctiveness, so the model treats a filler add as evidence of a different business. The US/India analog of that is a
record-only token that is filler (per A's add_LR) but has HIGH IDF (dba, formerly, aka, nee, fka, www, known, doing ...), or
an arbitrary high-IDF record-only token added to an otherwise identical name at the same street.
Question: does RL-27 NEW mis-handle those in V1?  y-rate / confusion by IDF of the record-only tokens.
IDF = A's pooled formula on train S1 names (log((N - df + .5) / (df + .5) + 1)).  READ-ONLY; writes VA_3_v1_idf.json."""
import os, sys, json, time, re, math
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
import e023_stage2 as S2
from rapidfuzz.distance import Levenshtein as LV

T0 = time.time()
ACR = re.compile(r"\b((?:[a-z]\.){2,}[a-z]?)\.?")


def ntoks(s):
    s = fold(s); s = ACR.sub(lambda m: m.group(1).replace(".", ""), s)
    return frozenset(re.findall(r"[a-z0-9]+", s))


def typo(a, b):
    return (not a.isdigit() and not b.isdigit()) and (LV.distance(a, b) <= 1 or LV.normalized_similarity(a, b) >= 0.75)


FILL = {C: set(v) for C, v in json.load(open(os.path.join(OUT, "VA_1_v1.json")))["fillers"].items()}
DT = load("train", verbose=False)
S1t = DT["s1"].set_index("id"); RECt = pd.concat([DT["s2"], DT["s3"]]).set_index("id")
df = Counter(t for nm in DT["s1"].name.values for t in ntoks(nm)); N = len(DT["s1"])
idf = lambda t: math.log((N - df.get(t, 0) + 0.5) / (df.get(t, 0) + 0.5) + 1.0)
V = S2.load_set("V1", "a50n10d10a")
p = np.load(PATHS["v1_p_new"]); y = V["y"].astype(int); acc = p >= NEW_TH
si = V["s1idx"]; s1ids = V["s1_ids"][si]; cand = V["cand"]; ctry = V["country"][si]; n = len(y)
s1n = S1t.name.reindex(s1ids).values; s1a = S1t.addr.reindex(s1ids).values
rn = RECt.name.reindex(cand).values; ra = RECt.addr.reindex(cand).values
cache, scache = {}, {}


def nt(s):
    s = s if isinstance(s, str) else ""
    v = cache.get(s)
    if v is None: v = cache[s] = ntoks(s)
    return v


def sp(a):
    a = a if isinstance(a, str) else ""
    v = scache.get(a)
    if v is None: v = scache[a] = street_parts(a)[:2]
    return v


rows = []
for k in range(n):
    A, B = nt(s1n[k]), nt(rn[k])
    ob = B - A
    if not ob: continue
    n1, st1 = sp(s1a[k]); n2, st2 = sp(ra[k])
    if not (n1 is not None and n1 == n2 and st1 and st1 == st2): continue
    pa, pb = set(A - B), set(ob)
    if pa and pb:
        for _, x, z in sorted(((LV.normalized_similarity(x, z), x, z) for x in pa for z in pb if typo(x, z)), reverse=True):
            if x in pa and z in pb: pa.discard(x); pb.discard(z)
    if not pb: continue
    rest = [t for t in pa if t not in LEGAL and t not in HONOR and t not in FILL[ctry[k]]]
    if rest: continue                                         # S1 side loses nothing but legal / honorific / filler: a PURE record-side add
    fl = all(t in FILL[ctry[k]] for t in pb)
    mx = max(idf(t) for t in pb)
    rows.append((k, ctry[k], fl, len(pb), mx, " ".join(sorted(pb))))
Rw = pd.DataFrame(rows, columns=["k", "country", "all_filler", "n_add", "idf_max", "added"])
Rw["y"] = y[Rw.k.values]; Rw["p"] = p[Rw.k.values]; Rw["acc"] = acc[Rw.k.values]
Rw["idf_bin"] = pd.cut(Rw.idf_max, [0, 5, 7, 9, 11, 100], labels=["<5", "5-7", "7-9", "9-11", ">=11"])
print("pure record-side adds, same street:", len(Rw), f"{time.time() - T0:.0f}s", flush=True)


def conf(g):
    return dict(n=int(len(g)), pos=int(g.y.sum()), yrate=round(float(g.y.mean()), 4),
                TP=int((g.acc & (g.y == 1)).sum()), FN=int((~g.acc & (g.y == 1)).sum()), FP=int((g.acc & (g.y == 0)).sum()),
                TN=int((~g.acc & (g.y == 0)).sum()), acc_rate_of_pos=round(float(g.acc[g.y == 1].mean()), 4) if g.y.sum() else None,
                median_p_pos=round(float(g.p[g.y == 1].median()), 4) if g.y.sum() else None)


out = {}
for (fl, b), g in Rw.groupby(["all_filler", "idf_bin"], observed=True):
    out[f"{'filler' if fl else 'nonfiller'}_add|idf{b}"] = conf(g)
for C, g in Rw.groupby("country"):
    out[f"{C}|all"] = conf(g)
    out[f"{C}|filler_idf>=9"] = conf(g[g.all_filler & (g.idf_max >= 9)])
    out[f"{C}|nonfiller_idf>=9"] = conf(g[~g.all_filler & (g.idf_max >= 9)])
hi = Rw[Rw.all_filler & (Rw.idf_max >= 9)]
out["top_high_idf_filler_adds"] = hi.added.value_counts().head(25).to_dict()
nf = Rw[~Rw.all_filler & (Rw.idf_max >= 9)]
out["examples_nonfiller_hi_idf_pos"] = [dict(s1=str(s1n[k]), rec=str(rn[k]), p=round(float(p[k]), 4)) for k in nf[nf.y == 1].k.values[:10]]
out["examples_nonfiller_hi_idf_neg"] = [dict(s1=str(s1n[k]), rec=str(rn[k]), p=round(float(p[k]), 4)) for k in nf[nf.y == 0].k.values[:10]]
out["examples_filler_hi_idf_FN"] = [dict(s1=str(s1n[k]), rec=str(rn[k]), p=round(float(p[k]), 4)) for k in hi[(hi.y == 1) & ~hi.acc].k.values[:10]]
out["secs"] = round(time.time() - T0, 1)
for k_, v in out.items(): print(k_, v, flush=True)
json.dump(out, open(os.path.join(OUT, "VA_3_v1_idf.json"), "w"), indent=1, ensure_ascii=False, default=float)
