"""WD (adversarial verifier of investigator D's 'content-word swap = decoy' claim), step 2: MODEL-FREE enumeration.
All S1 x record pairs sharing (house number, street-name token set) [rl30_lib.street_parts], typed with an independent,
simpler classifier (not D_transform). Saves rl30/WD_adv_pairs_{split}_{country}.pkl (NEW files). READ-ONLY otherwise.
Usage: nice -n 10 python WD_2_modelfree.py test France | test US | test India | train US | train India
"""
import sys, os, collections, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, toks, street_parts, LEGAL, HONOR, STOP
from rapidfuzz.distance import Levenshtein, OSA

split, cc = sys.argv[1], sys.argv[2]
t0 = time.time()
L = lambda *a: print(f"[{time.time()-t0:5.0f}s]", *a, flush=True)
D = load(split, verbose=False)


def core(n):
    return frozenset(t for t in toks(n) if t not in LEGAL and t not in HONOR and t not in STOP)


def akey(d):
    out = []
    for a in d.addr.values:
        n, s, _ = street_parts(a)
        out.append(f"{n}|{' '.join(sorted(s))}" if (n is not None and s) else None)
    return out


s1 = D["s1"][D["s1"].country == cc].copy()
rec = pd.concat([D["s2"], D["s3"]]); rec = rec[rec.country == cc].copy()
s1["key"] = akey(s1); rec["key"] = akey(rec)
s1["core"] = [core(n) for n in s1.name.values]; rec["core"] = [core(n) for n in rec.name.values]
rec["src"] = rec.id.str[:2]
L("parsed", len(s1), len(rec), "s1 key ok", s1.key.notna().mean().round(4), "rec key ok", rec.key.notna().mean().round(4))
df = collections.Counter(t for c in s1.core.values for t in c)

m = s1[s1.key.notna()][["id", "key", "core"]].merge(rec[rec.key.notna()][["id", "key", "core", "src"]], on="key", suffixes=("_s", "_r"))
L("pairs same num+street", len(m))


def near(a, b):
    if min(len(a), len(b)) >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    s, l = (a, b) if len(a) <= len(b) else (b, a)
    if len(s) <= 3 and s[0] == l[0] and all(ch in iter(l) for ch in s):
        return True
    if OSA.distance(a, b) <= 1 and min(len(a), len(b)) >= 2:
        return True
    if len(a) >= 4 and len(b) >= 4 and (Levenshtein.normalized_similarity(a, b) >= 0.6 or OSA.distance(a, b) <= 2):
        return True
    if min(len(a), len(b)) >= 4 and Levenshtein.normalized_similarity("".join(sorted(a)), "".join(sorted(b))) >= 0.75:
        return True
    return False


typ, ta, tb, nA, nB, nC = [], [], [], [], [], []
for cs, cr in zip(m.core_s.values, m.core_r.values):
    A, B = cs - cr, cr - cs
    C = len(cs & cr)
    nA.append(len(A)); nB.append(len(B)); nC.append(C)
    if C == 0:
        typ.append("DISJ"); ta.append(""); tb.append(""); continue
    if not A and not B:
        typ.append("SAME"); ta.append(""); tb.append(""); continue
    # greedy near matching
    Al, Bl = list(A), list(B)
    ua, ub = set(), set()
    for a in Al:
        for b in Bl:
            if b not in ub and near(a, b):
                ua.add(a); ub.add(b); break
    Ar = [a for a in Al if a not in ua]; Br = [b for b in Bl if b not in ub]
    if not Ar and not Br:
        typ.append("TYPO"); ta.append(""); tb.append(""); continue
    if len(A) == 1 and len(B) == 1 and Ar and Br:
        a, b = Ar[0], Br[0]
        ta.append(a); tb.append(b)
        if df.get(a, 0) >= 20 and df.get(b, 0) >= 20:
            typ.append("CSWAP")
        elif df.get(b, 0) < 20:
            typ.append("GSWAP")
        else:
            typ.append("RSWAP")
        continue
    ta.append(" ".join(sorted(Ar))); tb.append(" ".join(sorted(Br)))
    if Ar and not Br:
        typ.append("DROP")
    elif Br and not Ar:
        typ.append("ADD")
    else:
        typ.append("MULTI")
m["typ"] = typ; m["ta"] = ta; m["tb"] = tb; m["nA"] = nA; m["nB"] = nB; m["nC"] = nC
m = m.rename(columns={"id_s": "s1", "id_r": "rec"})
L("typed", m.typ.value_counts().to_dict())
if split == "train":
    gt = D["gt"]
    pos = set(zip(gt.s1, gt.rec)); linked = set(gt.rec)
    m["y"] = [(s, r) in pos for s, r in zip(m.s1, m.rec)]
    m["orphan"] = ~m.rec.isin(linked)
    L("labels", m.groupby("typ").agg(n=("y", "size"), y=("y", "mean"), orphan=("orphan", "mean")).to_string())
m = m.drop(columns=["core_s", "core_r"])
m.to_pickle(os.path.join(OUT, f"WD_adv_pairs_{split}_{cc}.pkl"))
# keep record-core lookups for the cluster test
rec[["id", "key", "src"]].assign(core=[" ".join(sorted(c)) for c in rec.core.values]).to_pickle(os.path.join(OUT, f"WD_adv_rec_{split}_{cc}.pkl"))
s1[["id", "key"]].assign(core=[" ".join(sorted(c)) for c in s1.core.values]).to_pickle(os.path.join(OUT, f"WD_adv_s1_{split}_{cc}.pkl"))
L("saved")
