"""RL-30 adversarial verification (WD), part 2: POPULATION view of the 'same core name, same street, shifted house number' pattern,
independent of the model's candidate pool/scores.
For every S2/S3 record of a country, find the S1s with the identical core name (WD core) and the same street (WD parser), and
classify the house-number delta. Train US/India: label each pattern pair (true link / unlinked distractor / linked to another S1).
Test US/France: share of pattern pairs that S005 accepted (final). Also the shape test by NEW-p bin (selection check).
READ-ONLY. Writes rl30/WD_pop.json (NEW file).  Usage: OMP_NUM_THREADS=4 nice -n 10 python WD_pop.py
"""
import sys, os, json, time, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import re
from rapidfuzz import fuzz
from rl30_lib import load, OUT, accepted, NEW_TH, fold, LEGAL, HONOR, STOP, STREET_TYPE, UNIT
src = open(os.path.join(OUT, "WD_verify.py")).read()
exec(src[src.index("_DOT = "):src.index("# =========================================================== TEST")])   # wcore, wstreet, same_street, dclass, CLASSES, f05, shape_*

t0 = time.time()
L = lambda *a: print(f"[{time.time() - t0:5.0f}s]", *a, flush=True)
R = {}


def pattern_pairs(s1df, recdf):
    s1_core = [wcore(n) for n in s1df.name.values]
    s1_st = {}
    by_core = collections.defaultdict(list)
    for i, c, a in zip(s1df.id.values, s1_core, s1df.addr.values):
        if c:
            by_core[c].append((i, a))
    out = []
    for r, n, a in zip(recdf.id.values, recdf.name.values, recdf.addr.values):
        c = wcore(n)
        lst = by_core.get(c)
        if not lst:
            continue
        nr, sr = wstreet(a)
        if nr is None:
            continue
        for s, sa in lst:
            v = s1_st.get(s)
            if v is None:
                v = s1_st[s] = wstreet(sa)
            ns, ss = v
            if ns is None or not same_street(ss, sr):
                continue
            out.append((s, r, dclass(ns, nr), len(lst)))
    return pd.DataFrame(out, columns=["s1", "rec", "cls", "n_samecore_s1"])


# ------------------------------------------------ train (labelled population)
Tr = load("train")
owner = dict(zip(Tr["gt"].rec, Tr["gt"].s1))
POP = {}
for cc in ("US", "India"):
    s1df = Tr["s1"][Tr["s1"].country == cc]
    recdf = pd.concat([Tr["s2"][Tr["s2"].country == cc], Tr["s3"][Tr["s3"].country == cc]])
    pp = pattern_pairs(s1df, recdf)
    ow = [owner.get(r) for r in pp.rec]
    pp["lab"] = ["true" if o == s else ("distractor" if o is None else "other_s1") for o, s in zip(ow, pp.s1)]
    t = pp.groupby("cls").lab.value_counts().unstack(fill_value=0)
    t["N"] = t.sum(axis=1)
    for k in ("true", "distractor", "other_s1"):
        if k in t:
            t[f"P_{k}"] = (t[k] / t.N).round(4)
    t["per_1k_S1"] = (1000 * t.N / len(s1df)).round(3)
    L(f"train {cc}: S1 {len(s1df):,}, pattern pairs {len(pp):,}"); L(t.to_string())
    POP[f"train_{cc}"] = dict(n_s1=int(len(s1df)), table=t.to_dict(orient="index"))
    # distractor-per-S1 rate for the shift classes
    sh = pp[pp.cls.isin(CLASSES)]
    POP[f"train_{cc}"]["distractor_shift_pairs_per_1k_S1"] = round(1000 * float((sh.lab == "distractor").sum()) / len(s1df), 3)
    POP[f"train_{cc}"]["true_shift_pairs_per_1k_S1"] = round(1000 * float((sh.lab == "true").sum()) / len(s1df), 3)
del Tr, owner

# ------------------------------------------------ test (accepted share of the population)
T = load("test")
for cc in ("US", "France", "India"):
    s1df = T["s1"][T["s1"].country == cc]
    recdf = pd.concat([T["s2"][T["s2"].country == cc], T["s3"][T["s3"].country == cc]])
    pp = pattern_pairs(s1df, recdf)
    a = accepted(f"S005_{cc}")
    fin = set(zip(a.s1[a.kept_final], a.rec[a.kept_final]))
    pp["acc_final"] = [(s, r) in fin for s, r in zip(pp.s1, pp.rec)]
    t = pp.groupby("cls").agg(N=("acc_final", "size"), A=("acc_final", "sum"))
    t["A_over_N"] = (t.A / t.N).round(4); t["N_per_1k_S1"] = (1000 * t.N / len(s1df)).round(3); t["A_per_1k_S1"] = (1000 * t.A / len(s1df)).round(3)
    L(f"test {cc}: S1 {len(s1df):,}, pattern pairs {len(pp):,}"); L(t.to_string())
    POP[f"test_{cc}"] = dict(n_s1=int(len(s1df)), table=t.to_dict(orient="index"))
    # shape test by NEW-p bin for pm1+pm2+odd3_10 final pairs (selection check)
    fk = a[a.kept_final]
    kk = fk.assign(src=fk.rec.str[:2]).groupby(["s1", "src"]).size()
    REP1, ADD1 = shape_refs(kk, s1df.id.values)
    sel = pp[pp.acc_final & pp.cls.isin(["pm1", "pm2", "odd3_10"])].merge(fk[["s1", "rec", "p"]], on=["s1", "rec"])
    pb = {}
    for lo, hi in ((0.78, 0.9), (0.9, 0.97), (0.97, 0.995), (0.995, 1.01)):
        g = sel[(sel.p >= lo) & (sel.p < hi)]
        if len(g) >= 50:
            pb[f"[{lo},{hi})"] = shape_f(kk, g.s1.values, g.rec.values, REP1, ADD1)
    POP[f"test_{cc}"]["shape_by_pbin_pm1_pm2_odd3_10"] = pb
    POP[f"test_{cc}"]["p_quantiles_pm1_pm2_odd3_10"] = sel.p.quantile([0.1, 0.25, 0.5, 0.75]).round(4).to_dict()
    L(cc, "shape by p bin:", json.dumps(pb)); L(cc, "p quantiles:", POP[f"test_{cc}"]["p_quantiles_pm1_pm2_odd3_10"])
R["population"] = POP
json.dump(R, open(os.path.join(OUT, "WD_pop.json"), "w"), indent=1, default=str)
L("wrote WD_pop.json")
