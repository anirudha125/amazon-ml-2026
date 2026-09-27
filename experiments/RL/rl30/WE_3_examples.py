"""RL-30 / WE -- concrete France test examples behind E's D / F counts for a few tokens (label-free). Writes rl30/WE_3_examples.json.
For token t: D examples = pairs of distinct S1 in the same locality whose name sets differ only by t (one-sided or substitution);
F examples = (S1, record) at a single-S1 (house number, street) key whose names differ by t."""
import os, re, sys, json, zlib, collections, random
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import load, fold, toks, akey, street_parts, LEGAL, HONOR, STOP
from WE_1_roles import my_loc, my_key2, nd, NONC

TOK = ["lille", "bordeaux", "nantes", "club", "ecole", "sarl", "ets", "etablissements"]
D = load("test", verbose=False)
s1 = D["s1"][D["s1"].country == "France"].reset_index(drop=True)
NT = [frozenset(toks(x)) for x in s1.name.values]; LOC = [my_loc(a) for a in s1.addr.values]
random.seed(0)
out = {}
grp = collections.defaultdict(list); full = collections.defaultdict(list)
for i, (A, L) in enumerate(zip(NT, LOC)):
    if L is None or not (A - NONC):
        continue
    full[(L, A)].append(i)
    if len(A) >= 2:
        for t in A:
            K = A - {t}
            if K - NONC:
                grp[(L, K)].append((i, t))
for t in TOK:
    ex = []
    for (L, K), lst in grp.items():
        mine = [i for i, u in lst if u == t]
        if not mine:
            continue
        others = [(i, u) for i, u in lst if u != t]
        one = full.get((L, K), [])
        if others or one:
            i = mine[0]
            mate = s1.name.values[others[0][0]] if others else s1.name.values[one[0]]
            ex.append(dict(locality=L, s1=s1.name.values[i], s1_addr=s1.addr.values[i], other_s1=mate,
                           other_addr=s1.addr.values[others[0][0]] if others else s1.addr.values[one[0]], kind="sub" if others else "one"))
    random.shuffle(ex)
    out[f"D_{t}"] = dict(n_groups=len(ex), examples=ex[:8])
    # how often is the D-pair at the SAME canonical address
    out[f"D_{t}"]["share_same_akey"] = round(float(np.mean([akey(e["s1_addr"]) == akey(e["other_addr"]) for e in ex])) if ex else 0, 4)
# F examples
KEY = collections.defaultdict(list)
K2 = [my_key2(a) for a in s1.addr.values]
for i in range(len(s1)):
    if K2[i] is not None and LOC[i] is not None:
        KEY[K2[i]].append(i)
LW = [set(l.split()) if l else set() for l in LOC]
R = pd.concat([D["s2"], D["s3"]], ignore_index=True); R = R[R.country == "France"]
fex = collections.defaultdict(list)
for nm, a in zip(R.name.values, R.addr.values):
    if not a.strip():
        continue
    k = my_key2(a)
    if k is None or k not in KEY:
        continue
    lw = my_loc(a); lw = set(lw.split()) if lw else set()
    L2 = [j for j in KEY[k] if LW[j] & lw]
    if len(L2) != 1:
        continue
    j = L2[0]; A = NT[j]; Rn = frozenset(toks(nm))
    if nd(A, Rn, True):
        for t in A ^ Rn:
            if t in TOK and len(fex[t]) < 200:
                fex[t].append(dict(s1=s1.name.values[j], s1_addr=s1.addr.values[j], rec=nm, rec_addr=a))
for t in TOK:
    L = fex.get(t, []); random.shuffle(L)
    out[f"F_{t}"] = L[:8]
json.dump(out, open(os.path.join(HERE, "WE_3_examples.json"), "w"), indent=1, ensure_ascii=False)
for k, v in out.items():
    if k.startswith("D_"):
        print(k, v["n_groups"], "share_same_akey", v["share_same_akey"])
        for e in v["examples"][:4]:
            print("   ", e["kind"], "|", e["s1"], "|", e["other_s1"], "|", e["s1_addr"], "||", e["other_addr"])
    else:
        print(k)
        for e in v[:4]:
            print("   ", e["s1"], "|", e["rec"], "|", e["s1_addr"], "||", e["rec_addr"])
