"""RL-26b -- map D2b's V1 decisions on EMPTY-ADDRESS candidates to the record-centric count n_compat (RL-26):
how many FNs sit in the n=1 cell (train posterior 0.99) and how many FPs sit in n>=2? Counterfactual: accept every
empty-address candidate whose record is compatible with exactly this S1 (n_compat==1 and the S1 is it), reject every
accepted empty-address candidate with n_compat>=2 whose S1 has an identical-full-name sibling (C1-strict).
Rule is fixed a priori from train statistics (no tuning on V1). Read-only; CPU."""
import os, sys, re, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import is_indic, toks
from rl26_record_centric import Inv, ctoks
from rl17_context_rules import f05_vec, boot
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))); OUT = os.path.dirname(os.path.abspath(__file__))
D = load("train", verbose=False); s1 = D["s1"].reset_index(drop=True); sid = s1.id.values
R = pd.concat([D["s2"], D["s3"]]).set_index("id")
m = np.load(os.path.join(ROOT, "experiments/E024/V1_a50n10d10a/meta.npz"), allow_pickle=True)
p = np.load(os.path.join(ROOT, "experiments/E024/p_D2b_union_rrUb_big_V1_s42.npy")); th = 0.72
s1_ids, s1idx, cand, y, ngt = m["s1_ids"], m["s1idx"], m["cand"], m["y"].astype(int), m["n_gt"]; n = len(s1_ids)
full = s1.name.map(lambda s: " ".join(toks(s))); dupfull = (s1.assign(f=full).groupby(["country", "f"]).f.transform("size") >= 2).values
row_of = {s: i for i, s in enumerate(sid)}
inv = {c: Inv(np.where(s1.country.values == c)[0], [ctoks(x) for x in s1.name.values[s1.country.values == c]]) for c in ("US", "India")}
empty = set(R.index[R.addr.str.strip() == ""])
sel = np.array([c in empty for c in cand]) & (p >= 0.01)
rows = []
for i in np.where(sel)[0]:
    a = s1_ids[s1idx[i]]; r = cand[i]; nm = R.at[r, "name"]; c = s1.country.values[row_of[a]]
    if is_indic(nm) or re.search(r"\.(com|in|net|org)|^[#@]", nm.lower()):
        nc, a_in = -1, False
    else:
        comp = inv[c].compat(ctoks(nm)); nc = len(comp) if comp is not None else -1
        a_in = comp is not None and row_of[a] in set(comp.tolist())
    rows.append((i, nc, a_in, bool(dupfull[row_of[a]])))
X = pd.DataFrame(rows, columns=["i", "n_compat", "a_in", "a_dupfull"]); X["y"] = y[X.i]; X["acc"] = p[X.i] >= th; X["p"] = p[X.i]
def cell(x): return "n=1 (a is it)" if (x.n_compat == 1 and x.a_in) else ("n>=2, a in set" if (x.n_compat >= 2 and x.a_in) else ("n=0/other" ))
X["cell"] = X.apply(cell, axis=1)
tab = X.groupby(["cell", "acc", "y"]).size().unstack(fill_value=0)
print(tab)
acc = p >= th
add = X[(X.cell == "n=1 (a is it)") & ~X.acc].i.values
rm = X[(X.n_compat >= 2) & X.a_in & X.a_dupfull & X.acc].i.values
new = acc.copy(); new[add] = True; new[rm] = False
f0 = f05_vec(np.bincount(s1idx, weights=acc & (y == 1), minlength=n), np.bincount(s1idx, weights=acc, minlength=n), ngt)
res = {}
for nm, A in (("accept n=1", np.where(np.isin(np.arange(len(p)), add), True, acc)), ("reject C1-strict n>=2", np.where(np.isin(np.arange(len(p)), rm), False, acc)), ("both", new)):
    f1 = f05_vec(np.bincount(s1idx, weights=A & (y == 1), minlength=n), np.bincount(s1idx, weights=A, minlength=n), ngt)
    d, lo, hi = boot(f1 - f0); res[nm] = dict(delta_pp=round(d, 3), ci95=[round(lo, 3), round(hi, 3)], macro=round(f1.mean() * 100, 3))
res["added"] = dict(n=len(add), true=int(y[add].sum())); res["removed"] = dict(n=len(rm), true=int(y[rm].sum()))
print(json.dumps(res, indent=1)); json.dump(res, open(os.path.join(OUT, "rl26b_v1_mapping.json"), "w"), indent=1)
