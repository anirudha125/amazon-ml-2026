"""RL-07 -- disjoint loss budget (pp of val macro F0.5) for E018C: counterfactual 'fix exactly this category'."""
import sys, os, pickle, collections, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl02_error_decomp import f05
OUT = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(OUT))
gv = pd.read_pickle(os.path.join(OUT, "cache", "val_links_annot.pkl"))
E = pd.read_pickle(os.path.join(OUT, "rl02_errors.pkl"))
d = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl"), "rb"))
th = d["th"]; val = sorted({m[0] for m in d["val_meta"]})
A = collections.defaultdict(set)
for (a, c, y), p in zip(d["val_meta"], d["p_va"]):
    if p >= th: A[a].add(c)
G = gv.groupby("s1").rec.apply(set).to_dict()
def cat(r):
    if r.addr_empty and r.k >= 2: return "L1 empty-addr & shared name (irreducible)"
    if r.addr_empty: return "L2 empty-addr & unique name"
    if r.name_rel == "indic": return "L3 Indic-script name"
    if r.name_rel == "fake": return "L4 fake/substituted name"
    if r.hn in ("d<=2", "1digit", "d<=20", "far") and r.wov >= 0.99: return "L5 house-number perturbed"
    return "L6 other (latin name, address present)"
lost = gv[gv.out != "TP"].copy(); lost["cat"] = lost.apply(cat, axis=1)
lost["stage"] = np.where(lost.out == "miss", "retrieval", "scorer")
base = np.mean([f05(G.get(a, set()), A[a]) for a in val]) * 100
print(f"E018C val macro F0.5 = {base:.2f}\n")
rows = []
for (c, st), grp in lost.groupby(["cat", "stage"]):
    add = collections.defaultdict(set)
    for a, r in zip(grp.s1, grp.rec): add[a].add(r)
    v = np.mean([f05(G.get(a, set()), A[a] | add.get(a, set())) for a in val]) * 100
    rows.append((c, st, len(grp), v - base))
fp = E[E.kind == "fp"].copy()
fp["cat"] = np.where(fp.owner.notna(), "F1 FP owned by another S1", np.where(fp.addr_empty, "F2 FP unlinked, empty addr", "F3 FP unlinked decoy (address present)"))
for c, grp in fp.groupby("cat"):
    rm = collections.defaultdict(set)
    for a, r in zip(grp.s1, grp.cand): rm[a].add(r)
    v = np.mean([f05(G.get(a, set()), A[a] - rm.get(a, set())) for a in val]) * 100
    rows.append((c, "scorer", len(grp), v - base))
T = pd.DataFrame(rows, columns=["category", "stage", "links", "pp_if_fixed"]).sort_values("pp_if_fixed", ascending=False)
pd.set_option("display.width", 200)
print(T.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
print(f"\nsum of pp (approx, non-additive): {T.pp_if_fixed.sum():.2f}  | total gap to 100: {100-base:.2f}")
T.to_csv(os.path.join(OUT, "rl07_budget.csv"), index=False)
