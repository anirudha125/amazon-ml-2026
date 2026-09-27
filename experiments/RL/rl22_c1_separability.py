"""RL-22 -- is the C1 class really symmetric? Measure how well D2b separates the OWNER from same-name NON-owners on
empty-address records, and which information breaks the symmetry.

Population (V1 union pool): pairs (a, r) with r's address empty, r linked to owner o, core(o) == core(a) (same country).
  positive: o == a ; negative: o != a (a same-name sibling of the owner)
Candidate symmetry breakers:
  full-name identity: normalized FULL names (suffixes kept) of a and o identical -> strict symmetry
  suffix agreement  : record's legal-suffix/honorific tokens == a's
  count context     : number of OTHER accepted candidates of a in r's source (generator prior on per-source counts)
Read-only; CPU only. Output: rl22_c1_separability.json
"""
import os, sys, re, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import toks, SUFFIX

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
E24 = os.path.join(ROOT, "experiments", "E024")


def auc(s, y):
    s, y = np.asarray(s), np.asarray(y).astype(bool)
    if y.all() or (~y).all():
        return float("nan")
    r = pd.Series(s).rank().values
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum()))


def main():
    D = load("train", verbose=False)
    C = pd.read_pickle(os.path.join(OUT, "cache", "ctx_train.pkl"))
    core_of = dict(zip(C["s1"].id, C["s1"].core)); ctry = dict(zip(D["s1"].id, D["s1"].country))
    s1name = dict(zip(D["s1"].id, D["s1"].name))
    R = pd.concat([D["s2"], D["s3"]]).set_index("id")
    empty = set(R.index[R.addr.str.strip() == ""])
    owner = dict(zip(D["gt"].rec, D["gt"].s1))
    full = lambda s: " ".join(toks(s))
    suff = lambda s: frozenset(t for t in toks(s) if t in SUFFIX)
    m = np.load(os.path.join(E24, "V1_a50n10d10a", "meta.npz"), allow_pickle=True)
    p = np.load(os.path.join(E24, "p_D2b_union_rrUb_big_V1_s42.npy")); th = 0.72
    s1_ids, s1idx, cand = m["s1_ids"], m["s1idx"], m["cand"]
    acc = p >= th
    # other accepted candidates per (S1, source)
    src = np.array([c[:2] for c in cand])
    df = pd.DataFrame({"j": s1idx, "src": src, "acc": acc})
    n_acc = df.groupby(["j", "src"]).acc.sum().to_dict()
    rows = []
    for i in range(len(cand)):
        r = cand[i]
        if r not in empty:
            continue
        o = owner.get(r); a = s1_ids[s1idx[i]]
        if o is None or core_of.get(o) != core_of.get(a) or ctry[o] != ctry[a] or not core_of.get(a):
            continue
        y = o == a
        other = n_acc.get((s1idx[i], src[i]), 0) - int(acc[i])
        rows.append(dict(a=a, r=r, y=y, p=float(p[i]), acc=bool(acc[i]),
                         strict=(full(s1name[a]) == full(s1name[o])) if not y else None,
                         suffix_agree=suff(R.at[r, "name"]) == suff(s1name[a]), other_acc_same_src=other))
    X = pd.DataFrame(rows)
    # strictness for positives: does ANY same-core sibling share the owner's full name?
    fam_full = pd.DataFrame({"id": C["s1"].id, "core": C["s1"].core, "c": C["s1"].country})
    fam_full["full"] = fam_full.id.map(lambda s: full(s1name[s]))
    dup_full = fam_full.groupby(["c", "core", "full"]).size()
    X["owner_full_name_duplicated"] = [dup_full.get((ctry[a], core_of[a], full(s1name[a])), 1) >= 2 for a in X.a]
    pos, neg = X[X.y], X[~X.y]
    res = dict(n_pairs=len(X), n_pos=len(pos), n_neg=len(neg),
               auc_p=round(auc(X.p, X.y), 3),
               accepted_pos=int(pos.acc.sum()), accepted_neg=int(neg.acc.sum()),
               precision_of_acceptance=round(pos.acc.sum() / max(1, X.acc.sum()), 3))
    # strict (identical full names across family) vs loose
    for nm, g in (("strict_identical_names", X[X.owner_full_name_duplicated | (X.strict == True)]),
                  ("loose_suffix_differs", X[~(X.owner_full_name_duplicated | (X.strict == True))])):
        res[nm] = dict(n=len(g), pos=int(g.y.sum()), auc_p=round(auc(g.p, g.y), 3),
                       accepted_pos=int((g.y & g.acc).sum()), accepted_neg=int((~g.y & g.acc).sum()))
    res["suffix_agree_rate_pos_vs_neg"] = [round(pos.suffix_agree.mean(), 3), round(neg.suffix_agree.mean(), 3)]
    res["auc_suffix_agree"] = round(auc(X.suffix_agree.astype(float), X.y), 3)
    res["auc_fewer_other_accepted_same_src"] = round(auc(-X.other_acc_same_src, X.y), 3)
    res["P(pos | other accepted in source = k)"] = {int(k): [int(len(g)), round(g.y.mean(), 3)]
                                                  for k, g in X.groupby(X.other_acc_same_src.clip(upper=4))}
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "rl22_c1_separability.json"), "w"), indent=1)
    X.to_pickle(os.path.join(OUT, "cache", "rl22_c1_pairs.pkl"))


if __name__ == "__main__":
    main()
