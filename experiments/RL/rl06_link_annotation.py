"""RL-06 -- annotate every train GT link with evidence classes and compute macro-F0.5 ceilings over ALL 2.2M train S1.

Classes per link (record vs its true S1):
  addr_empty, k (#S1 in country sharing the S1 name core), hn (house-number relation, RL-04), wov (address word overlap),
  name_rel: core_eq | overlap (shares >=1 core token) | indic | domain/hashtag/handle | fake (no shared token, Latin, not a url)
Ceilings (every non-ambiguous link found, ambiguous links rejected, no FPs):
  C1 = drop {addr_empty & k>=2}                                   (symmetry argument, near-rigorous)
  C2 = C1 + drop {hn in d<=2/1digit/d<=20/far & wov>=.99}          (numeric ambiguity, coarse posterior 0.2-0.6)
  C3 = C2 + drop {fake name & k-address ambiguity unknown}         (upper bound on fake-name loss)
"""
import sys, os, re, json, pickle
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core, toks, is_indic, f05, SUFFIX
from rl04_numeric_ambiguity import addr_feats, rel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))


def name_rel(rn, sn):
    if is_indic(rn):
        return "indic"
    low = rn.strip().lower()
    if low.startswith("#") or low.startswith("@") or re.search(r"\.(com|in|net|org|co)\b", low):
        return "url"
    a = set(t for t in toks(rn) if t not in SUFFIX); b = set(t for t in toks(sn) if t not in SUFFIX)
    if a == b:
        return "core_eq"
    if a & b:
        return "overlap"
    return "fake"


def ceilings(g, masks, s1_all):
    """macro F0.5 over all S1 (singletons score 1) when links in mask are lost and all others found."""
    res = {}
    n_links = g.groupby("s1").size()
    for nm, m in masks.items():
        lost = g[m].groupby("s1").size()
        df = pd.DataFrame({"n": n_links}).join(lost.rename("lost")).fillna(0)
        found = df.n - df.lost
        # F0.5 with P=1, R=found/n ; zero if found==0
        R = found / df.n
        F = np.where(found > 0, 1.25 * R / (0.25 + R), 0.0)
        tot = F.sum() + (len(s1_all) - len(df))          # singletons (no links) score 1
        res[nm] = 100 * tot / len(s1_all)
    return res


def main():
    ann = os.path.join(OUT, "cache", "gt_annot_full.pkl")
    if os.path.exists(ann):
        g = pd.read_pickle(ann)
    else:
        g = pd.read_pickle(os.path.join(OUT, "cache", "gt_annot.pkl"))       # from RL-03 (s1, rec, k, addr_empty, ...)
        D = load("train", verbose=False)
        S1 = D["s1"].set_index("id"); R = pd.concat([D["s2"], D["s3"]]).set_index("id")
        g["s1_name"] = S1.name.reindex(g.s1).values; g["s1_addr"] = S1.addr.reindex(g.s1).values
        g["r_name"] = R.name.reindex(g.rec).values; g["r_addr"] = R.addr.reindex(g.rec).values
        print("name relations ...", flush=True)
        g["name_rel"] = [name_rel(a, b) for a, b in zip(g.r_name, g.s1_name)]
        print("address relations ...", flush=True)
        F1 = [addr_feats(a) for a in g.s1_addr]; F2 = [addr_feats(a) for a in g.r_addr]
        g["hn"] = [rel(a[0], b[1], b[0]) for a, b in zip(F1, F2)]
        g["wov"] = [len(a[2] & b[2]) / max(1, min(len(a[2]), len(b[2]))) for a, b in zip(F1, F2)]
        g.loc[g.addr_empty, "hn"] = "empty"
        g = g.drop(columns=["s1_name", "s1_addr", "r_name", "r_addr"])
        g.to_pickle(ann)
    D = load("train", verbose=False)
    s1_all = D["s1"].id.values
    print(f"links {len(g):,}")
    print("\nname_rel distribution (% of links):", (g.name_rel.value_counts(normalize=True) * 100).round(2).to_dict())
    print("hn distribution (% of links):", (g.hn.value_counts(normalize=True) * 100).round(2).to_dict())
    amb_num = g.hn.isin(["d<=2", "1digit", "d<=20", "far"]) & (g.wov >= 0.99)
    m1 = g.addr_empty & (g.k >= 2)
    fake = g.name_rel.eq("fake")
    print(f"\nC1 class (addr empty & k>=2): {m1.mean()*100:.2f}% of links")
    print(f"numeric-ambiguous class: {amb_num.mean()*100:.2f}% of links (overlap with C1: {(amb_num & m1).mean()*100:.3f}%)")
    print(f"fake-name links: {fake.mean()*100:.2f}%  of which addr empty {(fake & g.addr_empty).mean()*100:.3f}%")
    masks = {"C0 perfect": np.zeros(len(g), bool), "C1": m1.values, "C2": (m1 | amb_num).values,
             "C3": (m1 | amb_num | (fake & g.addr_empty)).values}
    res = {"all_train": ceilings(g, masks, s1_all)}
    d = pickle.load(open(os.path.join(ROOT, "experiments/TEST_PIPELINE/valpreds_E018C_s42.pkl"), "rb"))
    val = np.array(sorted({m[0] for m in d["val_meta"]}))
    gv = g[g.s1.isin(set(val))]
    mv = {k: v[g.s1.isin(set(val)).values] for k, v in masks.items()}
    res["val_2001"] = ceilings(gv, mv, val)
    for c in ("US", "India"):
        ids = D["s1"].id[D["s1"].country == c].values
        gc = g[g.country == c]
        res["train_" + c] = ceilings(gc, {k: v[(g.country == c).values] for k, v in masks.items()}, ids)
    for k, v in res.items():
        print(k, {a: round(b, 2) for a, b in v.items()})
    json.dump(res, open(os.path.join(OUT, "rl06_ceilings.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
