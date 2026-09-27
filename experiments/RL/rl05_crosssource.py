"""RL-05 -- can cross-source agreement resolve the numeric ambiguity found in RL-04?

Hypothesis H-XS: address corruption is applied per (entity, source); a decoy entity carries its changed number in
BOTH sources. So for a candidate record r whose house number differs from the S1's, the existence of another record
(same country, same name core, OTHER source) carrying r's number indicates a separate entity (decoy), while its
absence indicates source noise on a true match. Also checks: does the OTHER source carry the S1's own number?
Falsified if P(match) barely differs between the two groups inside the ambiguous cells.
"""
import sys, os, re, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core
from rl04_numeric_ambiguity import addr_feats

OUT = os.path.dirname(os.path.abspath(__file__))


def main():
    P = pd.read_pickle(os.path.join(OUT, "cache", "rl04_pairs.pkl"))
    amb_cells = {"d<=2", "1digit", "d<=20", "far"}
    P = P[P.hn.isin(amb_cells) & (P.wov >= 0.99)].copy()
    print("ambiguous-cell pairs:", len(P), "positives", int(P.y.sum()))
    D = load("train", verbose=False)
    s1 = D["s1"].set_index("id")
    rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    rec["core"] = rec.name.map(core)
    rec = rec[rec.addr.str.strip() != ""]
    need_cores = set(s1.loc[P.s1.unique()].name.map(core))
    rec = rec[rec.core.isin(need_cores)].copy()
    rec["hn"] = rec.addr.map(lambda a: addr_feats(a)[0])
    rec["src"] = rec.id.str[:2]
    idx = collections.defaultdict(set)            # (country, core, number, src) -> record ids
    for i, c, cr, h, s in zip(rec.id, rec.country, rec.core, rec.hn, rec.src):
        if h:
            idx[(c, cr, h, s)].add(i)
    R = rec.set_index("id")
    out = []
    for a, r in zip(P.s1, P.id):
        c = s1.at[a, "country"]; cr = core(s1.at[a, "name"])
        h_r = R.at[r, "hn"]; src = r[:2]; other = "S3" if src == "S2" else "S2"
        h_s1 = addr_feats(s1.at[a, "addr"])[0]
        out.append((len(idx.get((c, cr, h_r, other), ())) > 0,
                    len(idx.get((c, cr, h_r, src), set()) - {r}) > 0,
                    len(idx.get((c, cr, h_s1, other), ())) > 0,
                    len(idx.get((c, cr, h_s1, src), ())) > 0))
    P["xs_same_num_other_src"], P["same_num_same_src"], P["s1num_in_other_src"], P["s1num_in_same_src"] = zip(*out)
    for col in ["xs_same_num_other_src", "same_num_same_src", "s1num_in_other_src", "s1num_in_same_src"]:
        t = P.groupby(col).y.agg(["size", "sum", "mean"])
        print(f"\n{col}\n{t}")
    t = P.groupby(["xs_same_num_other_src", "s1num_in_same_src"]).y.agg(["size", "sum", "mean"])
    print("\njoint (record number seen in other source) x (S1 number seen in record's own source):\n", t)
    P.to_pickle(os.path.join(OUT, "cache", "rl05_pairs.pkl"))


if __name__ == "__main__":
    main()
