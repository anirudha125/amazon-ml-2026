"""RL-30 investigator B -- label-backed V1 analog of the France 'generic business word substitution' pattern.
V1 pairs with p_new >= 0.1 whose folded name tokens differ by exactly one token on each side (sharing >= 1 token), excluding pairs where
either differing token is a legal form / honorific (rl30_lib LEGAL, HONOR). Bucket by the IDF (train S1 corpus, per country) of the
differing tokens; report y-rate vs mean p and accepted precision vs expected. Also the same for single-token add/drop.
READ-ONLY; writes rl30/B_v1_subst.json.
"""
import os, sys, json, time, math, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl30_lib import *  # noqa

t0 = time.time()


def main():
    m = np.load(PATHS["v1_meta"], allow_pickle=True)
    s1_ids, s1idx, cand, y = m["s1_ids"], m["s1idx"], m["cand"], m["y"].astype(int)
    ctry = m["country"][s1idx]; p = np.load(PATHS["v1_p_new"]).astype(float)
    D = load("train", verbose=False); s1 = D["s1"]; rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    idf = {}
    for c in ("US", "India"):
        dfc = collections.Counter(); names = s1.name.values[s1.country.values == c]
        for x in names:
            dfc.update(set(toks(x)))
        N = len(names); idf[c] = {t: math.log((N - d + 0.5) / (d + 0.5) + 1.0) for t, d in dfc.items()}; idf[c]["__N"] = N
    rel = np.flatnonzero(p >= 0.1)
    an = s1.name.values[pd.Index(s1.id.values).get_indexer(s1_ids[s1idx[rel]])]
    rn = rec.name.values[pd.Index(rec.id.values).get_indexer(cand[rel])]
    skip = LEGAL | HONOR
    rows = []
    for k, i in enumerate(rel):
        A, B = frozenset(toks(an[k])), frozenset(toks(rn[k]))
        d1, d2 = A - B, B - A
        if not (A & B):
            continue
        c = ctry[i]; unseen = math.log((idf[c]["__N"] + 0.5) / 0.5 + 1.0)
        if len(d1) == 1 and len(d2) == 1:
            t1, t2 = next(iter(d1)), next(iter(d2))
            if t1 in skip or t2 in skip or t1.isdigit() or t2.isdigit():
                continue
            rows.append(("sub", c, max(idf[c].get(t1, unseen), idf[c].get(t2, unseen)), y[i], p[i], t1, t2))
        elif (len(d1) + len(d2)) == 1:
            t = next(iter(d1 | d2))
            if t in skip or t.isdigit():
                continue
            rows.append(("adddrop", c, idf[c].get(t, unseen), y[i], p[i], t, ""))
    df = pd.DataFrame(rows, columns=["kind", "country", "idf", "y", "p", "t1", "t2"])
    df["idf_bin"] = pd.cut(df.idf, [0, 4, 5, 5.5, 6, 7, 8, 30])
    out = {}
    for (kind, c), g in df.groupby(["kind", "country"]):
        tab = []
        for b, h in g.groupby("idf_bin", observed=True):
            acc = h[h.p >= NEW_TH]
            tab.append(dict(idf_bin=str(b), n=len(h), y_rate=round(h.y.mean(), 4), mean_p=round(h.p.mean(), 4),
                            accepted=len(acc), acc_precision=round(acc.y.mean(), 4) if len(acc) else None,
                            acc_FP=int((acc.y == 0).sum()), acc_expected_FP=round(float((1 - acc.p).sum()), 1),
                            examples=[f"{a}->{b_}" if b_ else a for a, b_ in h.sample(min(4, len(h)), random_state=0)[["t1", "t2"]].values]))
        out[f"{kind}_{c}"] = tab
    out["runtime_s"] = round(time.time() - t0, 1)
    json.dump(out, open(os.path.join(OUT, "B_v1_subst.json"), "w"), indent=1, default=str)
    for k, v in out.items():
        if k == "runtime_s":
            continue
        print("==", k)
        for r in v:
            print("  ", r)


if __name__ == "__main__":
    main()
