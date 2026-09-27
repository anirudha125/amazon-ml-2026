"""RL-18 -- R1/R2 vetoes (RL-17 definitions) with and without max-claimer on the E020 dense slices (PRODUCTION E018C scores,
every train S1 of Jaipur / Oregon; eval S1 = slice minus the 11,994 stage-2/reranker training S1 -- same protocol as E020).
This is the labelled proxy for stacking the rules on S003 (test = lexical pool + E018C + max-claimer).
Parity gate: 'none' and 'max' must reproduce e020_results.json. Read-only inputs; CPU only.
"""
import os, sys, json, gzip, pickle, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core
from rl04_numeric_ambiguity import addr_feats, rel
from rl17_context_rules import corpus_index, AMB

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
E20 = os.path.join(ROOT, "experiments", "E020_maxclaimer")
TAGS = ["India_jaipur", "US_OR"]
TAB = chr(9)


def f05(pred, g):
    if not g:
        return 1.0 if not pred else 0.0
    tp = len(pred & g)
    return 0.0 if tp == 0 else 1.25 * tp / (0.25 * len(g) + len(pred))


def maxclaim(preds):
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, p in v:
            cl[c].append((-p, s))
    win = {c: min(L)[1] for c, L in cl.items()}
    return {s: [(c, p) for c, p in v if win[c] == s] for s, v in preds.items()}


def boot(a, b, n=10000, seed=0):
    d = np.asarray(b) - np.asarray(a); rng = np.random.default_rng(seed)
    bs = d[rng.integers(0, len(d), size=(n, len(d)))].mean(1)
    return round(d.mean() * 100, 3), [round(np.percentile(bs, 2.5) * 100, 3), round(np.percentile(bs, 97.5) * 100, 3)]


def main():
    D = load("train", verbose=False)
    S1 = D["s1"].set_index("id"); R = pd.concat([D["s2"], D["s3"]]).set_index("id")
    IX = corpus_index("train"); ridx, sidx = IX["ridx"], IX["sidx"]
    train_s1 = set()
    with gzip.open(os.path.join(ROOT, "experiments", "E017_embed", "pkg", "pairs.tsv.gz"), "rt") as f:
        f.readline()
        for l in f:
            p = l.split(TAB, 4)
            if p[0] == "train":
                train_s1.add(p[3])
    G = D["gt"].groupby("s1").rec.apply(set).to_dict()
    ref = json.load(open(os.path.join(E20, "e020_results.json")))
    out = {}
    for t in TAGS:
        d = pickle.load(open(os.path.join(E20, f"preds_{t}.pkl"), "rb"))
        P = d["preds"]; ev = sorted(s for s in d["ids"] if s not in train_s1)
        flags = {}
        for a, v in P.items():
            c = S1.at[a, "country"]; ca = core(S1.at[a, "name"]); fa = addr_feats(S1.at[a, "addr"])
            for r, p in v:
                ra = R.at[r, "addr"]; r1 = r2 = False
                if ca and fa[0] and ra.strip():
                    fr = addr_feats(ra); hn = rel(fa[0], fr[1], fr[0])
                    if fr[0] and hn != "exact":
                        if hn in AMB:
                            r1 = ridx.get((c, ca, fa[0], r[:2]), 0) > 0
                        r2 = sidx.get((c, ca, fr[0]), 0) > 0 and fr[0] != fa[0]
                flags[(a, r)] = (r1, r2)
        def veto(Pin, which):
            return {s: [(c, p) for c, p in v if not any(flags[(s, c)][k] for k in which)] for s, v in Pin.items()}
        arms = {"none": P, "max": maxclaim(P), "R1": veto(P, [0]), "R2": veto(P, [1]), "R1|R2": veto(P, [0, 1])}
        arms["max+R2"] = veto(arms["max"], [1]); arms["max+R1|R2"] = veto(arms["max"], [0, 1])
        sc = {k: np.array([f05({c for c, _ in A.get(s, [])}, G.get(s, set())) for s in ev]) for k, A in arms.items()}
        res = {"n_eval_s1": len(ev)}
        for k in arms:
            rm = [(s, c) for s in ev for c, _ in P.get(s, []) if c not in {x for x, _ in arms[k].get(s, [])}]
            tl = sum(1 for s, c in rm if c in G.get(s, set()))
            res[k] = dict(macro=round(sc[k].mean() * 100, 3), removed=len(rm), removed_true=tl, removed_false=len(rm) - tl)
        for k in ["max", "R1", "R2", "R1|R2", "max+R2", "max+R1|R2"]:
            res[k]["delta_vs_none"], res[k]["ci95_vs_none"] = boot(sc["none"], sc[k])
        for k in ["max+R2", "max+R1|R2"]:
            res[k]["delta_vs_max"], res[k]["ci95_vs_max"] = boot(sc["max"], sc[k])
        res["parity_vs_E020"] = dict(none=(res["none"]["macro"], ref[t]["E018C"]["macro_none"]),
                                     max=(res["max"]["macro"], ref[t]["E018C"]["max"]["macro"]))
        out[t] = res
        print(t, json.dumps(res), flush=True)
    json.dump(out, open(os.path.join(OUT, "rl18_slice_rules.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
