"""RL-19 -- label-free feasibility of R1/R2 on test: how often would each veto fire on S003's accepted pairs, per country?
Indexes are built from the unlabelled TEST corpus (test records / test S1), exactly as they would be in deployment.
No submission is written; nothing is tuned. Read-only inputs; CPU only.
"""
import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core
from rl04_numeric_ambiguity import addr_feats, rel
from rl17_context_rules import corpus_index, AMB

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
S003 = os.path.join(ROOT, "experiments", "TEST_PIPELINE", "submission_S003_E018C_s42_maxclaim", "matching_results.tsv")
TAB, NL = chr(9), chr(10)


def main():
    T = load("test", verbose=False)
    S1 = T["s1"].set_index("id"); R = pd.concat([T["s2"], T["s3"]]).set_index("id")
    IX = corpus_index("test"); ridx, sidx = IX["ridx"], IX["sidx"]
    st = {c: collections.Counter() for c in ("US", "India", "France")}
    s1_hit = {c: collections.Counter() for c in st}
    with open(S003, encoding="utf-8") as f:
        f.readline()
        for line in f:
            a, m = line.rstrip(NL).split(TAB)
            if not m:
                continue
            L = m.split(","); c = S1.at[a, "country"]; ca = core(S1.at[a, "name"]); fa = addr_feats(S1.at[a, "addr"])
            n1 = n2 = n12 = 0
            for r in L:
                st[c]["accepted"] += 1
                ra = R.at[r, "addr"]; r1 = r2 = False
                if ca and fa[0] and ra.strip():
                    fr = addr_feats(ra); hn = rel(fa[0], fr[1], fr[0])
                    if fr[0] and hn != "exact":
                        st[c]["num_mismatch"] += 1
                        if hn in AMB:
                            r1 = ridx.get((c, ca, fa[0], r[:2]), 0) > 0
                        r2 = sidx.get((c, ca, fr[0]), 0) > 0 and fr[0] != fa[0]
                n1 += r1; n2 += r2; n12 += (r1 or r2)
            st[c]["R1"] += n1; st[c]["R2"] += n2; st[c]["R1|R2"] += n12
            s1_hit[c]["R1|R2_s1_affected"] += n12 > 0
            s1_hit[c]["R1|R2_s1_emptied"] += n12 == len(L)
    res = {}
    for c in st:
        x = dict(st[c]); x.update(s1_hit[c]); a = x["accepted"]
        for k in ("R1", "R2", "R1|R2"):
            x[k + "_per_1k"] = round(1000 * x[k] / a, 2)
        x["test_s1"] = int((T["s1"].country == c).sum())
        x["R1|R2_s1_affected_pct"] = round(100 * x["R1|R2_s1_affected"] / x["test_s1"], 3)
        res[c] = x
    print(json.dumps(res, indent=1))
    json.dump(res, open(os.path.join(OUT, "rl19_test_firing.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
