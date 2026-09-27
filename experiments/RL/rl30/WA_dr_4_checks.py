"""RL-30 verifier WA, claim 'S1_side_token_drop_rate', PART 4 (READ-ONLY): (a) which record-only tokens 'explain' abbreviation drops
(checks the abbreviation detector is not fooled by the generic '.com' adder); (b) label-backed position effect in TRAIN GT (US/India).
Output: WA_dr_4_results.json"""
import os, sys, json
from collections import Counter
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *
from rapidfuzz.distance import Levenshtein as LV
R = {}
D = load("test", verbose=False)
S1 = D["s1"].set_index("id"); REC = pd.concat([D["s2"], D["s3"]]).set_index("id")
a = accepted("S005_France"); a = a[a.kept_final & (a.p >= 0.99) & (a.n_claims == 1)]
s1n = S1.name.reindex(a.s1.values).values; s1a = S1.addr.reindex(a.s1.values).values
rn = REC.name.reindex(a.rec.values).values; ra = REC.addr.reindex(a.rec.values).values
W = ("comite", "compagnie", "freres", "communale", "saint", "etablissements")
cnt = {w: Counter() for w in W}; ex = {w: [] for w in W}
def subseq(r, t):
    it = iter(t); return all(ch in it for ch in r)
for k in range(len(a)):
    if not isinstance(rn[k], str): continue
    A = set(toks(s1n[k])); B = set(toks(rn[k]))
    for w in W:
        if w in A and w not in B:
            n1, st1, _ = street_parts(s1a[k]); n2, st2, _ = street_parts(ra[k]) if isinstance(ra[k], str) else (None, frozenset(), None)
            if not (n1 is not None and n1 == n2 and st1 and st1 == st2): continue
            ab = [r for r in (B - A) if len(r) >= 2 and len(r) < len(w) and r[0] == w[0] and subseq(r, w) and not r.isdigit()]
            cnt[w]["drops"] += 1
            for r in ab: cnt[w][r] += 1
            if ab and len(ex[w]) < 4: ex[w].append((s1n[k], rn[k]))
R["abbrev_explainers_France_PP"] = {w: cnt[w].most_common(6) for w in W}
R["abbrev_examples"] = ex
P = pd.read_csv(os.path.join(OUT, "WA_dr_2_trainGT_pos.csv"))
P = P[(P.street) & (P.cls == "content")]
R["trainGT_content_drop_by_pos"] = {f"{c}_{p}_{n}": dict(n=int(r["has"]), dr=round(float(r["drop"] / r["has"]), 4))
                                    for (c, p, n), r in P.groupby(["country", "pos", "nlen"])[["has", "drop"]].sum().iterrows() if r["has"] >= 3000}
json.dump(R, open(os.path.join(OUT, "WA_dr_4_results.json"), "w"), indent=1, ensure_ascii=False)
print(json.dumps(R, ensure_ascii=False))
