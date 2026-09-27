"""RL-31 Phase 4 -- cross-source / sibling-consistency evidence on V1 for S006's model (RRL, th .72). Labels = evaluation only.
Question: do records of the same S1 corroborate (or contradict) each other beyond what the pairwise model already uses?
For every V1 S1: accepted set A (p >= th) and borderline rejected set B (0.2 <= p < th).
  sibling agreement (record-record, text only): same name core (non-empty, suffixes dropped) AND
      (same address token multiset, non-empty) OR (same first house number AND address-word Jaccard >= .5)  ->  "twin"
  R+ rescue   : accept b in B if b has a twin in A                       (weak pair score, strong joint evidence)
  R+x         : same, twin must come from the OTHER source (S2 <-> S3)   (cross-source agreement)
  R+w         : accept pairs of mutually-twin records both in B (two weak scores that agree)
  R- veto     : reject a in A if A contains a higher-p record from the other source that CONTRADICTS a
                (name cores disjoint AND both have house numbers that differ)   (strong scores that conflict)
Reported: firing counts, precision on V1 labels, paired delta vs RRL; plus the upper bound (oracle: apply only correct firings).
Output: p4_crosssource.json
"""
import os, sys, re, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *
from rl_data import load
from rl02_error_decomp import toks, SUFFIX, fold
from rl04_numeric_ambiguity import addr_feats

V = load_v1(); y = V["y"]; si = V["s1idx"]; n = len(V["s1_ids"]); p = V["p_RRL"]; th = V["th_RRL"]
D = load("train", verbose=False)
R = pd.concat([D["s2"], D["s3"]]).set_index("id")
rows = np.flatnonzero(p >= 0.2)
recs = V["cand"][rows]
nm = R.name.reindex(recs).values; ad = R.addr.reindex(recs).values
core = [frozenset(t for t in toks(x) if t not in SUFFIX) for x in nm]
akey = [" ".join(sorted(re.findall(r"[a-z0-9]+", fold(a)))) for a in ad]
af = [addr_feats(a) for a in ad]
src = np.array([c.startswith("S2") for c in recs])
by_s1 = collections.defaultdict(list)
for j, i in enumerate(rows):
    by_s1[si[i]].append(j)

def twin(a, b):
    if not core[a] or core[a] != core[b]:
        return False
    if akey[a] and akey[a] == akey[b]:
        return True
    na, nb = af[a][0], af[b][0]
    if na is not None and na == nb:
        wa, wb = af[a][2], af[b][2]
        return len(wa | wb) > 0 and len(wa & wb) / len(wa | wb) >= 0.5
    return False

def contra(a, b):
    na, nb = af[a][0], af[b][0]
    return bool(core[a]) and bool(core[b]) and not (core[a] & core[b]) and na is not None and nb is not None and na != nb

fire = {k: np.zeros(len(p), bool) for k in ("R+", "R+x", "R+w", "R-")}
for s, js in by_s1.items():
    A = [j for j in js if p[rows[j]] >= th]; B = [j for j in js if p[rows[j]] < th]
    for b in B:
        tw = [a for a in A if twin(a, b)]
        if tw:
            fire["R+"][rows[b]] = True
            if any(src[a] != src[b] for a in tw):
                fire["R+x"][rows[b]] = True
        for b2 in B:
            if b2 != b and twin(b, b2):
                fire["R+w"][rows[b]] = True
    for a in A:
        for a2 in A:
            if a2 != a and src[a2] != src[a] and p[rows[a2]] > p[rows[a]] and contra(a, a2):
                fire["R-"][rows[a]] = True; break

acc0 = p >= th; f0 = macro(V, acc0); out = dict(base=round(f0.mean() * 100, 3))
for k, fm in fire.items():
    acc = (acc0 | fm) if k != "R-" else (acc0 & ~fm)
    corr = fm & ((y == 1) if k != "R-" else (y == 0))
    acc_or = (acc0 | corr) if k != "R-" else (acc0 & ~corr)
    out[k] = dict(fires=int(fm.sum()), correct=int(corr.sum()), precision=round(float(corr.sum() / max(fm.sum(), 1)), 3),
                  delta=boot_delta(macro(V, acc) - f0), oracle_delta=boot_delta(macro(V, acc_or) - f0),
                  by_band={b: [int((fm & (p >= lo) & (p < hi)).sum()), int((corr & (p >= lo) & (p < hi)).sum())]
                           for b, lo, hi in (("0.2-0.4", .2, .4), ("0.4-0.6", .4, .6), ("0.6-th", .6, th), (">=th", th, 1.01))})
# how many rejected true links (FN, in pool, p>=.2) exist at all, and how many have a twin in A
fn = (y == 1) & (p < th) & (p >= 0.2)
out["FN_p_ge_0.2"] = int(fn.sum()); out["FN_all_inpool"] = int(((y == 1) & (p < th)).sum())
out["FN_p_ge_0.2_with_twin_in_A"] = int((fn & fire["R+"]).sum())
json.dump(out, open(os.path.join(HERE, "p4_crosssource.json"), "w"), indent=1)
print(json.dumps(out, indent=1))
