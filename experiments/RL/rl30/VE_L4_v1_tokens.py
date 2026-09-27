"""RL-30 / VE-L: label-backed token audit on V1 (US/India): for SAME-ADDRESS core near-dup pairs, per differing token,
P(match), NEW acceptance, NEW errors, and E's label-free s(t) (train corpus).  Tests whether the label-free 'filler' label
(low s) can tell noise tokens from systematic decoy tokens, and whether NEW already gets these cells right.
READ-ONLY; uses VE_L1_feats.npz (row-aligned with the V1 meta).  Writes rl30/VE_L4_v1_tokens.json"""
import os, sys, json, pickle, collections
import numpy as np, pandas as pd
from scipy.stats import spearmanr
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *

M = np.load(PATHS["v1_meta"]); s1_ids = M["s1_ids"]; s1idx = M["s1idx"]; cand = M["cand"]; y = M["y"] == 1; pc = M["country"][s1idx]
p = np.load(PATHS["v1_p_new"]); acc = p >= NEW_TH
F = np.load(os.path.join(HERE, "VE_L1_feats.npz"))
nd, sa = F["nd"], F["same_addr"]
D = load("train", verbose=False)
S1 = D["s1"].set_index("id"); recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
RO = pickle.load(open(os.path.join(HERE, "E_roles_train.pkl"), "rb"))
out = {}
for lab, m0 in (("same_addr_nd", nd & sa), ("not_same_addr_nd_p>=0.05", nd & ~sa & (p >= 0.05))):
    ii = np.flatnonzero(m0)
    ps1 = s1_ids[s1idx[ii]]
    A = [nset(n) for n in S1.name.loc[ps1].values]; R = [nset(n) for n in recs.name.loc[cand[ii]].values]
    st = collections.defaultdict(lambda: np.zeros(6, int))   # n, pos, acc, FP, FN, rec_adds
    for k, i in enumerate(ii):
        a, b = A[k] - R[k], R[k] - A[k]
        for t in a | b:
            key = (pc[i], t)
            v = st[key]; v[0] += 1; v[1] += y[i]; v[2] += acc[i]; v[3] += acc[i] & ~y[i]; v[4] += (~acc[i]) & y[i]; v[5] += int(t in b and not a)
    res = {}
    for c in ("US", "India"):
        T = RO["roles"][c]
        rows = [(t, *map(int, v), round(float(T.s.loc[t]), 3) if t in T.index else None) for (cc, t), v in st.items() if cc == c and v[0] >= 15]
        rows.sort(key=lambda r: -r[1])
        df = pd.DataFrame(rows, columns=["t", "n", "pos", "acc", "FP", "FN", "rec_adds_only", "s"])
        dk = df.dropna()
        res[c] = dict(n_tokens=int(len(dk)),
                      spearman_s_vs_Pmatch=round(float(spearmanr(dk.s, dk.pos / dk.n).correlation), 4) if len(dk) > 5 else None,
                      spearman_s_vs_NEWacc=round(float(spearmanr(dk.s, dk.acc / dk.n).correlation), 4) if len(dk) > 5 else None,
                      spearman_NEWacc_vs_Pmatch=round(float(spearmanr(dk.acc / dk.n, dk.pos / dk.n).correlation), 4) if len(dk) > 5 else None,
                      filler_s_lt_0_3_but_Pmatch_lt_0_5=[r for r in dk[(dk.s < 0.3) & (dk.pos / dk.n < 0.5)].head(15).itertuples(index=False, name=None)],
                      top=[r for r in df.head(30).itertuples(index=False, name=None)],
                      totals=dict(pairs=int(len(ii[pc[ii] == c])), FP=int((acc[ii] & ~y[ii] & (pc[ii] == c)).sum()), FN=int((~acc[ii] & y[ii] & (pc[ii] == c)).sum())))
    out[lab] = res
    print(lab, json.dumps(res, default=str)[:3500], flush=True)
json.dump(out, open(os.path.join(HERE, "VE_L4_v1_tokens.json"), "w"), indent=1, default=str)
