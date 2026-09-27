"""RL-30 Part 4 (investigator C): France stakes (ESTIMATED F0.5 bounds) + examples with the best same-name rival S1.  READ-ONLY."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street2 import street_sig, rmatch, log
A = accepted("S005_France"); kept = A.kept_final.values.astype(bool)
F = np.load(os.path.join(HERE, "C_test_France_robust.npz")); rs = F["rs"]; hn = F["hn2"]; rival = F["rival"]; EXF = F["EXF"]
full = np.load(os.path.join(HERE, "C_test_France_pairfeats.npz"))["full"]
TD = load("test", verbose=False); s1 = TD["s1"]; s1F = s1[s1.country == "France"]; N_FR = len(s1F); N_ALL = len(s1)
out = dict(n_france_s1=int(N_FR), n_test_s1=int(N_ALL))
K = A[kept].copy(); K["diff"] = (rs < 0.6)[kept]; K["key2"] = ((full >= 0.8) & (rs < 0.6))[kept]
K["rv11"] = (EXF[:, 10] == 11)[kept]
for flag in ("diff", "key2"):
    for extra in ("", "&rv11"):
        f = K[flag] & (K.rv11 if extra else True)
        g = K.assign(f=f).groupby("s1").agg(k=("f", "size"), fl=("f", "sum"))
        g = g[g.fl > 0]; k = g.k.values.astype(float); fl = g.fl.values.astype(float); t = k - fl
        # scenario FP: flagged are FP and the rest are all the GT (n_gt = t): current F vs 1 after removal
        cur_fp = np.where(t > 0, 1.25 * t / (0.25 * t + k), 0.0); gain = (1 - cur_fp).sum()
        # scenario TP: flagged are TP (n_gt = k): removal gives F = 1.25 t / (0.25 k + t)
        loss = (1 - np.where(t > 0, 1.25 * t / (0.25 * k + t), 0.0)).sum()
        out[f"kept_{flag}{extra}"] = dict(pairs=int(fl.sum()), s1=int(len(g)), share_france_s1=round(len(g) / N_FR, 5),
            ESTIMATED_france_macro_gain_if_all_FP_removed=round(gain / N_FR, 5), ESTIMATED_france_macro_loss_if_all_TP_removed=round(loss / N_FR, 5),
            ESTIMATED_overall_LB_gain_if_all_FP=round(gain / N_ALL, 6), ESTIMATED_overall_LB_loss_if_all_TP=round(loss / N_ALL, 6))
        log(flag + extra, out[f"kept_{flag}{extra}"])
# examples: kept KEY2 not flagged by the strict rival test; find the best-matching same-core France S1 (any house number)
s1F = s1F.assign(core=[core_tokens(x) for x in s1F.name.values])
by_core = {}
for i, c in enumerate(s1F.core.values):
    by_core.setdefault(c, []).append(i)
sig_cache = {}
def sig(a):
    v = sig_cache.get(a)
    if v is None:
        v = sig_cache[a] = street_sig(a)
    return v
s1m = s1.set_index("id"); rec = pd.concat([TD["s2"], TD["s3"]]).set_index("id")
idx = np.flatnonzero(kept & (full >= 0.8) & (rs < 0.6) & ~rival)
rng = np.random.default_rng(21); ex = []
n_rival_loose = 0; n_checked = 0
for i in idx:
    r = A.iloc[i]; ra = rec.addr[r.rec]; rsig = sig(ra); c = core_tokens(s1m.name[r.s1])
    best = (-1, None)
    for j in by_core.get(c, ()):
        sid = s1F.id.values[j]
        if sid == r.s1:
            continue
        v = rmatch(sig(s1F.addr.values[j]), rsig)
        if not np.isnan(v[0]) and v[0] > best[0]:
            best = (v[0] + 0.001 * (v[2] == 1), j)
    n_checked += 1
    if best[0] >= 0.8:
        n_rival_loose += 1
    if len(ex) < 14 and rng.random() < 0.02:
        j = best[1]
        ex.append(dict(s1=r.s1, s1_name=s1m.name[r.s1], s1_addr=s1m.addr[r.s1], rec=r.rec, rec_name=rec.name[r.rec], rec_addr=ra,
                       p=round(float(r.p), 4), n_claims=int(r.n_claims), rs=round(float(rs[i]), 3), hn2=int(hn[i]), rv_rank=float(EXF[i, 10]), dupf=float(EXF[i, 9]),
                       best_same_core_s1=None if j is None else s1F.id.values[j], best_s1_addr=None if j is None else s1F.addr.values[j],
                       best_s1_street_sim=round(float(best[0]), 3)))
out["kept_KEY2_not_strict_rival"] = dict(n=int(len(idx)), loose_rival_rs_ge_0_8=int(n_rival_loose), share=round(n_rival_loose / max(n_checked, 1), 4))
out["examples"] = ex
log(out["kept_KEY2_not_strict_rival"])
for e in ex:
    log("EX", e)
json.dump(out, open(os.path.join(HERE, "C_results_france_stakes.json"), "w"), indent=1, default=str)
log("saved")
