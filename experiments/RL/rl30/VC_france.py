"""RL-30 adversarial verifier VC (lens: ALREADY CAPTURED / NOT INCREMENTAL) -- France test (unlabelled) part.
For C's KEY2 flag on S005 France accepted pairs: (1) how does RL-27 NEW already treat them (p vs S004, kept vs S004),
(2) do existing stage-2 columns (LF from P3 chunks, RL-27 test feats, rrUb reranker logit) already encode the flag,
(3) how much of KEY2 lies inside cells that existing columns already isolate.  READ-ONLY; writes only rl30/VC_*.  Label-free.
Run: OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 10 python VC_france.py
"""
import os, sys, json, time, glob, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street2 import robust_pairs
from sklearn.metrics import roc_auc_score

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)
def auc(y, x):
    y = np.asarray(y); x = np.asarray(x, np.float64)
    if len(y) < 20 or y.min() == y.max():
        return None
    return round(float(roc_auc_score(y, np.nan_to_num(x, nan=-9.0))), 4)
RES = {}

A5 = accepted("S005_France"); n = len(A5)
P1 = np.load(os.path.join(HERE, "C_test_France_pairfeats.npz")); full = P1["full"]; ceq = P1["ceq"]
P2 = np.load(os.path.join(HERE, "C_test_France_robust.npz")); rs = P2["rs"]; hn2 = P2["hn2"]; rival = P2["rival"]; EXF = P2["EXF"]
RLC = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
R = {c: EXF[:, 4 + j] for j, c in enumerate(RLC)}
p5 = A5.p.values; kept = A5.kept_final.values.astype(bool); ncl = A5.n_claims.values
both = ~np.isnan(full) & ~np.isnan(rs)
KEY2 = both & (full >= 0.8) & (rs < 0.6); DIFF = both & (rs < 0.6)
log("S005 France accepted", n, "KEY2", int(KEY2.sum()), "kept KEY2", int((KEY2 & kept).sum()))

# ---------------------------------------------------------------- 1. how RL-27 NEW already treats KEY2 (p levels, S004 -> S005)
def pstats(m):
    return dict(n=int(m.sum()), mean_p=round(float(p5[m].mean()), 4), median_p=round(float(np.median(p5[m])), 4),
                share_p_lt_0_99=round(float((p5[m] < 0.99).mean()), 4), share_p_lt_0_9=round(float((p5[m] < 0.9).mean()), 4),
                ESTIMATED_fp_if_calibrated_sum_1_minus_p=round(float((1 - p5[m]).sum()), 1))
RES["p_levels"] = {"kept_all": pstats(kept), "kept_street_match": pstats(kept & both & (rs >= 0.8)), "kept_KEY2": pstats(kept & KEY2),
                   "kept_KEY2&same_core": pstats(kept & KEY2 & ceq), "kept_DIFF": pstats(kept & DIFF),
                   "accepted_KEY2": pstats(KEY2)}
for k, v in RES["p_levels"].items():
    log("P", k, v)

A4 = accepted("S004_France")
# S004 max-claimer (p3_test.maxclaim: winner = max p, ties -> smallest S1 id)
o = A4.sort_values(["rec", "p", "s1"], ascending=[True, False, True])
win = o.drop_duplicates("rec", keep="first").set_index("rec").s1
A4["kept"] = A4.s1.values == win.reindex(A4.rec.values).values
# sanity: same rule reproduces S005 kept_final?
o5 = A5.sort_values(["rec", "p", "s1"], ascending=[True, False, True]); w5 = o5.drop_duplicates("rec", keep="first").set_index("rec").s1
rule5 = A5.s1.values == w5.reindex(A5.rec.values).values
RES["maxclaim_rule_reproduces_S005_kept_final"] = round(float((rule5 == kept).mean()), 6)
k4 = pd.Series(np.arange(len(A4)), index=pd.MultiIndex.from_arrays([A4.s1.values, A4.rec.values]))
j4 = k4.reindex(pd.MultiIndex.from_arrays([A5.s1.values, A5.rec.values])).values
in4 = ~np.isnan(j4); jj = j4[in4].astype(np.int64)
p4 = np.full(n, np.nan); p4[in4] = A4.p.values[jj]; kept4 = np.zeros(n, bool); kept4[in4] = A4.kept.values[jj]
d = {}
for nm, m in [("KEY2", KEY2), ("KEY2&kept5", KEY2 & kept), ("street_match&kept5", both & (rs >= 0.8) & kept), ("all_accepted5", np.ones(n, bool))]:
    mm = m & in4
    d[nm] = dict(n=int(m.sum()), in_S004_accepted=int(mm.sum()), mean_p_S004=round(float(p4[mm].mean()), 4), mean_p_S005=round(float(p5[mm].mean()), 4),
                 share_p_decreased=round(float((p5[mm] < p4[mm]).mean()), 4), mean_delta_logit=round(float(np.mean(
                     np.log(p5[mm] / (1 - p5[mm] + 1e-9) + 1e-9) - np.log(p4[mm] / (1 - p4[mm] + 1e-9) + 1e-9))), 4),
                 kept_in_S004=int((m & kept4).sum()), kept_in_S005=int((m & kept).sum()))
    log("S004->S005", nm, d[nm])
# S004-accepted pairs that S005 rejected: compute KEY2 on them
m5 = pd.Series(1, index=pd.MultiIndex.from_arrays([A5.s1.values, A5.rec.values]))
only4 = ~pd.MultiIndex.from_arrays([A4.s1.values, A4.rec.values]).isin(m5.index)
TD = load("test", verbose=False); s1 = TD["s1"].set_index("id"); rec = pd.concat([TD["s2"], TD["s3"]]).set_index("id")
B = A4[only4]
sa = s1.addr.reindex(B.s1.values).fillna("").values.astype(object); ra = rec.addr.reindex(B.rec.values).fillna("").values.astype(object)
rsB, _, hnB = robust_pairs(sa, ra)
fB = np.full(len(B), np.nan, np.float32)
for k, (a, b) in enumerate(zip(sa, ra)):
    if a and b and a.strip().lower() != "null" and b.strip().lower() != "null":
        W1 = addr_words(a); W2 = addr_words(b)
        if W1 and W2:
            fB[k] = len(W1 & W2) / min(len(W1), len(W2))
K2B = ~np.isnan(fB) & ~np.isnan(rsB) & (fB >= 0.8) & (rsB < 0.6)
d["S004_accepted_S005_rejected"] = dict(n=int(len(B)), KEY2=int(K2B.sum()), KEY2_kept_in_S004=int((K2B & B.kept.values).sum()),
                                        kept_in_S004=int(B.kept.values.sum()), KEY2_share=round(float(K2B.mean()), 4))
kept4_KEY2_total = int((KEY2 & kept4).sum() + (K2B & B.kept.values).sum())
d["S004_kept_KEY2_total"] = kept4_KEY2_total; d["S005_kept_KEY2_total"] = int((KEY2 & kept).sum())
s1_changed = set(A5.s1.values[KEY2 & (kept != kept4)]) | set(B.s1.values[K2B & B.kept.values])
d["S1_whose_kept_KEY2_set_changed_S004_to_S005"] = int(len(s1_changed))
RES["S004_to_S005"] = d
log("S004->S005 extra", {k: v for k, v in d.items() if not isinstance(v, dict) or k.startswith("S004_acc")})

# ---------------------------------------------------------------- 2. existing columns (LF from P3 chunks + rrUb) -> AUC for the KEY2 flag
LFC = {"name_tset3": 3, "name_jacc4": 4, "log_s1freq5": 5, "addr_lev11": 11, "addr_jw12": 12, "addr_tsort13": 13, "addr_tset14": 14,
       "addr_jacc15": 15, "first_num_agree56": 56, "house_agree57": 57, "house_absdiff58": 58, "addr_n_s1_only83": 83, "addr_n_c_only84": 84,
       "addr_frac_s1_only85": 85, "addr_frac_c_only86": 86}
L = {k: np.full(n, np.nan, np.float32) for k in LFC}; rdense = np.full(n, np.nan, np.float32); dcos = np.full(n, np.nan, np.float32)
key = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([A5.s1.values, A5.rec.values]))
cols = list(LFC.values())
for f in sorted(glob.glob(PATHS["test_chunks"].format(country="France"))):
    z = np.load(f, allow_pickle=True)
    j = key.reindex(pd.MultiIndex.from_arrays([z["s1"], z["cand"]])).values
    ok = ~np.isnan(j)
    if not ok.any():
        continue
    jj = j[ok].astype(np.int64); LFc = z["LF"][ok][:, cols]
    for c, k in enumerate(LFC):
        L[k][jj] = LFc[:, c]
    rdense[jj] = z["rank_dense"][ok]; dcos[jj] = z["dcos"][ok]
log("LF joined", int((~np.isnan(L["addr_tset14"])).sum()))
rr = pickle.load(open(os.path.join(ROOT, "experiments/P3/rrcache_model_rrUb_a50n10d10a_France.pkl"), "rb"))
rrv = np.array([rr.get((a, b), np.nan) for a, b in zip(A5.s1.values, A5.rec.values)], np.float32); del rr
log("rrUb joined", int((~np.isnan(rrv)).sum()))
EXC = dict(L); EXC.update({k: R[k] for k in RLC}); EXC["rank_dense"] = rdense; EXC["dcos"] = dcos; EXC["rrUb"] = rrv; EXC["p_S005"] = p5
HIGH_IS_FLAG = {"addr_n_s1_only83", "addr_n_c_only84", "addr_frac_s1_only85", "addr_frac_c_only86", "house_absdiff58", "rv_rank", "af_n",
                "nf_n", "dupf", "rv_so", "rank_dense", "log_s1freq5"}
red = {}
for region, R0 in (("accepted_both", both), ("kept_both", both & kept), ("kept_both&same_core", both & kept & ceq),
                   ("kept_both&full>=0.8", both & kept & (full >= 0.8))):
    tgt = KEY2[R0].astype(np.int8)
    a = {k: auc(tgt, v[R0] if k in HIGH_IS_FLAG else -v[R0]) for k, v in EXC.items()}
    best = sorted(((v, k) for k, v in a.items() if v is not None and k != "p_S005"), reverse=True)[:5]
    red[region] = dict(n=int(R0.sum()), KEY2=int(tgt.sum()), top5_existing=best, aucs=a)
    log("RED", region, int(R0.sum()), int(tgt.sum()), best)
RES["france_redundancy_single_column_auc_for_KEY2"] = red

# 4-col lookup (same cells as VC_v1.py), fit on half of the S1 (hash split), AUC on the other half; label-free (target = the flag)
def cells(msk):
    t = np.clip((np.nan_to_num(L["addr_tset14"][msk], nan=0) * 20).astype(int), 0, 20)
    fc = np.clip((np.nan_to_num(L["addr_frac_c_only86"][msk], nan=-0.1) * 10).astype(int) + 1, 0, 11)
    fs = np.clip((np.nan_to_num(L["addr_frac_s1_only85"][msk], nan=-0.1) * 10).astype(int) + 1, 0, 11)
    ha = (np.nan_to_num(L["house_agree57"][msk], nan=-1) + 1).astype(int)
    af = np.where(np.isnan(R["af_a"][msk]), 2, R["af_a"][msk]).astype(int)
    return (((t * 12 + fc) * 12 + fs) * 3 + ha) * 3 + af
def cells_plus(msk):
    rv = np.where(R["rv_rank"][msk] == 11, 1, 0); du = np.clip(np.digitize(np.nan_to_num(R["dupf"][msk], nan=0), [1, 5, 20]), 0, 3)
    return (cells(msk) * 2 + rv) * 4 + du
h = pd.util.hash_array(A5.s1.values.astype(object)) % 2 == 0
look = {}
for region, R0 in (("accepted_both", both), ("kept_both", both & kept), ("kept_both&same_core", both & kept & ceq)):
    for nm, cf, nc in (("4col", cells, 21 * 12 * 12 * 9), ("4col+rv11+dupf", cells_plus, 21 * 12 * 12 * 9 * 8)):
        tr = R0 & h; te = R0 & ~h
        ct = cf(tr); ce = cf(te)
        num = np.bincount(ct, weights=KEY2[tr], minlength=nc); den = np.bincount(ct, minlength=nc); prior = KEY2[tr].mean()
        sc = (num[ce] + prior) / (den[ce] + 1.0)
        tgt = KEY2[te]
        # precision/recall of the best existing-cell selector at the KEY2 base size
        order = np.argsort(-sc); topk = order[:int(tgt.sum())]
        look[f"{region}|{nm}"] = dict(test_n=int(te.sum()), test_KEY2=int(tgt.sum()), auc_heldout=auc(tgt.astype(np.int8), sc),
                                      precision_at_KEY2_size=round(float(tgt[topk].mean()), 4) if len(topk) else None)
        log("LOOKUP", region, nm, look[f"{region}|{nm}"])
RES["france_redundancy_lookup_heldout"] = look

# ---------------------------------------------------------------- 3. how much of kept KEY2 is inside cells existing features already isolate
cellsets = {"rv11": R["rv_rank"] == 11, "rv11&af_a0": (R["rv_rank"] == 11) & (R["af_a"] == 0),
            "rv11&af_a0&dupf>=1": (R["rv_rank"] == 11) & (R["af_a"] == 0) & (R["dupf"] >= 1),
            "rv_gap<-0.3&af_a0": (R["rv_gap"] < -0.3) & (R["af_a"] == 0),
            "rv11&af_a0&addr_tsort13<0.8": (R["rv_rank"] == 11) & (R["af_a"] == 0) & (L["addr_tsort13"] < 0.8)}
cs = {}
for nm, C in cellsets.items():
    kc = kept & both & C
    cs[nm] = dict(kept_in_cell=int(kc.sum()), share_of_kept=round(float(kc.sum() / (kept & both).sum()), 5),
                  KEY2_in_cell=int((kc & KEY2).sum()), KEY2_coverage=round(float((kc & KEY2).sum() / max((kept & KEY2).sum(), 1)), 4),
                  KEY2_share_in_cell=round(float((kc & KEY2).mean()), 4) if kc.any() else None,
                  DIFF_share_in_cell=round(float((kc & DIFF).mean()), 4) if kc.any() else None,
                  mean_p_cell=round(float(p5[kc].mean()), 4) if kc.any() else None,
                  mean_p_cell_KEY2=round(float(p5[kc & KEY2].mean()), 4) if (kc & KEY2).any() else None,
                  mean_p_cell_notKEY2=round(float(p5[kc & ~KEY2].mean()), 4) if (kc & ~KEY2).any() else None,
                  n_s1_cell=int(len(np.unique(A5.s1.values[kc]))))
    log("CELL", nm, cs[nm])
RES["france_kept_existing_cells"] = cs
# examples: kept pairs inside rv11&af_a0 that are NOT KEY2 (what else the existing cell holds) and KEY2 examples with rrUb
rng = np.random.default_rng(3); ex = {}
for nm, m in [("cell_rv11_af0_notKEY2", kept & both & cellsets["rv11&af_a0"] & ~KEY2), ("cell_rv11_af0_KEY2", kept & cellsets["rv11&af_a0"] & KEY2)]:
    idx = np.flatnonzero(m)
    ex[nm] = [dict(s1=A5.s1.values[i], s1_name=s1.name[A5.s1.values[i]], s1_addr=s1.addr[A5.s1.values[i]], rec=A5.rec.values[i],
                   rec_name=rec.name[A5.rec.values[i]], rec_addr=rec.addr[A5.rec.values[i]], p=round(float(p5[i]), 4),
                   p_S004=None if np.isnan(p4[i]) else round(float(p4[i]), 4), rrUb=None if np.isnan(rrv[i]) else float(rrv[i]),
                   rs=round(float(rs[i]), 3), full=round(float(full[i]), 3), dupf=float(R["dupf"][i]), addr_tsort13=round(float(L["addr_tsort13"][i]), 3))
              for i in rng.choice(idx, size=min(8, len(idx)), replace=False)]
    for e in ex[nm]:
        log("EX", nm, e)
RES["examples"] = ex
json.dump(RES, open(os.path.join(HERE, "VC_france_results.json"), "w"), indent=1, default=str)
log("saved VC_france_results.json")
