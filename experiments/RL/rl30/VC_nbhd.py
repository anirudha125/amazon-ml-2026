"""RL-30 adversarial verifier VC: labelled neighbourhood test.  In train (T0+E014+T2X, what RL-27 NEW was fitted on) and V1,
take same-core pairs whose EXISTING features look like France's kept KEY2 pairs (high full-address token-set ratio, reverse-BM25
rank 11, af_a = 0, generic name) and ask whether the street flag (rs < 0.6) still separates y inside that neighbourhood,
and whether RL-27 NEW (V1, held out) is already right there.  READ-ONLY; writes only rl30/VC_*.  No model training.
Run: OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 10 python VC_nbhd.py
"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street2 import robust_pairs

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

D = load("train", verbose=False)
s1t = D["s1"].set_index("id"); rect = pd.concat([D["s2"], D["s3"]]).set_index("id")
cc = {}
def cores(names):
    o = np.empty(len(names), object)
    for i, x in enumerate(names):
        v = cc.get(x)
        if v is None:
            v = cc[x] = core_tokens(x)
        o[i] = v
    return o
sets = [("T0", PATHS["t_sets"][0]), ("E014", PATHS["t_sets"][1]), ("T2X", PATHS["t_sets"][2]), ("V1", PATHS["v1_meta"])]
cols = {"addr_tset14": 14, "addr_tsort13": 13, "house_agree57": 57, "addr_frac_c_only86": 86, "name_tset3": 3}
parts = []
for nm, mp_ in sets:
    M = np.load(mp_, allow_pickle=True)
    ti = M["s1idx"]; tc = M["cand"]; ty = M["y"].astype(np.int8); tid = M["s1_ids"][ti]
    sn = s1t.name.reindex(tid).fillna("").values.astype(object); rn = rect.name.reindex(tc).fillna("").values.astype(object)
    ce = np.fromiter((bool(a) and a == b for a, b in zip(cores(sn), cores(rn))), bool, len(ty))
    idx = np.flatnonzero(ce)
    sa = s1t.addr.reindex(tid[idx]).fillna("").values.astype(object); ra = rect.addr.reindex(tc[idx]).fillna("").values.astype(object)
    rs, _, hn = robust_pairs(sa, ra)
    fw = np.full(len(idx), np.nan, np.float32)
    for k, (a, b) in enumerate(zip(sa, ra)):
        if a and b and a.strip().lower() != "null" and b.strip().lower() != "null":
            W1 = addr_words(a); W2 = addr_words(b)
            if W1 and W2:
                fw[k] = len(W1 & W2) / min(len(W1), len(W2))
    LF = np.load(os.path.join(os.path.dirname(mp_), "LF.npy"), mmap_mode="r")
    rl = np.load(PATHS["v1_rl27"] if nm == "V1" else os.path.join(RL, "cache", f"rl27_{nm}.npy"), mmap_mode="r")
    df = pd.DataFrame({"set": nm, "y": ty[idx], "rs": rs, "hn2": hn, "full": fw, "rv_rank": np.asarray(rl[idx, 6]), "af_a": np.asarray(rl[idx, 3]),
                       "dupf": np.asarray(rl[idx, 5]), "rv_gap": np.asarray(rl[idx, 9]), "s1": tid[idx]})
    for k, j in cols.items():
        df[k] = np.asarray(LF[idx, j])
    df["p"] = np.load(PATHS["v1_p_new"])[idx] if nm == "V1" else np.nan
    parts.append(df); log(nm, len(ty), "same-core", len(idx))
X = pd.concat(parts, ignore_index=True)
X["flag"] = (~X.rs.isna()) & (X.rs < 0.6)
X["key2"] = X.flag & (X.full >= 0.8)
ok = ~X.rs.isna() & ~X.full.isna()
NB = {"N_A: tset14>=0.8 & rv11 & af_a0": ok & (X.addr_tset14 >= 0.8) & (X.rv_rank == 11) & (X.af_a == 0),
      "N_B: N_A & house_agree=1": ok & (X.addr_tset14 >= 0.8) & (X.rv_rank == 11) & (X.af_a == 0) & (X.house_agree57 == 1),
      "N_C: tset14>=0.8 & af_a0 & dupf>=5": ok & (X.addr_tset14 >= 0.8) & (X.af_a == 0) & (X.dupf >= 5),
      "N_D: tset14>=0.8 & rv_gap<-0.3": ok & (X.addr_tset14 >= 0.8) & (X.rv_gap < -0.3),
      "N_E: tset14>=0.8 (same-core)": ok & (X.addr_tset14 >= 0.8)}
RES = {"n_same_core_rows": int(len(X))}
for nm, m in NB.items():
    r = {}
    for part, pm in (("train", X.set != "V1"), ("V1", X.set == "V1")):
        mm = m & pm
        d = dict(n=int(mm.sum()), pos_rate=round(float(X.y[mm].mean()), 4) if mm.any() else None)
        for fl in ("flag", "key2"):
            a = mm & X[fl]; b = mm & ~X[fl]
            d[f"{fl}_n"] = int(a.sum()); d[f"{fl}_pos_rate"] = round(float(X.y[a].mean()), 4) if a.any() else None
            d[f"not_{fl}_n"] = int(b.sum()); d[f"not_{fl}_pos_rate"] = round(float(X.y[b].mean()), 4) if b.any() else None
        if part == "V1" and mm.any():
            acc = X.p[mm] >= NEW_TH
            d["V1_accepted"] = int(acc.sum()); d["V1_FP"] = int((acc & (X.y[mm] == 0)).sum()); d["V1_FN"] = int((~acc & (X.y[mm] == 1)).sum())
            a = mm & X.flag
            if a.any():
                accf = X.p[a] >= NEW_TH
                d["V1_flag_accepted"] = int(accf.sum()); d["V1_flag_FP"] = int((accf & (X.y[a] == 0)).sum()); d["V1_flag_FN"] = int((~accf & (X.y[a] == 1)).sum())
        r[part] = d
    RES[nm] = r
    log(nm, r)
# examples of labelled flagged pairs in N_A (train) to see what they look like
ex = []
sub = X[NB["N_A: tset14>=0.8 & rv11 & af_a0"] & X.flag]
for i, row in sub.sample(min(12, len(sub)), random_state=1).iterrows():
    ex.append(dict(set=row.set, y=int(row.y), s1=row.s1, rs=round(float(row.rs), 3), full=round(float(row.full), 3), dupf=float(row.dupf)))
RES["examples_N_A_flagged"] = ex
json.dump(RES, open(os.path.join(HERE, "VC_nbhd_results.json"), "w"), indent=1, default=str)
log("saved")
