"""WC step 6: are the RL-27 'anomaly' rates C cites for kept KEY2 (rv_rank==11 86%, dupf>=10 60%, af_a==0 99%) just mechanical consequences
of (generic name) x (different street)?  Uses C's joined EXF (France accepted, 14 cols) after an independent alignment spot-check.  READ-ONLY."""
import os, sys, json, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
A = accepted("S005_France"); kept = A.kept_final.values.astype(bool)
EXF = np.load(os.path.join(HERE, "C_test_France_robust.npz"))["EXF"]
W = np.load(os.path.join(HERE, "WC_03_France_pairs.npz")); full = W["full"]; rs = W["rsC"]; hn = W["hnC"]; dc = W["dupf_core"]
R = {}
# alignment spot-check: re-join 3 chunks independently
key = pd.Series(np.arange(len(A)), index=pd.MultiIndex.from_arrays([A.s1.values, A.rec.values]))
files = sorted(glob.glob(PATHS["test_chunks"].format(country="France")))
chk = []; nm_ = 0
for f in files[:3]:
    z = np.load(f, allow_pickle=True)
    j = key.reindex(pd.MultiIndex.from_arrays([z["s1"], z["cand"]])).values; ok = ~np.isnan(j)
    rl = np.load(f.replace(os.path.join("P3", "France"), os.path.join("RL", "test_feats", "France")).replace("chunk_", "rl27_chunk_").replace(".npz", ".npy"))
    jj = j[ok].astype(int)
    chk.append(float(np.nanmax(np.abs(np.nan_to_num(rl[ok], nan=-9) - np.nan_to_num(EXF[jj, 4:14], nan=-9)))))
    nm_ += int(ok.sum())
R["alignment_check"] = dict(rows_checked=nm_, max_abs_diff=chk)
rv11 = EXF[:, 10] == 11; dupf = EXF[:, 9]; afa = EXF[:, 7]; gap = EXF[:, 13]
K2 = (full >= 0.8) & (rs < 0.6); SM = (rs >= 0.8)
def rates(m):
    return dict(n=int(m.sum()), rv_rank11=round(float(rv11[m].mean()), 4), dupf_ge10=round(float((dupf[m] >= 10).mean()), 4),
                af_a0=round(float((afa[m] == 0).mean()), 4), rv_gap_lt_m03=round(float((gap[m] < -0.3).mean()), 4)) if m.any() else dict(n=0)
for nm, m in [("kept_all", kept), ("kept_KEY2", kept & K2), ("kept_street_match", kept & SM),
              ("kept_street_match&dupf>=10", kept & SM & (dupf >= 10)), ("kept_KEY2&dupf>=10", kept & K2 & (dupf >= 10)),
              ("kept_street_match&dupf<10", kept & SM & (dupf < 10)), ("kept_KEY2&dupf<10", kept & K2 & (dupf < 10)),
              ("kept_street_diff_any(rs<0.6)&dupf>=10", kept & (rs < 0.6) & (dupf >= 10)),
              ("kept_KEY2&dupf_core>=10", kept & K2 & (dc >= 10)), ("kept_street_match&dupf_core>=10", kept & SM & (dc >= 10))]:
    R[nm] = rates(m)
    print(nm, R[nm])
# how much of rv_rank==11 is explained by dupf (identical-name S1 count) among street-diff kept pairs?
m = kept & K2
tab = {}
for lo, hi in [(0, 0), (1, 4), (5, 9), (10, 10**9)]:
    b = m & (dupf >= lo) & (dupf <= hi)
    tab[f"dupf[{lo},{hi}]"] = dict(n=int(b.sum()), rv11=round(float(rv11[b].mean()), 4) if b.any() else None)
R["kept_KEY2_rv11_by_dupf"] = tab
print(tab)
json.dump(R, open(os.path.join(HERE, "WC_06_rl27_confound.json"), "w"), indent=1)
