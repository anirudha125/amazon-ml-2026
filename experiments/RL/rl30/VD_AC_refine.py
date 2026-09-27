"""VD_AC: refined labelled analog (genuine word-to-word swaps: both tokens >=4 chars, not typo/stem-related) and France flag composition.
READ-ONLY; writes VD_AC_refine.json"""
import os, sys, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
from rapidfuzz.distance import Levenshtein
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, NEW_TH
V = pickle.load(open(os.path.join(OUT, "VD_AC_v1_sub.pkl"), "rb")); F = pickle.load(open(os.path.join(OUT, "VD_AC_fr_sub.pkl"), "rb"))
vs = V["sub"].copy()
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc in")[0])
vs["arel"] = addr_rel(vs); vs["swapcls"] = ""
for cc in ("US", "India"):
    k = (vs.country == cc).values; vs.loc[k, "swapcls"] = swap_class(vs[k], cc)
vs["rr"] = V["rr_raw"]; vs["pnew"] = V["p"]
fl = vs[(vs.swapcls == "content_word") & vs.arel.isin(["exact_addr", "same_num_street"])].copy()
def genuine(a, b):
    return (len(a) >= 4) & (len(b) >= 4) & (Levenshtein.normalized_similarity(a, b) < 0.5) & (a[:3] != b[:3])
fl["genuine"] = [genuine(a, b) for a, b in zip(fl.sw_a, fl.sw_b)]
R = {}
for cc in ("US", "India", "all"):
    for gnm, gm in (("genuine", True), ("short_or_related", False)):
        g = fl[(fl.genuine == gm) & ((fl.country == cc) if cc != "all" else True)]
        acc = g.pnew >= NEW_TH
        R[f"{cc}|{gnm}"] = dict(n=int(len(g)), pos=int(g.y.sum()), p_match=round(float(g.y.mean()), 3) if len(g) else None,
                               TP=int((acc & (g.y == 1)).sum()), FP=int((acc & (g.y == 0)).sum()), FN=int((~acc & (g.y == 1)).sum()),
                               rr_median=round(float(np.nanmedian(g.rr)), 2) if g.rr.notna().any() else None,
                               frac_rr_le_m3=round(float((g.rr <= -3).mean()), 3))
print(json.dumps(R, indent=1))
print(fl[fl.genuine & (fl.y == 1)][["country", "sw_a", "sw_b", "pnew", "rr"]].to_string())
# France FLAG composition under the same 'genuine' definition
fm = F["meta"]; ff = fm[fm.grp == "FLAG"].copy(); ff["rr"] = F["X"][(fm.grp == "FLAG").values, 101]
ff["genuine"] = [genuine(a, b) for a, b in zip(ff.sw_a, ff.sw_b)]
R["France_FLAG"] = dict(n=int(len(ff)), genuine_share=round(float(ff.genuine.mean()), 4),
                        rr_median_genuine=round(float(np.nanmedian(ff[ff.genuine].rr)), 2),
                        frac_rr_le_m3_genuine=round(float((ff[ff.genuine].rr <= -3).mean()), 4),
                        p_median_genuine=round(float(ff[ff.genuine].p_ref.median()), 4),
                        accepted_genuine=int(ff.genuine.sum()))
# NEW p separates FLAG from NOISE in France?  (and V1 analog)
g = fm.grp.values; mm = (g == "FLAG") | (g == "NOISE")
R["France_auc_NEWp_FLAG_vs_NOISE"] = round(float(1 - roc_auc_score((g[mm] == "FLAG").astype(int), F["p"][mm])), 3)
R["France_p_median"] = {k: round(float(np.median(F["p"][g == k])), 4) for k in ("FLAG", "NOISE", "CLEAN")}
noise = vs[(vs.swapcls == "to_noise_suffix") & vs.arel.isin(["exact_addr", "same_num_street"])]
lab = np.r_[np.ones(len(fl)), np.zeros(len(noise))]; pp = np.r_[fl.pnew.values, noise.pnew.values]
R["V1_auc_NEWp_FLAG_vs_NOISE"] = round(float(1 - roc_auc_score(lab, pp)), 3)
print(json.dumps({k: R[k] for k in ("France_FLAG", "France_auc_NEWp_FLAG_vs_NOISE", "France_p_median", "V1_auc_NEWp_FLAG_vs_NOISE")}, indent=1))
json.dump(R, open(os.path.join(OUT, "VD_AC_refine.json"), "w"), indent=1, default=str)
