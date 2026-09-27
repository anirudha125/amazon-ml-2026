"""VD_AC: V1 flagged pairs listed with rrUb / p / tokens; rrUb separation FLAG vs NOISE in V1 and France. READ-ONLY; writes VD_AC_rr.json"""
import os, sys, json, pickle
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 300)
V = pickle.load(open(os.path.join(OUT, "VD_AC_v1_sub.pkl"), "rb")); F = pickle.load(open(os.path.join(OUT, "VD_AC_fr_sub.pkl"), "rb"))
vs = V["sub"].copy()
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc in")[0])
vs["arel"] = addr_rel(vs); vs["swapcls"] = ""
for cc in ("US", "India"):
    k = (vs.country == cc).values; vs.loc[k, "swapcls"] = swap_class(vs[k], cc)
same = vs.arel.isin(["exact_addr", "same_num_street"])
vs["rr"] = V["X"][:, 101]; vs["rr_raw"] = V["rr_raw"]; vs["pnew"] = V["p"]
fl = vs[(vs.swapcls == "content_word") & same].sort_values(["country", "y", "rr"])
print(fl[["country", "y", "pnew", "rr", "rr_raw", "sw_a", "sw_b", "df_a", "df_b", "arel"]].to_string())
R = {}
# y-rate by rrUb bin inside the V1 flag
fl2 = fl.assign(rb=pd.cut(fl.rr_raw, [-99, -3, 0, 3, 6, 99]))
t = fl2.groupby("rb", observed=True).agg(n=("y", "size"), pos=("y", "sum"), p_med=("pnew", "median"))
print(t); R["v1_flag_y_by_rr_bin"] = {str(k): r.to_dict() for k, r in t.iterrows()}
# business-word swaps (both tokens in >=200 train S1 names) vs others
bw = (fl.df_a >= 200) & (fl.df_b >= 200)
R["v1_flag_business_word"] = {nm: dict(n=int(len(g)), pos=int(g.y.sum()), rr_med=round(float(np.nanmedian(g.rr_raw)), 3),
                                       acc=int((g.pnew >= 0.78).sum()), FP=int(((g.pnew >= 0.78) & (g.y == 0)).sum()))
                              for nm, g in (("both_df>=200", fl[bw]), ("other", fl[~bw]))}
print(R["v1_flag_business_word"])
# rrUb AUC FLAG vs NOISE (V1 and France)
noise = vs[(vs.swapcls == "to_noise_suffix") & same]
a = np.r_[fl.rr_raw.values, noise.rr_raw.values]; lab = np.r_[np.ones(len(fl)), np.zeros(len(noise))]; ok = ~np.isnan(a)
R["v1_auc_rr_flag_vs_noise"] = round(float(1 - roc_auc_score(lab[ok], a[ok])), 3)
fm = F["meta"]; rrf = F["X"][:, 101]
g = fm.grp.values; mm = (g == "FLAG") | (g == "NOISE")
R["fr_auc_rr_flag_vs_noise"] = round(float(1 - roc_auc_score((g[mm] == "FLAG").astype(int), np.nan_to_num(rrf[mm], nan=-20))), 3)
R["fr_rr_quantiles"] = {k: np.nanquantile(rrf[g == k], [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist() for k in ("FLAG", "NOISE", "CLEAN")}
R["v1_rr_quantiles"] = {"FLAG_neg": np.nanquantile(fl[fl.y == 0].rr_raw, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
                        "FLAG_pos": np.nanquantile(fl[fl.y == 1].rr_raw, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist(),
                        "NOISE_pos": np.nanquantile(noise[noise.y == 1].rr_raw, [0.1, 0.25, 0.5, 0.75, 0.9]).round(3).tolist()}
# France FLAG: p by rr bin
ff = fm[g == "FLAG"].assign(rr=rrf[g == "FLAG"])
R["fr_flag_by_rr_bin"] = {str(k): dict(n=int(len(x)), p_med=round(float(x.p_ref.median()), 4)) for k, x in ff.groupby(pd.cut(ff.rr, [-99, -3, 0, 3, 6, 99]), observed=True)}
print(json.dumps(R, indent=1))
json.dump(R, open(os.path.join(OUT, "VD_AC_rr.json"), "w"), indent=1, default=str)
