"""WD step 4: is the France final-count histogram = US-test histogram + extra random FN (binomial thinning)?
Also: the same for train GT -> V1-pred (does thinning reproduce the known V1 deficit shape?), and pre-max-claimer counts.
Writes rl30/WD_step4_thinning.json (NEW)."""
import sys, os, json
import numpy as np, pandas as pd
from scipy.stats import binom
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH, accepted
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test", verbose=False)
sub = pd.read_csv(PATHS["s005_final"], sep="\t", dtype=str, keep_default_na=False)
sub["n"] = sub.matched_entity_ids.map(lambda s: 0 if not s.strip() else s.count(",") + 1)
sub["country"] = sub.source1_entity_id.map(T["s1"].set_index("id").country)
K = 13
def hist(n):
    return np.bincount(np.clip(n, 0, K - 1), minlength=K) / len(n)
def thin(h, rho):
    out = np.zeros(K)
    for k in range(K):
        out[:k + 1] += h[k] * binom.pmf(np.arange(k + 1), k, rho)
    return out
H = {cc: hist(sub.n[sub.country == cc].values) for cc in ("France", "US", "India")}
N = {cc: int((sub.country == cc).sum()) for cc in H}
mean = {cc: float((np.arange(K) * H[cc]).sum()) for cc in H}
rho = mean["France"] / mean["US"]
Hth = thin(H["US"], rho)
se = np.sqrt(H["France"] * (1 - H["France"]) / N["France"] + H["US"] * (1 - H["US"]) / N["US"])
tab = pd.DataFrame(dict(US=H["US"], France=H["France"], US_thinned=Hth, France_minus_thinned=H["France"] - Hth, z=(H["France"] - Hth) / se)).iloc[:10]
L("rho", round(rho, 5)); L((tab * [100, 100, 100, 100, 0.01]).round(3).to_string())
chi = float(((H["France"] - Hth) ** 2 / np.maximum(Hth, 1e-9))[:10].sum() * N["France"])
L("chi2 (France vs thinned US, 10 bins):", round(chi, 1))
R["thin_US_to_France"] = dict(rho=rho, table_pct=(tab.iloc[:, :4] * 100).round(4).to_dict(), z=tab.z.round(2).to_dict(), chi2_10bins=chi)
# pre-max-claimer counts
for tag, cc in (("S005_France", "France"), ("S005_US", "US"), ("S005_India", "India")):
    a = accepted(tag)
    ids = T["s1"].id[T["s1"].country == cc]
    pre = a.groupby("s1").size().reindex(ids, fill_value=0)
    R[f"pre_mc_{cc}"] = dict(mean=float(pre.mean()), removed_by_mc=int((~a.kept_final).sum()), removed_per_s1=float((~a.kept_final).sum() / len(ids)),
                             multi_claimed_recs=int(a[a.n_claims > 1].rec.nunique()), p0=float((pre == 0).mean()))
    L(cc, R[f"pre_mc_{cc}"])
# V1: GT hist -> thinning to V1 pred mean; compare with actual V1 pred hist (shape check of the thinning model where truth is known)
m = np.load(PATHS["v1_meta"]); p = np.load(PATHS["v1_p_new"])
acc = p >= NEW_TH
npred = np.bincount(m["s1idx"][acc], minlength=len(m["s1_ids"]))
ngt = m["n_gt"].astype(int)
for cc in ("US", "India"):
    mk = m["country"] == cc
    hg, hp = hist(ngt[mk]), hist(npred[mk])
    r_ = hp @ np.arange(K) / (hg @ np.arange(K))
    th = thin(hg, r_)
    se_ = np.sqrt(hp * (1 - hp) / mk.sum())
    t2 = pd.DataFrame(dict(gt=hg, pred=hp, gt_thinned=th, z=(hp - th) / np.maximum(se_, 1e-9))).iloc[:9]
    L("V1", cc, "rho", round(r_, 5)); L((t2 * [100, 100, 100, 1]).round(3).to_string())
    R[f"V1_{cc}_thin"] = dict(rho=float(r_), table=(t2 * [100, 100, 100, 1]).round(4).to_dict())
# test US vs train GT thinned
Tr = load("train", verbose=False)
gt = Tr["gt"]; c1 = Tr["s1"].set_index("id").country
for cc in ("US", "India", "France"):
    ids = Tr["s1"].id[Tr["s1"].country == ("US" if cc == "France" else cc)]
    g = gt[gt.s1.map(c1) == ("US" if cc == "France" else cc)].groupby("s1").size().reindex(ids, fill_value=0).values
    hg = hist(g); r_ = mean[cc] / (hg @ np.arange(K)); th = thin(hg, r_)
    t3 = pd.DataFrame(dict(trainGT=hg, test=H[cc], GT_thinned=th, diff=H[cc] - th)).iloc[:9]
    L("train GT ->", cc, "test; rho", round(r_, 5)); L((t3 * 100).round(3).to_string())
    R[f"trainGT_thin_to_test_{cc}"] = dict(rho=float(r_), table=(t3 * 100).round(4).to_dict())
json.dump(R, open(os.path.join(OUT, "WD_step4_thinning.json"), "w"), indent=1, default=str)
L("wrote WD_step4_thinning.json")
