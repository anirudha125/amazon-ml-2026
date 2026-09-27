"""RL-30 Part 5 (investigator D): ESTIMATED macro-F0.5 effect of dropping France same-address content-word swaps
(assumes the S1's other accepted records are correct; decoy fraction f from the label-free shape test). Writes rl30/D_impact.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted
T = load("test")
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl")); us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl"))
ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl")); v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read())
fk = fr[fr.kept_final]
cs = fk[(fk.swapcls == "content_word") & fk.arel.isin(["exact_addr", "same_num_street"])]
ntot = fk.groupby("s1").size(); m = cs.groupby("s1").size()
def f05(tp, fp, fn):
    if tp == 0: return 0.0 if (fp + fn) else 1.0
    p, r = tp / (tp + fp), tp / (tp + fn); return 1.25 * p * r / (0.25 * p + r)
gain = loss = 0.0
for s, k in m.items():
    n = ntot[s]
    # if the k swap records are decoys: current F=f05(n-k, k, 0) -> after drop 1.0
    gain += 1.0 - f05(n - k, k, 0)
    # if they are true: current 1.0 -> after drop f05(n-k, 0, k)
    loss += 1.0 - f05(n - k, 0, k)
nfr = int((T["s1"].country == "France").sum()); nall = len(T["s1"])
out = dict(n_pairs=int(len(cs)), n_s1=int(len(m)), mean_n_acc_of_those_s1=round(float(ntot[m.index].mean()), 3),
           gain_if_all_decoy_per_S1=round(gain / len(m), 4), loss_if_all_true_per_S1=round(loss / len(m), 4))
for f in (0.6, 0.75, 0.9):
    net = f * gain - (1 - f) * loss
    out[f"f={f}"] = dict(France_pts=round(100 * net / nfr, 3), overall_LB_pts=round(100 * net / nall, 3))
# S004 -> S005 differences for this class
a4only = pd.read_pickle(os.path.join(OUT, "D_enriched_FR4only.pkl"))
exec("fr_ = a4only")
a4only["arel"] = addr_rel(a4only); a4only["swapcls"] = swap_class(a4only, "France")
new5 = fr[~fr.in_s004]
out["S004only_n"] = int(len(a4only)); out["S004only_content_swap_same_addr"] = int(((a4only.swapcls == "content_word") & a4only.arel.isin(["exact_addr", "same_num_street"])).sum())
out["S005new_n"] = int(len(new5)); out["S005new_content_swap_same_addr"] = int(((new5.swapcls == "content_word") & new5.arel.isin(["exact_addr", "same_num_street"])).sum())
out["S005new_nt"] = new5.nt.value_counts().head(6).to_dict()
# counts of content-swap-same-addr among ALL accepted (before max-claimer) per country (US/IN scaled from 300k samples)
out["all_accepted_rate"] = {"France": round(float(((fr.swapcls == "content_word") & fr.arel.isin(["exact_addr", "same_num_street"])).mean()), 5),
                            "US": round(float(((us.swapcls == "content_word") & us.arel.isin(["exact_addr", "same_num_street"])).mean()), 5),
                            "India": round(float(((ind.swapcls == "content_word") & ind.arel.isin(["exact_addr", "same_num_street"])).mean()), 5)}
# V1 hard candidates of this class, per country (labelled analog) and NEW behaviour
for cc in ("US", "India"):
    g = v[(v.country == cc) & (v.swapcls == "content_word") & v.arel.isin(["exact_addr", "same_num_street"]) & v.hard]
    out[f"V1_{cc}_analog"] = dict(n_hard=len(g), n_pos=int(g.y.sum()), p_match=round(float(g.y.mean()), 3), n_acc=int(g.acc.sum()),
                                  FP=int((g.acc & (g.y == 0)).sum()), FN=int((~g.acc & (g.y == 1)).sum()), median_p_neg=round(float(g[g.y == 0].p.median()), 4))
print(json.dumps(out, indent=1, default=str))
json.dump(out, open(os.path.join(OUT, "D_impact.json"), "w"), indent=1, default=str)
