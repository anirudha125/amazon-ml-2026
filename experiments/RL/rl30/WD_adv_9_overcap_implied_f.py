"""WD step 9: decoy fraction implied by the over-capacity count (S2 cap 5 in train gt), under two decoy-placement models,
compared with D's shape-test f; EV range. Writes rl30/WD_adv_9_results.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, accepted, load
a = accepted("S005_France"); a = a[a.kept_final].copy(); a["src"] = a.rec.str[:2]
k = a.groupby(["s1", "src"]).size()
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc")[0])
fr["arel"] = addr_rel(fr); fr["swapcls"] = swap_class(fr, "France")
fk = fr[fr.kept_final].assign(src=lambda x: x.rec.str[:2])
cs = fk[(fk.swapcls == "content_word") & fk.arel.isin(["exact_addr", "same_num_street"])]
nS1 = int((load("test", verbose=False)["s1"].country == "France").sum())
R = {}
for src, cap in (("S2", 5), ("S3", 6)):
    kk = k.xs(src, level="src")
    n_cs = int((cs.src == src).sum()); n_rec = int((fk.src == src).sum())
    n_atcap = int((kk == cap).sum()); over = kk[kk == cap + 1]
    g_cs = set(cs[cs.src == src].s1)
    n_over_cs = int(over.index.isin(g_cs).sum())
    rate_group = n_cs / nS1; rate_rec = n_cs / n_rec
    exp_indep = n_atcap * rate_group; exp_perrec = n_atcap * cap * rate_rec
    R[src] = dict(cap=cap, n_content_swaps=n_cs, n_groups_at_cap=n_atcap, n_over_cap=int(len(over)), n_over_cap_with_cs=n_over_cs,
                  expected_over_cap_with_decoy_cs_if_f1_independent=round(exp_indep, 1), implied_f_max_independent=round(n_over_cs / exp_indep, 2),
                  expected_if_f1_per_record=round(exp_perrec, 1), implied_f_max_per_record=round(n_over_cs / exp_perrec, 2))
gain, loss = 0.2659, 0.1228; n_aff = 13792
R["breakeven_f"] = round(loss / (gain + loss), 3)
R["EV_France_pts"] = {str(f): round(100 * n_aff * (f * gain - (1 - f) * loss) / nS1, 3) for f in (0.1, 0.2, 0.316, 0.45, 0.63, 0.89)}
R["EV_overall_LB_pts"] = {kf: round(v * nS1 / 1732544, 3) for kf, v in R["EV_France_pts"].items()}
print(json.dumps(R, indent=1))
json.dump(R, open(os.path.join(OUT, "WD_adv_9_results.json"), "w"), indent=1)
