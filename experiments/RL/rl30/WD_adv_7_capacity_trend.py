"""WD step 7: over-capacity check done against the size trend. For each S005-final France (s1,source) group size k, the share of
groups that contain >=1 accepted same-address content swap (D's definition and WD's definition). If swaps are decoys, the share
should jump at k = cap+1 (S2 cap 5, S3 cap 6 in train gt) beyond the within-cap trend. Writes rl30/WD_adv_7_results.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, accepted
a = accepted("S005_France"); a = a[a.kept_final].copy(); a["src"] = a.rec.str[:2]
k = a.groupby(["s1", "src"]).size().rename("k")
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read().split("for _d, _cc")[0])
fr["arel"] = addr_rel(fr); fr["swapcls"] = swap_class(fr, "France")
fk = fr[fr.kept_final].assign(src=lambda x: x.rec.str[:2])
flags = {"D_content_swap_same_addr": (fk.swapcls == "content_word") & fk.arel.isin(["exact_addr", "same_num_street"]),
         "D_to_noise_suffix_same_addr": (fk.swapcls == "to_noise_suffix") & fk.arel.isin(["exact_addr", "same_num_street"]),
         "D_N_SAME_num_shift": (fk.nt == "N_SAME") & (fk.arel == "num_shift_same_street"),
         "D_garble_same_addr": (fk.swapcls == "garble_or_rare") & fk.arel.isin(["exact_addr", "same_num_street"])}
wd = pd.read_pickle(os.path.join(OUT, "WD_adv_4_pairs_test_France.pkl")); wd = wd[wd.fin & (wd.typ == "CSWAP")]
R = {}
for src in ("S2", "S3"):
    kk = k.xs(src, level="src")
    out = {}
    for nm, fl in flags.items():
        g = fk[fl.values & (fk.src == src).values].s1.unique()
        has = kk.index.isin(g)
        out[nm] = pd.Series(has, index=kk.index).groupby(kk.values).agg(["size", "mean"]).round(4)
    g = wd[wd.src == src].s1.unique(); has = kk.index.isin(g)
    out["WD_content_swap"] = pd.Series(has, index=kk.index).groupby(kk.values).agg(["size", "mean"]).round(4)
    t = pd.concat(out, axis=1)
    print("==", src); print(t.to_string())
    R[src] = {nm: v.reset_index().values.tolist() for nm, v in out.items()}
json.dump(R, open(os.path.join(OUT, "WD_adv_7_results.json"), "w"), indent=1)
