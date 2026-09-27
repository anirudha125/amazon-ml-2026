"""WD step 3: independent re-derivation of D's support count on S005 final France pairs + model acceptance of
MODEL-FREE same-(number,street) content swaps. Reads WD_pairs_test_*.pkl, accepted_S005_*.pkl. Writes WD_3_results.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, accepted, load
R = {}
NOISE_D = {"France": ["fils", "services", "associes", "developpement", "france", "compagnie", "fka", "labs", "one"],
           "US": ["incorporated", "center", "services", "www", "com", "service", "fka", "partners"]}
for cc in ("France", "US"):
    m = pd.read_pickle(os.path.join(OUT, f"WD_adv_pairs_test_{cc}.pkl"))
    a = accepted(f"S005_{cc}")
    m = m.merge(a[["s1", "rec", "p", "kept_final"]], on=["s1", "rec"], how="left")
    m["acc"] = m.p.notna(); m["fin"] = m.kept_final.fillna(False).astype(bool)
    cs = m[(m.typ == "CSWAP") & ~m.tb.isin(NOISE_D[cc])]
    n_s1 = int((load("test", verbose=False)["s1"].country == cc).sum())
    r = dict(modelfree_cswap_all=int((m.typ == "CSWAP").sum()), modelfree_cswap_exD_noise=int(len(cs)), modelfree_S1=int(cs.s1.nunique()),
             modelfree_S1_share=round(cs.s1.nunique() / n_s1, 4),
             accepted_rate=round(float(cs.acc.mean()), 4), final_rate=round(float(cs.fin.mean()), 4),
             final_n=int(cs.fin.sum()), final_S1=int(cs[cs.fin].s1.nunique()),
             final_share_of_final_pairs=round(float(cs.fin.sum() / a.kept_final.sum()), 5))
    # acceptance by to-token (top 30 by count)
    g = cs.groupby("tb").agg(n=("fin", "size"), fin=("fin", "mean"), acc=("acc", "mean")).sort_values("n", ascending=False)
    r["by_tb_top30"] = g.head(30).round(3).reset_index().values.tolist()
    # acceptance for other model-free types at same number+street (reference)
    r["final_rate_by_type"] = m.groupby("typ").fin.mean().round(4).to_dict()
    r["n_by_type"] = m.typ.value_counts().to_dict()
    R[cc] = r
    print(cc, json.dumps({k: v for k, v in r.items() if k != "by_tb_top30"}))
    print(g.head(30).round(3).to_string())
json.dump(R, open(os.path.join(OUT, "WD_adv_3_results.json"), "w"), indent=1)
