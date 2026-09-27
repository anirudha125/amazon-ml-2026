"""RL-30 / E (Part 6) -- label-free firing counts of F1/F2/F3 among S005 accepted test pairs (before max-claimer),
using token roles + co-located siblings estimated from the TEST corpus (E_roles_test.pkl). Output: rl30/E_test_firing.json"""
import os, sys, time, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from E_flags import *

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    t0 = time.time()
    T = load("test", verbose=False)
    S1 = T["s1"].set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    RO = pickle.load(open(os.path.join(HERE, "E_roles_test.pkl"), "rb"))
    S_role, DC = role_lookup(RO["roles"])
    n_s1_c = T["s1"].country.value_counts().to_dict()
    out = {}; ex = {}
    rng = np.random.default_rng(0)
    for tag in ("S005_France", "S005_US", "S005_India"):
        c = tag.split("_")[1]
        a = accepted(tag).reset_index(drop=True)
        us1 = a.s1.unique(); ur = a.rec.unique()
        SS = S1.loc[us1]; RR = recs.loc[ur]
        S1F = {s: s1_feats(nm, ad) for s, nm, ad in zip(us1, SS.name.values, SS.addr.values)}
        RF = {r: rec_feats(nm, ad) for r, nm, ad in zip(ur, RR.name.values, RR.addr.values)}
        fl = compute_flags(a.s1.values, a.rec.values, np.full(len(a), c), S1F, RF, S_role, DC, RO["sib"])
        F = derive(fl)
        kept = a.kept_final.values
        r = dict(accepted_pairs=int(len(a)), accepted_s1=int(len(us1)), n_s1_country=int(n_s1_c[c]), kept_final_pairs=int(kept.sum()))
        for name, f in F.items():
            r[name] = dict(pairs=int(f.sum()), rate_per_accepted_pair=round(float(f.mean()), 6),
                           s1=int(a.s1.values[f].size and np.unique(a.s1.values[f]).size),
                           rate_s1_per_country_s1=round(float(np.unique(a.s1.values[f]).size / n_s1_c[c]), 6),
                           kept_final_pairs=int((f & kept).sum()), multi_claimed_pairs=int((f & (a.n_claims.values > 1)).sum()),
                           p_quantiles=[round(float(q), 4) for q in np.quantile(a.p.values[f], [0.1, 0.5, 0.9])] if f.any() else None)
            ii = np.flatnonzero(f & kept)
            if len(ii):
                ii = rng.choice(ii, size=min(8, len(ii)), replace=False)
                ex.setdefault(tag, {})[name] = [dict(s1=a.s1[i], s1_name=SS.name.loc[a.s1[i]], s1_addr=SS.addr.loc[a.s1[i]], rec=a.rec[i],
                                                     rec_name=RR.name.loc[a.rec[i]], rec_addr=RR.addr.loc[a.rec[i]], p=round(float(a.p[i]), 4),
                                                     smax=None if np.isnan(fl["smax"][i]) else round(float(fl["smax"][i]), 3)) for i in ii]
        out[tag] = r
        log(tag, {k: (v["pairs"], v["rate_per_accepted_pair"]) if isinstance(v, dict) else v for k, v in r.items()}, f"{time.time()-t0:.0f}s")
    out["roles_test_summary"] = dict(nd_pairs=RO["nd_pairs"], nd_pairs_core=RO["nd_pairs_core"], coloc_pairs=RO["coloc_pairs"], f_hits=RO["f_hits"])
    json.dump(out, open(os.path.join(HERE, "E_test_firing.json"), "w"), indent=1, default=str)
    json.dump(ex, open(os.path.join(HERE, "E_test_examples.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
