"""RL-30 / E (Part 6, round 2) -- label-free firing of the typo-robust flags among S005 accepted test pairs (before max-claimer),
vocab + roles from the TEST corpus. Output: rl30/E_refined_test.json, rl30/E_refined_test_examples.json"""
import os, sys, time, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from E_flags import s1_feats, rec_feats, role_lookup
from E_flags2 import *

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def main():
    t0 = time.time()
    T = load("test", verbose=False)
    NV, SV = vocabs(T["s1"])
    RO = pickle.load(open(os.path.join(HERE, "E_roles_test.pkl"), "rb")); S_role, DC = role_lookup(RO["roles"])
    S1 = T["s1"].set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    n_s1_c = T["s1"].country.value_counts().to_dict()
    out, ex = {}, {}; rng = np.random.default_rng(2)
    for tag in ("S005_France", "S005_US", "S005_India"):
        c = tag.split("_")[1]
        a = accepted(tag).reset_index(drop=True)
        us1 = a.s1.unique(); ur = a.rec.unique(); SS = S1.loc[us1]; RR = recs.loc[ur]
        S1F = {s: s1_feats(nm, ad) for s, nm, ad in zip(us1, SS.name.values, SS.addr.values)}
        RF = {r: rec_feats(nm, ad) for r, nm, ad in zip(ur, RR.name.values, RR.addr.values)}
        FR = compute_refined(a.s1.values, a.rec.values, np.full(len(a), c), S1F, RF, S_role, NV, SV)
        kept = a.kept_final.values
        r = dict(accepted_pairs=int(len(a)), kept_final_pairs=int(kept.sum()), n_s1_country=int(n_s1_c[c]))
        for k, f in FR.items():
            r[k] = dict(pairs=int(f.sum()), rate_per_accepted_pair=round(float(f.mean()), 6), s1=int(np.unique(a.s1.values[f]).size),
                        rate_s1_per_country_s1=round(float(np.unique(a.s1.values[f]).size / n_s1_c[c]), 6),
                        kept_final_pairs=int((f & kept).sum()), kept_final_s1=int(np.unique(a.s1.values[f & kept]).size),
                        kept_rate_per_kept_pair=round(float((f & kept).sum() / kept.sum()), 6),
                        multi_claimed_pairs=int((f & (a.n_claims.values > 1)).sum()),
                        p_quantiles=[round(float(q), 4) for q in np.quantile(a.p.values[f], [0.1, 0.5, 0.9])] if f.any() else None)
            ii = np.flatnonzero(f & kept)
            if len(ii):
                ii = rng.choice(ii, size=min(10, len(ii)), replace=False)
                ex.setdefault(tag, {})[k] = [dict(s1_name=SS.name.loc[a.s1[i]], s1_addr=SS.addr.loc[a.s1[i]], rec_name=RR.name.loc[a.rec[i]],
                                                  rec_addr=RR.addr.loc[a.rec[i]], p=round(float(a.p[i]), 4)) for i in ii]
        out[tag] = r
        log(tag, {k: (v["pairs"], v["kept_final_pairs"], v["rate_per_accepted_pair"]) if isinstance(v, dict) else v for k, v in r.items()}, f"{time.time()-t0:.0f}s")
    json.dump(out, open(os.path.join(HERE, "E_refined_test.json"), "w"), indent=1, default=str)
    json.dump(ex, open(os.path.join(HERE, "E_refined_test_examples.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
