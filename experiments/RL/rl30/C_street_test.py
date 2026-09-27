"""RL-30 Part 4 (investigator C), test side: street-name vs full-address overlap among S005 accepted pairs (before max-claimer),
France vs US vs India.  Label-free.  READ-ONLY; writes only rl30/C_*.
Run: OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 10 python C_street_test.py
"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from C_street import pair_feats, core_eq, sbucket, hbucket, log

RNG = np.random.default_rng(1)
RES = {}


def share_table(m0, fb, SB, HB, kept, extra=None):
    out = {}
    n0 = int(m0.sum())
    for f_ in ("full>=0.8", "full<0.8", "nan"):
        for s_ in ("1", "part", "0", "miss"):
            for h_ in ("eq", "neq", "miss"):
                m = m0 & (fb == f_) & (SB == s_) & (HB == h_)
                if m.sum() == 0:
                    continue
                out[f"{f_}|street={s_}|hn={h_}"] = dict(n=int(m.sum()), share=round(float(m.sum() / n0), 5),
                                                        kept_share=round(float(kept[m].mean()), 4))
    return out


def main():
    TD = load("test", verbose=False)
    s1 = TD["s1"].set_index("id"); rec = pd.concat([TD["s2"], TD["s3"]]).set_index("id")
    del TD
    v1 = json.load(open(os.path.join(HERE, "C_results_v1.json")))
    v1acc = v1["accepted"]
    for c in ("France", "US", "India"):
        A = accepted(f"S005_{c}")
        sid = A.s1.values; cid = A.rec.values; p = A.p.values; kept = A.kept_final.values.astype(bool); ncl = A.n_claims.values
        sa = s1.addr.reindex(sid).fillna("").values.astype(object); sn = s1.name.reindex(sid).fillna("").values.astype(object)
        ra = rec.addr.reindex(cid).fillna("").values.astype(object); rn = rec.name.reindex(cid).fillna("").values.astype(object)
        F = pair_feats(sa, ra); ceq = core_eq(sn, rn)
        full, so, hn, sf = F["full"], F["so"], F["hn"], F["sf"]
        both = ~np.isnan(full)
        fb = np.where(np.isnan(full), "nan", np.where(full >= 0.8, "full>=0.8", "full<0.8"))
        SB = sbucket(so); HB = hbucket(hn)
        K = both & (full >= 0.8) & (SB == "0")
        n = len(p)
        R = dict(n_accepted=int(n), n_s1=int(len(np.unique(sid))), kept_final=int(kept.sum()), both_addr_words=int(both.sum()),
                 share_both=round(float(both.mean()), 4),
                 street_defined_share_of_both=round(float((both & (SB != "miss")).sum() / both.sum()), 4),
                 s1_street_missing_share=round(float((both & (F["nS1"] == 0)).sum() / both.sum()), 4),
                 rec_street_missing_share=round(float((both & (F["nS2"] == 0)).sum() / both.sum()), 4),
                 hn_defined_share_of_both=round(float((both & (hn >= 0)).sum() / both.sum()), 4))
        for nm, m in [("street=0", both & (SB == "0")), ("street=0&hn=neq", both & (SB == "0") & (HB == "neq")),
                      ("street=0&hn=eq", both & (SB == "0") & (HB == "eq")), ("street=1&hn=eq", both & (SB == "1") & (HB == "eq")),
                      ("KEY", K), ("KEY&hn=eq", K & (HB == "eq")), ("KEY&hn=neq", K & (HB == "neq")), ("KEY&hn=miss", K & (HB == "miss")),
                      ("KEY&street_fuzzy<0.8", K & (sf < 0.8)), ("KEY&street_fuzzy>=0.8", K & (sf >= 0.8)),
                      ("KEY&same_core", K & ceq), ("KEY&p<0.99", K & (p < 0.99))]:
            R[nm] = dict(n=int(m.sum()), share_of_accepted=round(float(m.sum() / n), 5), share_of_both=round(float(m.sum() / both.sum()), 5),
                         n_kept=int((m & kept).sum()), share_of_kept=round(float((m & kept).sum() / kept.sum()), 5),
                         mean_p=round(float(p[m].mean()), 4) if m.any() else None,
                         multi_claim_share=round(float((ncl[m] > 1).mean()), 4) if m.any() else None,
                         n_s1=int(len(np.unique(sid[m]))))
        R["baseline_multi_claim_share"] = round(float((ncl > 1).mean()), 4)
        R["p_quantiles_KEY"] = np.round(np.percentile(p[K], [5, 25, 50, 75, 95]), 4).tolist() if K.any() else None
        R["p_quantiles_all"] = np.round(np.percentile(p, [5, 25, 50, 75, 95]), 4).tolist()
        # label-free sibling proxy: does the S1 also have an accepted record whose street overlaps its own?
        g = pd.DataFrame({"s1": sid, "st_ok": both & (SB == "1")})
        has_street_ok = g.groupby("s1").st_ok.transform("max").values
        R["KEY&S1_has_other_street_match_rec"] = dict(n=int((K & has_street_ok).sum()), share_of_KEY=round(float((K & has_street_ok).sum() / max(K.sum(), 1)), 4))
        # ESTIMATED FP in KEY using V1 precision by hn bucket (transfer assumption; France has no labels)
        est = 0.0
        for h_ in ("eq", "neq", "miss"):
            pr = v1acc.get(f"KEY|hn={h_}", {}).get("precision")
            if pr is not None:
                est += (1 - pr) * int((K & kept & (HB == h_)).sum())
        R["ESTIMATED_fp_in_KEY_kept_via_V1_precision"] = round(est, 1)
        R["table"] = share_table(np.ones(n, bool), fb, SB, HB, kept)
        # examples
        exs = {}
        for nm, m in [("KEY&hn=eq", K & (HB == "eq") & kept), ("KEY&hn=neq", K & (HB == "neq") & kept), ("KEY&hn=miss", K & (HB == "miss") & kept),
                      ("street=0&hn=neq&full<0.8", both & (full < 0.8) & (SB == "0") & (HB == "neq") & kept)]:
            idx = np.flatnonzero(m)
            if not len(idx):
                continue
            sel = RNG.choice(idx, size=min(10, len(idx)), replace=False)
            exs[nm] = [dict(s1=sid[i], s1_name=sn[i], s1_addr=sa[i], rec=cid[i], rec_name=rn[i], rec_addr=ra[i], p=round(float(p[i]), 4),
                            n_claims=int(ncl[i]), full=round(float(full[i]), 3), street_fuzzy=round(float(sf[i]), 3), same_core=bool(ceq[i])) for i in sel]
        R["examples"] = exs
        RES[c] = R
        log(c, {k: v for k, v in R.items() if k not in ("table", "examples")})
        for nm, L in exs.items():
            for e in L[:6]:
                log("EX", c, nm, e)
        np.savez_compressed(os.path.join(HERE, f"C_test_{c}_pairfeats.npz"), **F, ceq=ceq)
    json.dump(RES, open(os.path.join(HERE, "C_results_test.json"), "w"), indent=1, default=str)
    log("saved C_results_test.json")


if __name__ == "__main__":
    main()
