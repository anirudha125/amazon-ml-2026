"""RL-30 / E (Part 6) -- label-backed sanity check of the test-derivable candidate features on V1, incremental over RL-27 NEW.
Flags are built exactly as they would be on test: token roles and co-located siblings from the split's own corpus
(E_roles_train.pkl), no labels. Labels (V1 meta y / n_gt) are used ONLY to score the flags.
Output: rl30/E_v1_results.json, rl30/E_v1_flags.npz (per-pair flag arrays, row-aligned with the V1 meta)."""
import os, sys, time, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from E_flags import *

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def maxclaim(acc, p, cand):
    idx = np.flatnonzero(acc)
    df = pd.DataFrame({"i": idx, "c": cand[idx], "p": p[idx]}).sort_values(["p", "i"], ascending=[False, True])
    out = np.zeros_like(acc); out[df.drop_duplicates("c").i.values] = True
    return out


def main():
    t0 = time.time()
    M = np.load(PATHS["v1_meta"]); s1idx, cand, y, n_gt, cs = M["s1idx"], M["cand"], M["y"] == 1, M["n_gt"], M["country"]
    s1_ids = M["s1_ids"]; ps1 = s1_ids[s1idx]; pc = cs[s1idx]
    p = np.load(PATHS["v1_p_new"]); acc = p >= NEW_TH
    rl27 = np.load(PATHS["v1_rl27"])
    D = load("train", verbose=False)
    S1 = D["s1"].set_index("id").loc[s1_ids]
    recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
    uc = np.unique(cand); RR = recs.loc[uc]
    S1F = {s: s1_feats(nm, ad) for s, nm, ad in zip(s1_ids, S1.name.values, S1.addr.values)}
    RF = {r: rec_feats(nm, ad) for r, nm, ad in zip(uc, RR.name.values, RR.addr.values)}
    log(f"features for {len(S1F):,} S1 / {len(RF):,} records {time.time()-t0:.0f}s")
    RO = pickle.load(open(os.path.join(HERE, "E_roles_train.pkl"), "rb"))
    S_role, DC = role_lookup(RO["roles"])
    fl = compute_flags(ps1, cand, pc, S1F, RF, S_role, DC, RO["sib"])
    F = derive(fl)
    np.savez_compressed(os.path.join(HERE, "E_v1_flags.npz"), **{k: v for k, v in fl.items()}, **{"flag_" + k: v for k, v in F.items()})
    log(f"flags {time.time()-t0:.0f}s")

    f0 = f05_per_s1(s1idx, y, acc, n_gt)
    acc_mc = maxclaim(acc, p, cand); f0_mc = f05_per_s1(s1idx, y, acc_mc, n_gt)
    isUS = cs == "US"
    out = dict(baseline=dict(macro=round(f0.mean() * 100, 4), US=round(f0[isUS].mean() * 100, 4), India=round(f0[~isUS].mean() * 100, 4),
                             macro_maxclaim_within_V1=round(f0_mc.mean() * 100, 4), accepted=int(acc.sum()), TP=int((acc & y).sum()),
                             FP=int((acc & ~y).sum()), n_gt=int(n_gt.sum()), pool_pos=int(y.sum()),
                             FN_in_pool_rejected=int((~acc & y).sum()), rejected_p03_pos=int((~acc & (p >= 0.3) & y).sum()),
                             rejected_p03=int((~acc & (p >= 0.3)).sum())),
               n_pairs=int(len(y)), n_s1=int(len(s1_ids)))
    log("baseline", out["baseline"])
    band = p >= 0.3
    res = {}
    ex = {}
    rng = np.random.default_rng(0)
    for name, f in F.items():
        r = dict(pairs=int(f.sum()), s1=int(np.unique(s1idx[f]).size), pos_pairs=int((f & y).sum()),
                 P_match_flag=round(float(y[f].mean()), 5) if f.any() else None, P_match_noflag=round(float(y[~f].mean()), 5),
                 P_match_flag_p03=round(float(y[f & band].mean()), 5) if (f & band).any() else None,
                 P_match_noflag_p03=round(float(y[~f & band].mean()), 5),
                 accepted=int((acc & f).sum()), accepted_s1=int(np.unique(s1idx[acc & f]).size),
                 acc_TP=int((acc & f & y).sum()), acc_FP=int((acc & f & ~y).sum()),
                 acc_precision_in_flag=round(float(y[acc & f].mean()), 5) if (acc & f).any() else None,
                 acc_precision_outside=round(float(y[acc & ~f].mean()), 5),
                 NEW_recall_in_flag=round(float((acc & f & y).sum() / max((f & y).sum(), 1)), 5),
                 NEW_recall_outside=round(float((acc & ~f & y).sum() / max((~f & y).sum(), 1)), 5),
                 rejected_p03=int((f & band & ~acc).sum()), rejected_p03_pos=int((f & band & ~acc & y).sum()),
                 by_country={c: dict(pairs=int((f & (pc == c)).sum()), accepted=int((acc & f & (pc == c)).sum()),
                                     acc_FP=int((acc & f & ~y & (pc == c)).sum())) for c in ("US", "India")},
                 firing_rate_among_accepted=round(float((acc & f).sum() / acc.sum()), 6),
                 firing_rate_s1_with_flagged_accepted=round(float(np.unique(s1idx[acc & f]).size / len(s1_ids)), 6))
        if (acc & f).any():
            # veto accepted flagged pairs
            fv = f05_per_s1(s1idx, y, acc & ~f, n_gt); r["veto"] = paired_boot(f0, fv)
            r["veto"]["US_pp"] = round((fv[isUS] - f0[isUS]).mean() * 100, 4); r["veto"]["India_pp"] = round((fv[~isUS] - f0[~isUS]).mean() * 100, 4)
            fo = f05_per_s1(s1idx, y, acc & ~(f & ~y), n_gt); r["veto_oracle"] = paired_boot(f0, fo)
            am = maxclaim(acc & ~f, p, cand); r["veto_then_maxclaim_vs_maxclaim"] = paired_boot(f0_mc, f05_per_s1(s1idx, y, am, n_gt))
            # per-pair effect of vetoing ONE flagged pair alone (others unchanged) -> break-even false fraction L/(L+G)
            ii = np.flatnonzero(acc & f); si = s1idx[ii]
            npd = np.bincount(s1idx, weights=acc, minlength=len(n_gt)); tpd = np.bincount(s1idx, weights=acc & y, minlength=len(n_gt))
            g, npr, tp = n_gt[si], npd[si], tpd[si]
            tp2 = tp - y[ii]; np2 = npr - 1
            f_new = np.where(g == 0, (np2 == 0).astype(float), np.where(tp2 > 0, 1.25 * tp2 / np.maximum(0.25 * g + np2, 1e-9), 0.0))
            dlt = f_new - f0[si]
            r["single_veto_mean_delta_if_TP"] = round(float(dlt[y[ii]].mean()), 5) if y[ii].any() else None
            r["single_veto_mean_delta_if_FP"] = round(float(dlt[~y[ii]].mean()), 5) if (~y[ii]).any() else None
            r["veto_TP_removed"] = int((acc & f & y).sum()); r["veto_FP_removed"] = int((acc & f & ~y).sum())
        if (f & band & ~acc).any():
            ar = acc | (f & band); fr = f05_per_s1(s1idx, y, ar, n_gt); r["rescue_p03"] = paired_boot(f0, fr)
            r["rescue_p03"]["added_TP"] = int((f & band & ~acc & y).sum()); r["rescue_p03"]["added_FP"] = int((f & band & ~acc & ~y).sum())
            ro = acc | (f & band & y); r["rescue_p03_oracle"] = paired_boot(f0, f05_per_s1(s1idx, y, ro, n_gt))
        # overlap with RL-27 columns (coloc = col 4, dupf = col 5, nf_n = col 0)
        if f.any():
            r["share_with_rl27_coloc_gt0"] = round(float((rl27[f, 4] > 0).mean()), 4)
            r["p_new_quantiles_in_flag"] = [round(float(q), 4) for q in np.quantile(p[f], [0.1, 0.5, 0.9])]
        res[name] = r
        # examples: accepted FPs and TPs inside the flag, rejected positives
        E = {}
        for lab, m in (("acc_FP", acc & f & ~y), ("acc_TP", acc & f & y), ("rej_pos_p03", f & band & ~acc & y), ("rej_neg_p03", f & band & ~acc & ~y)):
            ii = np.flatnonzero(m)
            if len(ii):
                ii = rng.choice(ii, size=min(6, len(ii)), replace=False)
                E[lab] = [dict(s1=ps1[i], s1_name=S1.name.loc[ps1[i]], s1_addr=S1.addr.loc[ps1[i]], rec=cand[i], rec_name=RR.name.loc[cand[i]],
                               rec_addr=RR.addr.loc[cand[i]], p=round(float(p[i]), 4), smax=None if np.isnan(fl["smax"][i]) else round(float(fl["smax"][i]), 3))
                          for i in ii]
        ex[name] = E
        log(name, {k: r[k] for k in ("pairs", "s1", "accepted", "acc_FP", "acc_precision_in_flag", "NEW_recall_in_flag")},
            r.get("veto", {}).get("delta_pp"), r.get("veto_oracle", {}).get("delta_pp"), r.get("rescue_p03", {}).get("delta_pp"))
    out["flags"] = res
    # combined best-case: union of vetoes
    json.dump(out, open(os.path.join(HERE, "E_v1_results.json"), "w"), indent=1, default=str)
    json.dump(ex, open(os.path.join(HERE, "E_v1_examples.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
