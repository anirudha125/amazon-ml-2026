"""RL-30 / VE_F2 (adversarial verification of E's F2 'near-duplicate co-located sibling' signal) -- labelled V1 checks.
Re-creation of the script that produced VE_1_v1.json (08:57; its .py was overwritten by a concurrent verifier), plus:
  - strict (F2_gt) vs tie (F2_ge & ~F2_gt) split of the accepted flagged pairs,
  - where the true owner of each flagged FP record sits (V1 / V0 / training pools) and, if in a pool, the owner's own RL-27 row.
READ-ONLY.  Output: rl30/VE_F2_1_v1.json"""
import os, sys, json, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from E_flags2 import logit_incremental

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
C = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
E24 = os.path.join(ROOT, "experiments", "E024")


def fit_multi(yv, X):
    w = np.zeros(X.shape[1])
    for _ in range(60):
        mu = 1 / (1 + np.exp(-(X @ w))); W = mu * (1 - mu) + 1e-12
        H = X.T @ (X * W[:, None]) + 1e-6 * np.eye(X.shape[1]); st = np.linalg.solve(H, X.T @ (yv - mu)); w += st
        if np.abs(st).max() < 1e-8:
            break
    mu = np.clip(1 / (1 + np.exp(-(X @ w))), 1e-12, 1 - 1e-12)
    return w, np.sqrt(np.diag(np.linalg.inv(H))), float((yv * np.log(mu) + (1 - yv) * np.log(1 - mu)).sum())


def main():
    t0 = time.time()
    M = np.load(PATHS["v1_meta"])
    s1_ids = M["s1_ids"]; s1idx = M["s1idx"]; cand = M["cand"]; y = M["y"] == 1
    p = np.load(PATHS["v1_p_new"]); R = np.load(PATHS["v1_rl27"])
    p43 = np.load(os.path.join(RL, "cache", "rl27_p_NEW_V1_s43.npy")); p44 = np.load(os.path.join(RL, "cache", "rl27_p_NEW_V1_s44.npy"))
    Fz = np.load(os.path.join(HERE, "E_v1_flags.npz"))
    f2 = Fz["f2_ge"]; f2g = Fz["f2_gt"]; hs = Fz["has_sib"]; tie = f2 & ~f2g
    D = load("train", verbose=False)
    S1 = D["s1"].set_index("id"); recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
    owner = dict(zip(D["gt"].rec.values, D["gt"].s1.values))
    inV1 = set(s1_ids)
    pools = {}
    for nm in ("V0", "T0", "E014", "T2X"):
        Mp = np.load(os.path.join(E24, f"{nm}_a50n10d10a", "meta.npz"))
        pools[nm] = dict(ids=set(Mp["s1_ids"]), ps1=Mp["s1_ids"][Mp["s1idx"]], cand=Mp["cand"], y=Mp["y"])
    ps1 = s1_ids[s1idx]
    acc = p >= NEW_TH
    out = {}
    log(f"loaded {time.time()-t0:.0f}s  f2_ge {int(f2.sum())} acc {int((f2 & acc).sum())}")

    # ---------------- Q1: all V1 accepted FPs -- is the record's true owner scored in V1?
    fp = np.flatnonzero(acc & ~y)
    own = [owner.get(c) for c in cand[fp]]
    has_owner = np.array([o is not None for o in own]); own_in_v1 = np.array([o in inV1 for o in own])
    out["Q1"] = dict(v1_accepted=int(acc.sum()), v1_acc_FP=int(len(fp)), FP_record_has_owner=int(has_owner.sum()),
                     FP_owner_in_V1=int(own_in_v1.sum()), FP_owner_not_in_V1=int((has_owner & ~own_in_v1).sum()))
    log("Q1", out["Q1"])
    rows = []
    for i in np.flatnonzero(f2 & (p >= 0.01)):
        o = owner.get(cand[i])
        orow = {}
        for nm, P in pools.items():
            if o in P["ids"]:
                jj = np.flatnonzero((P["ps1"] == o) & (P["cand"] == cand[i]))
                orow[nm] = dict(owner_has_rec_as_candidate=bool(len(jj)))
                if len(jj):
                    Rp = np.load(os.path.join(RL, "cache", f"rl27_{nm}.npy"), mmap_mode="r")
                    orow[nm].update(owner_rv_rank=float(Rp[jj[0], 6]), owner_rv_gap=float(Rp[jj[0], 9]), y=int(P["y"][jj[0]]))
        rows.append(dict(s1_name=S1.name.loc[ps1[i]], rec_name=recs.name.loc[cand[i]], rec_addr=recs.addr.loc[cand[i]],
                         p42=round(float(p[i]), 4), p43=round(float(p43[i]), 4), p44=round(float(p44[i]), 4), accepted=bool(acc[i]),
                         y=bool(y[i]), strict_gt=bool(f2g[i]), owner=o, owner_name=S1.name.loc[o] if o in S1.index else None,
                         owner_in_V1=o in inV1, owner_rows=orow, **{c: (None if np.isnan(R[i, k]) else round(float(R[i, k]), 4)) for k, c in enumerate(C)}))
    out["F2_ge_pairs_p_ge_0.01"] = rows
    for r in rows:
        log(r["accepted"], r["y"], "gt" if r["strict_gt"] else "tie", r["p42"], "rv", r["rv_rank"], "| ", r["s1_name"], "<>", r["rec_name"], "| owner", r["owner_name"], r["owner_in_V1"], r["owner_rows"])
    # strict vs tie among accepted
    out["accepted_split"] = {k: dict(acc=int((acc & f).sum()), TP=int((acc & f & y).sum()), FP=int((acc & f & ~y).sum()),
                                     FP_S1=int(np.unique(s1idx[acc & f & ~y]).size))
                             for k, f in (("F2_ge", f2), ("F2_gt_strict", f2g), ("F2_tie", tie))}
    log("accepted split", out["accepted_split"])

    # ---------------- Q2: existing RL-27 column rules among accepted pairs
    rv = R[:, 6]; gap = np.nan_to_num(R[:, 9], nan=0); coloc = R[:, 4]
    rules = {"rv_rank>1": rv > 1, "rv_gap<0": gap < 0, "coloc>=1": coloc >= 1, "rv_rank>1&coloc>=1": (rv > 1) & (coloc >= 1),
             "F2_ge": f2, "F2_gt": f2g, "F2_tie": tie}
    f2fp = acc & f2 & ~y; q2 = {}
    for k, f in rules.items():
        a = acc & f
        q2[k] = dict(accepted=int(a.sum()), acc_FP=int((a & ~y).sum()), acc_precision=round(float(y[a].mean()), 5) if a.any() else None,
                     covers_F2_FPs=int((f2fp & f).sum()), of_F2_FPs=int(f2fp.sum()), logit_incr=logit_incremental(y, p, f))
        log("Q2", k, {x: q2[k][x] for x in ("accepted", "acc_FP", "acc_precision", "covers_F2_FPs")}, (q2[k]["logit_incr"] or {}).get("z"))
    out["Q2"] = q2
    # ---------------- Q3: F2 beyond logit(p) + existing rule; and the same test clustered by S1 (drop-one-S1 jackknife of the coefficient)
    m = p >= 0.01
    pp = np.clip(p[m].astype(float), 1e-6, 1 - 1e-6); yy = y[m].astype(float); lg = np.log(pp / (1 - pp))
    q3 = {}
    for rk in ("rv_rank>1", "rv_rank>1&coloc>=1"):
        base = np.column_stack([np.ones(m.sum()), lg, rules[rk][m].astype(float)])
        w1, s1e, ll1 = fit_multi(yy, base); w2, s2e, ll2 = fit_multi(yy, np.column_stack([base, f2[m].astype(float)]))
        q3[rk] = dict(F2_coef_given_rule=round(float(w2[3]), 3), F2_z_given_rule=round(float(w2[3] / s2e[3]), 2), LR_F2_given_rule=round(2 * (ll2 - ll1), 2))
        log("Q3", rk, q3[rk])
    # which S1 drive the F2 LR? drop each flagged S1 in turn
    sidx_m = s1idx[m]; fm = f2[m]
    base = np.column_stack([np.ones(m.sum()), lg])
    jk = {}
    for s in np.unique(sidx_m[fm]):
        keep = sidx_m != s
        w1, _, ll1 = fit_multi(yy[keep], base[keep]); w2, se2, ll2 = fit_multi(yy[keep], np.column_stack([base[keep], fm[keep].astype(float)]))
        jk[str(s1_ids[s])] = dict(name=S1.name.loc[s1_ids[s]], n_flag_removed=int((fm & ~keep).sum()), LR_without=round(2 * (ll2 - ll1), 2),
                                  z_without=round(float(w2[2] / se2[2]), 2))
    q3["drop_one_S1"] = jk
    log("Q3 drop-one", jk)
    out["Q3"] = q3
    out["seeds"] = {nm: dict(F2_acc=int(((pv >= NEW_TH) & f2).sum()), F2_acc_FP=int(((pv >= NEW_TH) & f2 & ~y).sum()))
                    for nm, pv in (("s42", p), ("s43", p43), ("s44", p44))}
    json.dump(out, open(os.path.join(HERE, "VE_F2_1_v1.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
