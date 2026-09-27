"""RL-30 / VE (adversarial verification of investigator E's token-role table; lens: ALREADY CAPTURED / NOT INCREMENTAL).
READ-ONLY: reads V0/V1 metas, RL-27 NEW probabilities, V1 LF / RL-27 columns, E_roles_train.pkl (label-free role table
built from the train corpus exactly as on test). Labels are used ONLY to score.  Writes rl30/VE_L1_v1.json, rl30/VE_L1_feats.npz.

Continuous role feature (not just E's binary flags): for every pool pair, over the differing name tokens A^R
   llr(t) = log( ((D+.5)/sumD) / ((F+.5)/sumF) )   (support >= 5, else unknown)
   rl_sum, rl_max, rl_min, n_unk, n_diff, nd (near-dup), same_addr.
Checks:
 1 premise (label-backed): P(match | near-dup, llr bin) overall and at the same address; token-level Spearman between
   label-free s(t) and the labelled P(match) of near-dup pairs whose differing set contains t.
 2 incremental over NEW: logistic y ~ cubic(logit p_NEW)+country  vs  + role features (LR chi2), and cross-fit decision
   impact (fit V0 -> apply V1 and V1 -> V0, threshold 0.78 on the recalibrated p, paired bootstrap over S1).
 3 redundancy: AUC of role features vs existing TOK16 / RL-27 columns inside near-dup pairs, per NEW p band;
   OLS R^2 of rl_sum on TOK16 + RL-27 columns.
 4 NEW error census in near-dup pairs by llr bin.
"""
import os, sys, time, json, pickle
import numpy as np, pandas as pd
from scipy.stats import spearmanr, chi2
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from E_flags import s1_feats, rec_feats

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
E24 = os.path.join(ROOT, "experiments", "E024")
TOKN = ["n_s1_only", "n_c_only", "idf_s1_only_sum", "idf_c_only_sum", "idf_c_only_max", "idf_s1_only_max", "frac_idf_s1_only",
        "frac_idf_c_only", "n_s1_only_soft", "n_c_only_soft", "idf_s1_only_soft", "idf_c_only_soft", "addr_n_s1_only",
        "addr_n_c_only", "addr_frac_s1_only", "addr_frac_c_only"]
RLN = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]


def llr_tables(roles, min_supp=5, alpha=0.5):
    out = {}
    for c, T in roles.items():
        sd, sf = float(T.D.sum()), float(T.F.sum())
        t = T[T.supp >= min_supp]
        out[c] = dict(zip(t.index, np.log(((t.D + alpha) / sd) / ((t.F + alpha) / sf)).astype(float)))
    return out


def pair_feats(ps1, cand, pc, S1F, RF, LLR):
    n = len(ps1)
    F = {k: np.zeros(n, np.float32) for k in ("rl_sum", "rl_max", "rl_min")}
    for k in ("n_unk", "n_diff"):
        F[k] = np.zeros(n, np.int16)
    F["nd"] = np.zeros(n, bool); F["same_addr"] = np.zeros(n, bool)
    for k in range(n):
        A, _, snum, sst, _ = S1F[ps1[k]]
        R, rnum, rst, _ = RF[cand[k]]
        L = LLR.get(pc[k], {})
        d = A ^ R
        vals = [L[t] for t in d if t in L]
        F["n_diff"][k] = len(d); F["n_unk"][k] = len(d) - len(vals)
        if vals:
            F["rl_sum"][k] = sum(vals); F["rl_max"][k] = max(vals); F["rl_min"][k] = min(vals)
        F["nd"][k] = near_dup(A, R, need_core=True)
        F["same_addr"][k] = (snum is not None and snum == rnum and bool(sst & rst))
    return F


def auc(score, y):
    y = np.asarray(y, bool); s = np.asarray(score, float)
    n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0:
        return None
    r = pd.Series(s).rank().values
    return round(float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), 4)


def irls(X, y, w0=None, ridge=1e-6):
    w = np.zeros(X.shape[1]) if w0 is None else w0.copy()
    for _ in range(100):
        eta = np.clip(X @ w, -30, 30); mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu) + 1e-12
        H = X.T @ (X * W[:, None]) + ridge * np.eye(X.shape[1]); g = X.T @ (y - mu) - ridge * w
        st = np.linalg.solve(H, g); w += st
        if np.abs(st).max() < 1e-9:
            break
    mu = np.clip(1 / (1 + np.exp(-np.clip(X @ w, -30, 30))), 1e-12, 1 - 1e-12)
    return w, float((y * np.log(mu) + (1 - y) * np.log(1 - mu)).sum())


def design(p, isUS, F, with_role):
    lp = np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
    cols = [np.ones_like(lp), lp, lp ** 2 / 10, lp ** 3 / 100, isUS.astype(float), isUS * lp]
    if with_role:
        nd = F["nd"].astype(float)
        cols += [nd, nd * F["rl_sum"], nd * F["rl_max"], F["rl_sum"] / 5, F["rl_max"] / 5, (F["n_unk"] > 0).astype(float),
                 F["same_addr"].astype(float), nd * F["same_addr"] * F["rl_max"], F["rl_sum"] / 5 * isUS]
    return np.column_stack(cols)


def load_set(vn, S1, recs, LLR):
    M = np.load(os.path.join(E24, f"{vn}_a50n10d10a", "meta.npz"))
    s1_ids = M["s1_ids"]; s1idx = M["s1idx"]; cand = M["cand"]; ps1 = s1_ids[s1idx]; cs = M["country"]; pc = cs[s1idx]
    SS = S1.loc[s1_ids]; uc = np.unique(cand); RR = recs.loc[uc]
    S1F = {s: s1_feats(nm, ad) for s, nm, ad in zip(s1_ids, SS.name.values, SS.addr.values)}
    RF = {r: rec_feats(nm, ad) for r, nm, ad in zip(uc, RR.name.values, RR.addr.values)}
    F = pair_feats(ps1, cand, pc, S1F, RF, LLR)
    p = np.load(os.path.join(RL, "cache", f"rl27_p_NEW_{vn}_s42.npy"))
    return dict(s1_ids=s1_ids, s1idx=s1idx, cand=cand, y=M["y"] == 1, n_gt=M["n_gt"], cs=cs, pc=pc, p=p, F=F, ps1=ps1, SS=SS, RR=RR)


def main():
    t0 = time.time()
    D = load("train", verbose=False)
    S1 = D["s1"].set_index("id"); recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
    RO = pickle.load(open(os.path.join(HERE, "E_roles_train.pkl"), "rb"))
    LLR = llr_tables(RO["roles"])
    sets = {vn: load_set(vn, S1, recs, LLR) for vn in ("V1", "V0")}
    log(f"features {time.time()-t0:.0f}s")
    V = sets["V1"]
    np.savez_compressed(os.path.join(HERE, "VE_L1_feats.npz"), **V["F"])
    y, p, F, pc, s1idx, n_gt, cs = V["y"], V["p"], V["F"], V["pc"], V["s1idx"], V["n_gt"], V["cs"]
    acc = p >= NEW_TH
    out = dict(n_pairs=int(len(y)), n_nd=int(F["nd"].sum()), n_nd_pos=int((F["nd"] & y).sum()))

    # ---------- 1 premise
    bins = [-99, -2, -1, 0, 1, 2, 99]
    prem = {}
    for lab, m0 in (("near_dup_all", F["nd"]), ("near_dup_same_addr", F["nd"] & F["same_addr"]), ("near_dup_band_p>=0.3", F["nd"] & (p >= 0.3))):
        rows = []
        for lo, hi in zip(bins[:-1], bins[1:]):
            m = m0 & (F["rl_max"] >= lo) & (F["rl_max"] < hi) & (F["n_unk"] < F["n_diff"])
            rows.append(dict(llr_max_bin=f"[{lo},{hi})", pairs=int(m.sum()), P_match=round(float(y[m].mean()), 4) if m.any() else None,
                             accepted=int((m & acc).sum()), acc_FP=int((m & acc & ~y).sum()), FN=int((m & ~acc & y).sum())))
        prem[lab] = rows
    # token-level: label-free s(t) vs labelled P(match | t in differing set of near-dup pair), per country
    tokcor = {}
    ii = np.flatnonzero(F["nd"])
    for c in ("US", "India"):
        cnt, pos = {}, {}
        for k in ii[pc[ii] == c]:
            A = V["SS"].name.loc[V["ps1"][k]]; R = V["RR"].name.loc[V["cand"][k]]
            for t in nset(A) ^ nset(R):
                cnt[t] = cnt.get(t, 0) + 1; pos[t] = pos.get(t, 0) + int(y[k])
        T = RO["roles"][c]
        toks_ = [t for t, n in cnt.items() if n >= 30 and t in T.index and T.supp.loc[t] >= 5]
        sv = np.array([T.s.loc[t] for t in toks_]); pm = np.array([pos[t] / cnt[t] for t in toks_])
        rho = spearmanr(sv, pm).correlation if len(toks_) > 5 else None
        top = sorted(toks_, key=lambda t: -cnt[t])[:25]
        tokcor[c] = dict(n_tokens=len(toks_), spearman_s_vs_Pmatch=None if rho is None else round(float(rho), 4),
                         top=[(t, cnt[t], round(pos[t] / cnt[t], 3), round(float(T.s.loc[t]), 3)) for t in top])
    out["1_premise"] = dict(by_llr_bin=prem, token_level=tokcor)
    log("premise", json.dumps(out["1_premise"]["token_level"], default=str)[:1500])

    # ---------- 2 incremental LR test + cross-fit decision impact
    lr = {}
    for vn, S in sets.items():
        m = S["p"] >= 0.01
        isUS = (S["pc"] == "US")[m]; Fm = {k: v[m] for k, v in S["F"].items()}; yy = S["y"][m].astype(float)
        X0 = design(S["p"][m], isUS, Fm, False); X1 = design(S["p"][m], isUS, Fm, True)
        w0, l0 = irls(X0, yy); w1, l1 = irls(X1, yy)
        dfree = X1.shape[1] - X0.shape[1]; stat = 2 * (l1 - l0)
        lr[vn] = dict(n=int(m.sum()), n_pos=int(yy.sum()), LR_chi2=round(stat, 2), df=dfree, p_value=float(chi2.sf(stat, dfree)), coef=[round(float(x), 4) for x in w1])
        for c in ("US", "India"):
            mc = (S["pc"][m] == c)
            Xa, Xb = X0[mc][:, [0, 1, 2, 3]], np.column_stack([X0[mc][:, [0, 1, 2, 3]], X1[mc][:, 6:]])
            Xb = np.delete(Xb, Xb.shape[1] - 1, axis=1)   # drop rl_sum*isUS (constant within country)
            _, la = irls(Xa, yy[mc]); _, lb = irls(Xb, yy[mc])
            lr[vn][c] = dict(n=int(mc.sum()), LR_chi2=round(2 * (lb - la), 2), df=Xb.shape[1] - Xa.shape[1], p_value=float(chi2.sf(2 * (lb - la), Xb.shape[1] - Xa.shape[1])))
        S["w"] = (w0, w1)
    out["2_LR_incremental"] = lr
    log("LR", json.dumps(lr, default=str)[:1500])
    xf = {}
    for tr, te in (("V0", "V1"), ("V1", "V0")):
        S = sets[te]; w0, w1 = sets[tr]["w"]; m = S["p"] >= 0.01
        isUS = S["pc"] == "US"
        q0 = np.zeros(len(S["p"])); q1 = np.zeros(len(S["p"]))
        q0[m] = 1 / (1 + np.exp(-np.clip(design(S["p"][m], isUS[m], {k: v[m] for k, v in S["F"].items()}, False) @ w0, -30, 30)))
        q1[m] = 1 / (1 + np.exp(-np.clip(design(S["p"][m], isUS[m], {k: v[m] for k, v in S["F"].items()}, True) @ w1, -30, 30)))
        fn = f05_per_s1(S["s1idx"], S["y"], S["p"] >= NEW_TH, S["n_gt"])
        r = {}
        for th in (0.7, 0.78, 0.85):
            f0 = f05_per_s1(S["s1idx"], S["y"], q0 >= th, S["n_gt"]); f1 = f05_per_s1(S["s1idx"], S["y"], q1 >= th, S["n_gt"])
            r[f"th{th}"] = dict(NEW_raw=round(fn.mean() * 100, 4), recal_base=round(f0.mean() * 100, 4), recal_role=round(f1.mean() * 100, 4),
                                role_vs_base=paired_boot(f0, f1), role_vs_NEWraw=paired_boot(fn, f1),
                                flips_acc_to_rej=int(((q0 >= th) & (q1 < th)).sum()), flips_rej_to_acc=int(((q0 < th) & (q1 >= th)).sum()),
                                flips_acc_to_rej_TP=int(((q0 >= th) & (q1 < th) & S["y"]).sum()), flips_rej_to_acc_TP=int(((q0 < th) & (q1 >= th) & S["y"]).sum()))
        xf[f"fit_{tr}_apply_{te}"] = r
    out["2b_crossfit_decision_impact"] = xf
    log("crossfit", json.dumps(xf, default=str)[:2000])

    # ---------- 3 redundancy with existing columns (V1)
    LF = np.load(PATHS["v1_LF"], mmap_mode="r"); TOK = np.nan_to_num(np.asarray(LF[:, 71:87], np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    R27 = np.nan_to_num(np.load(PATHS["v1_rl27"]).astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    red_nan = dict(tok_nonfinite=int((~np.isfinite(np.asarray(LF[:, 71:87]))).sum()), rl27_nonfinite=int((~np.isfinite(np.load(PATHS["v1_rl27"]))).sum()))
    nd = F["nd"]
    red = dict(nonfinite=red_nan)
    # sanity: TOK n_s1_only vs our |A-R| (tokenisation differs slightly)
    red["sanity_corr_n_s1_only"] = round(float(np.corrcoef(TOK[nd, 0] + TOK[nd, 1], F["n_diff"][nd])[0, 1]), 4)
    bands = [(0.0, 0.01), (0.01, 0.3), (0.3, 0.78), (0.78, 0.95), (0.95, 0.999), (0.999, 1.01)]
    rows = []
    for lo, hi in bands:
        m = nd & (p >= lo) & (p < hi)
        row = dict(band=f"[{lo},{hi})", pairs=int(m.sum()), pos=int((m & y).sum()), auc_p=auc(p[m], y[m]),
                   auc_neg_rl_sum=auc(-F["rl_sum"][m], y[m]), auc_neg_rl_max=auc(-F["rl_max"][m], y[m]))
        for j in (2, 3, 4, 5, 10, 11):
            row[f"auc_neg_{TOKN[j]}"] = auc(-TOK[m, j], y[m])
        for j in (0, 4, 5, 6, 9):
            row[f"auc_{RLN[j]}"] = auc(R27[m, j], y[m])
        rows.append(row)
    red["auc_by_NEW_band_in_near_dup"] = rows
    # OLS R^2 of rl_sum / rl_max on TOK16 + RL-27 inside near-dup pairs
    Xr = np.column_stack([np.ones(nd.sum()), TOK[nd], R27[nd], np.log1p(np.clip(np.abs(R27[nd]), 0, 1e6))])
    for k in ("rl_sum", "rl_max"):
        yv = F[k][nd].astype(float); b, *_ = np.linalg.lstsq(Xr, yv, rcond=None); res = yv - Xr @ b
        red[f"R2_{k}_on_TOK16_RL27"] = round(1 - res.var() / yv.var(), 4)
        Xt = np.column_stack([np.ones(nd.sum()), TOK[nd]]); b, *_ = np.linalg.lstsq(Xt, yv, rcond=None); res = yv - Xt @ b
        red[f"R2_{k}_on_TOK16"] = round(1 - res.var() / yv.var(), 4)
    red["spearman_rl_sum_vs_idf_diff_sum"] = round(float(spearmanr(F["rl_sum"][nd], TOK[nd, 2] + TOK[nd, 3]).correlation), 4)
    out["3_redundancy"] = red
    log("redundancy", json.dumps(red, default=str)[:2500])

    # ---------- 4 NEW error census in near-dup pairs
    cen = {}
    for lab, m0 in (("all_pairs", np.ones(len(y), bool)), ("near_dup", nd), ("near_dup_llrmax>=1", nd & (F["rl_max"] >= 1)),
                    ("near_dup_llrmax>=2", nd & (F["rl_max"] >= 2)), ("near_dup_llrmax<0", nd & (F["rl_max"] < 0) & (F["n_unk"] < F["n_diff"])),
                    ("near_dup_same_addr_llrmax>=1", nd & F["same_addr"] & (F["rl_max"] >= 1))):
        cen[lab] = dict(pairs=int(m0.sum()), pos=int((m0 & y).sum()), accepted=int((m0 & acc).sum()), FP=int((m0 & acc & ~y).sum()),
                        FN=int((m0 & ~acc & y).sum()), FN_p03=int((m0 & ~acc & y & (p >= 0.3)).sum()),
                        error_rate_pairs=round(float(((m0 & acc & ~y) | (m0 & ~acc & y)).sum() / max(m0.sum(), 1)), 5),
                        oracle_fix_all_errors_pp=round(float((f05_per_s1(s1idx, y, np.where(m0, y, acc), n_gt) - f05_per_s1(s1idx, y, acc, n_gt)).mean() * 100), 4))
    out["4_NEW_error_census_V1"] = cen
    log("census", json.dumps(cen, default=str))
    # examples: accepted FP and FN inside near-dup llr>=1
    rng = np.random.default_rng(0); ex = {}
    for lab, m in (("acc_FP_nd_llr>=1", nd & (F["rl_max"] >= 1) & acc & ~y), ("FN_nd_llr>=1", nd & (F["rl_max"] >= 1) & ~acc & y & (p >= 0.05)),
                   ("acc_TP_nd_same_addr_llr>=2", nd & F["same_addr"] & (F["rl_max"] >= 2) & acc & y)):
        jj = np.flatnonzero(m); jj = rng.choice(jj, size=min(6, len(jj)), replace=False) if len(jj) else jj
        ex[lab] = [dict(s1_name=V["SS"].name.loc[V["ps1"][i]], s1_addr=V["SS"].addr.loc[V["ps1"][i]], rec_name=V["RR"].name.loc[V["cand"][i]],
                        rec_addr=V["RR"].addr.loc[V["cand"][i]], p=round(float(p[i]), 4), rl_max=round(float(F["rl_max"][i]), 2), y=bool(y[i])) for i in jj]
    out["examples"] = ex
    json.dump(out, open(os.path.join(HERE, "VE_L1_v1.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
