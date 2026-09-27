"""RL-30 / E -- refined (typo-robust) flags + shared scoring helpers.
Round-1 finding: exact-token F1 / F3 fire mostly on character-level typos ("Glgbgbal", "Chambreletn"), not on real word swaps.
Refinement (label-free): a differing token counts as a real word only if it occurs in >= MIN_DF S1 names (name vocab) /
S1 street names (street vocab) of the same country in the split's own corpus, and a substitution t->u must not be a
near-spelling (Levenshtein normalized similarity < 0.8)."""
import os, sys, collections
import numpy as np, pandas as pd
from rapidfuzz.distance import Levenshtein
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from E_flags import s1_feats, rec_feats, role_lookup, MIN_SUPP

MIN_DF = 5
SIM_MAX = 0.8


def vocabs(s1df):
    """country -> Counter(token -> #S1 names containing it); country -> Counter(street token -> #S1 street names containing it)"""
    NV, SV = collections.defaultdict(collections.Counter), collections.defaultdict(collections.Counter)
    for nm, ad, c in zip(s1df.name.values, s1df.addr.values, s1df.country.values):
        NV[c].update(nset(nm)); SV[c].update(street_parts(ad)[1])
    return NV, SV


def compute_refined(s1_ids, rec_ids, ctry, S1F, RF, S_role, NV, SV):
    n = len(s1_ids)
    out = {k: np.zeros(n, bool) for k in ("F1r_dist", "F1r_dist_sameaddr", "F1r_dist_subst", "F1r_dist_onesided", "F1r_filler",
                                          "F1r_filler_sameaddr", "F1_typo_only", "F3r_mismatch", "F3r_mismatch_numeq", "F3r_numeq_anyov")}
    for k in range(n):
        A, _, snum, sst, sw = S1F[s1_ids[k]]
        R, rnum, rst, rw = RF[rec_ids[k]]
        c = ctry[k]; nv = NV[c]; sv = SV[c]; sr = S_role.get(c, {})
        ne = snum is not None and snum == rnum
        same_addr = ne and bool(sst & rst)
        if near_dup(A, R, need_core=True):
            a, b = A - R, R - A
            real = all(nv.get(t, 0) >= MIN_DF for t in a | b)
            if a and b:
                t, u = next(iter(a)), next(iter(b))
                spelled = Levenshtein.normalized_similarity(t, u) >= SIM_MAX
            else:
                spelled = False
            if not real or spelled:
                out["F1_typo_only"][k] = True
            else:
                vals = [sr.get(t) for t in a | b]
                kn = [v for v in vals if v is not None]
                if kn and max(kn) >= 0.5:
                    out["F1r_dist"][k] = True
                    out["F1r_dist_sameaddr"][k] = same_addr
                    out["F1r_dist_subst"][k] = bool(a and b)
                    out["F1r_dist_onesided"][k] = not (a and b)
                if kn and len(kn) == len(vals) and max(kn) < 0.5:
                    out["F1r_filler"][k] = True
                    out["F1r_filler_sameaddr"][k] = same_addr
        if sst and rst and not (sst & rst) and ne:
            realst = all(sv.get(t, 0) >= MIN_DF for t in sst | rst)
            if realst and Levenshtein.normalized_similarity(" ".join(sorted(sst)), " ".join(sorted(rst))) < SIM_MAX:
                out["F3r_numeq_anyov"][k] = True
        if sst and rst and not (sst & rst) and sw and rw and len(sw & rw) / min(len(sw), len(rw)) >= 0.8:
            realst = all(sv.get(t, 0) >= MIN_DF for t in sst | rst)
            if realst and Levenshtein.normalized_similarity(" ".join(sorted(sst)), " ".join(sorted(rst))) < SIM_MAX:
                out["F3r_mismatch"][k] = True
                out["F3r_mismatch_numeq"][k] = ne
    return out


def maxclaim(acc, p, cand):
    idx = np.flatnonzero(acc)
    df = pd.DataFrame({"i": idx, "c": cand[idx], "p": p[idx]}).sort_values(["p", "i"], ascending=[False, True])
    out = np.zeros_like(acc); out[df.drop_duplicates("c").i.values] = True
    return out


def logit_incremental(y, p, f, lo=0.01):
    """IRLS logistic y ~ 1 + logit(p) + flag on pairs with p >= lo. Returns flag coef, SE, z, LR chi2 (1 df)."""
    m = p >= lo
    if f[m].sum() < 5:
        return None
    pp = np.clip(p[m].astype(float), 1e-6, 1 - 1e-6); yy = y[m].astype(float)
    X1 = np.column_stack([np.ones(m.sum()), np.log(pp / (1 - pp))]); X2 = np.column_stack([X1, f[m].astype(float)])

    def fit(X):
        w = np.zeros(X.shape[1])
        for _ in range(50):
            eta = X @ w; mu = 1 / (1 + np.exp(-eta)); W = mu * (1 - mu) + 1e-12
            H = X.T @ (X * W[:, None]) + 1e-8 * np.eye(X.shape[1]); g = X.T @ (yy - mu)
            st = np.linalg.solve(H, g); w += st
            if np.abs(st).max() < 1e-8:
                break
        mu = np.clip(1 / (1 + np.exp(-(X @ w))), 1e-12, 1 - 1e-12)
        ll = float((yy * np.log(mu) + (1 - yy) * np.log(1 - mu)).sum())
        return w, np.linalg.inv(H), ll
    w1, _, ll1 = fit(X1); w2, C2, ll2 = fit(X2)
    se = float(np.sqrt(C2[2, 2]))
    return dict(n=int(m.sum()), n_flag=int(f[m].sum()), coef_flag=round(float(w2[2]), 4), se=round(se, 4), z=round(float(w2[2]) / se, 2),
                LR_chi2=round(2 * (ll2 - ll1), 2), coef_logit_p=round(float(w2[1]), 4))


def score_flag(f, s1idx, y, n_gt, cs, p, cand, th, f0, f0_mc, band_lo=0.3, boot=True):
    acc = p >= th; band = p >= band_lo; pc = cs[s1idx]; isUS = cs == "US"
    r = dict(pairs=int(f.sum()), s1=int(np.unique(s1idx[f]).size), pos_pairs=int((f & y).sum()),
             P_match_flag=round(float(y[f].mean()), 5) if f.any() else None, P_match_noflag=round(float(y[~f].mean()), 5),
             P_match_flag_p03=round(float(y[f & band].mean()), 5) if (f & band).any() else None,
             P_match_noflag_p03=round(float(y[~f & band].mean()), 5),
             accepted=int((acc & f).sum()), acc_TP=int((acc & f & y).sum()), acc_FP=int((acc & f & ~y).sum()),
             acc_precision_in_flag=round(float(y[acc & f].mean()), 5) if (acc & f).any() else None,
             acc_precision_outside=round(float(y[acc & ~f].mean()), 5),
             NEW_recall_in_flag=round(float((acc & f & y).sum() / max((f & y).sum(), 1)), 5) if (f & y).any() else None,
             NEW_recall_outside=round(float((acc & ~f & y).sum() / max((~f & y).sum(), 1)), 5),
             rejected_p03=int((f & band & ~acc).sum()), rejected_p03_pos=int((f & band & ~acc & y).sum()),
             firing_rate_among_accepted=round(float((acc & f).sum() / acc.sum()), 6),
             by_country={c: dict(pairs=int((f & (pc == c)).sum()), accepted=int((acc & f & (pc == c)).sum()),
                                 acc_FP=int((acc & f & ~y & (pc == c)).sum())) for c in ("US", "India")})
    # FP rate inside / outside the flag by accepted p band (is the flag informative at the margin?)
    bands = [(th, 0.95), (0.95, 0.999), (0.999, 1.01)]
    r["acc_FP_rate_by_p_band_in_vs_out"] = {f"[{lo},{hi})": [int((acc & f & (p >= lo) & (p < hi)).sum()),
                                                            round(float((~y[acc & f & (p >= lo) & (p < hi)]).mean()), 5) if (acc & f & (p >= lo) & (p < hi)).any() else None,
                                                            round(float((~y[acc & ~f & (p >= lo) & (p < hi)]).mean()), 5)] for lo, hi in bands}
    r["logit_incremental_p>=0.01"] = logit_incremental(y, p, f)
    if boot and (acc & f).any():
        fv = f05_per_s1(s1idx, y, acc & ~f, n_gt); r["veto"] = paired_boot(f0, fv)
        r["veto"]["US_pp"] = round((fv[isUS] - f0[isUS]).mean() * 100, 4); r["veto"]["India_pp"] = round((fv[~isUS] - f0[~isUS]).mean() * 100, 4)
        r["veto_oracle"] = paired_boot(f0, f05_per_s1(s1idx, y, acc & ~(f & ~y), n_gt))
        r["veto_then_maxclaim_vs_maxclaim"] = paired_boot(f0_mc, f05_per_s1(s1idx, y, maxclaim(acc & ~f, p, cand), n_gt))
        ii = np.flatnonzero(acc & f); si = s1idx[ii]
        npd = np.bincount(s1idx, weights=acc, minlength=len(n_gt)); tpd = np.bincount(s1idx, weights=acc & y, minlength=len(n_gt))
        g, npr, tp = n_gt[si], npd[si], tpd[si]; tp2 = tp - y[ii]; np2 = npr - 1
        fn = np.where(g == 0, (np2 == 0).astype(float), np.where(tp2 > 0, 1.25 * tp2 / np.maximum(0.25 * g + np2, 1e-9), 0.0))
        dl = fn - f0[si]
        L = float(-dl[y[ii]].mean()) if y[ii].any() else None; G = float(dl[~y[ii]].mean()) if (~y[ii]).any() else None
        r["single_veto_loss_if_TP"] = None if L is None else round(L, 5); r["single_veto_gain_if_FP"] = None if G is None else round(G, 5)
        r["break_even_false_fraction"] = round(L / (L + G), 4) if (L is not None and G is not None and L + G > 0) else None
    if boot and (f & band & ~acc).any():
        r["rescue_p03"] = paired_boot(f0, f05_per_s1(s1idx, y, acc | (f & band), n_gt))
        r["rescue_p03"]["added_TP"] = int((f & band & ~acc & y).sum()); r["rescue_p03"]["added_FP"] = int((f & band & ~acc & ~y).sum())
        r["rescue_p03_oracle"] = paired_boot(f0, f05_per_s1(s1idx, y, acc | (f & band & y), n_gt))
    return r
