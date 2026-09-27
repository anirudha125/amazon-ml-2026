"""RL-30 investigator B -- Part 3 audit, step 2: compute the proposed signals on V1 pairs (train corpus, label-free definitions from
rl30_lib) and measure how much of each is already encoded by existing stage-2 columns (LF 87 + RL-27 10), plus a label-backed
residual check against the deployed RL-27 NEW s42 probabilities (does the signal still separate y at fixed model probability?).
READ-ONLY; writes rl30/B_v1_signals.json only.

Signals
 S1  co-located contrast: another train S1 at this S1's exact address (akey) whose name is closer (token_set_ratio on folded tokens)
     to the record than this S1's name.  cc_S (S1-address group), cc_R (group of S1 at the record's exact address), margins.
 S2  token role: per token, co-located near-duplicate contrast rate over the train S1 corpus (label-free) vs IDF vs the label-backed
     match rate when the token is the differing token in a V1 near-duplicate pair.
 S3  street-name overlap (rl30_lib.street_parts street tokens) vs full-address overlap.
 S4  decoy transformations: house-number edit category (street_parts numbers) vs NUM27; single-token substitutions / add-drops.
"""
import os, sys, time, json, math, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl30_lib import *  # noqa
from rapidfuzz import process, fuzz
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
RES = {}
import B_lfsem  # noqa: E402  (module import: defines LF_NAMES; its main() is not run)
LF_NAMES = B_lfsem.LF_NAMES
R27_NAMES = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
ALL_NAMES = [f"LF{j}:{n}" for j, n in enumerate(LF_NAMES)] + [f"RL27:{n}" for n in R27_NAMES]


def auc_dir(target, score):
    """AUC of score for binary target, direction-free (max(AUC, 1-AUC)); NaN -> sentinel -1e9."""
    s = np.nan_to_num(np.asarray(score, np.float64), nan=-1e9)
    if target.min() == target.max():
        return float("nan")
    a = roc_auc_score(target, s)
    return float(max(a, 1 - a))


def top_assoc(target, X, names, k=8, binary=True):
    out = []
    for j in range(X.shape[1]):
        col = X[:, j]
        if np.nanstd(col) == 0:
            continue
        if binary:
            v = auc_dir(target, col)
        else:
            ok = ~np.isnan(col) & ~np.isnan(target)
            v = abs(float(spearmanr(col[ok], target[ok]).correlation)) if ok.sum() > 50 else float("nan")
        out.append((names[j], round(v, 4)))
    out.sort(key=lambda x: -np.nan_to_num(x[1]))
    return out[:k]


def ols_r2(t, X):
    X = np.nan_to_num(np.asarray(X, np.float64), nan=-1.0); X = np.hstack([X, np.ones((len(X), 1))])
    beta, *_ = np.linalg.lstsq(X, t, rcond=None); r = t - X @ beta
    return float(1 - r.var() / t.var())


def resid_table(mask_sig, y, p, pmask, bins=(0.1, 0.3, 0.5, 0.78, 0.9, 0.97, 1.0001)):
    """y-rate vs mean p per probability bin for signal=1 / signal=0 (only rows in pmask)."""
    rows = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        b = pmask & (p >= lo) & (p < hi)
        r = dict(bin=f"[{lo:.2f},{min(hi,1):.2f})")
        for tag, s in (("sig1", b & mask_sig), ("sig0", b & ~mask_sig)):
            n = int(s.sum()); r[tag] = dict(n=n, y_rate=round(float(y[s].mean()), 4) if n else None, mean_p=round(float(p[s].mean()), 4) if n else None)
        rows.append(r)
    acc = pmask & (p >= NEW_TH)
    summ = {}
    for tag, s in (("sig1", acc & mask_sig), ("sig0", acc & ~mask_sig)):
        n = int(s.sum()); summ[tag] = dict(accepted=n, precision=round(float(y[s].mean()), 4) if n else None,
                                           FP=int((y[s] == 0).sum()), expected_FP_from_p=round(float((1 - p[s]).sum()), 1))
    rej = pmask & (p >= 0.1) & (p < NEW_TH)
    for tag, s in (("sig1", rej & mask_sig), ("sig0", rej & ~mask_sig)):
        n = int(s.sum()); summ[tag].update(rejected_p10_78=n, FN_in_rejected=int(y[s].sum()), expected_pos_from_p=round(float(p[s].sum()), 1))
    return dict(bins=rows, at_threshold=summ)


def main():
    m = np.load(PATHS["v1_meta"], allow_pickle=True)
    s1_ids, s1idx, cand, y = m["s1_ids"], m["s1idx"], m["cand"], m["y"].astype(np.int8)
    ctry_s1 = m["country"]; ctry = ctry_s1[s1idx]
    LF = np.load(PATHS["v1_LF"]); R = np.load(PATHS["v1_rl27"]); p = np.load(PATHS["v1_p_new"]).astype(np.float64)
    n = len(y); log(f"V1 {n:,} pairs, {len(s1_ids):,} S1, positives {int(y.sum()):,}")
    D = load("train", verbose=False)
    s1 = D["s1"]; rec = pd.concat([D["s2"], D["s3"]], ignore_index=True)
    S1i = pd.Index(s1.id.values).get_indexer(s1_ids); assert (S1i >= 0).all()
    uc, rinv = np.unique(cand, return_inverse=True)
    Ri = pd.Index(rec.id.values).get_indexer(uc); assert (Ri >= 0).all()
    rn, ra, rc = rec.name.values[Ri], rec.addr.values[Ri], rec.country.values[Ri]
    an, aa = s1.name.values[S1i], s1.addr.values[S1i]
    log(f"text joined; {len(uc):,} unique records")
    # ---------------- per-record / per-S1 parsed text
    r_tok = [name_tokens(x) for x in rn]; r_str = [" ".join(t) for t in r_tok]; r_set = [frozenset(t) for t in r_tok]
    r_ak = [akey(x) for x in ra]; r_sp = [street_parts(x) for x in ra]; r_aw = [addr_words(x) for x in ra]
    a_tok = [name_tokens(x) for x in an]; a_str = [" ".join(t) for t in a_tok]; a_set = [frozenset(t) for t in a_tok]
    a_ak = [akey(x) for x in aa]; a_sp = [street_parts(x) for x in aa]; a_aw = [addr_words(x) for x in aa]
    log("record/S1 parsing done")
    # ---------------- train S1 corpus address groups
    all_ak = np.array([akey(x) for x in s1.addr.values], dtype=object)
    keys = pd.Series(s1.country.values, dtype=object) + "|" + pd.Series(all_ak, dtype=object)
    codes, uniq = pd.factorize(keys); codes = codes.astype(np.int64)
    empty_code = set(np.flatnonzero(pd.Series(uniq).str.endswith("|").values).tolist())
    size = np.bincount(codes); order = np.argsort(codes, kind="stable"); starts = np.r_[0, np.cumsum(size)]
    members = lambda c: order[starts[c]:starts[c + 1]]
    log(f"train S1 {len(s1):,}; address groups size>=2 (non-empty addr): {int(sum(1 for c in np.flatnonzero(size>=2) if c not in empty_code)):,}")
    s1_all_names = s1.name.values; name_cache = {}
    def s1name(j):
        v = name_cache.get(j)
        if v is None:
            v = " ".join(toks(s1_all_names[j])); name_cache[j] = v
        return v
    a_code = codes[S1i]
    r_code = pd.Index(uniq).get_indexer(pd.Series(rc, dtype=object) + "|" + pd.Series(r_ak, dtype=object))
    CAP = 50
    # ================= S1: co-located contrast
    sim_a = process.cpdist([a_str[i] for i in s1idx], [r_str[j] for j in rinv], scorer=fuzz.token_set_ratio, workers=4, dtype=np.float32)
    log("sim(a, r) done")
    def contrast(code_of_pair, exclude_a=True):
        """max over S1 b in group(code) \\ {a} of tsr(b, r); NaN if group has no other member. Also group size (other members)."""
        best = np.full(n, np.nan, np.float32); ngrp = np.zeros(n, np.int32)
        pi, bj = [], []
        for i in range(n):
            c = code_of_pair[i]
            if c < 0 or c in empty_code or size[c] < 1:
                continue
            mem = members(c)
            if len(mem) > CAP + 1:
                mem = mem[:CAP + 1]
            ai = S1i[s1idx[i]]
            mem = mem[mem != ai] if exclude_a else mem
            if len(mem) == 0:
                continue
            ngrp[i] = size[c] - (1 if codes[ai] == c else 0)
            pi.extend([i] * len(mem)); bj.extend(mem.tolist())
        pi = np.asarray(pi, np.int64)
        if len(pi):
            sims = process.cpdist([s1name(j) for j in bj], [r_str[rinv[i]] for i in pi], scorer=fuzz.token_set_ratio, workers=4, dtype=np.float32)
            np.fmax.at(best, pi, sims)
        return best, ngrp, len(pi)
    best_S, ng_S, ncmp_S = contrast(a_code[s1idx])
    log(f"cc_S group sims: {ncmp_S:,} comparisons")
    best_R, ng_R, ncmp_R = contrast(r_code[rinv])
    log(f"cc_R group sims: {ncmp_R:,} comparisons")
    hasS = ~np.isnan(best_S); hasR = ~np.isnan(best_R)
    ccS = hasS & (best_S > sim_a); ccR = hasR & (best_R > sim_a)
    marS = np.where(hasS, best_S - sim_a, np.nan); marR = np.where(hasR, best_R - sim_a, np.nan)
    same_addr = np.array([a_ak[s1idx[i]] != "" and a_ak[s1idx[i]] == r_ak[rinv[i]] for i in range(n)])
    Xall = np.hstack([np.asarray(LF, np.float32), R]).astype(np.float32)
    log("Xall assembled")
    rel = p >= 0.1
    s1res = dict(
        n_pairs_S1_has_colocated=int(hasS.sum()), share_pairs=round(float(hasS.mean()), 4),
        share_V1_S1_with_colocated=round(float(np.mean([a_code[k] not in empty_code and size[a_code[k]] >= 2 for k in range(len(s1_ids))])), 4),
        coloc_col_consistency=dict(
            share_rl27_coloc_equals_groupsize_minus1_nonempty_addr=round(float(np.mean((R[:, 4] == (size[a_code[s1idx]] - 1))[np.array([a_ak[k] != "" for k in s1idx])])), 4),
            share_coloc_gt0_given_hasS=round(float((R[hasS, 4] > 0).mean()), 4),
            share_hasS_given_coloc_gt0_nonempty=round(float(hasS[(R[:, 4] > 0) & (LF[:, 8] > 0)].mean()), 4)),
        n_ccS=int(ccS.sum()), n_ccS_rel_p10=int((ccS & rel).sum()), n_hasS_rel_p10=int((hasS & rel).sum()),
        n_ccR=int(ccR.sum()), n_ccR_rel_p10=int((ccR & rel).sum()), n_hasR_rel_p10=int((hasR & rel).sum()),
        n_ccS_sameaddr_rel=int((ccS & rel & same_addr).sum()),
        by_country={c: dict(hasS_rel=int((hasS & rel & (ctry == c)).sum()), ccS_rel=int((ccS & rel & (ctry == c)).sum()),
                            ccR_rel=int((ccR & rel & (ctry == c)).sum())) for c in ("US", "India")},
    )
    # association with existing columns (target = ccS among pairs with a co-located S1; and relevant subset)
    for tag, msk, tgt, mar in (("ccS_all_hasS", hasS, ccS, marS), ("ccS_rel_hasS", hasS & rel, ccS, marS),
                               ("ccR_rel_hasR", hasR & rel, ccR, marR)):
        Xm = Xall[msk]; t = tgt[msk].astype(int)
        s1res[f"assoc_{tag}"] = dict(n=int(msk.sum()), pos=int(t.sum()), top_auc=top_assoc(t, Xm, ALL_NAMES, k=10),
                                     top_spearman_margin=top_assoc(mar[msk], Xm, ALL_NAMES, k=6, binary=False),
                                     r2_margin_all97=round(ols_r2(mar[msk].astype(np.float64), Xm), 4))
        rv = R[msk]
        s1res[f"rv_crosstab_{tag}"] = dict(
            P_rvrank_gt1_given_cc1=round(float((rv[t == 1, 6] > 1).mean()), 4) if t.sum() else None,
            P_rvrank_gt1_given_cc0=round(float((rv[t == 0, 6] > 1).mean()), 4),
            P_rvgap_lt0_given_cc1=round(float((np.nan_to_num(rv[t == 1, 9], nan=0) < 0).mean()), 4) if t.sum() else None,
            P_rvgap_lt0_given_cc0=round(float((np.nan_to_num(rv[t == 0, 9], nan=0) < 0).mean()), 4))
    s1res["residual_ccS"] = resid_table(ccS, y, p, hasS)
    s1res["residual_ccR"] = resid_table(ccR, y, p, hasR)
    s1res["residual_ccS_and_rvrank1"] = resid_table(ccS & (R[:, 6] == 1), y, p, hasS)
    acc = hasS & (p >= NEW_TH)
    s1res["auc_y_within_accepted_hasS"] = dict(marS=auc_dir(y[acc], marS[acc]) if acc.sum() else None,
                                               rv_gap=auc_dir(y[acc], R[acc, 9]), p_new=auc_dir(y[acc], p[acc]), n=int(acc.sum()),
                                               FP=int((y[acc] == 0).sum()))
    exs = []
    for i in np.flatnonzero(ccS & rel & same_addr)[:400:40]:
        exs.append(dict(s1_name=an[s1idx[i]], s1_addr=aa[s1idx[i]], rec_name=rn[rinv[i]], rec_addr=ra[rinv[i]], y=int(y[i]), p=round(float(p[i]), 3),
                        sim_a=float(sim_a[i]), best_other=float(best_S[i]), rv_rank=float(R[i, 6]), rv_gap=float(R[i, 9]), coloc=float(R[i, 4])))
    s1res["examples_ccS_rel_sameaddr"] = exs
    RES["S1_colocated_contrast"] = s1res
    log(f"S1 done: hasS rel {s1res['n_hasS_rel_p10']:,} ccS rel {s1res['n_ccS_rel_p10']:,}")
    del Xall
    # ================= S2: token role
    tokres = {}
    s1_tok_all = None
    for c in ("US", "India"):
        idx_c = np.flatnonzero(s1.country.values == c)
        tl = [frozenset(toks(x)) for x in s1_all_names[idx_c]]
        dfc = collections.Counter(); [dfc.update(t) for t in tl]
        Nc = len(tl)
        loc = {int(j): k for k, j in enumerate(idx_c)}
        cdf = collections.Counter(); con = collections.Counter(); npairs = 0; nnd = 0
        grp_codes = np.unique(codes[idx_c]); grp_codes = grp_codes[(size[grp_codes] >= 2)]
        for g in grp_codes:
            if g in empty_code:
                continue
            mem = members(g)[:30]
            sets = [tl[loc[int(j)]] for j in mem]
            for s_ in sets:
                cdf.update(s_)
            for i1 in range(len(sets)):
                for i2 in range(i1 + 1, len(sets)):
                    A, B = sets[i1], sets[i2]; npairs += 1
                    d1, d2 = A - B, B - A
                    if A & B and len(d1) <= 1 and len(d2) <= 1 and (d1 or d2):
                        nnd += 1; con.update(d1 | d2)
        idf = {t: math.log((Nc - d + 0.5) / (d + 0.5) + 1.0) for t, d in dfc.items()}
        crate = {t: con[t] / (cdf[t] + 10.0) for t in cdf}
        idf_unseen = math.log((Nc + 0.5) / 0.5 + 1.0)
        tokres[c] = dict(n_s1=Nc, colocated_pairs=npairs, near_dup_pairs=nnd,
                         top_contrast=[(t, con[t], cdf[t], round(crate[t], 3), round(idf[t], 2)) for t, _ in con.most_common(20)])
        # label-backed per-token match rate when t is THE differing token of a near-dup V1 pair
        mask_c = (ctry == c)
        ym = collections.defaultdict(list); pm = collections.defaultdict(list); fill = collections.Counter(); unio = collections.Counter()
        role_sum = np.full(n, np.nan); idf_sum = np.full(n, np.nan); nd_mask = np.zeros(n, bool)
        for i in np.flatnonzero(mask_c):
            A, B = a_set[s1idx[i]], r_set[rinv[i]]
            if y[i] == 1:
                unio.update(A | B); fill.update(A ^ B)
            d1, d2 = A - B, B - A
            if A & B and len(d1) <= 1 and len(d2) <= 1 and (d1 or d2):
                nd_mask[i] = True; sd = d1 | d2
                role_sum[i] = sum(crate.get(t, 0.0) for t in sd); idf_sum[i] = sum(idf.get(t, idf_unseen) for t in sd)
                for t in sd:
                    ym[t].append(int(y[i])); pm[t].append(p[i])
        T = [t for t in ym if len(ym[t]) >= 30 and t in cdf]
        mt = np.array([np.mean(ym[t]) for t in T]); rt = np.array([np.mean(np.array(ym[t]) - np.array(pm[t])) for t in T])
        it = np.array([idf.get(t, 0) for t in T]); ct_ = np.array([crate.get(t, 0) for t in T])
        ft = np.array([fill[t] / max(unio[t], 1) for t in T])
        sp = lambda a, b: round(float(spearmanr(a, b).correlation), 4)
        tokres[c].update(
            n_tokens_eval=len(T), n_neardup_pairs_V1=int(nd_mask.sum()), n_neardup_pos=int(y[nd_mask].sum()),
            spearman_matchrate_vs_idf=sp(mt, it), spearman_matchrate_vs_contrast=sp(mt, ct_), spearman_idf_vs_contrast=sp(it, ct_),
            spearman_fillrate_vs_idf=sp(ft, it), spearman_fillrate_vs_contrast=sp(ft, ct_),
            spearman_resid_vs_contrast=sp(rt, ct_), spearman_resid_vs_idf=sp(rt, it),
            pair_level=dict(
                spearman_rolesum_vs_idfsum=sp(role_sum[nd_mask], idf_sum[nd_mask]),
                spearman_rolesum_vs_TOK_idf_s1only_plus_conly=sp(role_sum[nd_mask], np.asarray(LF[np.flatnonzero(nd_mask), 73]) + np.asarray(LF[np.flatnonzero(nd_mask), 74])),
                r2_rolesum_on_LF71_86_29_34_0_4=round(ols_r2(role_sum[nd_mask], np.asarray(LF[np.flatnonzero(nd_mask)][:, list(range(71, 87)) + list(range(29, 35)) + list(range(0, 5))])), 4),
                auc_y_rolesum=auc_dir(y[nd_mask], role_sum[nd_mask]), auc_y_idfsum=auc_dir(y[nd_mask], idf_sum[nd_mask]),
                auc_y_p_new=auc_dir(y[nd_mask], p[nd_mask])),
            residual_rolesum_high=resid_table(nd_mask & (role_sum > np.nanpercentile(role_sum[nd_mask], 75)), y, p, nd_mask),
            tokens_lowest_matchrate=sorted([(t, len(ym[t]), round(float(np.mean(ym[t])), 3), round(idf[t], 2), round(crate[t], 3)) for t in T], key=lambda x: x[2])[:12],
            tokens_highest_matchrate=sorted([(t, len(ym[t]), round(float(np.mean(ym[t])), 3), round(idf[t], 2), round(crate[t], 3)) for t in T], key=lambda x: -x[2])[:12],
            tokens_largest_abs_resid=sorted([(t, len(ym[t]), round(float(np.mean(ym[t])), 3), round(float(np.mean(pm[t])), 3), round(idf[t], 2), round(crate[t], 3)) for t in T],
                                            key=lambda x: -abs(x[2] - x[3]))[:10])
        log(f"S2 {c}: tokens {len(T)} near-dup V1 pairs {int(nd_mask.sum()):,}")
    RES["S2_token_role"] = tokres
    # ================= S3: street-name overlap vs full-address overlap
    stres = {}
    a_st = [sp_[1] for sp_ in a_sp]; r_st = [sp_[1] for sp_ in r_sp]
    st_j = np.full(n, np.nan); st_eq = np.zeros(n, bool); aw_j = np.full(n, np.nan); rest_j = np.full(n, np.nan)
    for i in range(n):
        A, B = a_st[s1idx[i]], r_st[rinv[i]]
        if A and B:
            st_j[i] = len(A & B) / len(A | B); st_eq[i] = A == B
        W1, W2 = a_aw[s1idx[i]], r_aw[rinv[i]]
        if W1 and W2:
            aw_j[i] = len(W1 & W2) / len(W1 | W2)
        R1, R2 = a_sp[s1idx[i]][2], r_sp[rinv[i]][2]
        if R1 and R2:
            rest_j[i] = len(R1 & R2) / len(R1 | R2)
    both = ~np.isnan(st_j)
    addr_cols = [11, 12, 13, 14, 15, 83, 84, 85, 86, 57, 58, 47]
    Xa = np.hstack([np.asarray(LF[:, addr_cols], np.float32), R[:, [2, 3]]])
    anames = [ALL_NAMES[j] for j in addr_cols] + ["RL27:af_n", "RL27:af_a"]
    for c in ("US", "India", "all"):
        mc = both & ((ctry == c) if c != "all" else True)
        mr = mc & rel
        d = dict(n_both_street=int(mc.sum()), share_pairs_both_street=round(float(mc.sum() / max(((ctry == c) if c != 'all' else np.ones(n, bool)).sum(), 1)), 4),
                 n_rel=int(mr.sum()), st_eq_rate_rel=round(float(st_eq[mr].mean()), 4) if mr.sum() else None)
        for tag, msk in (("all", mc), ("rel", mr)):
            if msk.sum() < 100:
                continue
            Xm = Xa[msk]
            d[f"{tag}_top_auc_st_eq"] = top_assoc(st_eq[msk].astype(int), Xm, anames, k=6)
            d[f"{tag}_top_spearman_st_j"] = top_assoc(st_j[msk], Xm, anames, k=6, binary=False)
            d[f"{tag}_r2_st_j_on_addr_cols"] = round(ols_r2(st_j[msk], Xm), 4)
            d[f"{tag}_r2_st_j_on_all97"] = round(ols_r2(st_j[msk], np.hstack([np.asarray(LF[np.flatnonzero(msk)], np.float32), R[msk]])), 4)
            hi = msk & (np.asarray(LF[:, 14]) >= 0.9)
            d[f"{tag}_P_street_mismatch_given_addr_token_set_ge_0.9"] = round(float((~st_eq[hi]).mean()), 4) if hi.sum() else None
            d[f"{tag}_P_addr_token_set_lt_0.7_given_street_eq"] = round(float((np.asarray(LF[:, 14])[msk & st_eq] < 0.7).mean()), 4) if (msk & st_eq).sum() else None
        if c != "all":
            d["residual_street_mismatch"] = resid_table(~st_eq, y, p, mc)
            acc = mc & (p >= NEW_TH)
            d["auc_y_within_accepted"] = dict(n=int(acc.sum()), FP=int((y[acc] == 0).sum()), st_j=auc_dir(y[acc], st_j[acc]),
                                              addr_token_set=auc_dir(y[acc], np.asarray(LF[:, 14])[acc]), p_new=auc_dir(y[acc], p[acc]))
        stres[c] = d
    ex = []
    for i in np.flatnonzero(both & rel & ~st_eq & (np.asarray(LF[:, 14]) >= 0.9))[:600:60]:
        ex.append(dict(s1_addr=aa[s1idx[i]], rec_addr=ra[rinv[i]], y=int(y[i]), p=round(float(p[i]), 3), addr_token_set=float(LF[i, 14]),
                       st_s1=sorted(a_st[s1idx[i]]), st_rec=sorted(r_st[rinv[i]])))
    stres["examples_street_mismatch_addr_tset_ge_0.9"] = ex
    RES["S3_street_overlap"] = stres
    log("S3 done")
    # ================= S4: decoy transformations
    def hcat(h1, h2):
        if h1 is None or h2 is None:
            return "missing"
        if h1 == h2:
            return "same"
        a_, b_ = int(h1[:15]), int(h2[:15]); dd = abs(a_ - b_)
        if dd == 1:
            return "pm1"
        if len(h1) == len(h2) and sorted(h1) == sorted(h2):
            return "transposition"
        if len(h1) == len(h2) and sum(x != z for x, z in zip(h1, h2)) == 1:
            return "one_digit"
        if dd <= 10:
            return "pm2_10"
        if h1.startswith(h2) or h2.startswith(h1) or h1.endswith(h2) or h2.endswith(h1):
            return "prefix_suffix"
        return "other"
    cat = np.array([hcat(a_sp[s1idx[i]][0], r_sp[rinv[i]][0]) for i in range(n)], dtype=object)
    L = np.asarray(LF[:, [47, 53, 55, 69, 57]], np.float32); hd = np.round(np.asarray(LF[:, 58], np.float64), 2)
    tup = pd.Series(list(map(tuple, np.hstack([L, hd[:, None]]).tolist())))
    def H(labels):
        v = pd.Series(labels).value_counts(normalize=True).values; return float(-(v * np.log2(v)).sum())
    def Hcond(lab, cond):
        df_ = pd.DataFrame({"l": lab, "c": cond}); tot = 0.0
        for _, g in df_.groupby("c"):
            tot += len(g) / len(df_) * H(g.l.values)
        return tot
    hn = cat != "missing"; sub = np.flatnonzero(hn & rel)
    s4 = dict(house_cat_counts_rel={k: int(v) for k, v in pd.Series(cat[rel]).value_counts().items()},
              H_cat_rel=round(H(cat[sub]), 4), H_cat_given_NUMtuple_rel=round(Hcond(cat[sub], tup.values[sub]), 4),
              NUMtuple_cols="lnum_status(47), lnum_edit(53), lnum_transposition(55), small_offset_same_len(69), house_agree(57), round(house_absdiff_log(58),2)")
    same_st = both & st_eq
    s4["by_cat_same_street_rel"] = {k: dict(n=int(((cat == k) & same_st & rel).sum()), y_rate=round(float(y[(cat == k) & same_st & rel].mean()), 4),
                                            mean_p=round(float(p[(cat == k) & same_st & rel].mean()), 4)) for k in pd.unique(cat[same_st & rel])}
    # token substitution / add-drop identity
    subs = collections.defaultdict(list); adds = collections.defaultdict(list)
    for i in np.flatnonzero(rel):
        A, B = a_set[s1idx[i]], r_set[rinv[i]]
        d1, d2 = A - B, B - A
        if not (A & B):
            continue
        if len(d1) == 1 and len(d2) == 1:
            subs[(next(iter(d1)), next(iter(d2)))].append(i)
        elif (len(d1) == 1 and not d2) or (len(d2) == 1 and not d1):
            adds[("drop:" + next(iter(d1))) if d1 else ("add:" + next(iter(d2)))].append(i)
    rng = np.random.default_rng(0)
    def hetero(groups, min_n=20):
        G = [(k, np.array(v)) for k, v in groups.items() if len(v) >= min_n]
        if not G:
            return {}
        allidx = np.concatenate([v for _, v in G]); lab = np.concatenate([[j] * len(v) for j, (_, v) in enumerate(G)])
        def zs(ix_by_group):
            z = []
            for ix in ix_by_group:
                e = p[ix].sum(); v = (p[ix] * (1 - p[ix])).sum(); z.append((y[ix].sum() - e) / math.sqrt(max(v, 1e-6)))
            return np.array(z)
        z = zs([v for _, v in G]); perm = rng.permutation(allidx)
        zp = zs([perm[lab == j] for j in range(len(G))])
        rows = sorted([(str(k), len(v), round(float(y[v].mean()), 3), round(float(p[v].mean()), 3), round(float(zz), 2)) for (k, v), zz in zip(G, z)], key=lambda x: -abs(x[4]))
        return dict(n_groups=len(G), n_pairs=int(len(allidx)), share_abs_z_gt3=round(float(np.mean(np.abs(z) > 3)), 4),
                    share_abs_z_gt3_permuted=round(float(np.mean(np.abs(zp) > 3)), 4), sum_z2=round(float((z ** 2).sum()), 1),
                    sum_z2_permuted=round(float((zp ** 2).sum()), 1), top_by_abs_z=rows[:12],
                    top_by_n=sorted(rows, key=lambda x: -x[1])[:12])
    s4["token_substitution_rel"] = hetero(subs)
    s4["token_add_drop_rel"] = hetero(adds)
    RES["S4_decoy_transformations"] = s4
    log("S4 done")
    RES["runtime_s"] = round(time.time() - t0, 1)
    json.dump(RES, open(os.path.join(OUT, "B_v1_signals.json"), "w"), indent=1, default=str)
    log("saved B_v1_signals.json")


if __name__ == "__main__":
    main()
