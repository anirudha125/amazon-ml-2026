"""RL-30 Part 4 (investigator C): street-name overlap vs full-address overlap.  READ-ONLY; writes only rl30/C_*.
V1 (labelled, RL-27 NEW s42 probabilities, held-out) + S005 accepted tables for France / US / India (unlabelled).
Run: OMP_NUM_THREADS=4 NUMBA_NUM_THREADS=4 nice -n 10 python C_street.py
"""
import os, sys, json, time, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from sklearn.metrics import roc_auc_score
from rapidfuzz import process, fuzz

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

RNG = np.random.default_rng(0)
NB = 1000
RES = {}


def addr_ok(a):
    return bool(a) and a.strip().lower() != "null"


def parse_addr(a):
    if not addr_ok(a):
        return frozenset(), None, frozenset()
    n, s, _ = street_parts(a)
    return addr_words(a), n, s


def pair_feats(sa, ra):
    """sa, ra: arrays of address strings (S1, record). Returns dict of arrays."""
    u, inv = np.unique(np.concatenate([sa, ra]), return_inverse=True)
    P = [parse_addr(a) for a in u]
    n = len(sa); i1 = inv[:n]; i2 = inv[n:]
    full = np.full(n, np.nan, np.float32); so = np.full(n, np.nan, np.float32); sj = np.full(n, np.nan, np.float32)
    hn = np.full(n, -1, np.int8)                      # 1 equal, 0 different, -1 missing on a side
    nW1 = np.zeros(n, np.int16); nW2 = np.zeros(n, np.int16); nS1 = np.zeros(n, np.int16); nS2 = np.zeros(n, np.int16)
    for k in range(n):
        W1, h1, S1 = P[i1[k]]; W2, h2, S2 = P[i2[k]]
        nW1[k] = len(W1); nW2[k] = len(W2); nS1[k] = len(S1); nS2[k] = len(S2)
        if W1 and W2:
            full[k] = len(W1 & W2) / min(len(W1), len(W2))
        if S1 and S2:
            it = len(S1 & S2)
            so[k] = it / min(len(S1), len(S2)); sj[k] = it / len(S1 | S2)
        if h1 is not None and h2 is not None:
            hn[k] = 1 if h1 == h2 else 0
    # fuzzy street string similarity (typo / abbreviation vs genuinely different street)
    sf = np.full(n, np.nan, np.float32)
    d = np.flatnonzero(~np.isnan(so))
    if len(d):
        st = [" ".join(sorted(p[2])) for p in P]
        a = [st[i1[k]] for k in d]; b = [st[i2[k]] for k in d]
        sf[d] = process.cpdist(a, b, scorer=fuzz.token_set_ratio, workers=4, dtype=np.float32) / 100.0
    return dict(full=full, so=so, sj=sj, hn=hn, sf=sf, nW1=nW1, nW2=nW2, nS1=nS1, nS2=nS2)


def core_eq(sn, rn):
    u, inv = np.unique(np.concatenate([sn, rn]), return_inverse=True)
    C = [core_tokens(x) for x in u]
    n = len(sn)
    return np.fromiter((bool(C[a]) and C[a] == C[b] for a, b in zip(inv[:n], inv[n:])), bool, n)


def sbucket(so):
    b = np.full(len(so), "miss", object)
    d = ~np.isnan(so)
    b[d & (so == 0)] = "0"; b[d & (so > 0) & (so < 1)] = "part"; b[d & (so >= 1)] = "1"
    return b


def hbucket(hn):
    return np.where(hn == 1, "eq", np.where(hn == 0, "neq", "miss"))


# ------------------------------------------------------------------ bootstrap helpers (S1-cluster Poisson bootstrap)
def boot_rate(g, num, den, W):
    """g: S1 index per row (subset rows), num/den per row. W: (NB, n_s1) Poisson weights. -> (point, lo, hi)"""
    n_s1 = W.shape[1]
    N = np.bincount(g, weights=num, minlength=n_s1); Dn = np.bincount(g, weights=den, minlength=n_s1)
    pt = N.sum() / max(Dn.sum(), 1e-9)
    b = (W @ N) / np.maximum(W @ Dn, 1e-9)
    return float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def boot_mean(vals, W):
    pt = vals.mean(); b = (W @ vals) / W.sum(1)
    return float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def auc(y, x):
    if y.min() == y.max() or len(y) < 20:
        return None
    return round(float(roc_auc_score(y, x)), 4)


def main():
    meta = np.load(PATHS["v1_meta"], allow_pickle=True)
    s1_ids, s1idx, cand, y, n_gt, ctry = (meta[k] for k in ("s1_ids", "s1idx", "cand", "y", "n_gt", "country"))
    y = y.astype(np.int8); n_s1 = len(s1_ids)
    p = np.load(PATHS["v1_p_new"]).astype(np.float64)
    LF = np.load(PATHS["v1_LF"], mmap_mode="r")
    EX = {"addr_lev(11)": np.asarray(LF[:, 11]), "addr_jw(12)": np.asarray(LF[:, 12]), "addr_tsort(13)": np.asarray(LF[:, 13]),
          "addr_tset(14)": np.asarray(LF[:, 14]), "addr_jacc(15)": np.asarray(LF[:, 15]), "house_agree(57)": np.asarray(LF[:, 57]),
          "neg_house_absdiff(58)": -np.asarray(LF[:, 58]), "neg_addr_frac_s1_only(85)": -np.asarray(LF[:, 85]),
          "neg_addr_frac_c_only(86)": -np.asarray(LF[:, 86])}
    log("V1 rows", len(y), "S1", n_s1, "pos", int(y.sum()))
    D = load("train", verbose=False)
    s1 = D["s1"].set_index("id"); rec = pd.concat([D["s2"], D["s3"]]).set_index("id")
    sid = s1_ids[s1idx]
    sa = s1.addr.reindex(sid).fillna("").values.astype(object); sn = s1.name.reindex(sid).fillna("").values.astype(object)
    ra = rec.addr.reindex(cand).fillna("").values.astype(object); rn = rec.name.reindex(cand).fillna("").values.astype(object)
    del D
    log("texts mapped; parsing")
    F = pair_feats(sa, ra)
    ceq = core_eq(sn, rn)
    log("features done")
    # reranker logit (top-10 by base only)
    rr = pickle.load(open(os.path.join(ROOT, "experiments", "E023", "rr_rrUb_big.pkl"), "rb"))
    rrc = np.array([rr.get((a, b), np.nan) for a, b in zip(sid, cand)], np.float32); del rr
    log("rr mapped, non-nan", int((~np.isnan(rrc)).sum()))
    np.savez_compressed(os.path.join(HERE, "C_v1_pairfeats.npz"), **F, ceq=ceq, rr=rrc)

    full, so, hn, sf = F["full"], F["so"], F["hn"], F["sf"]
    both = ~np.isnan(full)
    sd = ~np.isnan(so)
    acc = p >= NEW_TH
    W = RNG.poisson(1.0, size=(NB, n_s1)).astype(np.float64)
    cS1 = ctry  # per S1
    cr = ctry[s1idx]

    RES["v1_basic"] = dict(rows=int(len(y)), pos=int(y.sum()), both_addr_words=int(both.sum()), pos_both=int(y[both].sum()),
                           street_defined=int((both & sd).sum()), pos_street_defined=int(y[both & sd].sum()),
                           hn_defined=int((both & (hn >= 0)).sum()), accepted=int(acc.sum()), acc_precision=float(y[acc].mean()),
                           accepted_both=int((acc & both).sum()))
    log(RES["v1_basic"])

    # ------------------------------------------------------------ AUC table
    so_imp = np.where(sd, so, -0.1)                    # missing street ranks below 0 overlap
    hn_imp = np.where(hn == 1, 1.0, np.where(hn == 0, 0.0, 0.5))
    combo = np.where(sd, so, 0.0) + hn_imp             # street overlap + house number agreement
    sf_imp = np.where(sd, sf, -0.1)
    feats = {"full_overlap": np.nan_to_num(full, nan=-1), "street_overlap": so_imp, "street_jacc": np.where(sd, F["sj"], -0.1),
             "street_fuzzy": sf_imp, "hn_eq": hn_imp, "street+hn": combo, "p_NEW": p, "rr_logit": rrc}
    feats.update(EX)
    band = (p >= 0.2) & (p < 0.95)
    subsets = {"all_both_addr": both, "same_core_name": both & ceq, "same_house_number": both & (hn == 1),
               "uncertain_0.2_0.95": both & band, "accepted_p>=0.78": both & acc, "rejected_0.2<=p<0.78": both & (p >= 0.2) & ~acc,
               "full>=0.8": both & (full >= 0.8), "full>=0.8_and_band": both & (full >= 0.8) & band,
               "street_defined_both": both & sd, "US": both & (cr == "US"), "India": both & (cr == "India"),
               "same_core_and_full>=0.8": both & ceq & (full >= 0.8)}
    T = {}
    for sn_, m in subsets.items():
        row = dict(n=int(m.sum()), pos=int(y[m].sum()), pos_rate=round(float(y[m].mean()), 4) if m.any() else None)
        for fn, v in feats.items():
            mm = m & ~np.isnan(v) if fn == "rr_logit" else m
            row[fn] = auc(y[mm], v[mm])
        row["rr_defined"] = int((m & ~np.isnan(rrc)).sum())
        T[sn_] = row
        log(sn_, {k: row[k] for k in ("n", "pos_rate", "full_overlap", "street_overlap", "hn_eq", "street+hn", "addr_tset(14)", "house_agree(57)", "p_NEW")})
    RES["auc"] = T

    # ------------------------------------------------------------ 2x2: full hi/lo x street
    fb = np.where(full >= 0.8, "full>=0.8", "full<0.8")
    SB = sbucket(so); HB = hbucket(hn)
    cross = {}
    for scope, m0 in [("all", both), ("band", both & band), ("accepted", both & acc), ("rejected_p>=0.2", both & (p >= 0.2) & ~acc)]:
        tab = {}
        for f_ in ("full>=0.8", "full<0.8"):
            for s_ in ("1", "part", "0", "miss"):
                for h_ in ("eq", "neq", "miss"):
                    m = m0 & (fb == f_) & (SB == s_) & (HB == h_)
                    if m.sum() == 0:
                        continue
                    tab[f"{f_}|street={s_}|hn={h_}"] = dict(n=int(m.sum()), pos=int(y[m].sum()), pos_rate=round(float(y[m].mean()), 4),
                                                            mean_p=round(float(p[m].mean()), 4))
        cross[scope] = tab
    RES["cross"] = cross

    # ------------------------------------------------------------ accepted: precision by bucket + calibration (bootstrap CI)
    g = s1idx
    def prec_block(m, name):
        if m.sum() == 0:
            return dict(n=0)
        pt, lo, hi = boot_rate(g[m], y[m].astype(float), np.ones(m.sum()), W)
        return dict(n=int(m.sum()), n_s1=int(len(np.unique(g[m]))), fp=int((y[m] == 0).sum()), precision=round(pt, 4), ci=[round(lo, 4), round(hi, 4)],
                    mean_p=round(float(p[m].mean()), 4), expected_fp=round(float((1 - p[m]).sum()), 1),
                    US=int((m & (cr == "US")).sum()), India=int((m & (cr == "India")).sum()))
    A = {}
    A["all_accepted"] = prec_block(acc, "all")
    A["accepted_both_addr"] = prec_block(acc & both, "")
    for s_ in ("1", "part", "0", "miss"):
        A[f"street={s_}"] = prec_block(acc & both & (SB == s_), "")
        for h_ in ("eq", "neq", "miss"):
            A[f"street={s_}|hn={h_}"] = prec_block(acc & both & (SB == s_) & (HB == h_), "")
    K = acc & both & (full >= 0.8) & (SB == "0")
    A["KEY full>=0.8 & street=0"] = prec_block(K, "")
    for h_ in ("eq", "neq", "miss"):
        A[f"KEY|hn={h_}"] = prec_block(K & (HB == h_), "")
    A["KEY & street_fuzzy<0.8"] = prec_block(K & (sf < 0.8), "")
    A["KEY & street_fuzzy>=0.8"] = prec_block(K & (sf >= 0.8), "")
    A["KEY & same_core"] = prec_block(K & ceq, "")
    A["KEY & diff_core"] = prec_block(K & ~ceq, "")
    for c in ("US", "India"):
        A[f"KEY|{c}"] = prec_block(K & (cr == c), "")
        A[f"all_accepted|{c}"] = prec_block(acc & (cr == c), "")
    A["full>=0.8 & street=1 & hn=eq"] = prec_block(acc & both & (full >= 0.8) & (SB == "1") & (HB == "eq"), "")
    A["full<0.8 & street=0"] = prec_block(acc & both & (full < 0.8) & (SB == "0"), "")
    A["street=0 (any full)"] = prec_block(acc & both & (SB == "0"), "")
    # difference KEY vs rest-of-accepted (paired bootstrap)
    rest = acc & ~K
    NK = np.bincount(g[K], weights=y[K], minlength=n_s1); DK = np.bincount(g[K], minlength=n_s1).astype(float)
    NR = np.bincount(g[rest], weights=y[rest], minlength=n_s1); DR = np.bincount(g[rest], minlength=n_s1).astype(float)
    diff = (W @ NK) / np.maximum(W @ DK, 1e-9) - (W @ NR) / np.maximum(W @ DR, 1e-9)
    A["KEY_minus_rest_precision"] = dict(point=round(float(NK.sum() / max(DK.sum(), 1) - NR.sum() / DR.sum()), 4),
                                         ci=[round(float(np.percentile(diff, 2.5)), 4), round(float(np.percentile(diff, 97.5)), 4)])
    # calibration within KEY: observed vs expected FP by p-bin
    cal = {}
    for lo_, hi_ in [(0.78, 0.9), (0.9, 0.99), (0.99, 0.999), (0.999, 1.01)]:
        for nm, mm in [("KEY", K), ("rest", rest)]:
            m = mm & (p >= lo_) & (p < hi_)
            cal[f"{nm}|p[{lo_},{hi_})"] = dict(n=int(m.sum()), obs_prec=round(float(y[m].mean()), 4) if m.any() else None,
                                               mean_p=round(float(p[m].mean()), 4) if m.any() else None)
    A["calibration"] = cal
    RES["accepted"] = A
    for k, v in A.items():
        if k != "calibration":
            log("ACC", k, v)
    log("CAL", cal)

    # ------------------------------------------------------------ rejected p>=0.2: positive density + recall by bucket
    R = {}
    rj = (p >= 0.2) & ~acc & both
    for s_ in ("1", "part", "0", "miss"):
        for h_ in ("eq", "neq", "miss", "any"):
            mb = both & (SB == s_) & ((HB == h_) if h_ != "any" else True)
            m = rj & mb
            if m.sum() == 0:
                continue
            posb = mb & (y == 1)
            R[f"street={s_}|hn={h_}"] = dict(n_rej=int(m.sum()), pos_rej=int(y[m].sum()), pos_rate=round(float(y[m].mean()), 4),
                                             mean_p=round(float(p[m].mean()), 4),
                                             bucket_recall=round(float(acc[posb].mean()), 4) if posb.any() else None, bucket_pos=int(posb.sum()))
    # street match & hn eq & full>=0.8 in rejected band: model possibly under-weights?
    for nm, m in [("rej & street=1 & hn=eq", rj & (SB == "1") & (HB == "eq")), ("rej & street=1 & hn=eq & p>=0.5", rj & (SB == "1") & (HB == "eq") & (p >= 0.5)),
                  ("rej & street=0", rj & (SB == "0")), ("rej all", rj)]:
        pt, lo, hi = boot_rate(g[m], y[m].astype(float), np.ones(m.sum()), W) if m.any() else (None, None, None)
        R[nm] = dict(n=int(m.sum()), pos_rate=pt, ci=[lo, hi], mean_p=float(p[m].mean()) if m.any() else None)
    RES["rejected"] = R
    for k, v in R.items():
        log("REJ", k, v)

    # ------------------------------------------------------------ stratified AUC within p-deciles of the band (incremental)
    strat = {}
    for lo_, hi_ in [(0.2, 0.5), (0.5, 0.78), (0.78, 0.95), (0.95, 0.99), (0.99, 0.999), (0.999, 1.01)]:
        m = both & (p >= lo_) & (p < hi_)
        strat[f"p[{lo_},{hi_})"] = dict(n=int(m.sum()), pos_rate=round(float(y[m].mean()), 4), street_overlap=auc(y[m], so_imp[m]),
                                        street_hn=auc(y[m], combo[m]), full=auc(y[m], np.nan_to_num(full[m], nan=-1)),
                                        addr_tset=auc(y[m], EX["addr_tset(14)"][m]), house_agree=auc(y[m], EX["house_agree(57)"][m]),
                                        p_within=auc(y[m], p[m]))
    RES["stratified_auc"] = strat
    log("STRAT", strat)

    # ------------------------------------------------------------ counterfactual rules on V1 macro F0.5 (threshold only)
    def macro(a):
        npred = np.bincount(s1idx, weights=a, minlength=n_s1); tp = np.bincount(s1idx, weights=a & (y == 1), minlength=n_s1)
        f = np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))
        return f
    f0 = macro(acc)
    CF = {"base_threshold_only": dict(macro=round(float(f0.mean()), 6), US=round(float(f0[cS1 == "US"].mean()), 6), India=round(float(f0[cS1 == "India"].mean()), 6))}
    rules = {"reject KEY": K, "reject KEY & hn!=eq": K & (HB != "eq"), "reject KEY & street_fuzzy<0.8": K & (sf < 0.8),
             "reject KEY & p<0.99": K & (p < 0.99), "reject KEY & diff_core": K & ~ceq,
             "reject accepted street=0 & hn=neq": acc & both & (SB == "0") & (HB == "neq"),
             "reject accepted street=0 & hn=neq & p<0.99": acc & both & (SB == "0") & (HB == "neq") & (p < 0.99)}
    for nm, m in rules.items():
        f1 = macro(acc & ~m); d = f1 - f0
        pt, lo, hi = boot_mean(d, W)
        CF[nm] = dict(n_flipped=int(m.sum()), tp_lost=int(y[m].sum()), fp_removed=int((y[m] == 0).sum()),
                      delta_macro=round(pt, 6), ci=[round(lo, 6), round(hi, 6)])
    add = {"accept rej p>=0.5 & street=1 & hn=eq & full>=0.8": (p >= 0.5) & ~acc & both & (SB == "1") & (HB == "eq") & (full >= 0.8),
           "accept rej p>=0.5 & street=1 & hn=eq": (p >= 0.5) & ~acc & both & (SB == "1") & (HB == "eq")}
    for nm, m in add.items():
        f1 = macro(acc | m); d = f1 - f0
        pt, lo, hi = boot_mean(d, W)
        CF[nm] = dict(n_flipped=int(m.sum()), tp_gained=int(y[m].sum()), fp_added=int((y[m] == 0).sum()),
                      delta_macro=round(pt, 6), ci=[round(lo, 6), round(hi, 6)])
    RES["counterfactual"] = CF
    for k, v in CF.items():
        log("CF", k, v)

    # ------------------------------------------------------------ what existing features say inside KEY (are KEY FPs separable already?)
    KE = {}
    for fn in ["full_overlap", "street_fuzzy", "street_jacc", "hn_eq", "addr_tset(14)", "addr_jacc(15)", "house_agree(57)", "neg_house_absdiff(58)",
               "neg_addr_frac_s1_only(85)", "p_NEW", "rr_logit"]:
        v = feats[fn]; mm = K & ~np.isnan(v)
        KE[fn] = dict(auc_in_KEY=auc(y[mm], v[mm]), mean_pos=float(np.nanmean(v[mm & (y == 1)])) if (mm & (y == 1)).any() else None,
                      mean_neg=float(np.nanmean(v[mm & (y == 0)])) if (mm & (y == 0)).any() else None)
    KE["rr_defined_in_KEY"] = int((K & ~np.isnan(rrc)).sum())
    RES["inside_KEY_existing_features"] = KE
    log("KEYFEATS", KE)

    # examples (V1): KEY negatives and positives
    ex = []
    for lab in (0, 1):
        idx = np.flatnonzero(K & (y == lab))
        for i in RNG.choice(idx, size=min(8, len(idx)), replace=False):
            ex.append(dict(y=int(y[i]), p=round(float(p[i]), 4), country=str(cr[i]), s1_name=sn[i], s1_addr=sa[i], rec_name=rn[i], rec_addr=ra[i],
                           full=round(float(full[i]), 3), hn=int(hn[i]), street_fuzzy=round(float(sf[i]), 3)))
    RES["v1_key_examples"] = ex
    for e in ex:
        log("EX", e)
    # accepted street=0 & hn=neq negatives
    ex2 = []
    idx = np.flatnonzero(acc & both & (SB == "0") & (HB == "neq") & (y == 0))
    for i in RNG.choice(idx, size=min(8, len(idx)), replace=False):
        ex2.append(dict(y=int(y[i]), p=round(float(p[i]), 4), country=str(cr[i]), s1_name=sn[i], s1_addr=sa[i], rec_name=rn[i], rec_addr=ra[i],
                        full=round(float(full[i]), 3), hn=int(hn[i]), street_fuzzy=round(float(sf[i]), 3)))
    RES["v1_street0_hnneq_fp_examples"] = ex2
    json.dump(RES, open(os.path.join(HERE, "C_results_v1.json"), "w"), indent=1, default=str)
    log("saved C_results_v1.json")


if __name__ == "__main__":
    main()
