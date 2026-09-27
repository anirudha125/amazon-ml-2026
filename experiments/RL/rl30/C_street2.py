"""RL-30 Part 4 (investigator C), pass 2: ROBUST street-name signal + label-free rival-S1 test.  READ-ONLY; writes only rl30/C_*.
Pass 1 (C_street.py) showed the street_parts token overlap is dominated by artefacts (US ordinals 11th/11ND, 6th/Sixth; component
re-ordering; French single-token street typos Constantine/Constantile).  Here:
  street_sig(addr): every comma component that looks like a street (has a street-type word or a house number, and is not a bare
    postcode+city) -> (numbers, street tokens); ordinals 76th/76nd/seventy -> 'o76'; unit words removed.
  rs     = max over component pairs of token_set_ratio(street tokens)   (typo-robust street-name similarity, NaN if undefined)
  hn2    = 1 if any house number shared, 0 if both have numbers and none shared, -1 otherwise
  KEY2   = full-address overlap >= 0.8 (pass-1 definition) and rs < 0.6   ("same city/area words, different street name")
  rival  = another S1 of the split (same country) with the SAME name core as this S1 (or as the record) that shares a house number
           with the record and whose street matches the record (rs >= 0.8) -> the record plausibly belongs to that other S1.
Run:  nice -n 10 python C_street2.py v1     |    nice -n 10 python C_street2.py test
"""
import os, sys, re, json, time, pickle, glob
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *
from rapidfuzz import fuzz
from sklearn.metrics import roc_auc_score

T0 = time.time()
def log(*a):
    print(f"[{time.time() - T0:6.0f}s]", *a, flush=True)

ORD_WORDS = {w: str(i + 1) for i, w in enumerate(
    "first second third fourth fifth sixth seventh eighth ninth tenth eleventh twelfth thirteenth fourteenth fifteenth "
    "sixteenth seventeenth eighteenth nineteenth twentieth".split())}
ORD_RE = re.compile(r"^(\d+)(st|nd|rd|th)$")
NUM_RE = re.compile(r"^(\d+)([a-z]{0,2})$")
UNIT2 = UNIT | set("apartment room rm bldg building flat shop office plot block blk door house hno sno trailer level lvl wing gala "
                   "sector ward dept department etage".split())
NB = 1000


def street_sig(addr):
    if not addr or addr.strip().lower() == "null":
        return ()
    out = []
    for comp in fold(addr).split(","):
        ts = re.findall(r"[a-z0-9]+", comp)
        if not ts:
            continue
        nums, st, typ = set(), set(), False
        for t in ts:
            m = ORD_RE.match(t)
            if m:
                st.add("o" + (m.group(1).lstrip("0") or "0")); continue
            if t in ORD_WORDS:
                st.add("o" + ORD_WORDS[t]); continue
            if t[0].isdigit():
                m = NUM_RE.match(t)
                if m:
                    nums.add(m.group(1).lstrip("0") or "0")
                continue
            if t in STREET_TYPE:
                typ = True; continue
            if t in UNIT2 or t in STOP or len(t) <= 1:
                continue
            st.add(t)
        if not st:
            continue
        if not typ and (not nums or all(len(n) >= 5 for n in nums)):
            continue                                        # bare city/region or postcode+city component
        out.append((frozenset(nums), frozenset(st), " ".join(sorted(st))))
    return tuple(out)


def sig_nums(sig):
    s = set()
    for n, _, _ in sig:
        s |= n
    return s


def rmatch(A, B):
    """-> (rs fuzzy, rs exact overlap, hn2)"""
    if not A or not B:
        return np.nan, np.nan, -1
    best = 0.0; bex = 0.0
    for na, sa, ja in A:
        for nb_, sb, jb in B:
            ex = len(sa & sb) / min(len(sa), len(sb))
            if ex > bex:
                bex = ex
            if ex >= 1.0:
                best = 1.0
            elif best < 1.0:
                r = fuzz.token_set_ratio(ja, jb) / 100.0
                if r > best:
                    best = r
    nA = sig_nums(A); nB = sig_nums(B)
    hn = -1 if (not nA or not nB) else (1 if nA & nB else 0)
    return best, bex, hn


def robust_pairs(sa, ra):
    u, inv = np.unique(np.concatenate([sa, ra]), return_inverse=True)
    S = [street_sig(a) for a in u]
    n = len(sa); i1 = inv[:n]; i2 = inv[n:]
    rs = np.full(n, np.nan, np.float32); rx = np.full(n, np.nan, np.float32); hn = np.full(n, -1, np.int8)
    for k in range(n):
        a, b, c = rmatch(S[i1[k]], S[i2[k]])
        rs[k] = a; rx[k] = b; hn[k] = c
    return rs, rx, hn


class RivalIndex:
    """(country, name core, house number) -> S1 rows, over ALL S1 of the split (label-free corpus statistic)."""
    def __init__(self, s1df):
        t = time.time()
        self.ids = s1df.id.values; self.pos = {s: i for i, s in enumerate(self.ids)}
        self.core = [core_tokens(x) for x in s1df.name.values]
        ua, ia = np.unique(s1df.addr.values.astype(object), return_inverse=True)
        us = [street_sig(a) for a in ua]
        self.sig = [us[j] for j in ia]; self.ctry = s1df.country.values
        self.idx = {}
        for i in range(len(self.ids)):
            if not self.core[i]:
                continue
            for num in sig_nums(self.sig[i]):
                self.idx.setdefault((self.ctry[i], self.core[i], num), []).append(i)
        log(f"RivalIndex: {len(self.ids):,} S1, {len(self.idx):,} keys, {time.time() - t:.0f}s")

    def rival(self, s1_id, rec_core, rec_sig):
        """-> (n_rival_candidates_checked, best rival street similarity to the record, n rivals with rs>=0.8)"""
        i = self.pos.get(s1_id)
        if i is None or not rec_sig:
            return 0, np.nan, 0
        c = self.ctry[i]; cands = set()
        for num in sig_nums(rec_sig):
            for core in {self.core[i], rec_core}:
                if core:
                    cands.update(self.idx.get((c, core, num), ()))
        cands.discard(i)
        best = np.nan; nr = 0
        for j in cands:
            r = rmatch(self.sig[j], rec_sig)[0]
            if not np.isnan(r):
                best = r if np.isnan(best) else max(best, r)
                if r >= 0.8:
                    nr += 1
        return len(cands), best, nr


def auc(y, x):
    if len(y) < 20 or y.min() == y.max():
        return None
    return round(float(roc_auc_score(y, x)), 4)


def boot_rate(g, num, W):
    n_s1 = W.shape[1]
    N = np.bincount(g, weights=num, minlength=n_s1); Dn = np.bincount(g, minlength=n_s1).astype(float)
    b = (W @ N) / np.maximum(W @ Dn, 1e-9)
    return round(float(N.sum() / max(Dn.sum(), 1)), 4), [round(float(np.percentile(b, 2.5)), 4), round(float(np.percentile(b, 97.5)), 4)]


def rsb(rs):
    return np.where(np.isnan(rs), "undef", np.where(rs >= 0.8, "match", np.where(rs >= 0.6, "mid", "diff")))


# =========================================================================================== V1
def run_v1():
    RES = {}
    meta = np.load(PATHS["v1_meta"], allow_pickle=True)
    s1_ids, s1idx, cand, y, n_gt, ctry = (meta[k] for k in ("s1_ids", "s1idx", "cand", "y", "n_gt", "country"))
    y = y.astype(np.int8); n_s1 = len(s1_ids); p = np.load(PATHS["v1_p_new"]).astype(np.float64)
    P1 = np.load(os.path.join(HERE, "C_v1_pairfeats.npz"))
    full, so, ceq, rrc = P1["full"], P1["so"], P1["ceq"], P1["rr"]
    LF = np.load(PATHS["v1_LF"], mmap_mode="r"); tset = np.asarray(LF[:, 14]); hag = np.asarray(LF[:, 57])
    RL = np.load(PATHS["v1_rl27"])
    D = load("train", verbose=False)
    s1 = D["s1"].set_index("id"); rec = pd.concat([D["s2"], D["s3"]]).set_index("id")
    sid = s1_ids[s1idx]
    sa = s1.addr.reindex(sid).fillna("").values.astype(object); ra = rec.addr.reindex(cand).fillna("").values.astype(object)
    sn = s1.name.reindex(sid).fillna("").values.astype(object); rn = rec.name.reindex(cand).fillna("").values.astype(object)
    rs, rx, hn = robust_pairs(sa, ra)
    log("robust street done; defined", int((~np.isnan(rs)).sum()))
    both = ~np.isnan(full); acc = p >= NEW_TH; cr = ctry[s1idx]
    W = np.random.default_rng(0).poisson(1.0, size=(NB, n_s1)).astype(np.float64)
    RB = rsb(rs); HB = np.where(hn == 1, "eq", np.where(hn == 0, "neq", "miss"))
    RES["defined"] = dict(both=int(both.sum()), rs_defined=int((both & ~np.isnan(rs)).sum()), pos_rs_defined=int(y[both & ~np.isnan(rs)].sum()),
                          pos_both=int(y[both].sum()), rs_defined_US=float(np.mean(~np.isnan(rs[both & (cr == 'US')]))),
                          rs_defined_India=float(np.mean(~np.isnan(rs[both & (cr == 'India')]))))
    # AUC robust vs token vs existing
    rs_imp = np.where(np.isnan(rs), -0.1, rs); so_imp = np.where(np.isnan(so), -0.1, so)
    hn_imp = np.where(hn == 1, 1.0, np.where(hn == 0, 0.0, 0.5))
    band = (p >= 0.2) & (p < 0.95)
    subs = {"all_both": both, "same_core": both & ceq, "hn2_eq": both & (hn == 1), "full>=0.8": both & (full >= 0.8),
            "same_core&full>=0.8&hn2_eq": both & ceq & (full >= 0.8) & (hn == 1), "band": both & band, "accepted": both & acc,
            "rejected_p>=0.2": both & (p >= 0.2) & ~acc}
    A = {}
    for k, m in subs.items():
        A[k] = dict(n=int(m.sum()), pos_rate=round(float(y[m].mean()), 4), rs_robust=auc(y[m], rs_imp[m]), so_token=auc(y[m], so_imp[m]),
                    rs_plus_hn2=auc(y[m], rs_imp[m] + hn_imp[m]), full=auc(y[m], np.nan_to_num(full[m], nan=-1)), addr_tset14=auc(y[m], tset[m]),
                    house_agree57=auc(y[m], hag[m]), p_NEW=auc(y[m], p[m]))
        log("AUC", k, A[k])
    RES["auc"] = A
    # all-pairs table: full>=0.8 x robust street x hn2 (pos rate, accepted share, precision)
    tab = {}
    for f_ in ("hi", "lo"):
        fm = both & ((full >= 0.8) if f_ == "hi" else (full < 0.8))
        for r_ in ("match", "mid", "diff", "undef"):
            for h_ in ("eq", "neq", "miss"):
                m = fm & (RB == r_) & (HB == h_)
                if m.sum() == 0:
                    continue
                ma = m & acc
                tab[f"full_{f_}|street={r_}|hn2={h_}"] = dict(n=int(m.sum()), pos=int(y[m].sum()), pos_rate=round(float(y[m].mean()), 4),
                                                             n_acc=int(ma.sum()), acc_prec=round(float(y[ma].mean()), 4) if ma.any() else None,
                                                             acc_mean_p=round(float(p[ma].mean()), 4) if ma.any() else None,
                                                             recall=round(float(acc[m & (y == 1)].mean()), 4) if (m & (y == 1)).any() else None)
    RES["table_all_pairs"] = tab
    for k, v in tab.items():
        log("TAB", k, v)
    # KEY2 accepted precision with CI
    K2 = both & acc & (full >= 0.8) & (RB == "diff")
    out = {}
    for nm, m in [("accepted_all", acc), ("accepted_street_match", both & acc & (RB == "match")), ("KEY2", K2),
                  ("KEY2|hn2=eq", K2 & (hn == 1)), ("KEY2|hn2=neq", K2 & (hn == 0)), ("KEY2|same_core", K2 & ceq),
                  ("KEY2|same_core&hn2=eq", K2 & ceq & (hn == 1)), ("accepted street=diff (any full)", both & acc & (RB == "diff")),
                  ("accepted street=diff & hn2=neq", both & acc & (RB == "diff") & (hn == 0)),
                  ("accepted street=diff & hn2=eq", both & acc & (RB == "diff") & (hn == 1)),
                  ("KEY2|US", K2 & (cr == "US")), ("KEY2|India", K2 & (cr == "India"))]:
        if m.sum() == 0:
            out[nm] = dict(n=0); continue
        pr, ci = boot_rate(s1idx[m], y[m].astype(float), W)
        out[nm] = dict(n=int(m.sum()), fp=int((y[m] == 0).sum()), precision=pr, ci=ci, mean_p=round(float(p[m].mean()), 4))
        log("PREC", nm, out[nm])
    RES["accepted_precision"] = out
    # rival test on accepted + band pairs
    RI = RivalIndex(D["s1"])
    sub = np.flatnonzero(both & (p >= 0.2))
    rcore = {}
    uS = {}
    rv_best = np.full(len(y), np.nan, np.float32); rv_n = np.zeros(len(y), np.int16); rv_c = np.zeros(len(y), np.int32)
    for i in sub:
        a = ra[i]
        sg = uS.get(a)
        if sg is None:
            sg = uS[a] = street_sig(a)
        c = rcore.get(rn[i])
        if c is None:
            c = rcore[rn[i]] = core_tokens(rn[i])
        nc, b, nr = RI.rival(sid[i], c, sg)
        rv_c[i] = nc; rv_best[i] = b; rv_n[i] = nr
    rival = (rv_n > 0) & ((np.nan_to_num(rv_best, nan=0) > np.nan_to_num(rs, nan=0)))
    log("rival computed on", len(sub), "pairs; rival flags", int(rival[sub].sum()))
    rv = {}
    for nm, m in [("accepted", acc & both), ("accepted&rival", acc & both & rival), ("accepted&~rival", acc & both & ~rival),
                  ("KEY2&rival", K2 & rival), ("KEY2&~rival", K2 & ~rival), ("band&rival", both & band & rival), ("band&~rival", both & band & ~rival),
                  ("rej_p>=0.2&rival", both & (p >= 0.2) & ~acc & rival)]:
        if m.sum() == 0:
            rv[nm] = dict(n=0); continue
        pr, ci = boot_rate(s1idx[m], y[m].astype(float), W)
        rv[nm] = dict(n=int(m.sum()), fp=int((y[m] == 0).sum()), pos_rate=pr, ci=ci, mean_p=round(float(p[m].mean()), 4))
        log("RIVAL", nm, rv[nm])
    RES["rival"] = rv
    # RL-27 features inside KEY2 vs accepted
    cols = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
    RES["rl27_means"] = {nm: {c: round(float(np.nanmean(RL[m, j])), 3) for j, c in enumerate(cols)} for nm, m in
                         [("accepted", acc & both), ("KEY2", K2), ("accepted&rival", acc & both & rival)] if m.any()}
    # examples
    rng = np.random.default_rng(5); ex = []
    for nm, m in [("KEY2_fp", K2 & (y == 0)), ("KEY2_tp", K2 & (y == 1)), ("acc_rival_fp", acc & both & rival & (y == 0)), ("acc_rival_tp", acc & both & rival & (y == 1))]:
        idx = np.flatnonzero(m)
        for i in rng.choice(idx, size=min(6, len(idx)), replace=False):
            ex.append(dict(kind=nm, y=int(y[i]), p=round(float(p[i]), 4), s1_name=sn[i], s1_addr=sa[i], rec_name=rn[i], rec_addr=ra[i],
                           full=round(float(full[i]), 3), rs=round(float(rs[i]), 3), hn2=int(hn[i]), rival_best=round(float(rv_best[i]), 3)))
    RES["examples"] = ex
    for e in ex:
        log("EX", e)
    np.savez_compressed(os.path.join(HERE, "C_v1_robust.npz"), rs=rs, rx=rx, hn2=hn, rival=rival, rv_best=rv_best, rv_n=rv_n)
    json.dump(RES, open(os.path.join(HERE, "C_results_v1_robust.json"), "w"), indent=1, default=str)
    log("saved C_results_v1_robust.json")


# =========================================================================================== TEST
def run_test():
    RES = {}
    TD = load("test", verbose=False)
    s1df = TD["s1"]; s1 = s1df.set_index("id"); rec = pd.concat([TD["s2"], TD["s3"]]).set_index("id")
    RI = RivalIndex(s1df)
    v1 = json.load(open(os.path.join(HERE, "C_results_v1_robust.json")))
    for c in ("France", "US", "India"):
        A = accepted(f"S005_{c}")
        sid = A.s1.values; cid = A.rec.values; p = A.p.values; kept = A.kept_final.values.astype(bool); ncl = A.n_claims.values
        sa = s1.addr.reindex(sid).fillna("").values.astype(object); ra = rec.addr.reindex(cid).fillna("").values.astype(object)
        sn = s1.name.reindex(sid).fillna("").values.astype(object); rn = rec.name.reindex(cid).fillna("").values.astype(object)
        P1 = np.load(os.path.join(HERE, f"C_test_{c}_pairfeats.npz")); full = P1["full"]; ceq = P1["ceq"]
        rs, rx, hn = robust_pairs(sa, ra)
        both = ~np.isnan(full); RB = rsb(rs)
        K2 = both & (full >= 0.8) & (RB == "diff")
        # rival: France all accepted; US/India KEY2 + 150k random sample
        n = len(p)
        if c == "France":
            sub = np.flatnonzero(both)
        else:
            rng = np.random.default_rng(7)
            sub = np.union1d(np.flatnonzero(K2), rng.choice(np.flatnonzero(both), size=min(150000, int(both.sum())), replace=False))
        samp = np.zeros(n, bool); samp[sub] = True
        rv_best = np.full(n, np.nan, np.float32); rv_n = np.zeros(n, np.int16)
        uS = {}; rcore = {}
        for i in sub:
            a = ra[i]; sg = uS.get(a)
            if sg is None:
                sg = uS[a] = street_sig(a)
            cc = rcore.get(rn[i])
            if cc is None:
                cc = rcore[rn[i]] = core_tokens(rn[i])
            _, b, nr = RI.rival(sid[i], cc, sg)
            rv_best[i] = b; rv_n[i] = nr
        rival = (rv_n > 0) & (np.nan_to_num(rv_best, nan=0) > np.nan_to_num(rs, nan=0))
        # sibling: S1 has another accepted record whose street matches its own
        g = pd.DataFrame({"s1": sid, "m": both & (RB == "match")})
        sib = g.groupby("s1").m.transform("sum").values > 0
        R = dict(n_accepted=int(n), n_kept=int(kept.sum()), rs_defined_share=round(float(np.mean(~np.isnan(rs[both]))), 4),
                 baseline_multi_claim=round(float((ncl > 1).mean()), 4))
        def blk(m, sample_only=False):
            mm = m & samp if sample_only else m
            d = dict(n=int(m.sum()), share_accepted=round(float(m.sum() / n), 5), n_kept=int((m & kept).sum()),
                     share_kept=round(float((m & kept).sum() / kept.sum()), 5), n_s1=int(len(np.unique(sid[m]))),
                     n_s1_kept=int(len(np.unique(sid[m & kept]))),
                     mean_p=round(float(p[m].mean()), 4) if m.any() else None, multi_claim=round(float((ncl[m] > 1).mean()), 4) if m.any() else None)
            if mm.any():
                d["rival_share"] = round(float(rival[mm].mean()), 4); d["rival_share_kept"] = round(float(rival[mm & kept].mean()), 4) if (mm & kept).any() else None
            return d
        for nm, m in [("accepted_both", both), ("street_match", both & (RB == "match")), ("street_mid", both & (RB == "mid")),
                      ("street_diff", both & (RB == "diff")), ("street_undef", both & (RB == "undef")),
                      ("KEY2", K2), ("KEY2&hn2=eq", K2 & (hn == 1)), ("KEY2&hn2=neq", K2 & (hn == 0)), ("KEY2&hn2=miss", K2 & (hn == -1)),
                      ("KEY2&same_core", K2 & ceq), ("KEY2&same_core&hn2=eq", K2 & ceq & (hn == 1)), ("KEY2&sibling", K2 & sib),
                      ("KEY2&p<0.99", K2 & (p < 0.99)), ("street_diff&hn2=neq", both & (RB == "diff") & (hn == 0))]:
            R[nm] = blk(m, sample_only=(c != "France"))
        if c == "France":
            for nm, m in [("KEY2&rival", K2 & rival), ("KEY2&rival&kept", K2 & rival & kept), ("KEY2&~rival&kept", K2 & ~rival & kept),
                          ("accepted&rival", both & rival), ("accepted&rival&kept", both & rival & kept), ("street_match&rival", both & (RB == "match") & rival)]:
                R[nm] = blk(m)
        # ESTIMATED FP among kept KEY2 using V1 precision (transfer assumption)
        pr = v1["accepted_precision"]
        est = sum((1 - pr[f"KEY2|hn2={h}"]["precision"]) * int((K2 & kept & (hn == hv)).sum())
                  for h, hv in (("eq", 1), ("neq", 0)) if pr.get(f"KEY2|hn2={h}", {}).get("precision") is not None)
        R["ESTIMATED_fp_kept_KEY2_via_V1_precision"] = round(float(est), 1)
        # France: join RL-27 + LF existing features for accepted pairs
        if c == "France":
            key = pd.Series(np.arange(n), index=pd.MultiIndex.from_arrays([sid, cid]))
            EXF = np.full((n, 14), np.nan, np.float32)
            for f in sorted(glob.glob(PATHS["test_chunks"].format(country=c))):
                z = np.load(f, allow_pickle=True)
                j = key.reindex(pd.MultiIndex.from_arrays([z["s1"], z["cand"]])).values
                ok = ~np.isnan(j)
                if not ok.any():
                    continue
                jj = j[ok].astype(np.int64)
                LFc = z["LF"]
                EXF[jj, 0] = LFc[ok, 14]; EXF[jj, 1] = LFc[ok, 57]; EXF[jj, 2] = LFc[ok, 15]; EXF[jj, 3] = LFc[ok, 85]
                rl = np.load(f.replace(os.path.join("P3", c), os.path.join("RL", "test_feats", c)).replace("chunk_", "rl27_chunk_").replace(".npz", ".npy"))
                EXF[jj, 4:14] = rl[ok]
            names = ["addr_tset14", "house_agree57", "addr_jacc15", "addr_frac_s1_only85", "nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf",
                     "rv_rank", "rv_sa", "rv_so", "rv_gap"]
            R["joined_existing_features"] = int((~np.isnan(EXF[:, 0])).sum())
            R["existing_feature_means"] = {nm: {k: round(float(np.nanmean(EXF[m, j])), 4) for j, k in enumerate(names)}
                                           for nm, m in [("accepted_kept", both & kept), ("street_match_kept", both & (RB == "match") & kept),
                                                         ("KEY2_kept", K2 & kept), ("KEY2_rival_kept", K2 & rival & kept),
                                                         ("KEY2_hn2eq_samecore_kept", K2 & (hn == 1) & ceq & kept)] if m.any()}
            np.savez_compressed(os.path.join(HERE, "C_test_France_robust.npz"), rs=rs, hn2=hn, rival=rival, rv_best=rv_best, EXF=EXF)
        # examples
        rng = np.random.default_rng(11); exs = {}
        for nm, m in [("KEY2&kept&rival", K2 & kept & rival), ("KEY2&kept&~rival&hn2=eq", K2 & kept & ~rival & (hn == 1) & samp),
                      ("KEY2&kept&hn2=neq", K2 & kept & (hn == 0))]:
            idx = np.flatnonzero(m)
            if len(idx):
                exs[nm] = [dict(s1=sid[i], s1_name=sn[i], s1_addr=sa[i], rec=cid[i], rec_name=rn[i], rec_addr=ra[i], p=round(float(p[i]), 4),
                                n_claims=int(ncl[i]), full=round(float(full[i]), 3), rs=round(float(rs[i]), 3), hn2=int(hn[i]),
                                rival_best=round(float(rv_best[i]), 3)) for i in rng.choice(idx, size=min(10, len(idx)), replace=False)]
        # for rival examples, show the rival S1
        if "KEY2&kept&rival" in exs:
            for e in exs["KEY2&kept&rival"]:
                i = RI.pos[e["s1"]]; sg = street_sig(e["rec_addr"]); cc = core_tokens(e["rec_name"]); cands = set()
                for num in sig_nums(sg):
                    for core in {RI.core[i], cc}:
                        if core:
                            cands.update(RI.idx.get((RI.ctry[i], core, num), ()))
                cands.discard(i)
                best = max(cands, key=lambda j: np.nan_to_num(rmatch(RI.sig[j], sg)[0], nan=-1)) if cands else None
                if best is not None:
                    e["rival_s1"] = RI.ids[best]; e["rival_name"] = s1.name[RI.ids[best]]; e["rival_addr"] = s1.addr[RI.ids[best]]
        R["examples"] = exs
        RES[c] = R
        log(c, {k: v for k, v in R.items() if k not in ("examples",)})
        for nm, L in exs.items():
            for e in L[:8]:
                log("EX", c, nm, e)
    json.dump(RES, open(os.path.join(HERE, "C_results_test_robust.json"), "w"), indent=1, default=str)
    log("saved C_results_test_robust.json")


if __name__ == "__main__":
    {"v1": run_v1, "test": run_test}[sys.argv[1]]()
