"""RL-30 investigator B -- Part 3 audit, step 3 (LABEL-FREE, test corpus): how often each proposed signal fires on the S005 accepted
pairs per country, how the France co-located token contrast compares with the production IDF weighting, and what the RL-27 test
features say on the France pairs where the co-located-contrast signal fires. READ-ONLY; writes rl30/B_france.json only.
Inputs: load('test'), accepted('S005_<country>') (pairs before max-claimer, with kept_final), experiments/P3/France chunk s1/cand/LF,
experiments/RL/test_feats/France rl27 chunks, experiments/_shared/corpus_stats_test.pkl (production IDF for test features).
"""
import os, sys, time, json, math, glob, pickle, collections
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl30_lib import *  # noqa
from recon05_baseline_scorer import normalize as norm05
from rapidfuzz import process, fuzz
from scipy.stats import spearmanr

t0 = time.time()
log = lambda *a: print(f"[{time.time()-t0:6.0f}s]", *a, flush=True)
RES = {}
FR_EXTRA_SUFFIX = sorted(LEGAL - set("llc inc co corp corporation company ltd limited private pvt pc plc lp llp".split()))


def main():
    T = load("test", verbose=False)
    s1 = T["s1"]; rec = pd.concat([T["s2"], T["s3"]], ignore_index=True)
    rec_ix = pd.Index(rec.id.values); s1_ix = pd.Index(s1.id.values)
    stats = pickle.load(open(os.path.join(ROOT, "experiments", "_shared", "corpus_stats_test.pkl"), "rb"))
    tdf, TN = stats["token_df"], stats["N"]
    pidf = lambda w: math.log((TN - tdf.get(w, 0) + 0.5) / (tdf.get(w, 0) + 0.5) + 1.0)
    log(f"test loaded; production IDF N={TN:,}")
    # ------------- per-country S1 address groups (test corpus)
    all_ak = np.array([akey(x) for x in s1.addr.values], dtype=object)
    keys = pd.Series(s1.country.values, dtype=object) + "|" + pd.Series(all_ak, dtype=object)
    codes, uniq = pd.factorize(keys); codes = codes.astype(np.int64)
    empty = np.array([u.endswith("|") for u in uniq])
    size = np.bincount(codes); order = np.argsort(codes, kind="stable"); starts = np.r_[0, np.cumsum(size)]
    members = lambda c: order[starts[c]:starts[c + 1]]
    s1_names = s1.name.values; s1_addrs = s1.addr.values
    log("address groups built")
    dens = {}
    for c in ("France", "US", "India"):
        ic = np.flatnonzero(s1.country.values == c)
        grp_ok = (~empty[codes[ic]]) & (size[codes[ic]] >= 2)
        dens[c] = dict(n_s1=int(len(ic)), share_s1_colocated=round(float(grp_ok.mean()), 4))
    RES["colocation_density"] = dens
    # ================= token role on France S1 (label-free) vs production IDF
    ic = np.flatnonzero(s1.country.values == "France")
    tl = {int(j): frozenset(toks(s1_names[j])) for j in ic}
    variants = collections.defaultdict(collections.Counter)
    for j in ic:
        for w in norm05(s1_names[j]).split():
            variants[fold(w)][w] += 1
    dff = collections.Counter(); [dff.update(tl[j]) for j in ic]
    cdf = collections.Counter(); con = collections.Counter(); con_pairs = collections.Counter(); npairs = nnd = 0
    gcodes = np.unique(codes[ic]); gcodes = gcodes[(size[gcodes] >= 2) & ~empty[gcodes]]
    nd_examples = []
    for g in gcodes:
        mem = members(g)[:30]; sets = [tl[int(j)] for j in mem]
        for s_ in sets:
            cdf.update(s_)
        for a in range(len(sets)):
            for b in range(a + 1, len(sets)):
                A, B = sets[a], sets[b]; npairs += 1; d1, d2 = A - B, B - A
                if A & B and len(d1) <= 1 and len(d2) <= 1 and (d1 or d2):
                    nnd += 1; con.update(d1 | d2)
                    con_pairs[tuple(sorted(d1 | d2))] += 1
                    if len(nd_examples) < 12 and nnd % 37 == 1:
                        nd_examples.append((s1_names[mem[a]], s1_names[mem[b]], s1_addrs[mem[a]]))
    def prod_idf_of_fold(t):
        v = variants.get(t)
        return pidf(v.most_common(1)[0][0]) if v else pidf(t)
    # IDF percentile among France S1 token occurrences (weighted by df)
    occ_idf = np.array([prod_idf_of_fold(t) for t in dff for _ in range(1)]); occ_w = np.array([dff[t] for t in dff], float)
    o = np.argsort(occ_idf); cw = np.cumsum(occ_w[o]) / occ_w.sum(); idf_sorted = occ_idf[o]
    pct = lambda v: round(float(cw[min(np.searchsorted(idf_sorted, v, side="right"), len(cw)) - 1]) if np.searchsorted(idf_sorted, v, side="right") > 0 else 0.0, 3)
    crate = {t: con[t] / (cdf[t] + 10.0) for t in cdf}
    # filler proxy: add/drop rate of token between France S1 and its high-confidence accepted records (p >= 0.97, kept_final)
    A = accepted("S005_France"); A = A[A.kept_final.astype(bool)]
    hi = A[A.p >= 0.97]
    ai = s1_ix.get_indexer(hi.s1.values); ri = rec_ix.get_indexer(hi.rec.values)
    uni = collections.Counter(); sym = collections.Counter()
    for x, z in zip(ai, ri):
        S, Rr = frozenset(toks(s1_names[x])), frozenset(toks(rec.name.values[z])); uni.update(S | Rr); sym.update(S ^ Rr)
    fill = lambda t: round(sym[t] / uni[t], 3) if uni[t] else None
    Tk = [t for t in cdf if cdf[t] >= 20]
    it = np.array([prod_idf_of_fold(t) for t in Tk]); ct_ = np.array([crate[t] for t in Tk]); ft = np.array([fill(t) if uni[t] >= 20 else np.nan for t in Tk], float)
    okf = ~np.isnan(ft)
    RES["S2_token_role_France"] = dict(
        n_france_s1=int(len(ic)), colocated_pairs=npairs, near_dup_colocated_pairs=nnd, n_tokens_cdf_ge20=len(Tk),
        spearman_prodidf_vs_contrast_rate=round(float(spearmanr(it, ct_).correlation), 4),
        spearman_prodidf_vs_fill_rate=round(float(spearmanr(it[okf], ft[okf]).correlation), 4),
        spearman_contrast_vs_fill_rate=round(float(spearmanr(ct_[okf], ft[okf]).correlation), 4),
        top_contrast_tokens=[dict(tok=t, contrast_pairs=con[t], colocated_s1_with_tok=cdf[t], france_s1_df=dff[t], contrast_rate=round(crate[t], 4),
                                  prod_idf=round(prod_idf_of_fold(t), 2), prod_idf_pctile_france_occ=pct(prod_idf_of_fold(t)), fill_rate_accepted=fill(t))
                             for t, _ in con.most_common(25)],
        top_contrast_token_pairs=[(list(k), v) for k, v in con_pairs.most_common(15)],
        near_dup_examples=nd_examples,
        legal_forms_not_in_rl27_SUFFIX=FR_EXTRA_SUFFIX,
        share_france_s1_with_legal_form_not_in_SUFFIX=round(float(np.mean([bool(tl[int(j)] & set(FR_EXTRA_SUFFIX)) for j in ic])), 4),
    )
    log(f"S2 France: {npairs:,} co-located pairs, {nnd:,} near-dup")
    # ================= per-country accepted-pair densities (label-free)
    acc_stats = {}
    fr_pairs = None
    for c in ("France", "US", "India"):
        A_all = accepted(f"S005_{c}")
        kf_all = A_all.kept_final.astype(bool).values if "kept_final" in A_all.columns else np.ones(len(A_all), bool)
        claimed = set(zip(A_all.s1.values, A_all.rec.values)); n_acc_all = len(A_all)
        A = A_all[kf_all].reset_index(drop=True); kf = np.ones(len(A), bool); del A_all
        ai = s1_ix.get_indexer(A.s1.values); ri = rec_ix.get_indexer(A.rec.values)
        assert (ai >= 0).all() and (ri >= 0).all()
        an = [" ".join(toks(s1_names[x])) for x in ai]; rn_ = [" ".join(toks(rec.name.values[z])) for z in ri]
        sim_a = process.cpdist(an, rn_, scorer=fuzz.token_set_ratio, workers=4, dtype=np.float32)
        # co-located contrast (S1-address group, other members)
        pi, bj = [], []
        for k, x in enumerate(ai):
            g = codes[x]
            if empty[g] or size[g] < 2:
                continue
            mem = members(g)[:31]; mem = mem[mem != x]
            pi.extend([k] * len(mem)); bj.extend(mem.tolist())
        pi = np.asarray(pi, np.int64); bj = np.asarray(bj, np.int64)
        sims = process.cpdist([" ".join(toks(s1_names[j])) for j in bj], [rn_[k] for k in pi], scorer=fuzz.token_set_ratio, workers=4, dtype=np.float32)
        best = np.full(len(A), np.nan, np.float32); np.fmax.at(best, pi, sims)
        has = ~np.isnan(best); cc = has & (best > sim_a)
        # does a closer co-located S1 also claim the record (accepted pair)?
        closer = sims > sim_a[pi]
        b_claims = np.zeros(len(A), bool)
        for k, j in zip(pi[closer], bj[closer]):
            if (s1.id.values[j], A.rec.values[k]) in claimed:
                b_claims[k] = True
        # street / house number / full-address overlap
        a_sp = [street_parts(s1_addrs[x]) for x in ai]; r_sp = [street_parts(rec.addr.values[z]) for z in ri]
        a_aw = [addr_words(s1_addrs[x]) for x in ai]; r_aw = [addr_words(rec.addr.values[z]) for z in ri]
        st_both = np.array([bool(u[1]) and bool(v[1]) for u, v in zip(a_sp, r_sp)])
        st_eq = np.array([u[1] == v[1] for u, v in zip(a_sp, r_sp)]) & st_both
        awj = np.array([len(u & v) / len(u | v) if (u and v) else np.nan for u, v in zip(a_aw, r_aw)])
        hn_both = np.array([u[0] is not None and v[0] is not None for u, v in zip(a_sp, r_sp)])
        hn_eq = np.array([u[0] == v[0] for u, v in zip(a_sp, r_sp)]) & hn_both
        # name near-dup relation and legal-form-only differences
        aset = [frozenset(x.split()) for x in an]; rset = [frozenset(x.split()) for x in rn_]
        sub1 = np.array([len(u - v) == 1 and len(v - u) == 1 and bool(u & v) for u, v in zip(aset, rset)])
        legal_only = np.array([(u != v) and (core_tokens(" ".join(u)) == core_tokens(" ".join(v))) for u, v in zip(aset, rset)])
        top_con = {d["tok"] for d in RES["S2_token_role_France"]["top_contrast_tokens"]}
        diff_is_contrast_tok = np.array([bool((u ^ v) & top_con) and len(u ^ v) <= 2 and bool(u & v) for u, v in zip(aset, rset)])
        n_s1 = A.s1.nunique()
        def rates(msk):
            nn = int(msk.sum())
            return dict(n_pairs=nn, per_1k_s1=round(1000 * nn / n_s1, 2))
        d = dict(n_accepted=int(n_acc_all), n_kept_final=int(kf.sum()), n_s1_with_kept=int(n_s1))
        for tag, msk in (("kept_final", kf),):
            d[tag] = dict(
                has_colocated_S1=rates(msk & has), cc_closer_colocated=rates(msk & cc), cc_closer_and_closer_claims=rates(msk & cc & b_claims),
                cc_closer_unclaimed=rates(msk & cc & ~b_claims),
                share_street_both=round(float(st_both[msk].mean()), 4), street_mismatch_given_both=round(float((~st_eq[msk & st_both]).mean()), 4),
                street_mismatch_with_addrwords_j_ge_0_8=rates(msk & st_both & ~st_eq & (np.nan_to_num(awj) >= 0.8)),
                house_mismatch_given_both=round(float((~hn_eq[msk & hn_both]).mean()), 4),
                house_mismatch_same_street=rates(msk & hn_both & ~hn_eq & st_eq),
                name_one_token_substitution=rates(msk & sub1), name_legal_form_only_difference=rates(msk & legal_only),
                name_diff_is_france_top_contrast_token=rates(msk & diff_is_contrast_tok))
        acc_stats[c] = d
        if c == "France":
            fr_pairs = pd.DataFrame(dict(s1=A.s1.values, rec=A.rec.values, p=A.p.values, n_claims=A.n_claims.values, kf=kf, has=has, cc=cc,
                                         b_claims=b_claims, sim_a=sim_a, best=best, st_both=st_both, st_eq=st_eq, awj=awj, hn_eq=hn_eq, hn_both=hn_both,
                                         legal_only=legal_only, sub1=sub1, ctok=diff_is_contrast_tok))
            fr_ex = []
            for k in np.flatnonzero(kf & cc & ~b_claims)[:2000:200]:
                g = codes[ai[k]]; mem = [j for j in members(g)[:31] if j != ai[k]]
                fr_ex.append(dict(s1_name=s1_names[ai[k]], s1_addr=s1_addrs[ai[k]], rec_name=rec.name.values[ri[k]], rec_addr=rec.addr.values[ri[k]],
                                  p=round(float(A.p.values[k]), 3), colocated_names=[s1_names[j] for j in mem[:4]]))
            RES["examples_France_kept_cc_closer_unclaimed"] = fr_ex
            ex2 = []
            for k in np.flatnonzero(kf & diff_is_contrast_tok)[:3000:300]:
                ex2.append(dict(s1_name=s1_names[ai[k]], rec_name=rec.name.values[ri[k]], s1_addr=s1_addrs[ai[k]], rec_addr=rec.addr.values[ri[k]], p=round(float(A.p.values[k]), 3)))
            RES["examples_France_kept_name_diff_is_contrast_token"] = ex2
        log(f"accepted {c}: {len(A):,} pairs")
    RES["accepted_pair_signal_density"] = acc_stats
    # ================= RL-27 test features on the France accepted pairs (rv_rank / coloc / nf_a) and LF addr_token_set
    key = pd.MultiIndex.from_arrays([fr_pairs.s1.values, fr_pairs.rec.values])
    rv = np.full((len(fr_pairs), 10), np.nan, np.float32); lf = np.full((len(fr_pairs), 4), np.nan, np.float32)
    chunks = sorted(glob.glob(os.path.join(ROOT, "experiments", "P3", "France", "chunk_*.npz")))
    found = 0
    for f in chunks:
        z = np.load(f); cs1, cc_ = z["s1"], z["cand"]
        pos = key.get_indexer(pd.MultiIndex.from_arrays([cs1, cc_]))
        sel = np.flatnonzero(pos >= 0)
        if len(sel):
            r27 = np.load(os.path.join(ROOT, "experiments", "RL", "test_feats", "France", "rl27_chunk_" + os.path.basename(f)[6:10] + ".npy"), mmap_mode="r")
            rv[pos[sel]] = r27[sel]; L = z["LF"]; lf[pos[sel]] = L[sel][:, [3, 14, 57, 47]]; found += len(sel)
    log(f"France chunk join: {found:,}/{len(fr_pairs):,} accepted pairs located")
    kf = fr_pairs.kf.values; cc = fr_pairs.cc.values; bcl = fr_pairs.b_claims.values
    def dist(msk, col):
        v = rv[msk, col]; v = v[~np.isnan(v)]
        return dict(n=int(msk.sum()), mean=round(float(v.mean()), 4) if len(v) else None, share_eq1=round(float((v == 1).mean()), 4) if len(v) else None,
                    share_gt0=round(float((v > 0).mean()), 4) if len(v) else None)
    RES["France_rl27_on_kept_final"] = dict(
        located=found,
        rv_rank_cc1_unclaimed=dist(kf & cc & ~bcl, 6), rv_rank_cc1_claimed=dist(kf & cc & bcl, 6), rv_rank_cc0_has=dist(kf & ~cc & fr_pairs.has.values, 6),
        rv_rank_all_kept=dist(kf, 6), coloc_all_kept=dist(kf, 4),
        nf_a_kept=dist(kf, 1), nf_a_kept_legal_only_diff=dist(kf & fr_pairs.legal_only.values, 1),
        nf_a_kept_not_legal_only=dist(kf & ~fr_pairs.legal_only.values, 1),
        rv_gap_lt0_cc1_unclaimed=round(float((np.nan_to_num(rv[kf & cc & ~bcl, 9]) < 0).mean()), 4) if (kf & cc & ~bcl).sum() else None,
        p_mean_cc1_unclaimed=round(float(fr_pairs.p.values[kf & cc & ~bcl].mean()), 4) if (kf & cc & ~bcl).sum() else None,
        p_share_ge_0_97_cc1_unclaimed=round(float((fr_pairs.p.values[kf & cc & ~bcl] >= 0.97).mean()), 4) if (kf & cc & ~bcl).sum() else None,
        addr_token_set_ge_0_9_given_street_mismatch=round(float((lf[kf & fr_pairs.st_both.values & ~fr_pairs.st_eq.values, 1] >= 0.9).mean()), 4),
        p_mean_street_mismatch=round(float(fr_pairs.p.values[kf & fr_pairs.st_both.values & ~fr_pairs.st_eq.values].mean()), 4),
        p_mean_street_match=round(float(fr_pairs.p.values[kf & fr_pairs.st_eq.values].mean()), 4),
        p_mean_name_diff_is_contrast_tok=round(float(fr_pairs.p.values[kf & fr_pairs.ctok.values].mean()), 4) if (kf & fr_pairs.ctok.values).sum() else None,
        p_share_ge_0_97_name_diff_is_contrast_tok=round(float((fr_pairs.p.values[kf & fr_pairs.ctok.values] >= 0.97).mean()), 4) if (kf & fr_pairs.ctok.values).sum() else None,
    )
    RES["runtime_s"] = round(time.time() - t0, 1)
    json.dump(RES, open(os.path.join(OUT, "B_france.json"), "w"), indent=1, default=str)
    log("saved B_france.json")


if __name__ == "__main__":
    main()
