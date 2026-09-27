"""RL-30 / WE -- adversarial re-derivation of investigator E's F2 'near-duplicate co-located sibling' signal (READ-ONLY; writes rl30/WE_*).
Independent re-implementation (does not import E_*): sibling = other S1 of the same country with the same non-empty akey whose
name token set differs by <=1 token per side, shares >=1 token and is not identical.  F2_ge: max_sib jac(R,B) > 0 and >= jac(R,A);
F2_gt: strictly greater.
Part A: V0+V1 (labelled, RL-27 NEW s42 p): counts, clustering by S1, sibling presence in the pool, post-max-claim precision,
        S1-cluster bootstrap of the veto, tie vs strict breakdown, every accepted flagged pair listed.
Part B: test: sibling prevalence per country, F2 firing among S005 accepted pairs, max-claimer interplay (does the sibling also claim the
        record?), tie vs strict, bound on the France / LB value of a veto of the kept pairs.
Output: rl30/WE_1_results.json"""
import os, sys, time, json, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import load, toks, akey, accepted, PATHS, NEW_TH, ROOT, RL

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
E24 = os.path.join(ROOT, "experiments", "E024")


def ns(s):
    return frozenset(toks(s))


def is_nd(A, B):
    if A == B:
        return False
    return len(A - B) <= 1 and len(B - A) <= 1 and len(A & B) > 0


def jac(A, B):
    u = len(A | B)
    return len(A & B) / u if u else 0.0


def sibling_table(s1df, restrict_ids=None):
    """returns dict s1_id -> list of sibling s1_ids (same country, same akey, near-dup)"""
    ak = [akey(a) for a in s1df.addr.values]
    grp = collections.defaultdict(list)
    for i, (k, c) in enumerate(zip(ak, s1df.country.values)):
        if k:
            grp[c + "|" + k].append(i)
    ids = s1df.id.values; nms = s1df.name.values
    sets = {}
    sib = {}
    for key, L in grp.items():
        if len(L) < 2:
            continue
        if restrict_ids is not None and not any(ids[i] in restrict_ids for i in L):
            continue
        for i in L:
            if i not in sets:
                sets[i] = ns(nms[i])
        for x in range(len(L)):
            for y in range(x + 1, len(L)):
                i, j = L[x], L[y]
                if is_nd(sets[i], sets[j]):
                    sib.setdefault(ids[i], []).append(ids[j]); sib.setdefault(ids[j], []).append(ids[i])
    return sib, grp, ak


def f05(s1idx, y, acc, n_gt):
    n = len(n_gt)
    npred = np.bincount(s1idx, weights=acc, minlength=n); tp = np.bincount(s1idx, weights=acc & y, minlength=n)
    return np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))


def maxclaim(acc, p, cand):
    idx = np.flatnonzero(acc)
    df = pd.DataFrame({"i": idx, "c": cand[idx], "p": p[idx]}).sort_values(["p", "i"], ascending=[False, True])
    out = np.zeros_like(acc); out[df.drop_duplicates("c").i.values] = True
    return out


def main():
    t0 = time.time(); R = {}
    # ======================= Part A: train / V0+V1
    D = load("train", verbose=False)
    s1 = D["s1"]; S1 = s1.set_index("id"); recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
    owner = dict(zip(D["gt"].rec.values, D["gt"].s1.values))
    sib_tr, _, _ = sibling_table(s1)
    n_sib_tr = collections.Counter(S1.country.loc[list(sib_tr.keys())].values)
    n_c_tr = s1.country.value_counts().to_dict()
    R["train_sibling_prevalence"] = {c: dict(s1_with_sib=int(n_sib_tr.get(c, 0)), n_s1=int(n_c_tr[c]), rate=round(n_sib_tr.get(c, 0) / n_c_tr[c], 6))
                                     for c in n_c_tr}
    R["train_sibling_prevalence"]["all"] = dict(s1_with_sib=len(sib_tr), n_s1=len(s1), rate=round(len(sib_tr) / len(s1), 6))
    log("train siblings", R["train_sibling_prevalence"], f"{time.time()-t0:.0f}s")

    pools = {}
    for vn, pp in (("V1", PATHS["v1_p_new"]), ("V0", os.path.join(RL, "cache", "rl27_p_NEW_V0_s42.npy"))):
        M = np.load(os.path.join(E24, f"{vn}_a50n10d10a", "meta.npz"))
        pools[vn] = dict(s1_ids=M["s1_ids"], s1idx=M["s1idx"], cand=M["cand"], y=M["y"] == 1, n_gt=M["n_gt"], cs=M["country"], p=np.load(pp))
    V1, V0 = pools["V1"], pools["V0"]; off = len(V1["s1_ids"])
    P = {k: np.concatenate([V1[k], V0[k]]) for k in ("s1_ids", "cand", "y", "n_gt", "cs", "p")}
    P["s1idx"] = np.concatenate([V1["s1idx"], V0["s1idx"] + off])
    P["src"] = np.concatenate([np.full(len(V1["cand"]), "V1"), np.full(len(V0["cand"]), "V0")])
    ps1 = P["s1_ids"][P["s1idx"]]
    pool_s1 = set(P["s1_ids"])
    other_pool_s1 = set()
    for tn in ("T0", "E014", "T2X"):
        other_pool_s1 |= set(np.load(os.path.join(E24, f"{tn}_a50n10d10a", "meta.npz"))["s1_ids"])
    hs = np.array([s in sib_tr for s in ps1])
    ii = np.flatnonzero(hs)
    ge = np.zeros(len(ps1), bool); gt = np.zeros(len(ps1), bool); jr_a = np.full(len(ps1), np.nan); js_a = np.full(len(ps1), np.nan)
    for k in ii:
        A = ns(S1.name.loc[ps1[k]]); Rn = ns(recs.name.loc[P["cand"][k]])
        jr = jac(Rn, A); js = max(jac(Rn, ns(S1.name.loc[j])) for j in sib_tr[ps1[k]])
        jr_a[k] = jr; js_a[k] = js
        ge[k] = js > 0 and js >= jr; gt[k] = js > jr
    y, p, cand = P["y"], P["p"], P["cand"]; acc = p >= NEW_TH
    A = {}
    A["s1_in_V0V1_with_sibling"] = int(np.unique(ps1[hs]).size)
    A["siblings_of_those_in_V0V1_pool"] = int(sum(1 for s in np.unique(ps1[hs]) for j in sib_tr[s] if j in pool_s1))
    A["siblings_of_those_in_T0_E014_T2X_pools"] = int(sum(1 for s in np.unique(ps1[hs]) for j in sib_tr[s] if j in other_pool_s1))
    A["n_siblings_total"] = int(sum(len(sib_tr[s]) for s in np.unique(ps1[hs])))
    for nm, f in (("F2_ge", ge), ("F2_gt", gt), ("F2_tie_only", ge & ~gt), ("has_sib", hs)):
        own_sib = np.array([owner.get(cand[k]) in sib_tr.get(ps1[k], []) for k in range(len(ps1))]) if nm == "has_sib" else \
            np.array([f[k] and owner.get(cand[k]) in sib_tr.get(ps1[k], []) for k in range(len(ps1))])
        fa = f & acc
        fp_s1 = collections.Counter(ps1[fa & ~y])
        r = dict(pairs=int(f.sum()), s1=int(np.unique(ps1[f]).size), pos=int((f & y).sum()), owner_sibling=int((f & own_sib).sum()),
                 accepted=int(fa.sum()), accepted_s1=int(np.unique(ps1[fa]).size), acc_TP=int((fa & y).sum()), acc_FP=int((fa & ~y).sum()),
                 acc_FP_owned_by_sibling=int((fa & ~y & own_sib).sum()), acc_FP_distinct_s1=len(fp_s1), acc_FP_per_s1=dict(fp_s1),
                 acc_FP_by_country=dict(collections.Counter(P["cs"][P["s1idx"][fa & ~y]])),
                 acc_by_country=dict(collections.Counter(P["cs"][P["s1idx"][fa]])),
                 acc_TP_s1=sorted(set(ps1[fa & y])))
        mc = maxclaim(acc, p, cand)
        r["after_pool_maxclaim"] = dict(kept=int((fa & mc).sum()), kept_FP=int((fa & mc & ~y).sum()))
        # was the owner (sibling) of each accepted FP even in V0+V1 (so a max-claimer could see its claim)?
        r["acc_FP_owner_in_V0V1_pool"] = int(sum(1 for k in np.flatnonzero(fa & ~y) if owner.get(cand[k]) in pool_s1))
        r["acc_FP_owner_in_T_pools"] = int(sum(1 for k in np.flatnonzero(fa & ~y) if owner.get(cand[k]) in other_pool_s1))
        A[nm] = r
    # veto with S1-level paired bootstrap (resample S1) and a leave-one-S1-out check
    f0 = f05(P["s1idx"], y, acc, P["n_gt"]); fv = f05(P["s1idx"], y, acc & ~ge, P["n_gt"]); d = fv - f0
    nz = np.flatnonzero(d)
    A["veto_ge"] = dict(delta_pp=round(d.mean() * 100, 5), n_s1_changed=int(len(nz)),
                        per_s1_delta=[dict(s1=P["s1_ids"][i], country=P["cs"][i], n_gt=int(P["n_gt"][i]), f_before=round(float(f0[i]), 4),
                                           f_after=round(float(fv[i]), 4)) for i in nz],
                        delta_if_drop_each_changed_s1_pp=[round((d.sum() - d[i]) / len(d) * 100, 5) for i in nz])
    rng = np.random.default_rng(7); N = len(d); K = len(nz)
    pv = np.full(K + 1, 1.0 / N); pv[-1] = 1 - K / N
    C = rng.multinomial(N, pv, size=20000)[:, :K]; bs = C @ d[nz] / N
    A["veto_ge"]["boot_ci95_pp"] = [round(float(np.percentile(bs, 2.5) * 100), 5), round(float(np.percentile(bs, 97.5) * 100), 5)]
    A["veto_ge"]["boot_p_le0"] = round(float((bs <= 0).mean()), 4)
    A["veto_ge"]["P_all_changed_s1_absent_in_resample"] = round(float(np.exp(-K)), 4)
    # every accepted flagged pair, with names
    rows = []
    for k in np.flatnonzero(ge & acc):
        rows.append(dict(src=P["src"][k], country=P["cs"][P["s1idx"][k]], s1=ps1[k], s1_name=S1.name.loc[ps1[k]], s1_addr=S1.addr.loc[ps1[k]],
                         siblings=[S1.name.loc[j] for j in sib_tr[ps1[k]]], rec=cand[k], rec_name=recs.name.loc[cand[k]], rec_addr=recs.addr.loc[cand[k]],
                         p=round(float(p[k]), 5), y=bool(y[k]), owner=owner.get(cand[k]), owner_is_sibling=owner.get(cand[k]) in sib_tr[ps1[k]],
                         owner_name=S1.name.loc[owner[cand[k]]] if cand[k] in owner else None, jr=round(jr_a[k], 3), js=round(js_a[k], 3), gt=bool(gt[k])))
    A["accepted_flagged_pairs"] = rows
    # S1-level logistic-free test: among S1 with siblings, share of accepted pairs that are FP, flagged vs not, per S1
    per = []
    for s in np.unique(ps1[hs]):
        m = ps1 == s
        per.append(dict(s1=s, name=S1.name.loc[s], country=S1.country.loc[s], acc_flag=int((m & ge & acc).sum()), acc_flag_FP=int((m & ge & acc & ~y).sum()),
                        acc_noflag=int((m & ~ge & acc).sum()), acc_noflag_FP=int((m & ~ge & acc & ~y).sum()), n_gt=int(P["n_gt"][np.flatnonzero(P["s1_ids"] == s)[0]]),
                        siblings=[S1.name.loc[j] for j in sib_tr[s]]))
    A["per_s1_with_sibling"] = per
    R["V0V1"] = A
    log("V0+V1", {k: A[k] for k in ("s1_in_V0V1_with_sibling", "siblings_of_those_in_V0V1_pool", "siblings_of_those_in_T0_E014_T2X_pools")},
        {k: {x: A[k][x] for x in ("pairs", "s1", "pos", "owner_sibling", "accepted", "accepted_s1", "acc_FP", "acc_FP_distinct_s1", "acc_FP_owner_in_V0V1_pool")}
         for k in ("F2_ge", "F2_gt", "F2_tie_only")}, A["veto_ge"]["delta_pp"], A["veto_ge"]["boot_ci95_pp"], f"{time.time()-t0:.0f}s")
    del D, s1, S1, recs, owner

    # ======================= Part B: test
    T = load("test", verbose=False)
    s1 = T["s1"]; S1 = s1.set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    sib_te, _, _ = sibling_table(s1)
    n_c = s1.country.value_counts().to_dict()
    cnt = collections.Counter(S1.country.loc[list(sib_te.keys())].values)
    R["test_sibling_prevalence"] = {c: dict(s1_with_sib=int(cnt.get(c, 0)), n_s1=int(n_c[c]), rate=round(cnt.get(c, 0) / n_c[c], 6)) for c in n_c}
    log("test siblings", R["test_sibling_prevalence"], f"{time.time()-t0:.0f}s")
    B = {}
    for tag in ("S005_France", "S005_US", "S005_India"):
        c = tag.split("_")[1]
        a = accepted(tag).reset_index(drop=True)
        hs = a.s1.map(lambda s: s in sib_te).values
        ii = np.flatnonzero(hs)
        ge = np.zeros(len(a), bool); gt = np.zeros(len(a), bool)
        sib_claims = np.zeros(len(a), bool); sib_wins = np.zeros(len(a), bool)
        claim = a.groupby("rec").s1.apply(set).to_dict() if len(ii) else {}
        pmap = {(s, r): pv_ for s, r, pv_ in zip(a.s1.values[ii], a.rec.values[ii], a.p.values[ii])}
        # map every accepted (s1, rec) of any S1 claiming records touched by flagged S1 -> p
        touched = set(a.rec.values[ii]); sub = a[a.rec.isin(touched)]
        pall = {(s, r): pv_ for s, r, pv_ in zip(sub.s1.values, sub.rec.values, sub.p.values)}
        jr_l = np.full(len(a), np.nan); js_l = np.full(len(a), np.nan)
        for k in ii:
            s = a.s1.values[k]; r = a.rec.values[k]
            A_ = ns(S1.name.loc[s]); Rn = ns(recs.name.loc[r])
            jr = jac(Rn, A_); js = max(jac(Rn, ns(S1.name.loc[j])) for j in sib_te[s])
            jr_l[k] = jr; js_l[k] = js
            ge[k] = js > 0 and js >= jr; gt[k] = js > jr
            sc = [j for j in sib_te[s] if j in claim.get(r, ())]
            sib_claims[k] = bool(sc)
            sib_wins[k] = any(pall[(j, r)] > a.p.values[k] for j in sc)
        kept = a.kept_final.values; mcl = a.n_claims.values > 1
        # records kept for a flagged S1: does it have other kept records?
        kept_per_s1 = a[kept].groupby("s1").size()
        r = dict(accepted_pairs=int(len(a)), n_s1=int(n_c[c]))
        for nm, f in (("F2_ge", ge), ("F2_gt", gt), ("F2_tie_only", ge & ~gt)):
            fk = f & kept
            s_k = np.unique(a.s1.values[fk])
            k_other = np.array([kept_per_s1.get(s, 0) - int(((a.s1.values == s) & fk).sum()) for s in s_k]) if len(s_k) else np.array([])
            r[nm] = dict(accepted=int(f.sum()), accepted_s1=int(np.unique(a.s1.values[f]).size), multi_claimed=int((f & mcl).sum()),
                         dropped_by_maxclaim=int((f & ~kept).sum()), kept=int(fk.sum()), kept_s1=int(len(s_k)),
                         accepted_sibling_also_claims=int((f & sib_claims).sum()), accepted_sibling_claims_with_higher_p=int((f & sib_wins).sum()),
                         kept_sibling_also_claims=int((fk & sib_claims).sum()), kept_single_claim=int((fk & ~mcl).sum()),
                         kept_s1_with_no_other_kept_record=int((k_other == 0).sum()) if len(s_k) else 0,
                         kept_p_quantiles=[round(float(q), 4) for q in np.quantile(a.p.values[fk], [0.1, 0.25, 0.5, 0.9])] if fk.any() else None)
            # value bound: veto flagged kept pairs. best case = all flagged FP and all other kept pairs TP (n_gt = #other kept);
            # worst case = all flagged TP and all other kept TP (n_gt = #kept).
            if len(s_k):
                v = np.array([int(((a.s1.values == s) & fk).sum()) for s in s_k]); kk = v + k_other
                before_best = np.where(k_other > 0, 1.25 * k_other / (0.25 * k_other + kk), 0.0)
                gain_best = 1.0 - before_best
                after_worst = np.where(k_other > 0, 1.25 * k_other / (0.25 * kk + k_other), 0.0)
                loss_worst = 1.0 - after_worst
                r[nm]["bound_gain_all_FP_S1eq"] = round(float(gain_best.sum()), 2)
                r[nm]["bound_gain_all_FP_country_pp"] = round(float(gain_best.sum() / n_c[c] * 100), 5)
                r[nm]["bound_gain_all_FP_LB_pp"] = round(float(gain_best.sum() / len(s1) * 100), 5)
                r[nm]["bound_loss_all_TP_S1eq"] = round(float(loss_worst.sum()), 2)
                r[nm]["bound_loss_all_TP_country_pp"] = round(float(loss_worst.sum() / n_c[c] * 100), 5)
                r[nm]["mean_gain_per_S1_if_FP"] = round(float(gain_best.mean()), 4); r[nm]["mean_loss_per_S1_if_TP"] = round(float(loss_worst.mean()), 4)
                r[nm]["break_even_false_fraction"] = round(float(loss_worst.sum() / (loss_worst.sum() + gain_best.sum())), 4)
        # examples of kept flagged pairs (France only)
        if c == "France":
            rng = np.random.default_rng(3); ex = []
            for k in rng.choice(np.flatnonzero(ge & kept), size=min(20, int((ge & kept).sum())), replace=False):
                s = a.s1.values[k]; rr = a.rec.values[k]
                ex.append(dict(s1_name=S1.name.loc[s], s1_addr=S1.addr.loc[s], siblings=[S1.name.loc[j] for j in sib_te[s]],
                               rec_name=recs.name.loc[rr], rec_addr=recs.addr.loc[rr], p=round(float(a.p.values[k]), 4), jr=round(jr_l[k], 3), js=round(js_l[k], 3),
                               gt=bool(gt[k]), sibling_claims=bool(sib_claims[k]), n_claims=int(a.n_claims.values[k])))
            r["kept_examples"] = ex
        B[tag] = r
        log(tag, {k: v for k, v in r.items() if k != "kept_examples"}, f"{time.time()-t0:.0f}s")
    R["test"] = B
    json.dump(R, open(os.path.join(HERE, "WE_1_results.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
