"""RL-32 B -- covariate shift, support and extrapolation of the S006 stage-2 on France (label-free on test; T/V1 labelled).
Inputs: B_cache/* from B_build.py (113-col S006 stage-2 matrices).  Outputs: B_shift.json, B_shift_tables.txt, B_cache/B_w*.npy
  1  parity of recomputed test p6 vs rl31/test_scores
  2  domain classifier A: France vs US/India TEST accepted pairs (p6 >= .72), S1-grouped 70/30 split; gain; univariate KS / shift
  3  S006 stage-2 SHAP (pred_contrib) by feature group: France vs US/India accepted + band pairs
  4  domain classifier B: France TEST vs T (p >= .05), 2-fold S1 cross-fitted -> density-ratio weights on T -> covariate-shift
     precision per p-band -> implied France FP / TP per S1; same with US/India TEST vs T as a control
  5  support: kNN (top-20 S006-gain features, robust-standardised) distance to T accepted pairs; pred_leaf T-count per tree
CPU, 6 threads.  Usage: nice -n 10 python B_shift.py
"""
import os, sys, json, time, pickle
os.environ["OMP_NUM_THREADS"] = "6"
import numpy as np, pandas as pd
import lightgbm as lgb
from scipy.stats import ks_2samp
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(os.path.dirname(HERE)))
C = os.path.join(HERE, "B_cache"); E26 = os.path.join(ROOT, "experiments", "E026_rrL")
sys.path.insert(0, HERE)
from B_build import NAMES
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
TH = 0.72; NJ = 6
GROUPS = {"X22_name": list(range(0, 8)), "X22_addr": list(range(8, 16)), "X22_src_ctry": [16, 17], "X22_rank": list(range(18, 22)),
          "A_block": list(range(22, 34)), "B_ret": list(range(34, 41)), "C_idf": list(range(41, 47)), "D_addrcnt": list(range(47, 53)),
          "E_xsrc": list(range(53, 56)), "NUM": list(range(56, 83)), "TOK": list(range(83, 99)), "dense": [99, 100], "rrUb": [101],
          "RL27": list(range(102, 112)), "rrL": [112]}
assert sorted(sum(GROUPS.values(), [])) == list(range(113))
G_OF = {i: g for g, ix in GROUPS.items() for i in ix}
BANDS = [(0.01, 0.05), (0.05, 0.25), (0.25, 0.5), (0.5, 0.72), (0.72, 0.9), (0.9, 0.99), (0.99, 1.01)]
R = {}; TXT = []


def out(s=""):
    TXT.append(s); print(s, flush=True)


def load_test(c):
    m = np.load(os.path.join(C, f"test_{c}_meta.npz"), allow_pickle=True)
    X = np.load(os.path.join(C, f"test_{c}_X.npy"), mmap_mode="r")
    return {k: m[k] for k in m.files}, X


def dom_params(seed=0):
    return dict(n_estimators=200, learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
                min_child_samples=100, reg_lambda=1.0, random_state=seed, n_jobs=NJ, verbose=-1)


def grouped_split(keys, frac=0.7, seed=0):
    u, inv = np.unique(keys, return_inverse=True); rng = np.random.default_rng(seed); tr = rng.random(len(u)) < frac
    return tr[inv]


def main():
    t0 = time.time()
    M6 = pickle.load(open(os.path.join(E26, "model_RRL_s42.pkl"), "rb")); st2 = M6["stage2"]; st2.set_params(n_jobs=NJ)
    gain = st2.booster_.feature_importance("gain"); gain = gain / gain.sum()
    TE = {c: load_test(c) for c in ("France", "US", "India")}
    # ---------------- 1 parity
    par = {}
    for c, (m, X) in TE.items():
        z = np.load(os.path.join(ROOT, "experiments", "RL", "rl31", "test_scores", f"{c}.npz"))
        a = pd.DataFrame(dict(s1=m["s1"].astype(str), cand=m["cand"].astype(str), p=m["p6"]))
        b = pd.DataFrame(dict(s1=z["s1"].astype(str), cand=z["cand"].astype(str), q=z["p6"]))
        j = a.merge(b, on=["s1", "cand"], how="inner")
        par[c] = dict(n_rebuilt=int(len(a)), n_joined=int(len(j)), max_abs_dp=float(np.abs(j.p - j.q).max()),
                      acc_mismatch=int(((j.p >= TH) != (j.q >= TH)).sum()), n_s1_rebuilt=int(len(m["u"])))
    R["parity"] = par; log("parity", json.dumps(par))
    # per-S1 densities (rebuilt sample) for reference
    dens = {}
    for c, (m, X) in TE.items():
        ns = len(m["u"]); p = m["p6"]
        dens[c] = {f"[{lo},{hi})": round(float(((p >= lo) & (p < hi)).sum() / ns), 4) for lo, hi in BANDS}
        dens[c]["sum_p"] = round(float((p.sum() + m["rest"].sum()) / ns), 4)
    R["density_per_s1"] = dens; out("pairs per S1 by p6 band (rebuilt sample): " + json.dumps(dens))

    # ---------------- 2 domain classifier A (test accepted, France vs US/India)
    acc = {c: np.flatnonzero(m["p6"] >= TH) for c, (m, X) in TE.items()}
    rng = np.random.default_rng(1)
    nF = min(len(acc["France"]), 300000)
    iF = np.sort(rng.choice(acc["France"], nF, replace=False))
    XF = np.asarray(TE["France"][1][iF]); XU = np.asarray(TE["US"][1][acc["US"]]); XI = np.asarray(TE["India"][1][acc["India"]])
    XD = np.vstack([XF, XU, XI]); yD = np.r_[np.ones(len(XF)), np.zeros(len(XU) + len(XI))].astype(np.int8)
    kD = np.r_[np.char.add("F", TE["France"][0]["s1"][iF].astype(str)), np.char.add("U", TE["US"][0]["s1"][acc["US"]].astype(str)),
               np.char.add("I", TE["India"][0]["s1"][acc["India"]].astype(str))]
    trm = grouped_split(kD)
    resA = {}
    for tag, cols in (("all113", list(range(113))), ("no_is_india", [i for i in range(113) if i != 17])):
        clf = lgb.LGBMClassifier(**dom_params()).fit(XD[trm][:, cols], yD[trm])
        pd_ = clf.predict_proba(XD[~trm][:, cols])[:, 1]; auc = roc_auc_score(yD[~trm], pd_)
        g = clf.booster_.feature_importance("gain"); g = g / g.sum()
        top = [(NAMES[cols[i]], round(float(g[i]), 4)) for i in np.argsort(-g)[:20]]
        gg = {}
        for i, c in enumerate(cols):
            gg[G_OF[c]] = gg.get(G_OF[c], 0) + float(g[i])
        resA[tag] = dict(auc=round(float(auc), 4), top20_gain=top, group_gain={k: round(v, 4) for k, v in sorted(gg.items(), key=lambda x: -x[1])})
        out(f"[A:{tag}] France vs US/India TEST accepted: AUC {auc:.4f}; group gain {resA[tag]['group_gain']}")
        out("   top gain: " + ", ".join(f"{a}={b}" for a, b in top[:12]))
    # France vs US only (cleanest: both is_india = 0)
    mFU = np.r_[np.ones(len(XF), bool), np.ones(len(XU), bool), np.zeros(len(XI), bool)]
    clf = lgb.LGBMClassifier(**dom_params()).fit(XD[trm & mFU], yD[trm & mFU])
    auc = roc_auc_score(yD[~trm & mFU], clf.predict_proba(XD[~trm & mFU])[:, 1]); resA["France_vs_US"] = dict(auc=round(float(auc), 4))
    g = clf.booster_.feature_importance("gain"); g = g / g.sum()
    resA["France_vs_US"]["top20_gain"] = [(NAMES[i], round(float(g[i]), 4)) for i in np.argsort(-g)[:20]]
    out(f"[A:France_vs_US] AUC {auc:.4f}; top: " + ", ".join(f"{a}={b}" for a, b in resA['France_vs_US']['top20_gain'][:12]))
    # US vs India control (without is_india)
    mUI = ~np.r_[np.ones(len(XF), bool), np.zeros(len(XU) + len(XI), bool)]
    yUI = np.r_[np.zeros(len(XF)), np.ones(len(XU)), np.zeros(len(XI))].astype(np.int8)
    cols = [i for i in range(113) if i != 17]
    clf = lgb.LGBMClassifier(**dom_params()).fit(XD[trm & mUI][:, cols], yUI[trm & mUI])
    auc = roc_auc_score(yUI[~trm & mUI], clf.predict_proba(XD[~trm & mUI][:, cols])[:, 1]); resA["US_vs_India_control"] = dict(auc=round(float(auc), 4))
    out(f"[A:US_vs_India control, no is_india] AUC {auc:.4f}")
    R["domainA"] = resA
    # univariate shift France-acc vs USI-acc (and US-acc)
    XUI = np.vstack([XU, XI]); uni = []
    sub = lambda A, n=100000: A[np.random.default_rng(2).choice(len(A), min(n, len(A)), replace=False)]
    sF, sUI, sU = sub(XF), sub(XUI), sub(XU)
    for j in range(113):
        a, b, cu = sF[:, j], sUI[:, j], sU[:, j]
        fa, fb, fc = np.isfinite(a), np.isfinite(b), np.isfinite(cu)
        ks = ks_2samp(a[fa], b[fb]).statistic if fa.sum() > 50 and fb.sum() > 50 else np.nan
        ksu = ks_2samp(a[fa], cu[fc]).statistic if fa.sum() > 50 and fc.sum() > 50 else np.nan
        sd = np.nanstd(b) + 1e-9
        uni.append(dict(col=NAMES[j], group=G_OF[j], ks_vs_USI=round(float(ks), 4), ks_vs_US=round(float(ksu), 4),
                        mean_F=round(float(np.nanmean(a)), 4), mean_USI=round(float(np.nanmean(b)), 4),
                        shift_sd=round(float((np.nanmean(a) - np.nanmean(b)) / sd), 3), s006_gain=round(float(gain[j]), 4),
                        nan_F=round(float(1 - fa.mean()), 4), nan_USI=round(float(1 - fb.mean()), 4)))
    uni.sort(key=lambda d: -(d["ks_vs_US"] if np.isfinite(d["ks_vs_US"]) else 0))
    R["univariate_accepted"] = uni
    out("\nUnivariate shift, accepted pairs (France vs US, KS; top 25):")
    out(f"{'col':24s} {'grp':12s} {'KS_US':>6s} {'KS_USI':>6s} {'mean_F':>9s} {'mean_USI':>9s} {'shift_sd':>8s} {'S006gain':>8s}")
    for d in uni[:25]:
        out(f"{d['col']:24s} {d['group']:12s} {d['ks_vs_US']:6.3f} {d['ks_vs_USI']:6.3f} {d['mean_F']:9.3f} {d['mean_USI']:9.3f} {d['shift_sd']:8.2f} {d['s006_gain']:8.4f}")
    del XD, XUI
    log(f"domain A done {time.time()-t0:.0f}s")

    # ---------------- 3 SHAP by group (S006 stage 2), accepted and band [0.25, 0.72)
    shp = {}
    for reg, lo, hi in (("accepted", TH, 1.01), ("band_.25_.72", 0.25, TH), ("acc_.72_.99", TH, 0.99)):
        row = {}
        for c, (m, X) in TE.items():
            ix = np.flatnonzero((m["p6"] >= lo) & (m["p6"] < hi))
            ix = np.sort(np.random.default_rng(3).choice(ix, min(40000, len(ix)), replace=False))
            sv = st2.booster_.predict(np.asarray(X[ix]), pred_contrib=True)
            row[c] = {g: round(float(sv[:, ix_].sum(1).mean()), 4) for g, ix_ in GROUPS.items()}
            row[c]["bias"] = round(float(sv[:, -1].mean()), 4); row[c]["logit"] = round(float(sv.sum(1).mean()), 4)
        shp[reg] = row
        out(f"\nS006 stage-2 mean SHAP (logit) by group, {reg}:")
        out(f"{'group':12s} " + " ".join(f"{c:>8s}" for c in TE) + "   F-USIavg")
        for g in list(GROUPS) + ["logit"]:
            f, u, i = row["France"][g], row["US"][g], row["India"][g]
            out(f"{g:12s} " + " ".join(f"{row[c][g]:8.3f}" for c in TE) + f"   {f-(u+i)/2:+.3f}")
    R["shap_group"] = shp
    log(f"SHAP done {time.time()-t0:.0f}s")

    # ---------------- 4 domain classifier B (France TEST vs T) -> density-ratio weights on T
    TX = np.load(os.path.join(C, "T_X.npy"), mmap_mode="r"); Tm = np.load(os.path.join(C, "T_meta.npz"), allow_pickle=True)
    pT = Tm["p_oof"].astype(np.float64); yT = Tm["y"].astype(np.int8); sT = Tm["s1idx"]; nS1T = len(Tm["n_gt"])
    iT = np.flatnonzero(pT >= 0.05); XTb = np.asarray(TX[iT]); ypT = yT[iT]; ppT = pT[iT]; skT = sT[iT]
    R["T_band_counts"] = dict(n_T_s1=int(nS1T), n_rows_p05=int(len(iT)))
    resB = {}
    for dom in ("France", "USI"):
        if dom == "France":
            m, X = TE["France"]; iq = np.flatnonzero(m["p6"] >= 0.05); XQ = np.asarray(X[iq]); kq = m["s1"][iq].astype(str)
            pq = m["p6"][iq]; n_s1_q = len(m["u"])
        else:
            parts = [(TE[c][1], np.flatnonzero(TE[c][0]["p6"] >= 0.05), c) for c in ("US", "India")]
            XQ = np.vstack([np.asarray(X[i]) for X, i, _ in parts]); kq = np.concatenate([np.char.add(c, TE[c][0]["s1"][i].astype(str)) for _, i, c in parts])
            pq = np.concatenate([TE[c][0]["p6"][i] for _, i, c in parts]); n_s1_q = sum(len(TE[c][0]["u"]) for c in ("US", "India"))
        # 2-fold S1-grouped cross-fit: every T row and every query row gets an out-of-fold domain probability
        fq = grouped_split(kq, 0.5, seed=5); ft = grouped_split(skT, 0.5, seed=6)
        pd_T = np.zeros(len(XTb)); pd_Q = np.zeros(len(XQ)); aucs = []
        for k in (0, 1):
            trq, trt = (fq == bool(k)), (ft == bool(k))
            Xtr = np.vstack([XQ[trq], XTb[trt]]); ytr = np.r_[np.ones(trq.sum()), np.zeros(trt.sum())]
            clf = lgb.LGBMClassifier(**dom_params(k)).fit(Xtr, ytr)
            pd_Q[~trq] = clf.predict_proba(XQ[~trq])[:, 1]; pd_T[~trt] = clf.predict_proba(XTb[~trt])[:, 1]
            aucs.append(roc_auc_score(np.r_[np.ones((~trq).sum()), np.zeros((~trt).sum())], np.r_[pd_Q[~trq], pd_T[~trt]]))
        g = clf.booster_.feature_importance("gain"); g = g / g.sum()
        eps = 1e-6; ratio = len(XTb) / len(XQ)
        w = np.clip(pd_T, eps, 1 - eps) / np.clip(1 - pd_T, eps, 1 - eps) * ratio
        wq = np.clip(pd_Q, eps, 1 - eps) / np.clip(1 - pd_Q, eps, 1 - eps) * ratio     # for the "support" view: large => few T neighbours
        np.save(os.path.join(C, f"B_w_{dom}.npy"), w.astype(np.float32))
        ess = float(w.sum() ** 2 / (w ** 2).sum())
        # the query rows' share with essentially no T mass: P(query|x) > 0.99 (odds > 99)
        nosup = float((pd_Q > 0.99).mean()); nosup_acc = float((pd_Q[pq >= TH] > 0.99).mean())
        band = []
        for lo, hi in BANDS[1:]:
            bt = (ppT >= lo) & (ppT < hi); bq = (pq >= lo) & (pq < hi)
            if bt.sum() == 0:
                continue
            wb = w[bt]; wsum = wb.sum()
            prec_w = float((wb * ypT[bt]).sum() / wsum); pm_w = float((wb * ppT[bt]).sum() / wsum)
            prec_u = float(ypT[bt].mean()); pm_u = float(ppT[bt].mean())
            ess_b = float(wsum ** 2 / (wb ** 2).sum())
            nq = float(bq.sum() / n_s1_q); pmq = float(pq[bq].mean()) if bq.any() else np.nan
            band.append(dict(band=f"[{lo},{hi})", n_T=int(bt.sum()), ess_w=round(ess_b, 1), prec_T=round(prec_u, 4), pmean_T=round(pm_u, 4),
                             prec_Tw=round(prec_w, 4), pmean_Tw=round(pm_w, 4), q_pairs_per_s1=round(nq, 4), q_pmean=round(pmq, 4),
                             q_nosupport=round(float((pd_Q[bq] > 0.99).mean()), 4) if bq.any() else np.nan,
                             implied_true_per_s1=round(nq * prec_w, 4), model_true_per_s1=round(nq * pmq, 4)))
        acc_b = [b for b in band if float(b["band"].split(",")[0][1:]) >= TH - 1e-9]
        fp_cs = sum(b["q_pairs_per_s1"] * (1 - b["prec_Tw"]) for b in acc_b); fp_model = sum(b["q_pairs_per_s1"] * (1 - b["q_pmean"]) for b in acc_b)
        fp_unw = sum(b["q_pairs_per_s1"] * (1 - b["prec_T"]) for b in acc_b)
        true_cs = sum(b["implied_true_per_s1"] for b in band); true_model = sum(b["model_true_per_s1"] for b in band)
        resB[dom] = dict(auc_2fold=[round(float(a), 4) for a in aucs], ess_total=round(ess, 1), n_T_rows=int(len(XTb)),
                         w_p50=round(float(np.median(w)), 4), w_p99=round(float(np.percentile(w, 99)), 3), w_max=round(float(w.max()), 1),
                         share_T_weight_top1pct=round(float(np.sort(w)[-max(1, len(w) // 100):].sum() / w.sum()), 4),
                         query_nosupport_share=round(nosup, 4), query_acc_nosupport_share=round(nosup_acc, 4),
                         top_gain=[(NAMES[i], round(float(g[i]), 4)) for i in np.argsort(-g)[:15]], bands=band,
                         fp_per_s1_covshift=round(fp_cs, 4), fp_per_s1_unweighted_T=round(fp_unw, 4), fp_per_s1_model_self=round(fp_model, 4),
                         true_links_per_s1_covshift_p05=round(true_cs, 4), true_links_per_s1_model_p05=round(true_model, 4))
        out(f"\n[B:{dom} TEST vs T, p>=.05] 2-fold AUC {aucs}; ESS {ess:.0f}/{len(XTb)}; query no-support (P>0.99) {nosup:.4f} "
            f"(accepted {nosup_acc:.4f}); top gain " + ", ".join(f"{a}={b}" for a, b in resB[dom]['top_gain'][:8]))
        out(f"{'band':14s} {'n_T':>7s} {'ESS':>8s} {'prec_T':>7s} {'pm_T':>7s} {'prec_Tw':>7s} {'pm_Tw':>7s} {'q/S1':>7s} {'q_pm':>7s} {'q_nosup':>7s}")
        for b in band:
            out(f"{b['band']:14s} {b['n_T']:7d} {b['ess_w']:8.1f} {b['prec_T']:7.4f} {b['pmean_T']:7.4f} {b['prec_Tw']:7.4f} {b['pmean_Tw']:7.4f} "
                f"{b['q_pairs_per_s1']:7.4f} {b['q_pmean']:7.4f} {b['q_nosupport']:7.4f}")
        out(f"  FP per S1 among accepted: covariate-shift {fp_cs:.4f} | unweighted-T precision {fp_unw:.4f} | model self-expectation {fp_model:.4f}")
        out(f"  true links per S1 (p>=.05 pairs): covariate-shift {true_cs:.4f} | model sum-p {true_model:.4f}")
    R["domainB"] = resB
    log(f"domain B done {time.time()-t0:.0f}s")

    # ---------------- 5 support: kNN in top-20 S006-gain space + pred_leaf T-counts
    from sklearn.neighbors import NearestNeighbors
    top20 = list(np.argsort(-gain)[:20]); R["knn_features"] = [NAMES[i] for i in top20]
    accT = np.flatnonzero(ppT >= TH); refT = XTb[accT][:, top20].astype(np.float64)
    med = np.nanmedian(refT, 0); iqr = np.nanpercentile(refT, 75, 0) - np.nanpercentile(refT, 25, 0)
    sd = np.nanstd(refT, 0); scale = np.where(iqr > 1e-6, iqr, np.where(sd > 1e-6, sd, 1.0))
    z = lambda A: np.nan_to_num((A.astype(np.float64) - med) / scale, nan=-5.0)
    rs = np.random.default_rng(7); refi = rs.choice(len(refT), min(150000, len(refT)), replace=False)
    nn = NearestNeighbors(n_neighbors=10, n_jobs=NJ).fit(z(refT[refi]))
    dq = {}
    for c, (m, X) in TE.items():
        ia = acc[c]; ia = np.sort(rs.choice(ia, min(30000, len(ia)), replace=False))
        d, _ = nn.kneighbors(z(np.asarray(X[ia])[:, top20])); dq[c] = d.mean(1)
    # T held-out accepted rows (not in the reference sample) as the in-distribution yardstick
    rest = np.setdiff1d(np.arange(len(refT)), refi); rest = rs.choice(rest, min(30000, len(rest)), replace=False)
    d, _ = nn.kneighbors(z(refT[rest])); dq["T_heldout"] = d.mean(1)
    q95, q99 = np.percentile(np.r_[dq["US"], dq["India"]], [95, 99])
    knn = {c: dict(median=round(float(np.median(v)), 4), p90=round(float(np.percentile(v, 90)), 4),
                   share_gt_USIp95=round(float((v > q95).mean()), 4), share_gt_USIp99=round(float((v > q99).mean()), 4)) for c, v in dq.items()}
    R["knn_support"] = dict(thresholds=dict(USI_p95=round(float(q95), 4), USI_p99=round(float(q99), 4)), by=knn)
    out("\nkNN support (mean dist to 10 NN among T accepted, top-20 S006 features, IQR-scaled): " + json.dumps(knn))
    # pred_leaf: T leaf counts per tree (all T rows)
    nT = TX.shape[0]; ntree = st2.booster_.num_trees(); maxl = 64
    cnt = np.zeros((ntree, maxl), np.int64); pos = np.zeros((ntree, maxl), np.int64)
    for a in range(0, nT, 1000000):
        L = st2.booster_.predict(np.asarray(TX[a:a + 1000000]), pred_leaf=True).astype(np.int64)
        yy = yT[a:a + 1000000]
        for t in range(ntree):
            cnt[t] += np.bincount(L[:, t], minlength=maxl); pos[t] += np.bincount(L[:, t], weights=yy, minlength=maxl).astype(np.int64)
    leaf = {}
    for c, (m, X) in TE.items():
        ia = acc[c]; ia = np.sort(np.random.default_rng(8).choice(ia, min(60000, len(ia)), replace=False))
        L = st2.booster_.predict(np.asarray(X[ia]), pred_leaf=True).astype(np.int64)
        lc = cnt[np.arange(ntree)[None, :], L]; lp = pos[np.arange(ntree)[None, :], L]
        leaf[c] = dict(min_leaf_T=float(np.median(lc.min(1))), trees_leaf_lt100=round(float((lc < 100).sum(1).mean()), 3),
                       share_any_leaf_lt20=round(float(((lc < 20).sum(1) > 0).mean()), 4),
                       share_ge5_trees_lt100=round(float(((lc < 100).sum(1) >= 5).mean()), 4),
                       mean_leaf_pos_rate_min=round(float((lp / np.maximum(lc, 1)).min(1).mean()), 4))
    R["leaf_support"] = leaf
    out("pred_leaf support (T rows per leaf, S006 stage-2 300 trees), accepted pairs: " + json.dumps(leaf))
    R["secs"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(HERE, "B_shift.json"), "w"), indent=1, default=float)
    open(os.path.join(HERE, "B_shift_tables.txt"), "w").write("\n".join(TXT) + "\n")
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
