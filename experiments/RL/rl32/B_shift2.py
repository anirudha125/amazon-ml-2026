"""RL-32 B (part 2) -- covariate-shift estimate in the MODEL-RELEVANT subspace.
The full-113 density ratio is degenerate (France vs T AUC 0.99997, ESS 229; even US/India TEST vs T AUC 0.999 through the
split-specific IDF columns), so it cannot carry an estimate. S006's stage 2 puts 99.0% of its gain on 7 columns
(rrL, rrUb, dcos, A_pb, A_rank, rv_so, A_p_m_top); if P(y|x) depends on x only through the stage-2's inputs that matter,
the density ratio in that subspace is the relevant covariate-shift weight.
Subspaces: K7 (99.0% gain) and K12 (+nf_n, rv_gap, rv_sa, s1_logfreq, af_n: 99.3%).
For each: 2-fold S1 cross-fitted France-TEST vs T (p >= .05) classifier -> w = odds * nT/nQ on T -> weighted precision per band
-> implied France FP per S1 among accepted and implied true links per S1; control: US/India TEST vs T.
Usage: nice -n 10 python B_shift2.py
"""
import os, sys, json, time
os.environ["OMP_NUM_THREADS"] = "6"
import numpy as np
import lightgbm as lgb
from sklearn.metrics import roc_auc_score

HERE = os.path.dirname(os.path.abspath(__file__)); C = os.path.join(HERE, "B_cache"); sys.path.insert(0, HERE)
from B_build import NAMES
from B_shift import BANDS, dom_params, grouped_split, TH
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
K7 = ["rrL", "rrUb", "dcos", "A_pb", "A_rank", "rv_so", "A_p_m_top"]
K12 = K7 + ["nf_n", "rv_gap", "rv_sa", "s1_logfreq", "af_n"]


def main():
    t0 = time.time(); R = {}; TXT = []
    pr = lambda s: (TXT.append(s), print(s, flush=True))
    TX = np.load(os.path.join(C, "T_X.npy"), mmap_mode="r"); Tm = np.load(os.path.join(C, "T_meta.npz"), allow_pickle=True)
    pT = Tm["p_oof"].astype(np.float64); iT = np.flatnonzero(pT >= 0.05)
    XT = np.asarray(TX[iT]); yT = Tm["y"][iT].astype(np.int8); ppT = pT[iT]; skT = Tm["s1idx"][iT]
    TE = {}
    for c in ("France", "US", "India"):
        m = np.load(os.path.join(C, f"test_{c}_meta.npz"), allow_pickle=True); X = np.load(os.path.join(C, f"test_{c}_X.npy"), mmap_mode="r")
        i = np.flatnonzero(m["p6"] >= 0.05); TE[c] = dict(X=np.asarray(X[i]), p=m["p6"][i], k=np.char.add(c, m["s1"][i].astype(str)), ns=len(m["u"]))
    for sub_name, feats in (("K7", K7), ("K12", K12)):
        cols = [NAMES.index(f) for f in feats]
        for dom, cs in (("France", ["France"]), ("USI", ["US", "India"])):
            XQ = np.vstack([TE[c]["X"] for c in cs])[:, cols]; pq = np.concatenate([TE[c]["p"] for c in cs])
            kq = np.concatenate([TE[c]["k"] for c in cs]); nsq = sum(TE[c]["ns"] for c in cs)
            XTs = XT[:, cols]
            fq = grouped_split(kq, 0.5, seed=5); ft = grouped_split(skT, 0.5, seed=6)
            pd_T = np.zeros(len(XTs)); pd_Q = np.zeros(len(XQ)); aucs = []
            for k in (0, 1):
                trq, trt = fq == bool(k), ft == bool(k)
                clf = lgb.LGBMClassifier(**dom_params(k)).fit(np.vstack([XQ[trq], XTs[trt]]), np.r_[np.ones(trq.sum()), np.zeros(trt.sum())])
                pd_Q[~trq] = clf.predict_proba(XQ[~trq])[:, 1]; pd_T[~trt] = clf.predict_proba(XTs[~trt])[:, 1]
                aucs.append(round(float(roc_auc_score(np.r_[np.ones((~trq).sum()), np.zeros((~trt).sum())], np.r_[pd_Q[~trq], pd_T[~trt]])), 4))
            eps = 1e-4; w = np.clip(pd_T, eps, 1 - eps) / np.clip(1 - pd_T, eps, 1 - eps) * (len(XTs) / len(XQ))
            ess = float(w.sum() ** 2 / (w ** 2).sum())
            band = []
            for lo, hi in BANDS[1:]:
                bt = (ppT >= lo) & (ppT < hi); bq = (pq >= lo) & (pq < hi); wb = w[bt]
                band.append(dict(band=f"[{lo},{hi})", n_T=int(bt.sum()), ess=round(float(wb.sum() ** 2 / (wb ** 2).sum()), 1),
                                 prec_T=round(float(yT[bt].mean()), 4), prec_Tw=round(float((wb * yT[bt]).sum() / wb.sum()), 4),
                                 pm_Tw=round(float((wb * ppT[bt]).sum() / wb.sum()), 4), q_per_s1=round(float(bq.sum() / nsq), 4),
                                 q_pm=round(float(pq[bq].mean()), 4), q_nosup=round(float((pd_Q[bq] > 0.99).mean()), 4)))
            acc = [b for b in band if float(b["band"].split(",")[0][1:]) >= TH - 1e-9]
            fp_cs = sum(b["q_per_s1"] * (1 - b["prec_Tw"]) for b in acc); fp_self = sum(b["q_per_s1"] * (1 - b["q_pm"]) for b in acc)
            tl_cs = sum(b["q_per_s1"] * b["prec_Tw"] for b in band); tl_m = sum(b["q_per_s1"] * b["q_pm"] for b in band)
            r = dict(auc_2fold=aucs, ess=round(ess, 1), n_T=int(len(XTs)), query_nosupport=round(float((pd_Q > 0.99).mean()), 4),
                     query_acc_nosupport=round(float((pd_Q[pq >= TH] > 0.99).mean()), 4), bands=band, fp_per_s1_covshift=round(fp_cs, 4),
                     fp_per_s1_model_self=round(fp_self, 4), true_links_p05_covshift=round(tl_cs, 4), true_links_p05_model=round(tl_m, 4))
            R[f"{sub_name}_{dom}"] = r
            pr(f"\n[{sub_name} {dom} TEST vs T] AUC {aucs} ESS {ess:.0f}/{len(XTs)} no-support(P>.99) {r['query_nosupport']} (acc {r['query_acc_nosupport']})")
            pr(f"{'band':14s} {'n_T':>7s} {'ESS':>8s} {'prec_T':>7s} {'prec_Tw':>7s} {'pm_Tw':>7s} {'q/S1':>7s} {'q_pm':>7s} {'q_nosup':>7s}")
            for b in band:
                pr(f"{b['band']:14s} {b['n_T']:7d} {b['ess']:8.1f} {b['prec_T']:7.4f} {b['prec_Tw']:7.4f} {b['pm_Tw']:7.4f} {b['q_per_s1']:7.4f} {b['q_pm']:7.4f} {b['q_nosup']:7.4f}")
            pr(f"  FP/S1 accepted: covariate-shift {fp_cs:.4f} | model self {fp_self:.4f};  true links/S1 (p>=.05): covariate-shift {tl_cs:.4f} | model {tl_m:.4f}")
    R["secs"] = round(time.time() - t0, 1)
    json.dump(R, open(os.path.join(HERE, "B_shift2.json"), "w"), indent=1, default=float)
    open(os.path.join(HERE, "B_shift2_tables.txt"), "w").write("\n".join(TXT) + "\n")
    log("done", R["secs"])


if __name__ == "__main__":
    main()
