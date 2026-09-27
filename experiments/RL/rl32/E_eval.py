"""RL-32 / Investigator E -- evaluate structural flags. Labelled L (V1 + T OOF, S006 model) and label-free test (US/India/France).
Outputs E_eval.json + printed tables."""
import os, sys, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(RL, "rl31")); import rl31_lib as RLL
OUT = {}
BANDS = [(0.72, 0.90), (0.90, 0.99), (0.99, 1.01)]


def flags(D):
    f = {}
    f["num_diff"] = D.num == "diff"
    f["nm_far"] = D.nm == "far"
    f["st_diff"] = D.st == "diff"
    f["lonely_nm"] = D.lonely_nm
    f["lonely_nmfar"] = D.lonely_nmfar
    f["lonely_num"] = D.lonely_num
    f["lonely_st"] = D.lonely_st
    f["grp2_num"] = D.grp2_num
    f["lonely_any"] = D.lonely_nm | D.lonely_num | D.lonely_st
    f["nosib_twin_no2"] = (D.sib_twin == 0) & (D.n_o >= 2)
    f["nacc_ge7"] = D.n_acc >= 7
    f["s1_twin"] = D.s1_twins >= 1
    f["prun_ge.3"] = D.p_run >= 0.3
    f["prun_ge.72"] = D.p_run >= 0.72
    f["lonely_num|nm_2plus"] = (D.lonely_num | D.lonely_nm) & (D.n_o >= 2)
    f["lonely_nm&s1_twin"] = D.lonely_nm & (D.s1_twins >= 1)
    return f


def band_table(D, nS1, lab):
    """per band x flag: flagged per 1k S1, FP rate flagged/unflagged, share of band FP captured."""
    A = D[D.acc]
    F = flags(A); rows = []
    for lo, hi in BANDS:
        b = (A.p >= lo) & (A.p < hi)
        nb = int(b.sum()); fpb = int(((A.y == 0) & b).sum()) if lab else None
        for k, v in F.items():
            m = b & v
            r = dict(band=f"{lo:.2f}-{min(hi,1):.2f}", flag=k, per1k=round(m.sum() / nS1 * 1000, 2), band_per1k=round(nb / nS1 * 1000, 1))
            if lab:
                fp = int(((A.y == 0) & m).sum()); fpu = int(((A.y == 0) & b & ~v).sum())
                r.update(fp_rate_flag=round(fp / max(1, m.sum()), 4), fp_rate_unflag=round(fpu / max(1, (b & ~v).sum()), 4),
                         lift=round((fp / max(1, m.sum())) / max(1e-9, fpb / max(1, nb)), 2), fp_capt=round(fp / max(1, fpb), 3),
                         n_flag=int(m.sum()), n_fp=fp)
            rows.append(r)
    return pd.DataFrame(rows)


def auc(score, y):
    from sklearn.metrics import roc_auc_score
    if len(np.unique(y)) < 2: return None
    return round(float(roc_auc_score(y, score)), 4)


def veto_F(D, s1acc, gt, sub, veto):
    """dF (pp) over S1 in `sub` (gt rows) when accepted pairs in `veto` are removed."""
    g = gt[gt.set.isin(sub)].set_index("s1")
    base = s1acc.reindex(g.index).fillna(0)
    V = D[D.acc & veto]
    rm = V.groupby("s").agg(na=("acc", "size"), tp=("y", "sum")).reindex(g.index).fillna(0)
    f0 = RLL.f05_vec(base.tp.values, base.acc.values, g.n_gt.values); f1 = RLL.f05_vec(base.tp.values - rm.tp.values, base.acc.values - rm.na.values, g.n_gt.values)
    return RLL.boot_delta(f1 - f0, n=2000), round(float(f0.mean() * 100), 3), int(rm.na.sum()), int(rm.na.sum() - rm.tp.sum())


def promo_F(D, s1acc, gt, sub, promo):
    g = gt[gt.set.isin(sub)].set_index("s1")
    base = s1acc.reindex(g.index).fillna(0)
    V = D[(~D.acc) & promo]
    ad = V.groupby("s").agg(na=("acc", "size"), tp=("y", "sum")).reindex(g.index).fillna(0)
    f0 = RLL.f05_vec(base.tp.values, base.acc.values, g.n_gt.values); f1 = RLL.f05_vec(base.tp.values + ad.tp.values, base.acc.values + ad.na.values, g.n_gt.values)
    return RLL.boot_delta(f1 - f0, n=2000), int(ad.na.sum()), int(ad.tp.sum())


def main():
    L = pickle.load(open(os.path.join(HERE, "E_feat_L.pkl"), "rb")); D, gt = L["D"], L["gt"]
    s1acc = pickle.load(open(os.path.join(HERE, "E_L_s1acc.pkl"), "rb"))
    cty = gt.set_index("s1").country; D["cty"] = D.s.map(cty).values
    nS1 = len(gt); A = D[D.acc]
    print(f"L: {nS1:,} S1, accepted {len(A):,}, FP {(A.y==0).sum():,} ({(A.y==0).sum()/nS1:.4f}/S1), V1 FP/S1 {((A.y==0)&(A.set=='V1')).sum()/20000:.4f}")
    # coverage of record-side on L
    fp = A[A.y == 0]
    OUT["L_fp_owner"] = dict(n_fp=len(fp), share_has_true_owner=round(float((fp.true_owner != "").mean()), 3),
                             share_owner_in_L=round(float(fp.owner_in_L.mean()), 3))
    print("L FP record owner:", OUT["L_fp_owner"])
    T = band_table(D, nS1, True); OUT["L_band_table"] = T.to_dict("records")
    pd.set_option("display.width", 250, "display.max_rows", 200)
    print(T.to_string(index=False))
    # AUC within accepted, per band, of continuous structural scores (higher = more FP-like)
    au = {}
    for lo, hi in BANDS:
        b = A[(A.p >= lo) & (A.p < hi)]; yfp = (b.y == 0).astype(int).values
        au[f"{lo}-{hi}"] = {"p(neg)": auc(-b.p.values, yfp), "-sib_twin": auc(-b.sib_twin.values, yfp), "-supp_s": auc(-b.supp_s.values, yfp),
                            "n_acc": auc(b.n_acc.values, yfp), "o_eq_num-(num==eq)": auc(b.o_eq_num.values - 2 * (b.num == 'eq').values, yfp),
                            "s1_twins": auc(b.s1_twins.values, yfp), "lonely_any": auc((b.lonely_nm | b.lonely_num | b.lonely_st).values.astype(int), yfp)}
    OUT["L_auc_fp"] = au; print(json.dumps(au, indent=1))
    # veto on V1 and T separately
    F = flags(D); vt = {}
    for k in ["lonely_num", "lonely_nm", "lonely_nmfar", "lonely_st", "grp2_num", "lonely_any", "lonely_num|nm_2plus", "s1_twin", "nosib_twin_no2", "nacc_ge7"]:
        for pmax in (0.99, 1.01):
            veto = F[k] & (D.p < pmax)
            vt[f"{k}|p<{pmax}"] = {"V1": veto_F(D, s1acc, gt, ["V1"], veto), "T": veto_F(D, s1acc, gt, ["T0", "E014", "T2X"], veto)}
    OUT["L_veto"] = vt
    for k, v in vt.items(): print("veto", k, "V1 dF", v["V1"][0], "rm", v["V1"][2], "fp", v["V1"][3], "| T dF", v["T"][0], "rm", v["T"][2], "fp", v["T"][3])
    # pseudo-positive purity and promotion (rejected pairs 0.3<=p<.72)
    R = D[~D.acc]; pr = {}
    P = {"exact3": (D.nm == "eq") & (D.num == "eq") & (D.st == "eq"),
         "exact3&supp2": (D.nm == "eq") & (D.num == "eq") & (D.st == "eq") & (D.supp_s >= 2),
         "sib_twin>=1": D.sib_twin >= 1, "sib_twin>=2": D.sib_twin >= 2,
         "exact3&prun<.3": (D.nm == "eq") & (D.num == "eq") & (D.st == "eq") & (D.p_run < 0.3)}
    for k, v in P.items():
        for lo in (0.3, 0.5):
            m = (~D.acc) & v & (D.p >= lo) & (D.p_run < 0.72)
            pr[f"{k}|p>={lo}"] = dict(per1k=round(m.sum() / nS1 * 1000, 2), prec=round(float(D.y[m].mean()), 3) if m.sum() else None,
                                      V1=promo_F(D, s1acc, gt, ["V1"], v & (D.p >= lo) & (D.p_run < 0.72)),
                                      T=promo_F(D, s1acc, gt, ["T0", "E014", "T2X"], v & (D.p >= lo) & (D.p_run < 0.72)))
    # pseudo-TRUE purity among accepted
    for k, v in {"supp_s>=2&nm_eq&num_eq": (D.supp_s >= 2) & (D.nm == "eq") & (D.num == "eq"), "sib_twin>=2": D.sib_twin >= 2}.items():
        for lo, hi in BANDS:
            m = D.acc & v & (D.p >= lo) & (D.p < hi)
            pr[f"ACC_{k}|{lo}-{hi}"] = dict(per1k=round(m.sum() / nS1 * 1000, 2), prec=round(float(D.y[m].mean()), 4) if m.sum() else None)
    OUT["L_pseudo"] = pr; print(json.dumps(pr, indent=1))
    # ---------- test ----------
    te = {}
    for c in ["US", "India", "France"]:
        Z = pickle.load(open(os.path.join(HERE, f"E_feat_{c}.pkl"), "rb"))
        for pk in ("p6", "p5"):
            Dt = Z[pk]; Dt["y"] = -1
            tt = band_table(Dt, Z["nS1"], False)
            te[f"{c}_{pk}"] = tt
            P2 = {k: v for k, v in P.items()}
            Pt = {"exact3": (Dt.nm == "eq") & (Dt.num == "eq") & (Dt.st == "eq")}
            Pt["exact3&supp2"] = Pt["exact3"] & (Dt.supp_s >= 2); Pt["sib_twin>=1"] = Dt.sib_twin >= 1; Pt["sib_twin>=2"] = Dt.sib_twin >= 2
            Pt["exact3&prun<.3"] = Pt["exact3"] & (Dt.p_run < 0.3)
            for k, v in Pt.items():
                for lo in (0.3, 0.5):
                    m = (~Dt.acc) & v & (Dt.p >= lo) & (Dt.p_run < 0.72)
                    OUT.setdefault("test_promo_per1k", {})[f"{c}_{pk}_{k}|p>={lo}"] = round(m.sum() / Z["nS1"] * 1000, 2)
            OUT.setdefault("test_nacc_perS1", {})[f"{c}_{pk}"] = round(Dt.acc.sum() / Z["nS1"], 4)
    comb = None
    for pk in ("p6", "p5"):
        U, I, Fr = te[f"US_{pk}"], te[f"India_{pk}"], te[f"France_{pk}"]
        t = U[["band", "flag"]].copy(); t["US"] = U.per1k; t["India"] = I.per1k; t["France"] = Fr.per1k
        t["USI"] = ((U.per1k * 663106 + I.per1k * 809986) / (663106 + 809986)).round(2)
        t["Fr/USI"] = (t.France / t.USI.replace(0, np.nan)).round(2)
        t["band_Fr"] = Fr.band_per1k; t["band_USI"] = ((U.band_per1k * 663106 + I.band_per1k * 809986) / (663106 + 809986)).round(1)
        OUT[f"test_flags_{pk}"] = t.to_dict("records")
        print(f"\n=== test flags per 1k S1 among accepted, {pk} ==="); print(t.to_string(index=False))
    print(json.dumps({k: OUT[k] for k in ["test_nacc_perS1", "test_promo_per1k"]}, indent=1))
    json.dump(OUT, open(os.path.join(HERE, "E_eval.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
