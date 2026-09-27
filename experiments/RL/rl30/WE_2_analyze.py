"""RL-30 / WE -- adversarial analysis of E's token-role table (reads WE_1_roles_{test,train}.pkl, E_roles_{test,train}.pkl,
accepted_S005_France.pkl). READ-ONLY; writes rl30/WE_2_results.json only.
Labels (train gt) are used only in section 4 (purity / predictive checks of the label-free table on US / India)."""
import os, sys, json, pickle, collections
import numpy as np, pandas as pd
from scipy.stats import spearmanr, pearsonr, beta
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import LEGAL, HONOR, accepted

NONC = LEGAL | HONOR
OUT = {}
W = {s: pickle.load(open(os.path.join(HERE, f"WE_1_roles_{s}.pkl"), "rb")) for s in ("test", "train")}
E = {s: pickle.load(open(os.path.join(HERE, f"E_roles_{s}.pkl"), "rb")) for s in ("test", "train")}
LIST = dict(France=["ecole", "club", "amicale", "comite", "sportive", "centre", "union", "amis", "sarl", "sas", "eurl", "sasu", "sa", "sci",
                    "france", "lille", "nantes", "bordeaux", "ets", "etablissements"],
            US=["inc", "llc", "the", "services", "dental", "family"], India=["limited", "ltd", "tech", "global", "private", "pvt"])


def table(w, c, Dk="D", Fk="F"):
    D = pd.Series(w[Dk].get(c, {}), dtype=float); F = pd.Series(w[Fk].get(c, {}), dtype=float)
    T = pd.DataFrame({"D": D, "F": F}).fillna(0)
    T["Dc"] = pd.Series(w["Dc"].get(c, {}), dtype=float).reindex(T.index).fillna(0)
    T["df"] = pd.Series(w["df"].get(c, {}), dtype=float).reindex(T.index).fillna(0)
    T["first"] = pd.Series(w["first"].get(c, {}), dtype=float).reindex(T.index).fillna(0)
    T["last"] = pd.Series(w["last"].get(c, {}), dtype=float).reindex(T.index).fillna(0)
    sd, sf = max(T.D.sum(), 1), max(T.F.sum(), 1)
    T["s"] = (T.D / sd) / (T.D / sd + T.F / sf)
    T["supp"] = T.D + T.F
    return T, sd, sf


def s_of(D, F, sd, sf):
    D = np.asarray(D, float); F = np.asarray(F, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return (D / sd) / (D / sd + F / sf)


def q2s(q, sd, sf):
    return (q / sd) / (q / sd + (1 - q) / sf)


def r4(x):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), 4)


# ---------------------------------------------------------------- 1. reproduction vs E
rep = {}
for sp in ("test", "train"):
    for c in sorted(W[sp]["D"].keys()):
        T, sd, sf = table(W[sp], c)
        ET = E[sp]["roles"][c]
        j = T.join(ET[["D", "F", "Dc", "s", "supp"]], rsuffix="_E", how="outer").fillna(0)
        rep[f"{sp}_{c}"] = dict(sumD=int(sd), sumF=int(sf), sumDc=int(T.Dc.sum()), n_tokens=int(len(T)), n_supp5=int((T.supp >= 5).sum()),
                                E_sumD=int(ET.D.sum()), E_sumF=int(ET.F.sum()), E_n_supp5=int((ET.supp >= 5).sum()),
                                tokens_D_differ=int((j.D != j.D_E).sum()), tokens_F_differ=int((j.F != j.F_E).sum()),
                                max_abs_s_diff_supp5=r4(np.nanmax(np.abs(j.s - j.s_E)[j.supp >= 5])) if (j.supp >= 5).any() else None,
                                nd_pairs=W[sp]["nd_pairs"].get(c), nd_pairs_core=W[sp]["nd_pairs_core"].get(c), coloc_pairs=W[sp]["coloc_pairs"].get(c),
                                E_nd_pairs=E[sp]["nd_pairs"].get(c),
                                listed={t: dict(D=int(T.D.get(t, 0)), F=int(T.F.get(t, 0)), Dc=int(T.Dc.get(t, 0)), s=r4(T.s.get(t, np.nan)))
                                        for t in LIST.get(c, []) if t in T.index})
OUT["1_reproduction"] = rep

# ---------------------------------------------------------------- 2. support / noise / reliability
noise = {}
for sp in ("test", "train"):
    w = W[sp]
    for c in sorted(w["D"].keys()):
        T, sd, sf = table(w, c)
        t5 = T[T.supp >= 5].copy()
        lo = q2s(beta.ppf(0.025, t5.D + 0.5, t5.F + 0.5), sd, sf); hi = q2s(beta.ppf(0.975, t5.D + 0.5, t5.F + 0.5), sd, sf)
        t5["lo"], t5["hi"] = lo, hi
        excl = (t5.lo > 0.5) | (t5.hi < 0.5)
        # split-half by locality
        Dh0 = pd.Series(w["Dh"].get((c, 0), {}), dtype=float); Dh1 = pd.Series(w["Dh"].get((c, 1), {}), dtype=float)
        Fh0 = pd.Series(w["Fh"].get((c, 0), {}), dtype=float); Fh1 = pd.Series(w["Fh"].get((c, 1), {}), dtype=float)
        H = pd.DataFrame({"D0": Dh0, "D1": Dh1, "F0": Fh0, "F1": Fh1}).fillna(0)
        H["s0"] = s_of(H.D0, H.F0, H.D0.sum(), H.F0.sum()); H["s1"] = s_of(H.D1, H.F1, H.D1.sum(), H.F1.sum())
        sh = {}
        for m in (5, 30, 100):
            ok = ((H.D0 + H.F0) >= m) & ((H.D1 + H.F1) >= m)
            if ok.sum() >= 10:
                a, b = H.s0[ok], H.s1[ok]
                sh[f"both_halves_supp>={m}"] = dict(n_tokens=int(ok.sum()), spearman=r4(spearmanr(a, b)[0]), pearson=r4(pearsonr(a, b)[0]),
                                                    side_of_half_agree=r4(((a > 0.5) == (b > 0.5)).mean()),
                                                    mean_abs_diff=r4(np.abs(a - b).mean()))
        ev_share = {m: r4(T.supp[T.supp >= m].sum() / T.supp.sum()) for m in (5, 30, 100, 1000)}
        noise[f"{sp}_{c}"] = dict(n_supp5=int(len(t5)), supp_quantiles_among_supp5=[int(x) for x in np.quantile(t5.supp, [0.1, 0.25, 0.5, 0.75, 0.9])],
                                  n_supp30=int((T.supp >= 30).sum()), n_supp100=int((T.supp >= 100).sum()), n_supp1000=int((T.supp >= 1000).sum()),
                                  event_share_by_min_supp=ev_share,
                                  n_supp5_ci95_excludes_half=int(excl.sum()), frac_supp5_ci95_excludes_half=r4(excl.mean()),
                                  n_supp5_ci95_width_gt_0p3=int(((t5.hi - t5.lo) > 0.3).sum()),
                                  n_supp5_with_F0=int((t5.F == 0).sum()), n_supp5_with_D0=int((t5.D == 0).sum()),
                                  n_supp5_s_in_0p3_0p7=int(((t5.s >= 0.3) & (t5.s <= 0.7)).sum()),
                                  split_half=sh)
OUT["2_noise"] = noise

# train vs test (same country, disjoint corpora)
tt = {}
for c in ("US", "India"):
    A, _, _ = table(W["train"], c); B, _, _ = table(W["test"], c)
    J = A[["s", "supp"]].join(B[["s", "supp"]], lsuffix="_tr", rsuffix="_te", how="inner")
    d = {}
    for m in (5, 30, 100):
        ok = (J.supp_tr >= m) & (J.supp_te >= m)
        d[f"supp>={m}"] = dict(n=int(ok.sum()), spearman=r4(spearmanr(J.s_tr[ok], J.s_te[ok])[0]),
                               side_agree=r4(((J.s_tr[ok] > 0.5) == (J.s_te[ok] > 0.5)).mean()), mean_abs_diff=r4(np.abs(J.s_tr[ok] - J.s_te[ok]).mean()))
    d["listed"] = {t: [r4(J.s_tr.get(t, np.nan)), r4(J.s_te.get(t, np.nan))] for t in LIST[c]}
    d["sumD_train_vs_test"] = [int(A.D.sum()), int(B.D.sum())]; d["sumF_train_vs_test"] = [int(A.F.sum()), int(B.F.sum())]
    tt[c] = d
OUT["2b_train_vs_test"] = tt

# ---------------------------------------------------------------- 3. artifact checks (France test)
art = {}
for c in ("France", "US", "India"):
    T, sd, sf = table(W["test"], c)
    t5 = T[T.supp >= 30].copy()
    t5["first_rate"] = t5["first"] / t5.df.clip(lower=1); t5["last_rate"] = t5["last"] / t5.df.clip(lower=1)
    t5["legal"] = t5.index.isin(NONC).astype(float); t5["logdf"] = np.log1p(t5.df)
    lg = np.log(np.clip(t5.s, 1e-3, 1 - 1e-3) / (1 - np.clip(t5.s, 1e-3, 1 - 1e-3)))
    X = np.column_stack([np.ones(len(t5)), t5.first_rate, t5.last_rate, t5.legal, t5.logdf])
    coef, *_ = np.linalg.lstsq(X, lg, rcond=None); r2 = 1 - ((lg - X @ coef) ** 2).sum() / ((lg - lg.mean()) ** 2).sum()
    X2 = np.column_stack([np.ones(len(t5)), t5.first_rate, t5.legal]); c2, *_ = np.linalg.lstsq(X2, lg, rcond=None)
    r2b = 1 - ((lg - X2 @ c2) ** 2).sum() / ((lg - lg.mean()) ** 2).sum()
    # normalisation sensitivity: exclude legal/honorific tokens from the normalising sums
    nl = ~T.index.isin(NONC)
    s_nl = s_of(T.D, T.F, T.D[nl].sum(), T.F[nl].sum())
    # D is locality-level: share of D that happens at the exact same address
    Tc = T[T.supp >= 5]
    a = dict(n_tokens_supp30=int(len(t5)), spearman_s_vs_first_rate=r4(spearmanr(t5.s, t5.first_rate)[0]),
             spearman_s_vs_logdf=r4(spearmanr(t5.s, t5.logdf)[0]), spearman_s_vs_last_rate=r4(spearmanr(t5.s, t5.last_rate)[0]),
             R2_logit_s_on_first_last_legal_logdf=r4(r2), R2_logit_s_on_first_rate_legal=r4(r2b),
             sumDc=int(T.Dc.sum()), Dc_share_of_D=r4(T.Dc.sum() / T.D.sum()), n_tokens_Dc_ge5=int((T.Dc >= 5).sum()), n_tokens_Dc_ge20=int((T.Dc >= 20).sum()),
             normalisation_nonlegal_sums=dict(sumD=int(T.D[nl].sum()), sumF=int(T.F[nl].sum()),
                                              listed={t: [r4(T.s.get(t)), r4(s_nl[T.index.get_loc(t)])] for t in LIST.get(c, []) if t in T.index},
                                              n_supp5_side_flips=int(((T.s > 0.5) != (s_nl > 0.5))[T.supp >= 5].sum())),
             D_one_share=r4(sum(W["test"]["D_one"].get(c, {}).values()) / T.D.sum()),
             D_sub_share=r4(sum(W["test"]["D_sub"].get(c, {}).values()) / T.D.sum()))
    # F population (single-S1 keys) vs M population (multi-S1 keys, where a same-address feature would act)
    M = pd.Series(W["test"]["M"].get(c, {}), dtype=float)
    J = pd.DataFrame({"F": T.F, "M": M}).fillna(0)
    ok = (J.F + J.M) >= 30
    a["F_vs_M_population"] = dict(sumF=int(J.F.sum()), sumM=int(J.M.sum()), n_tok=int(ok.sum()),
                                  spearman_token_share=r4(spearmanr(J.F[ok] / J.F.sum(), J.M[ok] / J.M.sum())[0]) if ok.sum() > 10 else None)
    ev = {k[1]: v for k, v in W["test"]["ev"].items() if k[0] == c}
    a["record_events"] = ev
    art[c] = a
OUT["3_artifacts_test"] = art

# ---------------------------------------------------------------- 4. label checks in train (US / India)
lab = {}
w = W["train"]
for c in ("US", "India"):
    ev = {k[1]: v for k, v in w["ev"].items() if k[0] == c}
    T, sd, sf = table(w, c)
    Fm1 = pd.Series(w["Fm"].get((c, True), {}), dtype=float); Fm0 = pd.Series(w["Fm"].get((c, False), {}), dtype=float)
    Mm1 = pd.Series(w["Mm"].get((c, True), {}), dtype=float); Mm0 = pd.Series(w["Mm"].get((c, False), {}), dtype=float)
    # s from the OTHER half would be ideal; halves of F are by locality, labels are per event -> use s_half0 to predict half-1? events
    # (per-token label rates are pooled over halves; we report both in-sample and the half-0-s version)
    Dh0 = pd.Series(w["Dh"].get((c, 0), {}), dtype=float); Fh0 = pd.Series(w["Fh"].get((c, 0), {}), dtype=float)
    H0 = pd.DataFrame({"D": Dh0, "F": Fh0}).fillna(0); s_h0 = pd.Series(s_of(H0.D, H0.F, H0.D.sum(), H0.F.sum()), index=H0.index)
    J = pd.DataFrame({"s": T.s, "supp": T.supp, "m1": Fm1, "m0": Fm0, "M1": Mm1, "M0": Mm0, "s_h0": s_h0}).fillna({"m1": 0, "m0": 0, "M1": 0, "M0": 0})
    J["nF"] = J.m1 + J.m0; J["nM"] = J.M1 + J.M0
    J["nonmatch_F"] = J.m0 / J.nF.replace(0, np.nan); J["nonmatch_M"] = J.M0 / J.nM.replace(0, np.nan)
    d = dict(F_records=ev.get("F_records"), F_record_match_rate=r4(ev.get("F_records_match", 0) / max(ev.get("F_records", 1), 1)),
             M_records=ev.get("M_records"), M_pair_match_rate=r4(ev.get("M_pairs_match", 0) / max(ev.get("M_pairs_match", 0) + ev.get("M_pairs_nonmatch", 0), 1)),
             single_key_records=ev.get("single_key_records"), multi_key_records=ev.get("multi_key_records"))
    for nm, col, n_ in (("F", "nonmatch_F", "nF"), ("M", "nonmatch_M", "nM")):
        ok = (J.supp >= 5) & (J[n_] >= 20)
        if ok.sum() > 10:
            d[f"{nm}_spearman_s_vs_nonmatch_rate"] = dict(n_tok=int(ok.sum()), rho=r4(spearmanr(J.s[ok], J[col][ok])[0]),
                                                           rho_s_half0=r4(spearmanr(J.s_h0[ok].fillna(0.5), J[col][ok])[0]))
        # event-weighted nonmatch rate by s-bin
        bins = [0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.01]
        J["bin"] = pd.cut(J.s, bins, right=False)
        g = J[J.supp >= 5].groupby("bin", observed=False)
        d[f"{nm}_nonmatch_by_s_bin"] = {str(k): dict(events=int(v[n_].sum()), nonmatch=r4(v[("m0" if nm == "F" else "M0")].sum() / max(v[n_].sum(), 1)))
                                         for k, v in g}
    d["listed"] = {t: dict(s=r4(J.s.get(t)), F_events=int(J.nF.get(t, 0)), F_nonmatch=r4(J.nonmatch_F.get(t)),
                           M_events=int(J.nM.get(t, 0)), M_nonmatch=r4(J.nonmatch_M.get(t))) for t in LIST[c] if t in J.index}
    lab[c] = d
OUT["4_labels_train"] = lab

# ---------------------------------------------------------------- 5. France impact bound
Tf, sd, sf = table(W["test"], "France")
lo = pd.Series(q2s(beta.ppf(0.025, Tf.D + 0.5, Tf.F + 0.5), sd, sf), index=Tf.index)
hi = pd.Series(q2s(beta.ppf(0.975, Tf.D + 0.5, Tf.F + 0.5), sd, sf), index=Tf.index)
Dh0 = pd.Series(W["test"]["Dh"].get(("France", 0), {}), dtype=float); Dh1 = pd.Series(W["test"]["Dh"].get(("France", 1), {}), dtype=float)
Fh0 = pd.Series(W["test"]["Fh"].get(("France", 0), {}), dtype=float); Fh1 = pd.Series(W["test"]["Fh"].get(("France", 1), {}), dtype=float)
H = pd.DataFrame({"D0": Dh0, "D1": Dh1, "F0": Fh0, "F1": Fh1}).fillna(0)
H["s0"] = s_of(H.D0, H.F0, H.D0.sum(), H.F0.sum()); H["s1"] = s_of(H.D1, H.F1, H.D1.sum(), H.F1.sum())
P = [p for p in W["test"]["pairs"] if p[0] == "France"]
ids = set(); ntok = 0; n5 = 0; nci = 0; nstable = 0; nstrong = 0; pair_has_strong = 0; pair_all_neutral = 0; tokc = collections.Counter()
for c, i, j, a, b, core in P:
    ids.add(i); ids.add(j)
    strong = False; neutral = True
    for t in a + b:
        ntok += 1; tokc[t] += 1
        if Tf.supp.get(t, 0) >= 5:
            n5 += 1
            ex = lo[t] > 0.5 or hi[t] < 0.5
            nci += ex
            st = t in H.index and (H.D0[t] + H.F0[t]) >= 5 and (H.D1[t] + H.F1[t]) >= 5 and ((H.s0[t] > 0.5) == (H.s1[t] > 0.5))
            nstable += bool(st)
            if ex and (Tf.s[t] >= 0.8 or Tf.s[t] <= 0.2):
                nstrong += 1; strong = True
            if not (0.3 <= Tf.s[t] <= 0.7):
                neutral = False
        else:
            neutral = False
    pair_has_strong += strong; pair_all_neutral += neutral
acc = accepted("S005_France")
acc_in = acc[acc.s1.isin(ids)]
N_FR = W["test"]["n_s1"]["France"]; N_ALL = sum(W["test"]["n_s1"].values())
imp = dict(france_nd_pairs=len(P), france_nd_pairs_core=int(sum(p[5] for p in P)), france_s1_in_pairs=len(ids), n_s1_france=N_FR, n_s1_test=N_ALL,
           share_of_france_s1=r4(len(ids) / N_FR), max_france_gain_pp_if_all_from_0_to_1=r4(100 * len(ids) / N_FR),
           max_LB_gain_pp_if_all_from_0_to_1=r4(100 * len(ids) / N_ALL),
           france_gap_pp_vs_US_IN=round(98.6 - 95.3, 2), france_gap_S1_equivalents=round((98.6 - 95.3) / 100 * N_FR),
           differing_token_events=ntok, with_supp5=n5, ci95_excludes_0p5=int(nci), split_half_side_stable=int(nstable),
           strong_and_significant_s_le0p2_or_ge0p8=int(nstrong), pairs_with_any_strong_token=int(pair_has_strong),
           pairs_all_tokens_neutral_0p3_0p7=int(pair_all_neutral),
           top_differing_tokens=[(t, n, r4(Tf.s.get(t)), int(Tf.supp.get(t, 0))) for t, n in tokc.most_common(25)],
           s005_accepted_pairs_of_these_s1=int(len(acc_in)), s005_accepted_s1=int(acc_in.s1.nunique()),
           s005_accepted_multi_claimed=int((acc_in.n_claims > 1).sum()),
           s005_s1_with_zero_accepted=int(len(ids - set(acc_in.s1.unique()))))
OUT["5_france_impact"] = imp

json.dump(OUT, open(os.path.join(HERE, "WE_2_results.json"), "w"), indent=1, default=str)
print(json.dumps(OUT, indent=1, default=str))
