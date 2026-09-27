"""Read-only control check: effect of filling the rrUb scores that are missing for the RL-27 base's top-10 pairs.
Same stored model (experiments/RL/rl27_model_NEW_s42.pkl = S005's model), same threshold (.78); ONLY the rrUb column of the
missing top-10 pairs changes (NaN = S005 / RL-27 convention -> rrUb score). Pre-registered gate: PASS iff V1 |delta| <= 0.03 pp
with a 95% CI containing 0 AND the fill changes <= 1% of S1 rows (after max-claimer) in every test country.
  V (labelled): V1 / V0 predictions NaN vs filled; paired bootstrap; TP / FP / precision / recall; changed pairs / S1.
               Also RRL (E026) vs the filled stored model.
  T (test)    : every P3 chunk through the S005 recipe; stage 2 re-predicted only for the missing top-10 pairs (all other rows have
               identical inputs); S005's saved accepted pairs (experiments/RL/test_preds) edited for those pairs; max-claimer on
               both; per-S1 comparison. Also checks that max-claimer(S005 saved preds) reproduces S005's matching_results.tsv.
Reads only; writes experiments/E026_rrL/fill_check.json.
"""
import os, sys, json, time, pickle, glob, collections
import multiprocessing as mp
import numpy as np
from boot import H, S2, HERE, ROOT
import step2 as ST

RLD = os.path.join(ROOT, "experiments", "RL"); P3 = os.path.join(ROOT, "experiments", "P3"); P3L = os.path.join(ROOT, "experiments", "P3_rrL")
S005_TAG = "S005_RL27NEW_s42"; TH = None
TAB, NL = chr(9), chr(10)
log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
G = {}


def val_part(M, out):
    rrub = pickle.load(open(os.path.join(ROOT, "experiments", "E023", "rr_rrUb_big.pkl"), "rb"))
    fill = pickle.load(open(os.path.join(HERE, "cache", "rr_rrUb_fill.pkl"), "rb"))
    rrfull = dict(rrub); rrfull.update(fill)
    clf, th = M["stage2"], float(M["th"]); clf.set_params(n_jobs=8)
    Rres = json.load(open(os.path.join(HERE, "step2_results.json")))
    for vn in ("V1", "V0"):
        V = ST.load_val(vn); V["pb"] = np.load(os.path.join(HERE, "cache", f"pb_{vn}.npy")); V["sel"] = S2.topk_mask(V["s1idx"], V["pb"], 10)
        common = [V["LF"][:, :22], S2.block_a_vec(V["s1idx"], V["pb"]), V["LF"][:, 22:], V["rank_dense"].astype(np.float32)[:, None],
                  V["dcos"].astype(np.float32)[:, None]]
        c_nan, c_full = S2.rr_col(V, V["sel"], rrub), S2.rr_col(V, V["sel"], rrfull)
        pn = clf.predict_proba(np.hstack(common + [c_nan[:, None], V["rl"]]).astype(np.float32))[:, 1]
        pf = clf.predict_proba(np.hstack(common + [c_full[:, None], V["rl"]]).astype(np.float32))[:, 1]
        stored = np.load(os.path.join(RLD, "cache", f"rl27_p_NEW_{vn}_s42.npy"))
        sn, sf = S2.summarize(V, pn, th), S2.summarize(V, pf, th)
        d, lo, hi, pneg = H.paired_bootstrap(sn["scores"], sf["scores"])
        an, af = pn >= th, pf >= th; y = V["y"] == 1; ng = int(V["n_gt"].sum())
        changed_pair = an != af; ch_s1 = np.unique(V["s1idx"][changed_pair])
        prr = np.load(os.path.join(HERE, "cache", f"p_RRL_{vn}_s42.npy")); srr = S2.summarize(V, prr, Rres["RRL_s42"]["th_oof"])
        d2, lo2, hi2, _ = H.paired_bootstrap(sf["scores"], srr["scores"])
        pc = np.load(os.path.join(HERE, "cache", f"p_CTRL_{vn}_s42.npy")); sc = S2.summarize(V, pc, Rres["CTRL_s42"]["th_oof"])
        d3, lo3, hi3, _ = H.paired_bootstrap(sf["scores"], sc["scores"])
        out[vn] = dict(
            missing_top10_pairs=int((np.isnan(c_nan) & V["sel"]).sum()), top10_pairs=int(V["sel"].sum()),
            reproduces_stored_S005_model_preds_max_abs_dp=float(np.abs(pn - stored).max()),
            S005_recipe_NaN=round(sn["macro"] * 100, 3), full_rrUb=round(sf["macro"] * 100, 3),
            fill_delta_pp=round(d * 100, 4), ci95=[round(lo * 100, 4), round(hi * 100, 4)], p_delta_le0=round(pneg, 4),
            TP=[sn["tp"], sf["tp"]], FP=[sn["fp"], sf["fp"]], FN=[ng - sn["tp"], ng - sf["tp"]],
            precision=[round(sn["precision"] * 100, 4), round(sf["precision"] * 100, 4)],
            recall=[round(sn["e2e_recall"] * 100, 4), round(sf["e2e_recall"] * 100, 4)],
            changed_pair_decisions=int(changed_pair.sum()), changed_true=int((changed_pair & y).sum()),
            changed_S1=int(len(ch_s1)), n_S1=int(len(V["s1_ids"])),
            US=[round(sn["us"] * 100, 3), round(sf["us"] * 100, 3)], India=[round(sn["india"] * 100, 3), round(sf["india"] * 100, 3)],
            RRL_vs_full_rrUb_S005_model=[round(d2 * 100, 3), round(lo2 * 100, 3), round(hi2 * 100, 3)],
            CTRL_refit_vs_full_rrUb_S005_model=[round(d3 * 100, 3), round(lo3 * 100, 3), round(hi3 * 100, 3)])
        log(vn, json.dumps(out[vn]))


def _w(path):
    M, fill = G["M"], G["fill"]
    z = np.load(path); s1, cand, LF = z["s1"], z["cand"], z["LF"]; rd, dc = z["rank_dense"], z["dcos"]
    Xb = np.hstack([LF[:, :22], rd.astype(np.float32)[:, None], dc.astype(np.float32)[:, None]]).astype(np.float32)
    u, s1idx = np.unique(s1, return_inverse=True); s1idx = s1idx.astype(np.int32)
    pb = M["base"].predict_proba(Xb)[:, 1]; sel = S2.topk_mask(s1idx, pb, 10)
    rows = np.array([i for i in np.flatnonzero(sel) if (str(s1[i]), str(cand[i])) in fill], dtype=np.int64)
    if len(rows) == 0:
        return []
    F = np.load(os.path.join(RLD, "test_feats", G["country"], "rl27_" + os.path.basename(path).replace(".npz", ".npy")))
    A = S2.block_a_vec(s1idx, pb)
    common = [LF[rows, :22], A[rows], LF[rows, 22:], rd[rows].astype(np.float32)[:, None], dc[rows].astype(np.float32)[:, None]]
    vf = np.array([fill[(str(s1[i]), str(cand[i]))] for i in rows], np.float32)
    pn = M["clf"].predict_proba(np.hstack(common + [np.full((len(rows), 1), np.nan, np.float32), F[rows]]).astype(np.float32))[:, 1]
    pf = M["clf"].predict_proba(np.hstack(common + [vf[:, None], F[rows]]).astype(np.float32))[:, 1]
    return [(str(s1[i]), str(cand[i]), float(a), float(b)) for i, a, b in zip(rows, pn, pf)]


def maxclaim(preds):
    cl = collections.defaultdict(list)
    for s, v in preds.items():
        for c, p in v:
            cl[c].append((-p, s))
    win = {c: min(L)[1] for c, L in cl.items()}
    return {s: [(c, p) for c, p in v if win[c] == s] for s, v in preds.items()}


def test_part(Mraw, out, workers=8):
    base = Mraw["base"]; base.set_params(n_jobs=1); clf = Mraw["stage2"]; clf.set_params(n_jobs=1)
    th = float(Mraw["th"]); G["M"] = dict(base=base, clf=clf)
    m5 = {}
    with open(os.path.join(RLD, "submissions", f"{S005_TAG}_mc", "matching_results.tsv"), encoding="utf-8") as f:
        f.readline()
        for l in f:
            s, m = l.rstrip(NL).split(TAB); m5[s] = m
    for country in ("France", "US", "India"):
        t0 = time.time(); G["country"] = country
        G["fill"] = pickle.load(open(os.path.join(P3L, f"rrcache_rrUb_fill_{country}.pkl"), "rb"))
        paths = sorted(glob.glob(os.path.join(P3, country, "chunk_*.npz")))
        rows = []
        with mp.get_context("fork").Pool(workers) as pool:
            for r in pool.imap_unordered(_w, paths):
                rows += r
        assert len(rows) == len(G["fill"]), (len(rows), len(G["fill"]))
        d5 = pickle.load(open(os.path.join(RLD, "test_preds", f"preds_{S005_TAG}_{country}.pkl"), "rb")); pr5 = d5["preds"]; del d5
        # consistency: S005-recipe probabilities of accepted missing pairs equal S005's saved probabilities
        acc = {(s, c): p for s, v in pr5.items() for c, p in v}
        n_acc_nan = sum(pn >= th for _, _, pn, _ in rows); mism = sum(1 for s, c, pn, _ in rows if pn >= th and acc.get((s, c)) != pn)
        miss_in_s005 = sum(1 for s, c, pn, _ in rows if pn < th and (s, c) in acc)
        prf = {s: list(v) for s, v in pr5.items()}
        by_s1 = collections.defaultdict(list)
        for s, c, pn, pf in rows:
            by_s1[s].append((c, pf))
        for s, L in by_s1.items():
            drop = {c for c, _ in L}
            keep = [(c, p) for c, p in prf.get(s, []) if c not in drop] + [(c, pf) for c, pf in L if pf >= th]
            if keep:
                prf[s] = keep
            else:
                prf.pop(s, None)
        f5, ff = maxclaim(pr5), maxclaim(prf)
        ctry_s1 = [s for s in m5 if G["c"][s] == country]
        repro = sum(1 for s in ctry_s1 if sorted(x for x, _ in f5.get(s, [])) != sorted(filter(None, m5[s].split(","))))
        ch_s1 = add = rem = 0
        for s in ctry_s1:
            a = {x for x, _ in f5.get(s, [])}; b = {x for x, _ in ff.get(s, [])}
            if a != b:
                ch_s1 += 1; add += len(b - a); rem += len(a - b)
        flips = collections.Counter("accept->reject" if pn >= th and pf < th else "reject->accept" for _, _, pn, pf in rows if (pn >= th) != (pf >= th))
        out[f"test_{country}"] = dict(n_S1=len(ctry_s1), missing_top10_pairs=len(rows), accepted_under_S005_recipe=int(n_acc_nan),
                                      prob_mismatch_vs_saved_S005=int(mism), rejected_but_in_saved_S005=int(miss_in_s005),
                                      maxclaim_S005_saved_reproduces_S005_tsv_rows_differ=int(repro),
                                      pair_flips_before_maxclaim=dict(flips), changed_S1_after_maxclaim=int(ch_s1),
                                      changed_S1_pct=round(100 * ch_s1 / len(ctry_s1), 4), links_added=int(add), links_removed=int(rem),
                                      secs=round(time.time() - t0, 1))
        log(country, json.dumps(out[f"test_{country}"]))
        del pr5, prf, f5, ff, acc, rows, G["fill"]


def main():
    out = {"gate": "PASS iff V1 |fill delta| <= 0.03 pp with 95% CI containing 0 AND changed S1 <= 1% in every test country"}
    M = pickle.load(open(os.path.join(RLD, "rl27_model_NEW_s42.pkl"), "rb"))
    out["model"] = dict(path=os.path.join(RLD, "rl27_model_NEW_s42.pkl"), th=float(M["th"]), n_feat=int(M["stage2"].n_features_in_))
    val_part(M, out)
    G["c"] = {}
    with open(os.path.join(ROOT, "student_resource", "dataset", "test", "test_source1.tsv"), encoding="utf-8") as f:
        f.readline()
        for l in f:
            p = l.rstrip(chr(13) + NL).split(TAB); G["c"][p[0]] = p[3]
    test_part(pickle.load(open(os.path.join(RLD, "rl27_model_NEW_s42.pkl"), "rb")), out)
    v1 = out["V1"]
    ok_v1 = abs(v1["fill_delta_pp"]) <= 0.03 and v1["ci95"][0] <= 0 <= v1["ci95"][1]
    ok_t = all(out[f"test_{c}"]["changed_S1_pct"] <= 1.0 for c in ("France", "US", "India"))
    out["verdict"] = "PASS" if (ok_v1 and ok_t) else "STOP"; out["ok_v1"] = ok_v1; out["ok_test"] = ok_t
    json.dump(out, open(os.path.join(HERE, "fill_check.json"), "w"), indent=1, default=float)
    log("VERDICT", out["verdict"], "ok_v1", ok_v1, "ok_test", ok_t)


if __name__ == "__main__":
    main()
