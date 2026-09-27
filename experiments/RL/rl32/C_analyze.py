"""RL-32 C -- model disagreement as a detector of confident errors.
Inputs (read-only): C_lab_{V1,T}.npz (labelled, built by C_lab.py), rl31/test_scores/<c>.npz + C_test_<c>.npz (label-free).
Part 1  labelled: conditional AUC (FP = y==0) of each disagreement signal inside p-bands of the deployed model, cell precision,
        FP captured vs TP lost, exact V1 macro delta of vetoing each cell (cells chosen on T, evaluated on V1).
Part 2  test: cell frequency per S1 among kept pairs, France vs US/India.
Part 3  France FP captured under (a) equal cell precision, (b) cell error odds scaled uniformly to the count-prior FP budget.
Part 4  S005 -> S006 transitions (removals / additions) by cell on France vs the V1 (and T) analogue, reweighted estimate.
Writes C_analyze.json. Usage: nice -n 10 python C_analyze.py
"""
import os, sys, json
import numpy as np
from scipy.stats import rankdata
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(EXP, "RL", "rl31"))
from rl31_lib import f05_vec, boot_delta
lg = lambda p: np.log(np.clip(p, 1e-7, 1 - 1e-7) / (1 - np.clip(p, 1e-7, 1 - 1e-7)))
OUT = {}


def auc(score, bad):
    """P(score_bad > score_good); 0.5 = no signal."""
    ok = ~np.isnan(score); s, b = score[ok], bad[ok]
    nb, ng = int(b.sum()), int((~b).sum())
    if nb < 5 or ng < 5:
        return None
    r = rankdata(s); return round(float((r[b].sum() - nb * (nb + 1) / 2) / (nb * ng)), 3)


def signals(D, model):
    ps, po = (D["p6"], D["p5"]) if model == "S006" else (D["p5"], D["p6"])
    rrl, rrub = D["rrl"].astype(float), D["rrub"].astype(float)
    dc = np.where(np.isnan(D["dcos"]), 0.0, D["dcos"]).astype(float)
    # per-S1 max dcos among this model's kept pairs (dense outlier = far below its S1-mates)
    keep = D["k6"] if model == "S006" else D["k5"]
    mx = np.full(D["s1i"].max() + 1, -1.0); np.maximum.at(mx, D["s1i"][keep], dc[keep])
    return {
        "other_model_gap": lg(ps) - lg(po),            # this model much more confident than the other model
        "base_gap": lg(ps) - lg(D["pb"]),              # stage-2 much more confident than base
        "rrL_neg": -rrl, "rrUb_neg": -rrub,
        "rr_min_neg": -np.fmin(rrl, rrub),             # the more sceptical reranker
        "rr_absdis": np.abs(rrl - rrub),               # reranker disagreement magnitude
        "rrUb_minus_rrL": rrub - rrl,
        "dcos_neg": -dc, "dcos_gap_s1": mx[D["s1i"]] - dc,
        "rank_dense": D["rank_dense"].astype(float), "base_rank": D["rk"].astype(float),
    }


BANDS = [(".99-1", .99, 1.01), (".95-.99", .95, .99), (".90-.95", .90, .95), ("th-.90", 0.0, .90)]


def cells(D, model):
    """disjoint agreement cells for this model's kept pairs."""
    po, tho = (D["p5"], D["th5"]) if model == "S006" else (D["p6"], D["th6"])
    rrl, rrub = D["rrl"], D["rrub"]
    c = {}
    c["rr++"] = D["sel"] & (rrl > 0) & (rrub > 0)
    c["rrL+ rrUb-"] = D["sel"] & (rrl > 0) & (rrub <= 0)
    c["rrL- rrUb+"] = D["sel"] & (rrl <= 0) & (rrub > 0)
    c["rr--"] = D["sel"] & (rrl <= 0) & (rrub <= 0)
    c["not top10"] = ~D["sel"]
    x = {}
    x["other model rejects"] = po < tho
    x["base pb<.5"] = D["pb"] < .5
    x["dense miss (rank_dense>10)"] = D["rank_dense"] > 10
    x["any rr<=0"] = D["sel"] & ((rrl <= 0) | (rrub <= 0))
    x["rr<=0 or other rejects"] = x["any rr<=0"] | x["other model rejects"]
    x["rr-- or (rr<=0 & other rejects)"] = c["rr--"] | (x["any rr<=0"] & x["other model rejects"])
    return c, x


def band_of(p):
    b = np.full(len(p), "", object)
    for nm, lo, hi in BANDS:
        b[(p >= lo) & (p < hi)] = nm
    return b


def macro_delta(D, keep, veto):
    """per-S1 F0.5 delta of dropping `veto` pairs from `keep` (exact on the labelled set)."""
    n = len(D["n_gt"]); s = D["s1i"]; y = D["y"] == 1
    tp0 = np.bincount(s, (keep & y).astype(float), n); na0 = np.bincount(s, keep.astype(float), n)
    k1 = keep & ~veto; tp1 = np.bincount(s, (k1 & y).astype(float), n); na1 = np.bincount(s, k1.astype(float), n)
    return f05_vec(tp1, na1, D["n_gt"]) - f05_vec(tp0, na0, D["n_gt"])


LAB = {t: dict(np.load(os.path.join(HERE, f"C_lab_{t}.npz"), allow_pickle=True)) for t in ("V1", "T")}
for t, D in LAB.items():
    D["th5"], D["th6"] = float(D["th5"]), float(D["th6"])

# ---------------- Part 1: labelled -----------------
P1 = {}
for t, D in LAB.items():
    for model in ("S006", "S005"):
        keep = D["k6"] if model == "S006" else D["k5"]; p = D["p6"] if model == "S006" else D["p5"]
        bad = D["y"] == 0; S = signals(D, model); b = band_of(p); n_s1 = len(D["n_gt"])
        r = {"n_kept": int(keep.sum()), "n_fp": int((keep & bad).sum()), "fp_per_s1": round(float((keep & bad).sum() / n_s1), 5),
             "auc_all_kept": {k: auc(v[keep], bad[keep]) for k, v in S.items()}, "bands": {}}
        for nm, lo, hi in BANDS:
            m = keep & (b == nm)
            r["bands"][nm] = {"n": int(m.sum()), "fp": int((m & bad).sum()), "prec": round(float(1 - bad[m].mean()), 5) if m.any() else None,
                              "auc": {k: auc(v[m], bad[m]) for k, v in S.items()}}
        c, x = cells(D, model); r["cells"] = {}
        for nm, msk in list(c.items()) + list(x.items()):
            m = keep & msk; d = macro_delta(D, keep, m)
            r["cells"][nm] = {"n": int(m.sum()), "per_s1": round(float(m.sum() / n_s1), 5), "fp": int((m & bad).sum()),
                              "tp": int((m & ~bad).sum()), "prec": round(float(1 - bad[m].mean()), 4) if m.any() else None,
                              "share_of_all_fp": round(float((m & bad).sum() / max(1, (keep & bad).sum())), 4),
                              "veto_dF_pp": boot_delta(d, n=2000) if m.any() else None}
            # cell x band (precision only)
            r["cells"][nm]["by_band"] = {bn: [int((m & (b == bn)).sum()), int((m & (b == bn) & bad).sum())] for bn, _, _ in BANDS}
        P1[f"{t}_{model}"] = r
        print(t, model, json.dumps({k: r[k] for k in ("n_kept", "n_fp", "fp_per_s1")}), flush=True)
        print("  AUC all kept:", r["auc_all_kept"], flush=True)
        for nm in r["bands"]:
            print("  band", nm, r["bands"][nm]["n"], r["bands"][nm]["fp"], r["bands"][nm]["prec"], r["bands"][nm]["auc"], flush=True)
        for nm, v in r["cells"].items():
            print(f"  cell {nm:34s} n {v['n']:6d} fp {v['fp']:4d} prec {v['prec']} fp-share {v['share_of_all_fp']} vetoF {v['veto_dF_pp']}", flush=True)
OUT["part1"] = P1

# ---------------- Part 2: test cell frequencies ----------------
def test_table(c):
    z = np.load(os.path.join(EXP, "RL", "rl31", "test_scores", f"{c}.npz")); f = np.load(os.path.join(HERE, f"C_test_{c}.npz"))
    u = z["u"]; pos = {s: i for i, s in enumerate(u)}; idx = f["idx"]
    s1i_all = np.array([pos[s] for s in z["s1"]], np.int64); _, rec = np.unique(z["cand"], return_inverse=True)
    keep = {}
    for tag, p, th in (("S005", z["p5"].astype(float), .78), ("S006", z["p6"].astype(float), .72)):
        acc = p >= th; order = np.lexsort((s1i_all, -p, rec)); rs = rec[order]; first = np.r_[True, rs[1:] != rs[:-1]]
        win = np.zeros(len(p), bool); win[order[first]] = True; keep[tag] = (acc & win)[idx]
    D = dict(s1i=s1i_all[idx], p5=z["p5"][idx].astype(float), p6=z["p6"][idx].astype(float), pb=z["pb"][idx].astype(float),
             sel=z["sel"][idx], rrl=z["rrl"][idx], rrub=f["rrub"], dcos=f["dcos"], rank_dense=f["rank_dense"], rk=z["rk"][idx],
             k5=keep["S005"], k6=keep["S006"], th5=.78, th6=.72, n_s1=len(u))
    return D


TEST = {c: test_table(c) for c in ("France", "US", "India")}
P2 = {}
for model in ("S006", "S005"):
    rows = {}
    for c, D in TEST.items():
        keep = D["k6"] if model == "S006" else D["k5"]; p = D["p6"] if model == "S006" else D["p5"]; b = band_of(p)
        cc, xx = cells(D, model)
        rows[c] = {nm: round(float((keep & m).sum() / D["n_s1"]), 5) for nm, m in list(cc.items()) + list(xx.items())}
        rows[c]["_kept_per_s1"] = round(float(keep.sum() / D["n_s1"]), 4)
        rows[c]["_band_per_s1"] = {bn: round(float((keep & (b == bn)).sum() / D["n_s1"]), 5) for bn, _, _ in BANDS}
        # cell x band per S1 (for the reweighted estimates)
        rows[c]["_cell_band_per_s1"] = {nm: {bn: round(float((keep & m & (b == bn)).sum() / D["n_s1"]), 6) for bn, _, _ in BANDS}
                                        for nm, m in cc.items()}
    usi_s1 = TEST["US"]["n_s1"] + TEST["India"]["n_s1"]
    rows["USI"] = {nm: round((rows["US"][nm] * TEST["US"]["n_s1"] + rows["India"][nm] * TEST["India"]["n_s1"]) / usi_s1, 5)
                   for nm in rows["France"] if not nm.startswith("_")}
    rows["ratio_France_over_USI"] = {nm: (round(rows["France"][nm] / rows["USI"][nm], 2) if rows["USI"][nm] > 0 else None) for nm in rows["USI"]}
    P2[model] = rows
    print("\nTEST", model, "kept/S1", {c: rows[c]["_kept_per_s1"] for c in TEST})
    for nm in rows["USI"]:
        print(f"  {nm:34s} France {rows['France'][nm]:.5f}  USI {rows['USI'][nm]:.5f}  ratio {rows['ratio_France_over_USI'][nm]}")
OUT["part2"] = P2

# ---------------- Part 3: France FP captured estimates ----------------
# cell x band precision from V1 (+T pooled for stability); France composition from test.
P3 = {}
for model in ("S006", "S005"):
    est = {}
    for src in ("V1", "V1+T"):
        sets = ["V1"] if src == "V1" else ["V1", "T"]
        err = {}
        for nm in ["rr++", "rrL+ rrUb-", "rrL- rrUb+", "rr--", "not top10"]:
            for bn, _, _ in BANDS:
                n = sum(P1[f"{s}_{model}"]["cells"][nm]["by_band"][bn][0] for s in sets)
                fp = sum(P1[f"{s}_{model}"]["cells"][nm]["by_band"][bn][1] for s in sets)
                err[(nm, bn)] = ((fp + 0.5) / (n + 1.0), n, fp)      # Jeffreys-ish smoothing
        comp = P2[model]["France"]["_cell_band_per_s1"]; compU = P2[model]["USI"] if False else None
        fr_fp = sum(comp[nm][bn] * err[(nm, bn)][0] for (nm, bn) in err)
        usi = {nm: {bn: (P2[model]["US"]["_cell_band_per_s1"][nm][bn] * TEST["US"]["n_s1"] + P2[model]["India"]["_cell_band_per_s1"][nm][bn] * TEST["India"]["n_s1"])
                    / (TEST["US"]["n_s1"] + TEST["India"]["n_s1"]) for bn, _, _ in BANDS} for nm in comp}
        usi_fp = sum(usi[nm][bn] * err[(nm, bn)][0] for (nm, bn) in err)
        e = {"France_fp_per_s1_equal_cell_precision": round(fr_fp, 5), "USI_fp_per_s1_equal_cell_precision": round(usi_fp, 5)}
        # veto candidates (cells chosen on T in part 1): rr-- and any rr<=0 in bands; per-S1 France volume, equal-precision gain
        for vname, vcells, vbands in (("veto rr-- (all bands)", ["rr--"], [b for b, _, _ in BANDS]),
                                      ("veto rr-- & p<.95", ["rr--"], [".90-.95", "th-.90"]),
                                      ("veto any rr<=0 & p<.95", ["rr--", "rrL+ rrUb-", "rrL- rrUb+"], [".90-.95", "th-.90"]),
                                      ("veto any rr<=0 (all bands)", ["rr--", "rrL+ rrUb-", "rrL- rrUb+"], [b for b, _, _ in BANDS])):
            vol = sum(comp[nm][bn] for nm in vcells for bn in vbands)
            fpv = sum(comp[nm][bn] * err[(nm, bn)][0] for nm in vcells for bn in vbands)
            tpv = vol - fpv
            # (b) uniform error-odds multiplier k so that total France FP/S1 hits the count-prior budget B
            out_b = {}
            for B in (0.10, 0.18):
                lo, hi = 0.0, 1e4
                for _ in range(100):
                    k = (lo + hi) / 2
                    tot = sum(comp[nm][bn] * (k * err[(nm, bn)][0] / (1 - err[(nm, bn)][0] + k * err[(nm, bn)][0])) for (nm, bn) in err)
                    lo, hi = (k, hi) if tot < B else (lo, k)
                fpb = sum(comp[nm][bn] * (k * err[(nm, bn)][0] / (1 - err[(nm, bn)][0] + k * err[(nm, bn)][0])) for nm in vcells for bn in vbands)
                out_b[f"budget_{B}"] = {"odds_mult": round(k, 2), "fp_captured_per_s1": round(fpb, 5), "tp_lost_per_s1": round(vol - fpb, 5),
                                        "dF_France_pp": round(22.1 * fpb - 9.2 * (vol - fpb), 3), "dLB": round(0.14975 * (22.1 * fpb - 9.2 * (vol - fpb)) / 100, 5)}
            e[vname] = {"France_vol_per_s1": round(vol, 5), "a_equal_precision": {"fp_captured_per_s1": round(fpv, 5), "tp_lost_per_s1": round(tpv, 5),
                        "dF_France_pp": round(22.1 * fpv - 9.2 * tpv, 3), "dLB": round(0.14975 * (22.1 * fpv - 9.2 * tpv) / 100, 5)}, "b_uniform_odds": out_b}
        est[src] = e
    P3[model] = est
    print("\nPART3", model, json.dumps(est, indent=1))
OUT["part3"] = P3

# ---------------- Part 4: S005 -> S006 transitions ----------------
P4 = {}
def tcells(D, which):
    """cells for transitions: which='rem' (S005 kept, S006 dropped) uses p5 band; 'add' uses p6 band."""
    p = D["p5"] if which == "rem" else D["p6"]
    b = band_of(p); rr = np.where(~D["sel"], "ns", np.where(D["rrl"] > 0, "L+", "L-")); rr = np.char.add(rr.astype(str), np.where(~D["sel"], "", np.where(D["rrub"] > 0, "U+", "U-")))
    return np.char.add(np.char.add(b.astype(str), "|"), rr)


for t in ("V1", "T"):
    D = LAB[t]; bad = D["y"] == 0
    rem = D["k5"] & ~D["k6"]; add = D["k6"] & ~D["k5"]
    cr, ca = tcells(D, "rem"), tcells(D, "add")
    P4[t] = {"removed": int(rem.sum()), "removed_false": int((rem & bad).sum()), "added": int(add.sum()), "added_true": int((add & ~bad).sum()),
             "per_s1": [round(float(rem.sum() / len(D["n_gt"])), 5), round(float(add.sum() / len(D["n_gt"])), 5)],
             "rem_cells": {k: [int((rem & (cr == k)).sum()), int((rem & (cr == k) & bad).sum())] for k in np.unique(cr[rem])},
             "add_cells": {k: [int((add & (ca == k)).sum()), int((add & (ca == k) & ~bad).sum())] for k in np.unique(ca[add])},
             "exact_V1_macro_delta_pp_S006_minus_S005": None}
    n = len(D["n_gt"]); s = D["s1i"]; y = D["y"] == 1
    f = lambda k: f05_vec(np.bincount(s, (k & y).astype(float), n), np.bincount(s, k.astype(float), n), D["n_gt"])
    P4[t]["exact_macro_delta_pp_S006_minus_S005"] = boot_delta(f(D["k6"]) - f(D["k5"]), n=2000)
    print("\nPART4", t, {k: P4[t][k] for k in ("removed", "removed_false", "added", "added_true", "per_s1", "exact_macro_delta_pp_S006_minus_S005")})
for c, D in TEST.items():
    rem = D["k5"] & ~D["k6"]; add = D["k6"] & ~D["k5"]; cr, ca = tcells(D, "rem"), tcells(D, "add")
    r = {"removed_per_s1": round(float(rem.sum() / D["n_s1"]), 5), "added_per_s1": round(float(add.sum() / D["n_s1"]), 5),
         "rem_cells": {k: int((rem & (cr == k)).sum()) for k in np.unique(cr[rem])}, "add_cells": {k: int((add & (ca == k)).sum()) for k in np.unique(ca[add])}}
    # reweighted estimate: false-rate per cell from V1+T pooled; unseen cell -> pooled overall rate
    for src in (("V1",), ("V1", "T")):
        remF = sum(P4[t]["removed_false"] for t in src) / sum(P4[t]["removed"] for t in src)
        addT = sum(P4[t]["added_true"] for t in src) / sum(P4[t]["added"] for t in src)
        def rate(cells, kind, k, default):
            n_ = sum(P4[t][cells].get(k, [0, 0])[0] for t in src); x_ = sum(P4[t][cells].get(k, [0, 0])[1] for t in src)
            return (x_ + 2 * default) / (n_ + 2)                 # shrink to the pooled rate with 2 pseudo-counts
        fr_rem_false = sum(v * rate("rem_cells", "rem", k, remF) for k, v in r["rem_cells"].items()) / max(1, rem.sum())
        fr_add_true = sum(v * rate("add_cells", "add", k, addT) for k, v in r["add_cells"].items()) / max(1, add.sum())
        R, A = r["removed_per_s1"], r["added_per_s1"]
        naive = 22.1 * (R * remF - A * (1 - addT)) + 9.2 * (A * addT - R * (1 - remF))
        rew = 22.1 * (R * fr_rem_false - A * (1 - fr_add_true)) + 9.2 * (A * fr_add_true - R * (1 - fr_rem_false))
        r["+".join(src)] = {"pooled_rem_false": round(remF, 4), "pooled_add_true": round(addT, 4),
                            "reweighted_rem_false": round(fr_rem_false, 4), "reweighted_add_true": round(fr_add_true, 4),
                            "dF_pp_naive": round(naive, 3), "dF_pp_reweighted": round(rew, 3),
                            "dLB_reweighted_if_France": round(0.14975 * rew / 100, 5) if c == "France" else None}
    P4[c] = r
    print("PART4", c, json.dumps({k: v for k, v in r.items() if not k.endswith("cells")}))
OUT["part4"] = P4
json.dump(OUT, open(os.path.join(HERE, "C_analyze.json"), "w"), indent=1, default=str)
