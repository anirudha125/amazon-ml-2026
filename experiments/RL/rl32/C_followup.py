"""RL-32 C follow-up (reads C_lab_*.npz, C_analyze.json and the test tables via C_analyze helpers; writes C_followup.json).
 (1) V1/T exact macro delta of band-restricted disagreement vetoes + matched-volume p-threshold comparison (does disagreement add
     information beyond p?).
 (2) scenario (c): France cell x band error odds = V1(+T) odds x exp(count-matched logit shift) (S005 1.11, S006 0.52).
 (3) >=.99 band: low-reranker tail (rr_min below the V1 FP median) -- V1 precision there, France vs US/India frequency.
 (4) S005->S006 France transitions under odds multipliers 1 / exp(.52) / exp(1.11).
"""
import os, sys, json
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); EXP = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(EXP, "RL", "rl31"))
from rl31_lib import f05_vec, boot_delta
OUT = {}
A = json.load(open(os.path.join(HERE, "C_analyze.json")))
LAB = {t: dict(np.load(os.path.join(HERE, f"C_lab_{t}.npz"), allow_pickle=True)) for t in ("V1", "T")}
BANDS = [(".99-1", .99, 1.01), (".95-.99", .95, .99), (".90-.95", .90, .95), ("th-.90", 0.0, .90)]


def f_s1(D, keep):
    n = len(D["n_gt"]); s = D["s1i"]; y = D["y"] == 1
    return f05_vec(np.bincount(s, (keep & y).astype(float), n), np.bincount(s, keep.astype(float), n), D["n_gt"])


# (1) vetoes vs matched-volume threshold raise
R1 = {}
for t, D in LAB.items():
    for model in ("S006", "S005"):
        keep = D["k6"] if model == "S006" else D["k5"]; p = D["p6"] if model == "S006" else D["p5"]
        po, tho = (D["p5"], float(D["th5"])) if model == "S006" else (D["p6"], float(D["th6"]))
        anyneg = D["sel"] & ((D["rrl"] <= 0) | (D["rrub"] <= 0)); both = D["sel"] & (D["rrl"] <= 0) & (D["rrub"] <= 0)
        lneg = D["sel"] & (D["rrl"] <= 0); dmiss = D["rank_dense"] > 10
        rules = {"any rr<=0 & p<.95": anyneg & (p < .95), "rr-- & p<.95": both & (p < .95), "rrL<=0 & p<.95": lneg & (p < .95),
                 "any rr<=0 & p<.90": anyneg & (p < .90), "dense miss & p<.95": dmiss & (p < .95),
                 "other rejects & any rr<=0": anyneg & (po < tho), ">=.99 & rr_min<=0": anyneg & (p >= .99)}
        base = f_s1(D, keep); r = {}
        for nm, v in rules.items():
            m = keep & v; nv = int(m.sum()); fp = int((m & (D["y"] == 0)).sum())
            d = boot_delta(f_s1(D, keep & ~m) - base, n=2000)
            # matched-volume threshold raise: drop the nv lowest-p kept pairs
            kix = np.flatnonzero(keep); low = kix[np.argsort(p[kix], kind="stable")[:nv]]; mt = np.zeros(len(p), bool); mt[low] = True
            dt = boot_delta(f_s1(D, keep & ~mt) - base, n=2000)
            # same-band random-equivalent: expected FP if the veto had the precision of its own p-band (kept pairs)
            exp_fp = 0.0
            for bn, lo, hi in BANDS:
                bm = keep & (p >= lo) & (p < hi); vb = m & (p >= lo) & (p < hi)
                if vb.any():
                    exp_fp += vb.sum() * (D["y"][bm] == 0).mean()
            r[nm] = {"n": nv, "fp": fp, "prec": round(1 - fp / max(1, nv), 4), "fp_expected_at_band_precision": round(float(exp_fp), 1),
                     "veto_dF_pp": d, "matched_threshold_fp": int((mt & (D["y"] == 0)).sum()), "matched_threshold_dF_pp": dt}
        R1[f"{t}_{model}"] = r
        print(t, model); [print(f"   {k:28s} {v}") for k, v in r.items()]
OUT["vetoes_vs_threshold"] = R1

# test tables (same construction as C_analyze.test_table)
def test_table(c):
    z = np.load(os.path.join(EXP, "RL", "rl31", "test_scores", f"{c}.npz")); f = np.load(os.path.join(HERE, f"C_test_{c}.npz"))
    u = z["u"]; pos = {s: i for i, s in enumerate(u)}; idx = f["idx"]
    s1i_all = np.array([pos[s] for s in z["s1"]], np.int64); _, rec = np.unique(z["cand"], return_inverse=True); keep = {}
    for tag, p, th in (("S005", z["p5"].astype(float), .78), ("S006", z["p6"].astype(float), .72)):
        acc = p >= th; order = np.lexsort((s1i_all, -p, rec)); rs = rec[order]; first = np.r_[True, rs[1:] != rs[:-1]]
        win = np.zeros(len(p), bool); win[order[first]] = True; keep[tag] = (acc & win)[idx]
    return dict(p5=z["p5"][idx].astype(float), p6=z["p6"][idx].astype(float), sel=z["sel"][idx], rrl=z["rrl"][idx], rrub=f["rrub"],
                rank_dense=f["rank_dense"], k5=keep["S005"], k6=keep["S006"], n_s1=len(u))
TEST = {c: test_table(c) for c in ("France", "US", "India")}

# (2) scenario (c): count-matched odds multiplier on V1+T cell x band error odds
CELLS = ["rr++", "rrL+ rrUb-", "rrL- rrUb+", "rr--", "not top10"]
R2 = {}
for model, shift in (("S006", 0.52), ("S005", 1.11)):
    k = float(np.exp(shift)); comp = A["part2"][model]["France"]["_cell_band_per_s1"]; r = {"odds_mult": round(k, 3)}
    err = {}
    for nm in CELLS:
        for bn, _, _ in BANDS:
            n = sum(A["part1"][f"{s}_{model}"]["cells"][nm]["by_band"][bn][0] for s in ("V1", "T"))
            fp = sum(A["part1"][f"{s}_{model}"]["cells"][nm]["by_band"][bn][1] for s in ("V1", "T"))
            e = (fp + .5) / (n + 1); o = k * e / (1 - e); err[(nm, bn)] = o / (1 + o)
    r["France_fp_per_s1"] = round(sum(comp[nm][bn] * err[(nm, bn)] for (nm, bn) in err), 5)
    for vname, vc, vb in (("any rr<=0 & p<.95", CELLS[1:4], [".90-.95", "th-.90"]), ("rr-- & p<.95", ["rr--"], [".90-.95", "th-.90"]),
                          ("any rr<=0 (all)", CELLS[1:4], [b for b, _, _ in BANDS]),
                          ("threshold raise to .95 (all cells)", CELLS, [".90-.95", "th-.90"])):
        vol = sum(comp[nm][bn] for nm in vc for bn in vb); fpv = sum(comp[nm][bn] * err[(nm, bn)] for nm in vc for bn in vb)
        dF = 22.1 * fpv - 9.2 * (vol - fpv)
        r[vname] = {"vol_per_s1": round(vol, 5), "fp_captured_per_s1": round(fpv, 5), "tp_lost_per_s1": round(vol - fpv, 5),
                    "dF_France_pp": round(dF, 3), "dLB": round(0.14975 * dF / 100, 5)}
    R2[model] = r; print("scenario c", model, json.dumps(r))
OUT["scenario_c_count_shift"] = R2

# (3) >=.99 band low-reranker tail
R3 = {}
for model in ("S006", "S005"):
    D = LAB["V1"]; keep = D["k6"] if model == "S006" else D["k5"]; p = D["p6"] if model == "S006" else D["p5"]
    hi = keep & (p >= .99) & D["sel"]; rmin = np.fmin(D["rrl"], D["rrub"]).astype(float)
    fpm = hi & (D["y"] == 0); q = float(np.median(rmin[fpm])); q25 = float(np.percentile(rmin[fpm], 75))
    r = {"V1_fp_in_band": int(fpm.sum()), "V1_rrmin_fp_median": round(q, 3), "V1_rrmin_tp_p05": round(float(np.percentile(rmin[hi & (D['y'] == 1)], 5)), 3)}
    for nm, cut in (("rr_min<fp_median", q), ("rr_min<fp_p75", q25)):
        tail = hi & (rmin < cut)
        r[nm] = {"cut": round(cut, 3), "V1_n": int(tail.sum()), "V1_fp": int((tail & (D["y"] == 0)).sum()),
                 "V1_per_s1": round(float(tail.sum() / len(D["n_gt"])), 5)}
        T = LAB["T"]; kT = T["k6"] if model == "S006" else T["k5"]; pT = T["p6"] if model == "S006" else T["p5"]
        tT = kT & (pT >= .99) & T["sel"] & (np.fmin(T["rrl"], T["rrub"]) < cut)
        r[nm]["T_n"] = int(tT.sum()); r[nm]["T_fp"] = int((tT & (T["y"] == 0)).sum()); r[nm]["T_per_s1"] = round(float(tT.sum() / len(T["n_gt"])), 5)
        for c, Z in TEST.items():
            kz = Z["k6"] if model == "S006" else Z["k5"]; pz = Z["p6"] if model == "S006" else Z["p5"]
            r[nm][f"{c}_per_s1"] = round(float((kz & (pz >= .99) & Z["sel"] & (np.fmin(Z["rrl"], Z["rrub"]) < cut)).sum() / Z["n_s1"]), 5)
    R3[model] = r; print(">=.99 tail", model, json.dumps(r))
OUT["band99_tail"] = R3

# (4) transitions under odds multipliers (removal false-odds x k, addition true-odds / k)
R4 = {}
P4 = A["part4"]; Fr = P4["France"]["V1+T"]; R, Ad = P4["France"]["removed_per_s1"], P4["France"]["added_per_s1"]
for nm, k in (("k=1 (V1-like precision)", 1.0), ("k=exp(.52)", float(np.exp(.52))), ("k=exp(1.11)", float(np.exp(1.11)))):
    of = Fr["reweighted_rem_false"] / (1 - Fr["reweighted_rem_false"]) * k; rf = of / (1 + of)
    ot = Fr["reweighted_add_true"] / (1 - Fr["reweighted_add_true"]) / k; at = ot / (1 + ot)
    dF = 22.1 * (R * rf - Ad * (1 - at)) + 9.2 * (Ad * at - R * (1 - rf))
    R4[nm] = {"rem_false": round(rf, 3), "add_true": round(at, 3), "dF_France_pp": round(dF, 3), "dLB_France_part": round(0.14975 * dF / 100, 5)}
print("transitions", json.dumps(R4, indent=1))
OUT["transitions_odds"] = R4
json.dump(OUT, open(os.path.join(HERE, "C_followup.json"), "w"), indent=1)
