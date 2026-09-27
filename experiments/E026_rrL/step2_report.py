"""E026 gate-2 report (read-only): RRL (RL-27 NEW + rrL column) vs CTRL (RL-27 NEW refit, same run), seed 42, OOF thresholds.
Paired bootstrap over S1 (harness.paired_bootstrap, 10k resamples), US / India, TP / FP / FN, pair transitions, empty-address slice,
singleton-FP S1; references: PROD (S002 model, experiments/E023/f_PROD_V*.npy) and the stored RL-27 NEW predictions (parity).
Writes experiments/E026_rrL/step2_report.json."""
import os, sys, json
import numpy as np, pandas as pd
from boot import H, S2, HERE, ROOT

RL = os.path.join(ROOT, "experiments", "RL"); C = os.path.join(HERE, "cache"); GATE = 0.10
import pickle

R = json.load(open(os.path.join(HERE, "step2_results.json")))
D = pickle.load(open(os.path.join(RL, "cache", "train.pkl"), "rb"))     # RL session's train-table cache (read-only)
recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
out = {"gate_pp": GATE}
for vn in ("V1", "V0"):
    V = S2.load_set(vn, "a50n10d10a"); V["country_s1"] = V["country"]
    empty = recs.addr.reindex(V["cand"]).str.strip().eq("").values
    pc, pr = np.load(os.path.join(C, f"p_CTRL_{vn}_s42.npy")), np.load(os.path.join(C, f"p_RRL_{vn}_s42.npy"))
    tc, tr = R["CTRL_s42"]["th_oof"], R["RRL_s42"]["th_oof"]
    sc, sr = S2.summarize(V, pc, tc), S2.summarize(V, pr, tr)
    d, lo, hi, pneg = H.paired_bootstrap(sc["scores"], sr["scores"])
    ac, ar = pc >= tc, pr >= tr; y = V["y"] == 1; ng = int(V["n_gt"].sum())
    res = dict(th=[tc, tr], CTRL=round(sc["macro"] * 100, 3), RRL=round(sr["macro"] * 100, 3), delta_pp=round(d * 100, 3),
               ci95=[round(lo * 100, 3), round(hi * 100, 3)], p_delta_le0=round(pneg, 4),
               US=[round(sc["us"] * 100, 3), round(sr["us"] * 100, 3)], India=[round(sc["india"] * 100, 3), round(sr["india"] * 100, 3)],
               TP=[sc["tp"], sr["tp"]], FP=[sc["fp"], sr["fp"]], FN=[ng - sc["tp"], ng - sr["tp"]],
               precision=[round(sc["precision"] * 100, 3), round(sr["precision"] * 100, 3)],
               singleton_fp_s1=[sc["sing_fp_entities"], sr["sing_fp_entities"]],
               transitions=dict(new_TP=int((ar & ~ac & y).sum()), lost_TP=int((ac & ~ar & y).sum()),
                                new_FP=int((ar & ~ac & ~y).sum()), removed_FP=int((ac & ~ar & ~y).sum())),
               empty_address_slice=dict(CTRL=dict(TP=int((ac & y & empty).sum()), FP=int((ac & ~y & empty).sum())),
                                        RRL=dict(TP=int((ar & y & empty).sum()), FP=int((ar & ~y & empty).sum())),
                                        in_pool_true=int((y & empty).sum())))
    if vn == "V1":      # RL-21 categories of D2b's lost links (read-only), as in experiments/RL/rl27_report.py
        Lk = pd.read_pickle(os.path.join(RL, "cache", "rl21_links_V1.pkl"))
        key = pd.Series(V["s1_ids"][V["s1idx"]]).str.cat(pd.Series(V["cand"]), sep="|")
        cat = key.map(dict(zip(Lk.s1 + "|" + Lk.rec, Lk.cat))).values
        ch = pd.DataFrame({"cat": cat, "c": ac & y, "r": ar & y}); ch = ch[ch.cat.notna()]
        res["lost_link_categories_fixed_minus_broken"] = {k: int((g.r & ~g.c).sum() - (g.c & ~g.r).sum()) for k, g in ch.groupby("cat")}
    c = V["country_s1"]
    for ctry in ("US", "India"):
        m = c == ctry; dd, l2, h2, _ = H.paired_bootstrap(sc["scores"][m], sr["scores"][m])
        res[f"delta_{ctry}"] = [round(dd * 100, 3), round(l2 * 100, 3), round(h2 * 100, 3)]
    fprod = np.load(os.path.join(ROOT, "experiments", "E023", f"f_PROD_{vn}.npy"))
    for nm, s in (("CTRL", sc), ("RRL", sr)):
        dd, l2, h2, _ = H.paired_bootstrap(fprod, s["scores"]); res[f"{nm}_vs_PROD"] = [round(dd * 100, 3), round(l2 * 100, 3), round(h2 * 100, 3)]
    res["PROD"] = round(float(fprod.mean()) * 100, 3)
    pst = np.load(os.path.join(RL, "cache", f"rl27_p_NEW_{vn}_s42.npy")); sst = S2.summarize(V, pst, 0.78)
    dd, l2, h2, _ = H.paired_bootstrap(sst["scores"], sc["scores"])
    res["parity_CTRL_vs_stored_RL27_NEW"] = dict(stored=round(sst["macro"] * 100, 3), max_abs_dp=round(float(np.abs(pst - pc).max()), 4),
                                                 mean_abs_dp=round(float(np.abs(pst - pc).mean()), 6), delta_pp=[round(dd * 100, 3), round(l2 * 100, 3), round(h2 * 100, 3)])
    dd, l2, h2, _ = H.paired_bootstrap(sst["scores"], sr["scores"])
    res["RRL_vs_stored_RL27_NEW"] = [round(dd * 100, 3), round(l2 * 100, 3), round(h2 * 100, 3)]
    out[vn] = res
out["gate_pass"] = bool(out["V1"]["delta_pp"] >= GATE * 1.0 - 1e-9)
out["gain_share"] = {k: {kk: v[kk] for kk in v if kk.startswith("gain_share")} for k, v in R.items() if isinstance(v, dict) and "th_oof" in v}
out["fit_s"] = {k: v.get("fit_s") for k, v in R.items() if isinstance(v, dict) and "fit_s" in v}
out["rrUb_nan_share_top10"] = R.get("rrUb_nan_share_top10")
for f in ("model_rrL/train_info.json", "cache/score_info.json", "cache/base_info.json"):
    p = os.path.join(HERE, f)
    if os.path.exists(p):
        out[os.path.basename(f).replace(".json", "")] = json.load(open(p))
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "step2_report.json"), "w"), indent=1, default=float)
