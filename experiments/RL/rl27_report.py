"""RL-27 report -- NEW (D2b + record-centric features) vs BASE (D2b refit, parity-checked) on V1 / V0.
Per seed and seed-averaged: macro F0.5 (OOF-protocol thresholds from rl27_arm_results.json), paired bootstrap (harness),
TP / FP / FN, pair transitions, empty-address slice, US / India, and which RL-21 error categories moved. Read-only."""
import os, sys, json
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "src")); sys.path.insert(0, HERE)
import harness as H
import e023_stage2 as S2
from rl_data import load

R = json.load(open(os.path.join(HERE, "rl27_arm_results.json")))
seeds = sorted({int(k.split("_s")[1]) for k in R if k.startswith("NEW_s") and f"BASE_s{k.split('_s')[1]}" in R})
D = load("train", verbose=False)
recs = pd.concat([D["s2"], D["s3"]]).set_index("id")
out = {"seeds": seeds}
for vn in ("V1", "V0"):
    V = S2.load_set(vn, "a50n10d10a"); V["country_s1"] = V["country"]
    empty = recs.addr.reindex(V["cand"]).str.strip().eq("").values
    cat = None
    if vn == "V1":
        L = pd.read_pickle(os.path.join(HERE, "cache", "rl21_links_V1.pkl"))
        key = pd.Series(V["s1_ids"][V["s1idx"]]).str.cat(pd.Series(V["cand"]), sep="|")
        cmap = dict(zip(L.s1 + "|" + L.rec, L.cat)); cat = key.map(cmap).values
    fb_all, fn_all = [], []
    res = {}
    for s in seeds:
        pb = np.load(os.path.join(HERE, "cache", f"rl27_p_BASE_{vn}_s{s}.npy")); pn = np.load(os.path.join(HERE, "cache", f"rl27_p_NEW_{vn}_s{s}.npy"))
        tb, tn = R[f"BASE_s{s}"]["th_oof"], R[f"NEW_s{s}"]["th_oof"]
        sb, sn = S2.summarize(V, pb, tb), S2.summarize(V, pn, tn)
        d, lo, hi, pneg = H.paired_bootstrap(sb["scores"], sn["scores"])
        ab, an = pb >= tb, pn >= tn; y = V["y"] == 1
        trans = dict(new_TP=int((an & ~ab & y).sum()), lost_TP=int((ab & ~an & y).sum()),
                     new_FP=int((an & ~ab & ~y).sum()), removed_FP=int((ab & ~an & ~y).sum()))
        ng = int(V["n_gt"].sum())
        slice_ = dict(BASE=dict(TP=int((ab & y & empty).sum()), FP=int((ab & ~y & empty).sum())),
                      NEW=dict(TP=int((an & y & empty).sum()), FP=int((an & ~y & empty).sum())), in_pool_true=int((y & empty).sum()))
        r = dict(th=[tb, tn], BASE=round(sb["macro"] * 100, 3), NEW=round(sn["macro"] * 100, 3), delta_pp=round(d * 100, 3),
                 ci95=[round(lo * 100, 3), round(hi * 100, 3)], p_delta_le0=round(pneg, 4),
                 US=[round(sb["us"] * 100, 3), round(sn["us"] * 100, 3)], India=[round(sb["india"] * 100, 3), round(sn["india"] * 100, 3)],
                 TP=[sb["tp"], sn["tp"]], FP=[sb["fp"], sn["fp"]], FN=[ng - sb["tp"], ng - sn["tp"]],
                 singleton_fp_s1=[sb["sing_fp_entities"], sn["sing_fp_entities"]], transitions=trans, empty_address_slice=slice_)
        if cat is not None:
            ch = pd.DataFrame({"cat": cat, "b": ab & y, "n": an & y})
            ch = ch[ch.cat.notna()]
            r["lost_link_categories_fixed_minus_broken"] = {c: int((g.n & ~g.b).sum() - (g.b & ~g.n).sum()) for c, g in ch.groupby("cat")}
        res[f"s{s}"] = r; fb_all.append(sb["scores"]); fn_all.append(sn["scores"])
    if len(seeds) > 1:
        d, lo, hi, pneg = H.paired_bootstrap(np.mean(fb_all, 0), np.mean(fn_all, 0))
        res["seedavg"] = dict(BASE=round(np.mean(fb_all) * 100, 3), NEW=round(np.mean(fn_all) * 100, 3), delta_pp=round(d * 100, 3),
                              ci95=[round(lo * 100, 3), round(hi * 100, 3)])
    out[vn] = res
if "NEW_s42" in R:
    out["feature_gain_rl27_s42"] = R["NEW_s42"].get("feature_importance_gain_rl27"); out["gain_share_rl27_s42"] = R["NEW_s42"].get("gain_share_rl27")
out["runtime_base_s"] = R.get("runtime_base_s"); out["fit_s"] = {k: v.get("fit_s") for k, v in R.items() if isinstance(v, dict) and "fit_s" in v}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "rl27_report.json"), "w"), indent=1, default=float)
