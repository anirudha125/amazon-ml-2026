"""RL-32: LOCO US->India errors split by candidate-name script (Latin vs Indic). France is Latin-script, so the Latin part is the
relevant France analogue. Compares LOCO_US (G_p_LOCO_US_V1.npy, th from results) with the in-domain S006 model (p_RRL) on V1 India."""
import sys, os, json, re, numpy as np
sys.path.insert(0, "../../GAP"); sys.path.insert(0, "../rl31"); sys.path.insert(0, "../../../src")
import gap_lib as G, rl31_lib as L
from translit import has_indic
V = L.load_v1(); tr = G.records("train")
res = json.load(open("G_loco_s2_results.json")); th = res["LOCO_US"]["th_oof"]
p_loco = np.load("G_p_LOCO_US_V1.npy"); p_in = V["p_RRL"]
cty = V["country"][V["s1idx"]]; ind = cty == "India"
indic = np.array([has_indic(tr[c][0]) for c in V["cand"]])
out = {}
for nm, p, t in (("LOCO_US", p_loco, th), ("S006_indomain", p_in, 0.72)):
    acc = p >= t; y = V["y"] == 1
    for sc, m in (("latin", ind & ~indic), ("indic", ind & indic)):
        out[f"{nm}_{sc}"] = dict(pairs=int(m.sum()), pos=int((y & m).sum()), fp=int((acc & ~y & m).sum()), fn_inpool=int((~acc & y & m).sum()),
                                 tp=int((acc & y & m).sum()))
    # India S1 whose candidate pools contain no Indic-name positives: macro on that subset
    s1_indic_pos = np.bincount(V["s1idx"], weights=(indic & y), minlength=len(V["s1_ids"])) > 0
    f = L.macro(V, acc); c1 = V["country"]
    out[f"{nm}_macro_India_S1_without_indic_positive"] = round(float(f[(c1 == "India") & ~s1_indic_pos].mean() * 100), 3)
    out[f"{nm}_macro_India_S1_with_indic_positive"] = round(float(f[(c1 == "India") & s1_indic_pos].mean() * 100), 3)
    out[f"{nm}_n_S1_with_indic_positive"] = int(((c1 == "India") & s1_indic_pos).sum())
json.dump(out, open("G_loco_script.json", "w"), indent=1); print(json.dumps(out, indent=1))
