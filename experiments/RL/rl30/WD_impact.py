"""RL-30 adversarial verification (WD), part 3: per-class ORACLE upper bound on France macro F0.5 (ESTIMATED: the S1's other
accepted pairs assumed correct, no FN) and a label-free parity-mixture estimate of the decoy share of France accepted
3-10 / >10 shift pairs, using the train US/India generator parities from WD_pop.json. READ-ONLY. Writes rl30/WD_impact.json."""
import sys, os, json, re
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rapidfuzz import fuzz
from rl30_lib import load, OUT, accepted, fold, LEGAL, HONOR, STOP, STREET_TYPE, UNIT
src = open(os.path.join(OUT, "WD_verify.py")).read()
exec(src[src.index("_DOT = "):src.index("# =========================================================== TEST")])
R = {}
T = load("test")
s1df = T["s1"]; fr_ids = s1df.id[s1df.country == "France"]
rec = pd.concat([T["s2"][T["s2"].country == "France"], T["s3"][T["s3"].country == "France"]])
name_of = dict(zip(s1df.id, s1df.name)); name_of.update(zip(rec.id, rec.name))
addr_of = dict(zip(s1df.id, s1df.addr)); addr_of.update(zip(rec.id, rec.addr))
P = Parser(name_of, addr_of)
a = accepted("S005_France"); fk = a[a.kept_final].reset_index(drop=True)
sc, cls, nlen = classify(P, fk.s1.values, fk.rec.values)
fk["cls"] = cls
ntot = fk.groupby("s1").size(); nfr = len(fr_ids); nall = len(s1df)
out = {}
for c in CLASSES:
    m = fk[fk.cls == c].groupby("s1").size()
    gain = sum(1.0 - f05(ntot[s] - k, k, 0) for s, k in m.items())
    loss = sum(1.0 - f05(ntot[s] - k, 0, k) for s, k in m.items())
    out[c] = dict(n_pairs=int(m.sum()), n_s1=int(len(m)), oracle_all_decoy_France_pts=round(100 * gain / nfr, 4),
                  oracle_all_decoy_LB_pts=round(100 * gain / nall, 4), loss_if_all_true_dropped_France_pts=round(100 * loss / nfr, 4),
                  gain_per_pair=round(gain / max(1, m.sum()), 4))
R["France_oracle_per_class"] = out
print(pd.DataFrame(out).T.to_string())
# parity mixture (label-free for France; parities from train labels)
W = json.load(open(os.path.join(OUT, "WD_pop.json")))["population"]
def cnt(tab, c, keys):
    return sum(tab.get(c, {}).get(k, 0) for k in keys)
pm = {}
for cc in ("US", "India"):
    tab = W[f"train_{cc}"]["table"]
    for grp, (o, e) in (("3_10", ("odd3_10", "even3_10")), ("gt10", ("odd_gt10", "even_gt10"))):
        t_o, t_e = cnt(tab, o, ["true"]), cnt(tab, e, ["true"])
        d_o, d_e = cnt(tab, o, ["distractor", "other_s1"]), cnt(tab, e, ["distractor", "other_s1"])
        pm[f"{cc}_{grp}"] = dict(true_odd_share=round(t_o / (t_o + t_e), 4), nontrue_odd_share=round(d_o / (d_o + d_e), 4))
ft = W["test_France"]["table"]
est = {}
for grp, (o, e) in (("3_10", ("odd3_10", "even3_10")), ("gt10", ("odd_gt10", "even_gt10"))):
    A_o, A_e = ft[o]["A"], ft[e]["A"]; s = A_o / (A_o + A_e)
    for cc in ("US", "India"):
        q = pm[f"{cc}_{grp}"]; x = (s - q["true_odd_share"]) / (q["nontrue_odd_share"] - q["true_odd_share"])
        est[f"{grp}_parities_from_{cc}"] = dict(France_accepted_odd_share=round(s, 4), decoy_share_ESTIMATED=round(float(np.clip(x, 0, 1)), 3),
                                               decoy_pairs_ESTIMATED=int(round(float(np.clip(x, 0, 1)) * (A_o + A_e))), n_accepted=int(A_o + A_e))
R["train_parities"] = pm; R["France_parity_mixture_ESTIMATED"] = est
print(json.dumps(pm, indent=1)); print(json.dumps(est, indent=1))
json.dump(R, open(os.path.join(OUT, "WD_impact.json"), "w"), indent=1, default=str)
print("wrote WD_impact.json")
