"""RL-30 VD part 2: is the France shape-test excess of the 'same name, same street, neighbouring number' class (RULE3 = +-1/+-2/odd 3-10)
explained by its NEW probability? p-matched controls on the test accepted pairs (label-free) + V1 check of the same p-matched logic with labels.
READ-ONLY. Writes rl30/VD_check2.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, NEW_TH, accepted
L = lambda *a: print(*a, flush=True)
pd.set_option("display.width", 250)
SHIFT = ["H_D1", "H_D2", "H_D3_10_ODD", "H_D3_10_EVEN", "H_DIGSUB", "H_GT10_ODD", "H_GT10_EVEN", "H_DIGINDEL", "H_PERM"]
RULE3 = ["H_D1", "H_D2", "H_D3_10_ODD"]
PB = np.array([0.78, 0.82, 0.86, 0.9, 0.93, 0.95, 0.97, 0.98, 0.99, 0.995, 0.999, 1.0001])
T = load("test"); R = {}


def make_refs(kk, ids):
    allk = pd.concat([kk.xs(s, level="src").reindex(ids, fill_value=0) for s in ("S2", "S3")]).values
    pk = np.bincount(allk) / len(allk); kv_ = np.arange(len(pk)); sb_ = kv_ * pk / (kv_ * pk).sum()
    return dict(mean=float((kv_ * sb_).sum()), p1=float(sb_[1])), dict(mean=float((kv_ * pk).sum() + 1), p1=float(pk[0]))


def shape(kk, sel, REP, ADD, w=None):
    g = sel[["s1", "src"]].copy(); g["w"] = 1.0 if w is None else w
    g = g.groupby(["s1", "src"]).w.mean()
    kv = kk.reindex(g.index).values; ww = g.values
    p1 = float(np.average(kv == 1, weights=ww)); neff = ww.sum() ** 2 / (ww ** 2).sum()
    return dict(n=int(len(kv)), neff=round(float(neff), 1), f_p1=round((REP["p1"] - p1) / (REP["p1"] - ADD["p1"]), 3),
                se=round(float(np.sqrt(p1 * (1 - p1) / neff) / (REP["p1"] - ADD["p1"])), 3))


def pmatched_weights(target_p, pool_p):
    bt = np.clip(np.digitize(target_p, PB) - 1, 0, len(PB) - 2); bp = np.clip(np.digitize(pool_p, PB) - 1, 0, len(PB) - 2)
    ht = np.bincount(bt, minlength=len(PB) - 1) / len(bt); hp = np.bincount(bp, minlength=len(PB) - 1) / len(bp)
    return np.where(hp[bp] > 0, ht[bp] / np.maximum(hp[bp], 1e-12), 0.0)


for tag, cc, fn in (("S005_France", "France", "D_enriched_FR5.pkl"), ("S005_US", "US", "D_enriched_US5.pkl"), ("S005_India", "India", "D_enriched_IN5.pkl")):
    a = accepted(tag); a = a[a.kept_final]
    kk = a.assign(src=a.rec.str[:2]).groupby(["s1", "src"]).size()
    REP, ADD = make_refs(kk, T["s1"].id[T["s1"].country == cc])
    d = pd.read_pickle(os.path.join(OUT, fn)); d = d[d.kept_final].assign(src=lambda x: x.rec.str[:2])
    samest = d.st.isin(["S_SAME", "S_TYPO"])
    fl = (d.nt == "N_SAME") & samest & d.ht.isin(SHIFT)
    r3 = fl & d.ht.isin(RULE3)
    tgt = d[r3]
    out = {"RULE3": shape(kk, tgt, REP, ADD)}
    controls = {"all_nonflagged": d[~fl], "N_SAME_same_num_street": d[(d.nt == "N_SAME") & samest & (d.ht == "H_SAME")],
                "N_SAME_exact_addr": d[(d.nt == "N_SAME") & d.same_akey], "any_type_not_numshift": d[~d.hnum_shift],
                "N_SAME_GT10_or_DIGINDEL_samestreet": d[fl & d.ht.isin(["H_GT10_ODD", "H_GT10_EVEN", "H_DIGINDEL"])]}
    for nm, c in controls.items():
        out[f"{nm}_raw"] = shape(kk, c, REP, ADD)
        out[f"{nm}_pmatched"] = shape(kk, c, REP, ADD, w=pmatched_weights(tgt.p.values, c.p.values))
    # by p band: RULE3 vs non-flagged in the same band
    for lo, hi in ((0.78, 0.9), (0.9, 0.98), (0.98, 1.0001)):
        t_ = tgt[(tgt.p >= lo) & (tgt.p < hi)]; c_ = d[~fl & (d.p >= lo) & (d.p < hi)]
        if len(t_) >= 30:
            out[f"band[{lo},{hi})_RULE3"] = shape(kk, t_, REP, ADD); out[f"band[{lo},{hi})_nonflagged"] = shape(kk, c_, REP, ADD)
    R[cc] = out
    L(f"\n== {cc}  (RULE3 n_pairs={len(tgt)}, p median {tgt.p.median():.4f})"); L(pd.DataFrame(out).T.to_string())

# Labelled analogue on V1: FP rate of RULE3 accepted vs p-matched non-flagged accepted (US, India)
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
fl = (v.nt == "N_SAME") & v.st.isin(["S_SAME", "S_TYPO"]) & v.ht.isin(SHIFT)
V = {}
for cc in ("US", "India"):
    acc = v[(v.country == cc) & (v.p >= NEW_TH)]
    fla = fl[acc.index]
    t_ = acc[fla & acc.ht.isin(RULE3)]; c_ = acc[~fla]
    w = pmatched_weights(t_.p.values, c_.p.values)
    V[cc] = dict(RULE3_n=len(t_), RULE3_FP_rate=round(float((t_.y == 0).mean()), 4), RULE3_expected_FP_from_p=round(float((1 - t_.p).sum()), 2),
                 RULE3_FP=int((t_.y == 0).sum()),
                 control_pmatched_FP_rate=round(float(np.average(c_.y == 0, weights=w)), 4), control_raw_FP_rate=round(float((c_.y == 0).mean()), 4))
L("\nV1 accepted RULE3 vs p-matched non-flagged:", json.dumps(V, indent=1)); R["V1_pmatched_FP"] = V
json.dump(R, open(os.path.join(OUT, "VD_check2.json"), "w"), indent=1, default=str)
L("wrote VD_check2.json")
