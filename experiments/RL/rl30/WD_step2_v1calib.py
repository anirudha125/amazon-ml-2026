"""WD step 2: V1 (labelled) calibration of the count argument + F0.5 cost of the claimed France FN excess.
(a) V1 RL-27 NEW s42 @0.78 (+max-claimer within V1): mean predicted count vs GT count, TP/FP/FN per S1, p0/p1, macro F0.5, per country.
(b) Does test-US final count minus train-GT mean (-0.087) match the V1 net deficit?  (validates the comparison for US only)
(c) Simulated cost: remove extra TPs at random from V1 predictions so that the mean count drops by 0.079 (and 0.166-0.087 variants),
    report macro F0.5 delta (10 random seeds). Also the FN-vs-FP asymmetry: cost of +0.079 FP/S1 added at random.
(d) Bootstrap SE of the France and US mean final counts (259k / 663k S1).
Writes rl30/WD_step2_v1calib.json (NEW)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH
L = lambda *a: print(*a, flush=True)
R = {}
m = np.load(PATHS["v1_meta"]); p = np.load(PATHS["v1_p_new"])
s1_ids = m["s1_ids"]; ctry = m["country"]; n_gt = m["n_gt"].astype(int)
idx = m["s1idx"]; y = m["y"].astype(int); cand = m["cand"]
Tr = load("train", verbose=False)
gt = Tr["gt"]; gt = gt[gt.s1.isin(set(s1_ids))]
ngt2 = gt.groupby("s1").size().reindex(s1_ids, fill_value=0).values
L("n_gt check: meta n_gt == train gt count:", bool((ngt2 == n_gt).all()), "pool recall", y.sum() / n_gt.sum())
R["pool_recall_V1"] = float(y.sum() / n_gt.sum())
acc = p >= NEW_TH
# max-claimer within V1
d = pd.DataFrame(dict(i=idx[acc], rec=cand[acc], p=p[acc], y=y[acc]))
d["rk"] = d.groupby("rec").p.rank(ascending=False, method="first")
L("V1 multi-claimed accepted rows:", int((d.groupby("rec").i.transform("size") > 1).sum()))
d = d[d.rk == 1]
N = len(s1_ids)
npred = np.bincount(d.i, minlength=N); tp = np.bincount(d.i, weights=d.y, minlength=N).astype(int)
fp = npred - tp; fn = n_gt - tp


def f05(tp, npred, ngt):
    out = np.zeros(len(tp))
    e = ngt == 0
    out[e] = (npred[e] == 0).astype(float)
    ok = (~e) & (npred > 0) & (tp > 0)
    P = tp[ok] / npred[ok]; Rr = tp[ok] / ngt[ok]
    out[ok] = 1.25 * P * Rr / (0.25 * P + Rr)
    return out


base = f05(tp, npred, n_gt)
for cc in ("US", "India", "ALL"):
    mk = np.ones(N, bool) if cc == "ALL" else ctry == cc
    r = dict(n_s1=int(mk.sum()), gt_mean=float(n_gt[mk].mean()), pred_mean=float(npred[mk].mean()), net_deficit=float((n_gt[mk] - npred[mk]).mean()),
             FP_per_s1=float(fp[mk].mean()), FN_per_s1=float(fn[mk].mean()), FN_outside_pool_per_s1=float((n_gt[mk] - np.bincount(idx[y == 1], minlength=N)[mk]).mean()),
             p0_gt=float((n_gt[mk] == 0).mean()), p0_pred=float((npred[mk] == 0).mean()), p1_gt=float((n_gt[mk] == 1).mean()), p1_pred=float((npred[mk] == 1).mean()),
             macroF05=float(base[mk].mean()),
             loss_from_FN_only_S1=float((1 - base[mk])[(fp[mk] == 0) & (fn[mk] > 0)].sum() / mk.sum()),
             loss_from_FP_only_S1=float((1 - base[mk])[(fp[mk] > 0) & (fn[mk] == 0)].sum() / mk.sum()),
             loss_from_both=float((1 - base[mk])[(fp[mk] > 0) & (fn[mk] > 0)].sum() / mk.sum()))
    R[f"V1_{cc}"] = r
    L(cc, {k: round(v, 5) if isinstance(v, float) else v for k, v in r.items()})
# (b) test-US deficit vs train GT and V1 deficit
test_final = dict(France=3.293079, US=3.372248, India=3.368393)   # re-derived in WD_step1_raw.json from matching_results.tsv
gtm = dict(US=3.459057, India=3.464543)
for cc in ("US", "India"):
    R[f"test_{cc}_minus_trainGT"] = test_final[cc] - gtm[cc]
    L(cc, "test final - train GT mean =", round(test_final[cc] - gtm[cc], 4), " V1 pred - V1 GT =", round(-R[f'V1_{cc}']['net_deficit'], 4))
# (c) simulated extra FN: drop TPs uniformly at random until mean count falls by delta
rng = np.random.default_rng(0)
dd = d.reset_index(drop=True)
sims = {}
for cc in ("US", "India", "ALL"):
    mk = np.ones(N, bool) if cc == "ALL" else ctry == cc
    base_m = base[mk].mean()
    for delta in (0.031, 0.079, 0.166):
        res = []
        for sd in range(10):
            r_ = np.random.default_rng(sd)
            tp_rows = np.where((dd.y.values == 1) & mk[dd.i.values])[0]
            k = int(round(delta * mk.sum()))
            drop = r_.choice(tp_rows, size=k, replace=False)
            tp2 = tp - np.bincount(dd.i.values[drop], minlength=N); np2 = npred - np.bincount(dd.i.values[drop], minlength=N)
            res.append(base_m - f05(tp2, np2, n_gt)[mk].mean())
        sims[f"{cc}_extraFN_{delta}"] = dict(mean_loss_pts=round(100 * float(np.mean(res)), 4), sd=round(100 * float(np.std(res)), 4))
    # extra FP at random: add delta FP per S1 to random S1 (one each)
    for delta in (0.079,):
        res = []
        for sd in range(10):
            r_ = np.random.default_rng(100 + sd)
            ids = np.where(mk)[0]; k = int(round(delta * mk.sum()))
            add = r_.choice(ids, size=k, replace=False)
            np2 = npred.copy(); np2[add] += 1
            res.append(base_m - f05(tp, np2, n_gt)[mk].mean())
        sims[f"{cc}_extraFP_{delta}"] = dict(mean_loss_pts=round(100 * float(np.mean(res)), 4), sd=round(100 * float(np.std(res)), 4))
L(pd.DataFrame(sims).T.to_string())
R["sim"] = sims
# (d) bootstrap-free analytic SE of mean final counts from the submission
T = load("test", verbose=False)
sub = pd.read_csv(PATHS["s005_final"], sep="\t", dtype=str, keep_default_na=False)
sub["n"] = sub.matched_entity_ids.map(lambda s: 0 if s.strip() == "" else s.count(",") + 1)
sub["country"] = sub.source1_entity_id.map(T["s1"].set_index("id").country)
for cc in ("France", "US", "India"):
    n = sub.n[sub.country == cc]
    R[f"se_mean_{cc}"] = float(n.std() / np.sqrt(len(n)))
    L(cc, "mean", round(n.mean(), 4), "SE", round(R[f'se_mean_{cc}'], 5))
json.dump(R, open(os.path.join(OUT, "WD_step2_v1calib.json"), "w"), indent=1, default=str)
L("wrote WD_step2_v1calib.json")
