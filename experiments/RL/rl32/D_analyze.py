"""RL-32 D -- analysis of the LOCO arms on V1 (labelled). Output: D_analyze.json
For each model x V1 country: F0.5 at th .72 / th_dom (argmax on the TRAINING country's V1 rows) / oracle th*; mass per S1 vs true
in-pool count; label-free count-matched logit shift delta (target = training rows' in-pool true count per S1, no held-out labels);
F at th' = expit(logit(th)+delta); FP/FN links per S1; p-band mass and precision.
"""
import os, sys, json
import numpy as np
from scipy.special import expit, logit
from scipy.optimize import brentq
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, os.path.join(os.path.dirname(HERE), "rl31"))
from rl31_lib import load_v1, f05_vec, boot_delta
SCR = "/tmp/claude-1000/-teamspace-studios-this-studio/ceb7ed10-c913-40c4-9d44-c5d0057e9046/scratchpad/D"
V = load_v1(); s = V["s1idx"]; y = V["y"].astype(float); n = len(V["s1_ids"]); cty = V["country"]; ngt = V["n_gt"].astype(float)
inpool = np.bincount(s, weights=y, minlength=n)
M = np.load(os.path.join(SCR, "Tmeta.npz")); Ts, Ty, Tc = M["s1idx"], M["y"].astype(float), M["country"]
Tin = np.bincount(Ts, weights=Ty, minlength=len(Tc))
TGT = {"US": float(Tin[Tc == "US"].mean()), "India": float(Tin[Tc == "India"].mean()), "ALL": float(Tin.mean())}
GRID = np.round(np.arange(0.30, 0.955, 0.01), 2)


def fvec(p, th):
    a = p >= th
    tp = np.bincount(s, weights=a * y, minlength=n); na = np.bincount(s, weights=a, minlength=n)
    return f05_vec(tp, na, ngt), tp, na


def mass_delta(p, m, target):
    lg = logit(np.clip(p[m], 1e-7, 1 - 1e-7)); si = s[m]; nS = len(np.unique(si))
    f = lambda d: expit(lg - d).sum() / nS - target
    return float(brentq(f, -6, 6))


arms = {"S006": np.load(os.path.join(os.path.dirname(os.path.dirname(HERE)), "E026_rrL", "cache", "p_RRL_V1_s42.npy")).astype(float)}
train_c = {"S006": "ALL", "US": "US", "IN": "India", "MIX": "ALL", "ST": "US", "USB": "US"}
for a in ("US", "IN", "MIX", "ST", "USB"):
    fp = os.path.join(HERE, f"D_p_{a}_V1.npy")
    if os.path.exists(fp):
        arms[a] = np.load(fp).astype(float)
out = dict(target_inpool_per_s1=TGT, v1_true_inpool={c: float(inpool[cty == c].mean()) for c in ("US", "India")},
           v1_ngt={c: float(ngt[cty == c].mean()) for c in ("US", "India")})
bands = [(0.01, 0.05), (0.05, 0.5), (0.5, 0.72), (0.72, 0.99), (0.99, 1.01)]
F72 = {}
for a, p in arms.items():
    tc = train_c[a]; r = {}
    # threshold chosen on the training country's V1 rows (held-out S1 of the training country) -- for ALL: whole V1
    dom = (cty == tc) if tc != "ALL" else np.ones(n, bool)
    curve = {float(t): fvec(p, t)[0] for t in GRID}
    th_dom = max(GRID, key=lambda t: curve[float(t)][dom].mean())
    for c in ("US", "India"):
        cm = cty == c; rm = cm[s]
        f72, tp, na = fvec(p, 0.72); fd = curve[float(th_dom)]
        th_star = max(GRID, key=lambda t: curve[float(t)][cm].mean())
        mass = float(np.bincount(s[rm], weights=p[rm], minlength=n)[cm].mean())
        d_lf = mass_delta(p, rm, TGT[tc] if tc != "ALL" else TGT["ALL"]); d_true = mass_delta(p, rm, float(inpool[cm].mean()))
        res = dict(n_s1=int(cm.sum()), F_th72=round(f72[cm].mean() * 100, 3), th_dom=float(th_dom), F_th_dom=round(fd[cm].mean() * 100, 3),
                   th_star=float(th_star), F_th_star=round(curve[float(th_star)][cm].mean() * 100, 3),
                   mass_per_s1=round(mass, 4), true_inpool=round(float(inpool[cm].mean()), 4), mass_ratio=round(mass / inpool[cm].mean(), 4),
                   delta_labelfree=round(d_lf, 3), delta_true=round(d_true, 3))
        for tag, th0 in (("72", 0.72), ("dom", float(th_dom))):
            thn = float(expit(logit(th0) + d_lf)); fn_, _, _ = fvec(p, thn); base_f = fvec(p, th0)[0]
            res[f"th_countmatched_{tag}"] = round(thn, 4); res[f"F_countmatched_{tag}"] = round(fn_[cm].mean() * 100, 3)
            res[f"dF_countmatched_{tag}_boot"] = boot_delta((fn_ - base_f)[cm], n=2000)
        for tag, dd in (("lf", d_lf), ("true", d_true)):
            thn = float(expit(logit(0.72) + dd)); band = rm & (p >= 0.72) & (p < thn)
            res[f"removed_{tag}"] = dict(th=round(thn, 4), links_per_s1=round(band.sum() / cm.sum(), 4), precision=round(float(y[band].mean()) if band.any() else float("nan"), 4),
                                        mean_p=round(float(p[band].mean()) if band.any() else float("nan"), 4),
                                        dF=round((fvec(p, thn)[0] - f72)[cm].mean() * 100, 4))
        fpl = (na - tp)[cm].mean(); fnl = (ngt - tp)[cm].mean(); fnl_in = (inpool - tp)[cm].mean()
        res.update(FP_links_per_s1=round(float(fpl), 4), FN_links_per_s1=round(float(fnl), 4), FN_inpool_per_s1=round(float(fnl_in), 4))
        bm = {}
        for lo, hi in bands:
            bmask = rm & (p >= lo) & (p < hi)
            bm[f"{lo}-{hi}"] = dict(pairs_per_s1=round(bmask.sum() / cm.sum(), 4), mass_per_s1=round(p[bmask].sum() / cm.sum(), 4),
                                   precision=round(float(y[bmask].mean()) if bmask.any() else float("nan"), 4), mean_p=round(float(p[bmask].mean()) if bmask.any() else float("nan"), 4))
        res["bands"] = bm
        r[c] = res
        F72[(a, c)] = f72
    r["curve_India"] = {str(t): round(curve[float(t)][cty == "India"].mean() * 100, 3) for t in GRID[::5]}
    r["curve_US"] = {str(t): round(curve[float(t)][cty == "US"].mean() * 100, 3) for t in GRID[::5]}
    out[a] = r
# paired comparisons on the held-out country (th .72)
pairs = {}
for (a, b, c) in (("US", "IN", "India"), ("US", "MIX", "India"), ("US", "S006", "India"), ("IN", "US", "US"), ("IN", "MIX", "US"),
                  ("IN", "S006", "US"), ("MIX", "S006", "India"), ("MIX", "S006", "US"), ("ST", "US", "India"), ("ST", "IN", "India"), ("USB", "US", "India"), ("USB", "IN", "India"), ("USB", "S006", "India")):
    if (a, c) in F72 and (b, c) in F72:
        cm = cty == c; pairs[f"{a}-{b}@{c}_th72"] = boot_delta((F72[(a, c)] - F72[(b, c)])[cm], n=2000)
out["paired_th72"] = pairs
json.dump(out, open(os.path.join(HERE, "D_analyze.json"), "w"), indent=1, default=float)
for a in arms:
    for c in ("US", "India"):
        r = out[a][c]
        print(f"{a:5s} {c:6s} F72 {r['F_th72']:.3f} Fdom({r['th_dom']:.2f}) {r['F_th_dom']:.3f} F*({r['th_star']:.2f}) {r['F_th_star']:.3f} "
              f"mass {r['mass_per_s1']:.4f} true {r['true_inpool']:.4f} ratio {r['mass_ratio']:.4f} dLF {r['delta_labelfree']:+.3f} dT {r['delta_true']:+.3f} "
              f"th'72 {r['th_countmatched_72']:.3f} F' {r['F_countmatched_72']:.3f} FP {r['FP_links_per_s1']:.4f} FN {r['FN_links_per_s1']:.4f}")
print(json.dumps(pairs, indent=0))
