"""Gate report: paired bootstrap (harness, 10k) NEW vs CTRL, and vs S006's stored model (E026 p_RRL), per country."""
import os, sys, json, numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, os.path.join(ROOT, "experiments", "E026_rrL"))
from boot import H, S2
import glob
R = {}
for f in [os.path.join(HERE, "p5_results.json")] + sorted(glob.glob(os.path.join(HERE, "p5_res_*.json"))):
    if os.path.exists(f): R.update(json.load(open(f)))
out = {}
S6 = json.load(open(os.path.join(ROOT, "experiments", "E026_rrL", "step2_results.json")))["RRL_s42"]
for vn in ("V1", "V0"):
    V = S2.load_set(vn, "a50n10d10a"); c = V["country"]
    ps6 = np.load(os.path.join(ROOT, "experiments", "E026_rrL", "cache", f"p_RRL_{vn}_s42.npy")); f6 = S2.summarize(dict(V, country_s1=c), ps6, S6["th_oof"])["scores"]
    F = {a: np.load(os.path.join(HERE, f"f_{a}_{vn}.npy")) for a in R if os.path.exists(os.path.join(HERE, f"f_{a}_{vn}.npy"))}
    F["S006"] = f6; res = {a: round(float(f.mean()) * 100, 3) for a, f in F.items()}
    for a in F:
        for ref in ("CTRL", "S006"):
            if a in (ref,) or ref not in F: continue
            d, lo, hi, pn = H.paired_bootstrap(F[ref], F[a]); res[f"{a}_vs_{ref}"] = [round(d * 100, 3), round(lo * 100, 3), round(hi * 100, 3)]
            for ct in ("US", "India"):
                m = c == ct; d, lo, hi, _ = H.paired_bootstrap(F[ref][m], F[a][m]); res[f"{a}_vs_{ref}_{ct}"] = [round(d * 100, 3), round(lo * 100, 3), round(hi * 100, 3)]
    out[vn] = res
out["th"] = {a: R[a]["th_oof"] for a in R}
print(json.dumps(out, indent=1)); json.dump(out, open(os.path.join(HERE, "p5_report.json"), "w"), indent=1)
