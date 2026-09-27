"""RL-30 adversarial verification VD, part 3.
(a) Does D's DIST/FILL flag carry information about y BEYOND RL-27 NEW's p on labelled data?  Per-flag logit offset delta
    (MLE of y ~ sigmoid(logit(p) + delta)), S1-bootstrap 95% CI, on V1 and on V0 (V0 typed here with D_transform, same rules),
    overall and in the decision region p >= 0.5.  delta ~ 0  ->  the flag is not incremental for a learner trained on US/India.
(b) Composition of France DIST same-address final pairs by record-only token (is 'france' - a filler by D's own NOISE list - inside DIST?).
READ-ONLY; writes rl30/VD_3_offset.json.
"""
import sys, os, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH, ROOT, RL
import D_analyze as DA
from D_transform import classify_pairs
L = lambda *a: print(*a, flush=True)
R = {}
TAU = 0.2
T = load("test")
tabs = {}
fr_n = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
fr_n = fr_n[fr_n.kept_final]
nfr = len(fr_n); nsfr = int((T["s1"].country == "France").sum())
for cc, f in (("US", "D_enriched_US5.pkl"), ("India", "D_enriched_IN5.pkl")):
    d = pd.read_pickle(os.path.join(OUT, f)); dk = d[d.kept_final]
    s1c = T["s1"][T["s1"].country == cc]
    tabs[cc] = (DA.df_vocab(s1c), collections.Counter(" ".join(dk[dk.nt == "N_ADD"]["add"]).split()),
                collections.Counter(dk[dk.nt == "N_SWAP1"].sw_b), (len(dk) / len(s1c)) / (nfr / nsfr))
    del d


def flag(country, nt, add, swap):
    vv, a_, b_, sc = tabs[country]
    ts = add.split() if nt == "N_ADD" else ([swap.split("|")[1]] if (nt == "N_SWAP1" and "|" in swap) else [])
    ts = [t for t in ts if t]
    if not ts:
        return ""
    rs = [((a_[t] + b_[t]) / max(1, vv.get(t, 0)) / sc, vv.get(t, 0)) for t in ts]
    words = [(q, d) for q, d in rs if d >= 20]
    if any(q < TAU for q, d in words):
        return "DIST"
    return "FILL" if len(words) == len(rs) else "OTHER"


def prep(d):
    st_ = d.st.isin(["S_SAME", "S_TYPO"])
    d["same_addr"] = d.same_akey | ((d.ht == "H_SAME") & st_)
    d["flag"] = [flag(*x) for x in zip(d.country.values, d.nt.values, d["add"].values, d.swap.values)]
    return d


v1 = prep(pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl")))
# ---- V0: type hard rows (p>=0.001 or y==1) + nothing else (the offset test only needs rows with non-negligible p or positives)
Tr = load("train")
s1t = Tr["s1"].set_index("id"); rect = pd.concat([Tr["s2"], Tr["s3"]]).set_index("id")
m0 = np.load(os.path.join(ROOT, "experiments/E024/V0_a50n10d10a/meta.npz")); p0 = np.load(os.path.join(RL, "cache/rl27_p_NEW_V0_s42.npy"))
y0 = m0["y"]; idx = np.where((p0 >= 0.001) | (y0 == 1))[0]
s1 = m0["s1_ids"][m0["s1idx"][idx]].astype(object); cand = m0["cand"][idx].astype(object)
ty = classify_pairs(s1, cand, s1t, rect, None)
v0 = pd.concat([pd.DataFrame(dict(row=idx, s1=s1, rec=cand, y=y0[idx], p=p0[idx], country=m0["country"][m0["s1idx"][idx]], w=1.0)), ty], axis=1)
v0 = prep(v0)
L(f"V0 typed {len(v0):,} rows")
R["V0_flag_counts_same_addr"] = {"|".join(map(str, k)): int(n) for k, n in v0[v0.same_addr].groupby(["country", "flag", "y"]).size().items()}


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))


def mle_offset(z, y, w):
    d = 0.0
    for _ in range(50):
        q = 1 / (1 + np.exp(-(z + d)))
        g = (w * (y - q)).sum(); h = (w * q * (1 - q)).sum() + 1e-9
        d += np.clip(g / h, -2, 2)
        if abs(g / h) < 1e-7:
            break
    return d


def offset_test(d, nm):
    out = {}
    for reg, sel0 in (("all", np.ones(len(d), bool)), ("p>=0.5", d.p.values >= 0.5), ("0.01<=p<0.78", (d.p.values >= 0.01) & (d.p.values < NEW_TH))):
        for fl in ("DIST", "FILL"):
            for sa in ("same_addr", "any_addr"):
                sel = sel0 & (d.flag.values == fl) & (d.same_addr.values if sa == "same_addr" else True)
                g = d[sel]
                if len(g) < 5 or g.y.min() == g.y.max():
                    out[f"{reg}|{fl}|{sa}"] = dict(n=int(len(g)), n_pos=int(g.y.sum()), note="degenerate")
                    continue
                z, y, w = logit(g.p.values), g.y.values.astype(float), g.w.values
                dl = mle_offset(z, y, w)
                # S1 bootstrap
                s1s = g.s1.values; us, inv = np.unique(s1s, return_inverse=True)
                rng = np.random.default_rng(0); bs = []
                for _ in range(300):
                    cnt = np.bincount(rng.integers(0, len(us), len(us)), minlength=len(us))[inv]
                    ww = w * cnt
                    if ww.sum() == 0 or (ww * y).sum() in (0, ww.sum()):
                        continue
                    bs.append(mle_offset(z, y, ww))
                lo, hi = (np.percentile(bs, [2.5, 97.5]) if bs else (np.nan, np.nan))
                ll0 = -(w * (y * np.log(np.clip(1 / (1 + np.exp(-z)), 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - 1 / (1 + np.exp(-z)), 1e-9, 1)))).sum()
                q = 1 / (1 + np.exp(-(z + dl)))
                ll1 = -(w * (y * np.log(np.clip(q, 1e-9, 1)) + (1 - y) * np.log(np.clip(1 - q, 1e-9, 1)))).sum()
                acc = g.p.values >= NEW_TH; acc2 = q >= NEW_TH
                out[f"{reg}|{fl}|{sa}"] = dict(n=int(len(g)), n_pos=int(y.sum()), sum_p=round(float((w * g.p.values).sum()), 2),
                                               sum_y=round(float((w * y).sum()), 2), delta=round(float(dl), 3),
                                               ci95=[round(float(lo), 3), round(float(hi), 3)], ll_gain_nats=round(float(ll0 - ll1), 2),
                                               errors_at_TH=int(((acc & (y == 0)) | (~acc & (y == 1))).sum()),
                                               errors_after_offset=int(((acc2 & (y == 0)) | (~acc2 & (y == 1))).sum()))
    L(f"\n== offset test {nm}"); L(pd.DataFrame(out).T.to_string())
    return out


R["offset_V1"] = offset_test(v1, "V1")
R["offset_V0"] = offset_test(v0, "V0")
both = pd.concat([v1[["s1", "y", "p", "w", "flag", "same_addr"]], v0[["s1", "y", "p", "w", "flag", "same_addr"]]], ignore_index=True)
R["offset_V1+V0"] = offset_test(both, "V1+V0")

# ---- (b) France DIST composition
fr = pd.read_pickle(os.path.join(OUT, "VD_2_france_pairs.pkl"))
dist = fr[fr.flag == "DIST"]
tok = [(a.split() if nt == "N_ADD" else [b]) for nt, a, b in zip(dist.nt, dist["add"], dist.sw_b)]
c = collections.Counter(t for ts in tok for t in ts)
R["France_DIST_record_tokens_top30"] = dict(c.most_common(30))
NOISE_FR = ["fils", "services", "associes", "developpement", "france", "compagnie", "fka", "labs", "one"]
has_noise = np.array([any(t in NOISE_FR for t in ts) for ts in tok])
R["France_DIST_with_D_NOISE_token"] = dict(n=int(has_noise.sum()), share=round(float(has_noise.mean()), 4),
                                          n_france_token=int(sum("france" in ts for ts in tok)))
L("\nFrance DIST record-only tokens:", R["France_DIST_record_tokens_top30"]); L(R["France_DIST_with_D_NOISE_token"])
json.dump(R, open(os.path.join(OUT, "VD_3_offset.json"), "w"), indent=1, default=str)
L("wrote VD_3_offset.json")
