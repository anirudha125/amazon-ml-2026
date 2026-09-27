"""RL-17 -- can the RL-05 within-source number context and S1-side reverse competition act as a post-processing layer
(no retraining) on E018C (PROD) and E024-B2 outputs? Parameter-free rules, applied to ACCEPTED pairs only:

  R1 within-source veto : num_r != num_a (hn in d<=2/1digit/d<=20/far), and the record's OWN source contains another record
                          with the S1's name core and the S1's house number (RL-05: P(match) 0.11 vs 0.45)
  R2 family-owner veto  : num_a not in record numbers, and another S1 b != a (same country, same name core as a) has
                          house number == num_r  (b explains the record exactly; a only through noise)
Rescue probe (not a rule): rejected pairs with p in [0.3, th) in the RL-05 favourable cell -> precision.
Indexes are built from the UNLABELLED corpus of the split (train records / train S1 here; test at test time).
Evaluation: V1 (20k S1) and V0 (2,001 S1), macro F0.5 with out-of-pool GT counted (meta n_gt), paired bootstrap over S1.
Nothing is tuned: rules and thresholds are fixed a priori (model thresholds from arm_B2_union.json / 0.72 for PROD).
"""
import os, sys, json, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl_data import load
from rl02_error_decomp import core
from rl04_numeric_ambiguity import addr_feats, rel

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT = os.path.dirname(os.path.abspath(__file__))
E24 = os.path.join(ROOT, "experiments", "E024"); E23 = os.path.join(ROOT, "experiments", "E023")
AMB = {"d<=2", "1digit", "d<=20", "far"}


def corpus_index(split):
    """(country, core, num, src) -> #records and (country, core, num) -> #S1, from the unlabelled corpus of the split.
    Built from the parallel cache written by rl_ctx_precompute.py (records with an empty address have num=None)."""
    cp = os.path.join(OUT, "cache", f"ctx_index_{split}.pkl")
    if os.path.exists(cp):
        return pd.read_pickle(cp)
    C = pd.read_pickle(os.path.join(OUT, "cache", f"ctx_{split}.pkl"))
    rec, s1 = C["rec"].dropna(subset=["num"]), C["s1"].dropna(subset=["num"])
    ridx = rec.groupby(["country", "core", "num", "src"]).size()
    sidx = s1.groupby(["country", "core", "num"]).size()
    out = dict(ridx=ridx.to_dict(), sidx=sidx.to_dict())
    pd.to_pickle(out, cp)
    return out


def f05_vec(tp, na, ng):
    tp, na, ng = map(np.asarray, (tp, na, ng))
    out = np.zeros(len(tp))
    sing = ng == 0
    out[sing] = (na[sing] == 0).astype(float)
    ok = (~sing) & (tp > 0)
    P = tp[ok] / na[ok]; R = tp[ok] / ng[ok]
    out[ok] = 1.25 * P * R / (0.25 * P + R)
    return out


def boot(d, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1)
    return float(d.mean() * 100), float(np.percentile(bs, 2.5) * 100), float(np.percentile(bs, 97.5) * 100)


def main():
    D = load("train", verbose=False)
    S1 = D["s1"].set_index("id"); R = pd.concat([D["s2"], D["s3"]]).set_index("id")
    IX = corpus_index("train"); ridx, sidx = IX["ridx"], IX["sidx"]
    thB2 = {s: v["th_oof"] for s, v in json.load(open(os.path.join(E24, "arm_B2_union.json")))["seeds"].items()}
    arms = []
    for vs in ("V1", "V0"):
        arms.append((f"PROD_{vs}", f"{vs}_a50n10", np.load(os.path.join(E23, f"p_PROD_{vs}.npy")), 0.72))
        for s in ("42", "43", "44"):
            arms.append((f"B2s{s}_{vs}", f"{vs}_a50n10d10a", np.load(os.path.join(E24, f"p_B2_union_{vs}_s{s}.npy")), thB2[s]))
    feat_cache = {}
    results = {}
    for name, pool, p, th in arms:
        m = np.load(os.path.join(E24, pool, "meta.npz"), allow_pickle=True)
        s1_ids, s1idx, cand, y, ngt, ctry = m["s1_ids"], m["s1idx"], m["cand"], m["y"].astype(int), m["n_gt"], m["country"]
        assert len(p) == len(cand)
        sel = np.where(p >= 0.3)[0]
        rows = []
        for i in sel:
            key = (pool, int(i))
            if key not in feat_cache:
                a = s1_ids[s1idx[i]]; r = cand[i]
                c = S1.at[a, "country"]; ca = core(S1.at[a, "name"])
                fa = addr_feats(S1.at[a, "addr"]); ra = R.at[r, "addr"]
                fr = addr_feats(ra) if ra.strip() else (None, set(), set())
                hn = rel(fa[0], fr[1], fr[0]) if ra.strip() else "empty"
                src = r[:2]
                r1 = r2 = False; fav = False
                if ca and fa[0] and fr[0] and hn != "exact":
                    if hn in AMB:
                        r1 = ridx.get((c, ca, fa[0], src), 0) > 0
                        fav = (not r1) and ridx.get((c, ca, fr[0], src), 0) > 1
                    r2 = sidx.get((c, ca, fr[0]), 0) > 0 and fr[0] != fa[0]
                feat_cache[key] = (hn, r1, r2, fav, c)
            rows.append(feat_cache[key])
        F = pd.DataFrame(rows, columns=["hn", "r1", "r2", "fav", "country"]); F["i"] = sel
        F["p"] = p[sel]; F["y"] = y[sel]; F["acc"] = F.p >= th
        acc = np.zeros(len(p), bool); acc[p >= th] = True
        n_s1 = len(s1_ids)
        tp0 = np.bincount(s1idx, weights=(acc & (y == 1)), minlength=n_s1)
        na0 = np.bincount(s1idx, weights=acc, minlength=n_s1)
        base = f05_vec(tp0, na0, ngt)
        res = {"th": th, "base_macro": round(base.mean() * 100, 3), "accepted": int(acc.sum())}
        for rule, mask in (("R1", F.r1), ("R2", F.r2), ("R1|R2", F.r1 | F.r2)):
            v = F[F.acc & mask]
            veto = np.zeros(len(p), bool); veto[v.i.values] = True
            a2 = acc & ~veto
            tp1 = np.bincount(s1idx, weights=(a2 & (y == 1)), minlength=n_s1)
            na1 = np.bincount(s1idx, weights=a2, minlength=n_s1)
            new = f05_vec(tp1, na1, ngt)
            dm, lo, hi = boot(new - base)
            by_c = {c: int((v.country == c).sum()) for c in ("US", "India")}
            res[rule] = dict(vetoed=len(v), vetoed_true=int(v.y.sum()), vetoed_false=int((v.y == 0).sum()),
                             precision_of_veto=round(float((v.y == 0).mean()) if len(v) else float("nan"), 3),
                             macro=round(new.mean() * 100, 3), delta_pp=round(dm, 3), ci95=[round(lo, 3), round(hi, 3)],
                             vetoed_by_country=by_c)
        rj = F[(~F.acc) & F.fav]
        res["rescue_probe_fav_cell"] = dict(n=len(rj), precision=round(float(rj.y.mean()), 3) if len(rj) else None)
        results[name] = res
        print(name, json.dumps(res), flush=True)
    json.dump(results, open(os.path.join(OUT, "rl17_context_rules.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
