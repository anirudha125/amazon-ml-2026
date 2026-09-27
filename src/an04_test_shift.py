"""
AN04 -- Test-vs-validation score-distribution comparison (diagnostic only; nothing is tuned on test data).

Same statistics for validation (2,001 S1, full stage-2 scores) and each test country.
Test sources: full prediction files (accepted predictions + candidate lists for every S1) and, for top/second
scores, the per-S1 top2 recorded by the run (US) or a random shard sample re-scored with the same model
(France, India; `predict_test.py sample`).
"""
import os, sys, json, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import harness as H

T = os.path.join(H.ROOT, "experiments", "TEST_PIPELINE")
MODEL = "E014B_s42"


def stats(cands_n, acc_scores_per_s1, top2, th):
    cn = np.array(cands_n); na = np.array([len(v) for v in acc_scores_per_s1])
    acc = np.array([p for v in acc_scores_per_s1 for p in v]) if na.sum() else np.array([0.0])
    t1 = np.array([t[0] for t in top2]); t2 = np.array([t[1] for t in top2]); mg = t1 - t2
    q = lambda a, qs: [round(float(x), 3) for x in np.quantile(a, qs)] if len(a) else None
    return {
        "n_S1": len(cn), "cands_per_S1": round(float(cn.mean()), 1), "empty_pool_pct": round(float((cn == 0).mean() * 100), 2),
        "pred_nonempty_pct": round(float((na > 0).mean() * 100), 1), "preds_per_S1": round(float(na.mean()), 2),
        "ge2_accepted_pct": round(float((na >= 2).mean() * 100), 1),
        "accepted_hist_0..8+": [int(x) for x in np.bincount(np.minimum(na, 8), minlength=9)],
        "accepted_score_P5_P25_P50": q(acc, [.05, .25, .5]),
        "n_top2": len(t1),
        "top_score_P10_P25_P50": q(t1, [.1, .25, .5]),
        "top_below_th_pct": round(float((t1 < th).mean() * 100), 1) if len(t1) else None,
        "top_in_[th,0.9)_pct": round(float(((t1 >= th) & (t1 < 0.9)).mean() * 100), 1) if len(t1) else None,
        "margin_top1_top2_P10_P25_P50": q(mg, [.1, .25, .5]),
        "second_ge_th_pct": round(float((t2 >= th).mean() * 100), 1) if len(t2) else None,
    }


def main():
    v = pickle.load(open(os.path.join(T, f"valpreds_{MODEL}.pkl"), "rb")); th = v["th"]
    g = collections.defaultdict(list)
    for (s, c, y), p in zip(v["val_meta"], v["p_va"]):
        g[s].append(float(p))
    out = {"validation": stats([len(L) for L in g.values()], [[p for p in L if p >= th] for L in g.values()],
                               [tuple(sorted(L, reverse=True)[:2] + [0.0])[:2] for L in g.values()], th)}
    for ctry in ["US", "India", "France"]:
        pf = os.path.join(T, f"pred_{MODEL}_{ctry}_full.pkl")
        if not os.path.exists(pf):
            continue
        d = pickle.load(open(pf, "rb"))
        top2 = d.get("top2") or {}
        sp = os.path.join(T, f"sample_{MODEL}_{ctry}.pkl")
        src = "full run"
        if not top2 and os.path.exists(sp):
            top2 = pickle.load(open(sp, "rb"))["top2"]; src = "shard sample"
        acc = [[p for _, p in d["preds"].get(s, [])] for s in d["cands"]]
        out[ctry] = stats([len(x) for x in d["cands"].values()], acc, [t[:2] for t in top2.values()], th)
        out[ctry]["top2_source"] = src
    json.dump(dict(threshold=th, stats=out), open(os.path.join(H.ROOT, "experiments", "AN04_test_shift.json"), "w"), indent=1)
    keys = list(out["validation"].keys())
    cols = list(out)
    print(f"{'statistic':<32}" + "".join(f"{c:>26}" for c in cols))
    for k in keys + ["top2_source"]:
        print(f"{k:<32}" + "".join(f"{str(out[c].get(k, '-')):>26}" for c in cols))


if __name__ == "__main__":
    main()
