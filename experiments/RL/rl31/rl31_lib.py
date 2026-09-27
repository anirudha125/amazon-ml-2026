"""RL-31 shared loaders (READ-ONLY on every existing artifact; writes only into experiments/RL/rl31/).

V1 = experiments/E024/V1_a50n10d10a (20,000 train S1 held out, union pool a50n10d10a, 2,550,505 pairs).
Models (all seed 42, row-aligned with the V1 meta):
  RRL  = E026 model_RRL_s42 (S006's model)          th .72   experiments/E026_rrL/cache/p_RRL_V1_s42.npy
  CTRL = E026 same-run RL-27 NEW refit                th .76   experiments/E026_rrL/cache/p_CTRL_V1_s42.npy
  NEW  = stored RL-27 NEW (S005's model)              th .78   experiments/RL/cache/rl27_p_NEW_V1_s42.npy
Metric = per-S1 F0.5 over ALL ground truth (unretrieved GT counts as FN), macro over S1 (singleton S1: 1 iff nothing predicted).
"""
import os, sys, json, pickle
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
RL = os.path.dirname(HERE)
ROOT = os.path.dirname(os.path.dirname(RL))
E24 = os.path.join(ROOT, "experiments", "E024")
E26 = os.path.join(ROOT, "experiments", "E026_rrL")
E23 = os.path.join(ROOT, "experiments", "E023")
sys.path.insert(0, RL)

MODELS = {
    "RRL": (os.path.join(E26, "cache", "p_RRL_V1_s42.npy"), 0.72),
    "CTRL": (os.path.join(E26, "cache", "p_CTRL_V1_s42.npy"), 0.76),
    "NEW": (os.path.join(RL, "cache", "rl27_p_NEW_V1_s42.npy"), 0.78),
}
RL27_COLS = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]


def f05_vec(tp, na, ng):
    tp, na, ng = (np.asarray(x, float) for x in (tp, na, ng))
    out = np.zeros(len(tp)); sing = ng == 0
    out[sing] = (na[sing] == 0).astype(float)
    ok = (~sing) & (tp > 0)
    out[ok] = 1.25 * tp[ok] / (0.25 * ng[ok] + na[ok])
    return out


def boot_delta(d, n=10000, seed=0):
    """mean and 95% CI (pp) of a per-S1 delta vector."""
    rng = np.random.default_rng(seed); idx = rng.integers(0, len(d), size=(n, len(d)))
    bs = d[idx].mean(1) * 100
    return round(float(d.mean() * 100), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)


def load_v1():
    m = np.load(os.path.join(E24, "V1_a50n10d10a", "meta.npz"), allow_pickle=True)
    V = {k: m[k] for k in m.files}
    V["y"] = V["y"].astype(np.int8)
    for nm, (fp, th) in MODELS.items():
        V["p_" + nm] = np.load(fp).astype(np.float64); V["th_" + nm] = th
    V["pb"] = np.load(os.path.join(E26, "cache", "pb_V1.npy"))
    V["rl27"] = np.load(os.path.join(RL, "cache", "rl27_V1.npy"))
    assert len(V["rl27"]) == len(V["y"]) == len(V["p_RRL"]) == len(V["pb"])
    return V


def per_s1(V, acc):
    n = len(V["s1_ids"]); s = V["s1idx"]; y = V["y"]
    tp = np.bincount(s, weights=acc & (y == 1), minlength=n); na = np.bincount(s, weights=acc, minlength=n)
    return tp, na


def macro(V, acc):
    tp, na = per_s1(V, acc)
    return f05_vec(tp, na, V["n_gt"])


def rank_in_s1(s1idx, p):
    """1-based rank of each row inside its S1 by descending p (ties: row order)."""
    n = len(p); order = np.lexsort((np.arange(n), -p, s1idx)); g = s1idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, n])
    r = np.arange(n) - np.repeat(starts, K) + 1; out = np.empty(n, np.int32); out[order] = r
    return out


def topk_mask(s1idx, p, k):
    """copy of src/e023_stage2.topk_mask (that module imports harness -> golden, which is missing on this machine)."""
    order = np.lexsort((-p, s1idx)); g = s1idx[order]
    starts = np.r_[0, np.flatnonzero(np.diff(g)) + 1]; K = np.diff(np.r_[starts, len(p)])
    rank = np.arange(len(p)) - np.repeat(starts, K); m = np.zeros(len(p), bool); m[order[rank < k]] = True
    return m
