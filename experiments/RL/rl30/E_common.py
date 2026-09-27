"""RL-30 investigator E -- shared helpers for the Part-6 label-backed sanity check (READ-ONLY; writes only rl30/E_*).
All functions are label-free: they use only S1 / record text of the split being analysed."""
import os, re, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import *          # noqa: F401,F403  (fold, toks, core_tokens, akey, street_parts, addr_words, LEGAL, HONOR, STOP, ...)

NONCORE = LEGAL | HONOR


def nset(s):
    """name token set (legal forms / honorifics kept)"""
    return frozenset(toks(s))


def loc_key(addr):
    """locality = word set of the last two comma components after the first (city / region / state); None if <2 components"""
    comps = [c.strip() for c in fold(addr).split(",") if c.strip()]
    if len(comps) < 2:
        return None
    w = set()
    for c in comps[1:][-2:]:
        w |= {t for t in re.findall(r"[a-z]{2,}", c) if t not in STOP}
    return " ".join(sorted(w)) if w else None


def ak2(addr):
    """(house number, street-name token set) key; None if either is missing"""
    num, st, _ = street_parts(addr)
    if num is None or not st:
        return None
    return num + "|" + " ".join(sorted(st))


def near_dup(A, B, need_core=True):
    """<=1 token per side, not identical, shared part non-empty (with >=1 core token if need_core)"""
    if A == B:
        return False
    a, b = A - B, B - A
    if len(a) > 1 or len(b) > 1:
        return False
    sh = A & B
    if not sh:
        return False
    if need_core and not (sh - NONCORE):
        return False
    return True


def jac(A, B):
    u = len(A | B)
    return len(A & B) / u if u else 0.0


def f05_per_s1(s1idx, y, acc, n_gt):
    n_s1 = len(n_gt)
    npred = np.bincount(s1idx, weights=acc, minlength=n_s1)
    tp = np.bincount(s1idx, weights=acc & (y == 1), minlength=n_s1)
    return np.where(n_gt == 0, (npred == 0).astype(float), np.where(tp > 0, 1.25 * tp / np.maximum(0.25 * n_gt + npred, 1e-9), 0.0))


def paired_boot(a, b, n=10000, seed=0):
    """paired bootstrap over S1 (n resamples of all S1 with replacement). Exact in distribution: only S1 whose score
    changed contribute, so their resample counts are drawn jointly from Multinomial(N, 1/N each)."""
    d = np.asarray(b, float) - np.asarray(a, float); N = len(d)
    nz = np.flatnonzero(d != 0); K = len(nz)
    if K == 0:
        return dict(delta_pp=0.0, ci95_pp=[0.0, 0.0], p_delta_le0=1.0, n_s1_changed=0)
    rng = np.random.default_rng(seed)
    pv = np.full(K + 1, 1.0 / N); pv[-1] = max(0.0, 1.0 - K / N)
    C = rng.multinomial(N, pv, size=n)[:, :K]
    bs = C @ d[nz] / N
    return dict(delta_pp=round(d.mean() * 100, 4), ci95_pp=[round(np.percentile(bs, 2.5) * 100, 4), round(np.percentile(bs, 97.5) * 100, 4)],
                p_delta_le0=round(float((bs <= 0).mean()), 4), n_s1_changed=int(K))
