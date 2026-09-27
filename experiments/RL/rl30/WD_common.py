"""RL-30 adversarial verifier WD (lens: statistical artifact / support). Independent re-implementation of the
single-token name-difference typing (does NOT import D_transform). READ-ONLY investigation; new files only (WD_ prefix)."""
import os, sys, re, collections
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import fold, toks, akey, street_parts, LEGAL, HONOR, STOP
from rapidfuzz.distance import Levenshtein, OSA
from rapidfuzz import fuzz

DROP = LEGAL | HONOR | STOP


def core(name):
    return frozenset(t for t in toks(name) if t not in DROP and len(t) > 1)


def _typo(a, b):
    la, lb = len(a), len(b)
    m = min(la, lb)
    if m >= 3 and (a.startswith(b) or b.startswith(a)):
        return True
    d = OSA.distance(a, b)
    if d <= 1 and m >= 2:
        return True
    if la >= 4 and lb >= 4 and (d <= 2 or Levenshtein.normalized_similarity(a, b) >= 0.6):
        return True
    if m >= 4 and Levenshtein.normalized_similarity("".join(sorted(a)), "".join(sorted(b))) >= 0.75:
        return True
    return False


def name_diff(cs, cr):
    """cs, cr: frozensets of core tokens. Returns (kind, a, b):
    SAME / ADD1 (b=added token) / DROP1 (a) / SWAP1 (a->b) / OTHER / EMPTY. typo-related token pairs are cancelled first."""
    if not cr:
        return "EMPTY", "", ""
    A, B = sorted(cs - cr), sorted(cr - cs)
    if len(A) > 3 or len(B) > 3:
        return "OTHER", "", ""
    ua, ub = set(), set()
    for a in A:
        for b in B:
            if b not in ub and a not in ua and _typo(a, b):
                ua.add(a); ub.add(b)
    A = [a for a in A if a not in ua]; B = [b for b in B if b not in ub]
    if not A and not B:
        return "SAME", "", ""
    if not A and len(B) == 1:
        return "ADD1", "", B[0]
    if len(A) == 1 and not B:
        return "DROP1", A[0], ""
    if len(A) == 1 and len(B) == 1:
        return "SWAP1", A[0], B[0]
    return "OTHER", "", ""


def addr_key(addr):
    """(house number, street key) or None"""
    n, st, _ = street_parts(addr)
    if n is None or not st:
        return None
    return n + "|" + " ".join(sorted(st))


def same_addr(ka, kb, aa, ab):
    """same address = identical canonical akey, or same house number and street tokens equal / fuzz>=80"""
    if aa == ab and aa:
        return True
    if ka is None or kb is None:
        return False
    na, sa = ka.split("|", 1); nb, sb = kb.split("|", 1)
    return na == nb and (sa == sb or fuzz.ratio(sa, sb) >= 80)


def s1_df(names):
    c = collections.Counter()
    for n in names:
        c.update(core(n))
    return c
