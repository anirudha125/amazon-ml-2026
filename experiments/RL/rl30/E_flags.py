"""RL-30 / E -- per-pair label-free flags F1 / F2 / F3 (shared by the V1 evaluation and the test firing counts)."""
import os, sys, math
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *

MIN_SUPP = 5


def s1_feats(name, addr):
    num, st, _ = street_parts(addr)
    return nset(name), akey(addr), num, st, addr_words(addr)


def rec_feats(name, addr):
    num, st, _ = street_parts(addr) if addr.strip() else (None, frozenset(), frozenset())
    return nset(name), num, st, addr_words(addr) if addr.strip() else frozenset()


def role_lookup(roles):
    """country -> {token: (s, Dc)} for tokens with support >= MIN_SUPP; and country -> {token: Dc} for all tokens"""
    S, DC = {}, {}
    for c, T in roles.items():
        t = T[T.supp >= MIN_SUPP]
        S[c] = dict(zip(t.index, t.s.astype(float)))
        DC[c] = dict(zip(T.index[T.Dc > 0], T.Dc[T.Dc > 0].astype(int)))
    return S, DC


def compute_flags(s1_ids, rec_ids, ctry, S1F, RF, S_role, DC, sib):
    """s1_ids, rec_ids, ctry: per-pair arrays. S1F[s1] = s1_feats, RF[rec] = rec_feats, sib[s1] = [sibling name sets]."""
    n = len(s1_ids)
    nd = np.zeros(n, bool); typ = np.zeros(n, np.int8); smax = np.full(n, np.nan, np.float32); smin = np.full(n, np.nan, np.float32)
    n_unknown = np.zeros(n, np.int8); anydc = np.zeros(n, bool); same_addr = np.zeros(n, bool)
    f2_ge = np.zeros(n, bool); f2_gt = np.zeros(n, bool); has_sib = np.zeros(n, bool)
    f3 = np.zeros(n, bool); ov = np.full(n, np.nan, np.float32); st_ov0 = np.zeros(n, bool); num_eq = np.zeros(n, bool)
    for k in range(n):
        A, _, snum, sst, sw = S1F[s1_ids[k]]
        R, rnum, rst, rw = RF[rec_ids[k]]
        c = ctry[k]
        ne = snum is not None and snum == rnum
        num_eq[k] = ne
        so = bool(sst & rst)
        same_addr[k] = ne and so
        # F3
        if sw and rw:
            o = len(sw & rw) / min(len(sw), len(rw)); ov[k] = o
            if sst and rst and not so:
                st_ov0[k] = True
                if o >= 0.8:
                    f3[k] = True
        # F1
        if near_dup(A, R, need_core=True):
            nd[k] = True
            a, b = A - R, R - A
            typ[k] = 3 if (a and b) else (1 if a else 2)
            sr = S_role.get(c, {}); dc = DC.get(c, {})
            vals = []; unk = 0
            for t in a | b:
                v = sr.get(t)
                if v is None:
                    unk += 1
                else:
                    vals.append(v)
                if t in dc:
                    anydc[k] = True
            n_unknown[k] = unk
            if vals:
                smax[k] = max(vals); smin[k] = min(vals)
        # F2
        L = sib.get(s1_ids[k])
        if L:
            has_sib[k] = True
            jr = jac(R, A); js = max(jac(R, B) for B in L)
            if js > 0 and js >= jr:
                f2_ge[k] = True
                if js > jr:
                    f2_gt[k] = True
    return dict(nd=nd, typ=typ, smax=smax, smin=smin, n_unknown=n_unknown, anydc=anydc, same_addr=same_addr, num_eq=num_eq,
                has_sib=has_sib, f2_ge=f2_ge, f2_gt=f2_gt, f3=f3, ov=ov, st_ov0=st_ov0)


def derive(fl):
    """named boolean flags from the raw arrays"""
    nd, sm = fl["nd"], fl["smax"]
    known = ~np.isnan(sm)
    F = {}
    for th in (0.5, 0.7, 0.9):
        F[f"F1_dist_s{th}"] = nd & known & (np.nan_to_num(sm, nan=-1) >= th)
    F["F1_dist_s0.5_sameaddr"] = F["F1_dist_s0.5"] & fl["same_addr"]
    F["F1_dist_s0.5_subst"] = F["F1_dist_s0.5"] & (fl["typ"] == 3)
    F["F1_dist_s0.5_onesided"] = F["F1_dist_s0.5"] & (fl["typ"] != 3)
    F["F1_coloc_Dc"] = nd & fl["anydc"]
    F["F1_coloc_Dc_sameaddr"] = F["F1_coloc_Dc"] & fl["same_addr"]
    F["F1_filler_all"] = nd & known & (fl["n_unknown"] == 0) & (np.nan_to_num(fl["smin"], nan=9) < 0.5) & (np.nan_to_num(sm, nan=9) < 0.5)
    F["F1_filler_all_sameaddr"] = F["F1_filler_all"] & fl["same_addr"]
    F["F2_sib_ge"] = fl["f2_ge"]; F["F2_sib_gt"] = fl["f2_gt"]; F["F2_has_sib"] = fl["has_sib"]
    F["F3_street_mismatch"] = fl["f3"]
    F["F3_street_mismatch_numeq"] = fl["f3"] & fl["num_eq"]
    F["ND_any"] = nd
    return F
