"""RL-31 Phase 3 -- does entity-level (one-owner-per-record) assignment beyond max-claimer recover F0.5?
Labelled DENSE slices (E025: every train S1 of Oregon 29,826 / Jaipur 17,154, all pairs scored by D2b s42, th .72);
evaluation S1 = slice S1 not used in D2b's stage-2 training (as src/e025_slices.py::evaluate). READ-ONLY; CPU only.

Rules (all parameter-free: they reuse the model threshold th; nothing is tuned on the evaluation S1):
  none      : accept p >= th
  max       : max-claimer (production)                        dropall : drop every contested record
  soft      : exclusive-owner posterior over ALL S1 scoring the record (incl. sub-threshold):
              q(a,r) = o_a / (1 + sum_b o_b),  o = p/(1-p)   (independent pairwise evidence + "at most one owner");
              accept iff q >= th
  soft_mc   : soft, then max-claimer on what is left (identical when th >= .5; kept as a check)
  margin    : max-claimer, but also drop the winner if the runner-up (any p) is within 0.05 of it
Diagnostics: owner-margin anatomy of accepted pairs (A-E of the brief): runner-up p, #S1 with p>=.3, rank of the S1 among
the record's scorers, and whether the runner-up has other accepted records (independent support).
Outputs: p3_dense_assign.json
"""
import os, sys, json, pickle, collections
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl31_lib import *

E25 = os.path.join(ROOT, "experiments", "E025")
sets = json.load(open(os.path.join(ROOT, "experiments", "E021", "sets.json")))["sets"]
train_s1 = set().union(*[set(sets[n]) for n in ("T0", "E014", "T2X")])
TH = 0.72
res = {}
for tag in ("US_OR", "India_jaipur"):
    d = os.path.join(E24, f"SL_{tag}_a50n10d10a"); m = np.load(os.path.join(d, "meta.npz"), allow_pickle=True)
    ids, si, cand, y, ng = m["s1_ids"], m["s1idx"], m["cand"], m["y"].astype(np.int8), m["n_gt"]
    p = np.load(os.path.join(E25, f"p_D2b_union_rrUb_big_{tag}_s42.npy")).astype(np.float64); assert len(p) == len(y)
    ev = np.array([s not in train_s1 for s in ids]); n = len(ids)
    # record -> rows
    u, rec = np.unique(cand, return_inverse=True)
    o = np.clip(p, 1e-9, 1 - 1e-9); o = o / (1 - o)
    so = np.bincount(rec, weights=o, minlength=len(u))
    q = o / (1 + so[rec])
    acc = p >= TH
    # max over record, runner-up over record (all rows)
    order = np.lexsort((-p, rec)); rs = rec[order]; starts = np.r_[0, np.flatnonzero(np.diff(rs)) + 1]
    top1 = np.full(len(u), -1.0); top2 = np.zeros(len(u)); top1_row = np.full(len(u), -1)
    top1[rs[starts]] = p[order[starts]]; top1_row[rs[starts]] = order[starts]
    nxt = starts + 1; okn = (nxt < len(rs)); okn[okn] = rs[nxt[okn]] == rs[starts[okn]]
    top2[rs[starts[okn]]] = p[order[nxt[okn]]]
    is_top = top1_row[rec] == np.arange(len(p))
    runner = np.where(is_top, top2[rec], top1[rec])                   # best OTHER S1's p for this record
    n_ge3 = np.bincount(rec, weights=(p >= 0.3), minlength=len(u))[rec]
    n_acc_rec = np.bincount(rec, weights=acc, minlength=len(u))[rec]
    rules = {"none": acc, "max": acc & is_top, "dropall": acc & (n_acc_rec == 1), "soft": q >= TH}
    rules["soft_mc"] = rules["soft"] & is_top
    rules["margin05"] = acc & is_top & (p - runner >= 0.05)
    def f(mask):
        tp = np.bincount(si, weights=mask & (y == 1), minlength=n); na = np.bincount(si, weights=mask, minlength=n)
        return f05_vec(tp, na, ng)
    F = {k: f(v) for k, v in rules.items()}; evr = ev[si]
    out = dict(n_eval_s1=int(ev.sum()), n_pairs=int(len(p)))
    for k, v in rules.items():
        r = dict(macro=round(F[k][ev].mean() * 100, 3), tp=int((v & (y == 1) & evr).sum()), fp=int((v & (y == 0) & evr).sum()))
        if k != "none":
            r["vs_none"] = boot_delta(F[k][ev] - F["none"][ev]); r["vs_max"] = boot_delta(F[k][ev] - F["max"][ev])
        out[k] = r
    # anatomy of max-claimer-kept pairs on eval S1
    kept = rules["max"] & evr; kfp = kept & (y == 0); ktp = kept & (y == 1)
    an = {}
    for nm, msk in (("TP", ktp), ("FP", kfp)):
        an[nm] = dict(n=int(msk.sum()), runner_ge_0_3=int((msk & (runner >= 0.3)).sum()), runner_ge_0_1=int((msk & (runner >= 0.1)).sum()),
                      margin_lt_0_2=int((msk & (p - runner < 0.2)).sum()), n_ge3_ge2=int((msk & (n_ge3 >= 2)).sum()),
                      p_lt_0_9=int((msk & (p < 0.9)).sum()))
    # FP owner: is the true owner in the slice and scoring the record?
    own = {}
    for s_, g_ in zip(ids, range(n)):
        pass
    out["anatomy_kept"] = an
    # oracle: perfect record-level owner choice among scorers (upper bound of any assignment layer on this pool)
    #   accept (a,r) iff y==1 and p>=th  -> removes every FP, keeps TP; separate: remove only FPs whose record has a y==1 row in slice
    has_owner_row = np.bincount(rec, weights=(y == 1), minlength=len(u))[rec] > 0
    orc_owned = rules["max"] & ~((y == 0) & has_owner_row)          # drop FPs whose true owner is a slice S1 scoring the record
    orc_owned_scored = rules["max"] & ~((y == 0) & has_owner_row)
    F_o = f(orc_owned)
    out["oracle_drop_FPs_with_owner_in_slice"] = dict(macro=round(F_o[ev].mean() * 100, 3), vs_max=boot_delta(F_o[ev] - F["max"][ev]),
                                                      fp_left=int((orc_owned & (y == 0) & evr).sum()))
    F_all = f(rules["max"] & (y == 1))
    out["oracle_drop_all_FP"] = dict(macro=round(F_all[ev].mean() * 100, 3), vs_max=boot_delta(F_all[ev] - F["max"][ev]))
    res[tag] = out
    print(tag, json.dumps(out, indent=1), flush=True)
json.dump(res, open(os.path.join(HERE, "p3_dense_assign.json"), "w"), indent=1)
