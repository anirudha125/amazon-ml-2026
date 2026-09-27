"""RL-32 / E: 'another S1 owns r's name+address' -- structural record-side signal over the FULL S1 universe (not the pool),
computable on labelled train (V1+T) as well as on test. For each accepted pair (s, r):
  alt_owner = # S1 s' != s (same country) with core(s')==core(r), num(s')==num(r) (non-empty) and addr-token Jaccard(s', r) >= .5
  s_self    = s itself satisfies the same test (r matches s on core+num+street)
Flag OWN_ELSE = alt_owner>=1 & not s_self. Output E_owner.json"""
import os, sys, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE)
sys.path.insert(0, HERE); import E_struct as ES
sys.path.insert(0, os.path.join(RL, "rl31")); import rl31_lib as RLL
BANDS = [(0.72, 0.90), (0.90, 0.99), (0.99, 1.01)]
OUT = {}


def owner_index(s1df):
    P = ES.parse(s1df); P = P[(P.core != "") & (P.num != "")]
    idx = {}
    for sid, c, co, nu, a in zip(P.index.values, P.cty.values, P.core.values, P.num.values, P.ast.values):
        idx.setdefault((c, co, nu), []).append((sid, a))
    return idx


def feats(A, rec, idx):
    ids = list(set(A.r))
    P = ES.parse(rec.loc[ids].reset_index(drop=True))
    alt = np.zeros(len(A), int); self_ = np.zeros(len(A), bool)
    Pr = P.loc[A.r.values]
    for i, (s, c, co, nu, a) in enumerate(zip(A.s.values, Pr.cty.values, Pr.core.values, Pr.num.values, Pr.ast.values)):
        if not co or not nu: continue
        for sid, a2 in idx.get((c, co, nu), ()):
            if ES.jac(a, a2) >= 0.5:
                if sid == s: self_[i] = True
                else: alt[i] += 1
    A = A.copy(); A["alt_owner"] = alt; A["s_self"] = self_; A["own_else"] = (alt >= 1) & ~self_
    return A


def table(A, nS1, lab):
    rows = []
    for lo, hi in BANDS:
        b = A[(A.p >= lo) & (A.p < hi)]
        for k in ["own_else", "alt_owner_any"]:
            f = b.own_else if k == "own_else" else (b.alt_owner >= 1)
            r = dict(band=f"{lo:.2f}-{min(hi,1):.2f}", flag=k, per1k=round(f.sum() / nS1 * 1000, 3), band_per1k=round(len(b) / nS1 * 1000, 1))
            if lab:
                fpb = (b.y == 0).mean(); ff = (b.y[f] == 0).mean() if f.sum() else np.nan
                r.update(n_flag=int(f.sum()), n_fp=int((b.y[f] == 0).sum()), fp_rate_flag=round(float(ff), 4), fp_rate_band=round(float(fpb), 4),
                         lift=round(float(ff / fpb), 2) if fpb > 0 else None, fp_capt=round(float((b.y[f] == 0).sum() / max(1, (b.y == 0).sum())), 3))
            rows.append(r)
    return rows


def main():
    tr = pickle.load(open(os.path.join(RL, "cache", "train.pkl"), "rb"))
    L = pickle.load(open(os.path.join(HERE, "E_feat_L.pkl"), "rb")); D, gt = L["D"], L["gt"]
    s1acc = pickle.load(open(os.path.join(HERE, "E_L_s1acc.pkl"), "rb"))
    idx = owner_index(tr["s1"]); ES.log("train owner index", len(idx))
    rec = pd.concat([tr["s2"], tr["s3"]]).set_index("id", drop=False)
    A = feats(D[D.acc], rec, idx); ES.log("L feats")
    OUT["L"] = table(A, len(gt), True)
    for r in OUT["L"]: print("L", r)
    # veto effect on V1 / T
    for pmax in (0.99, 1.01):
        for sub, nm in ((["V1"], "V1"), (["T0", "E014", "T2X"], "T")):
            g = gt[gt.set.isin(sub)].set_index("s1"); base = s1acc.reindex(g.index).fillna(0)
            V = A[A.own_else & (A.p < pmax)]
            rm = V.groupby("s").agg(na=("acc", "size"), tp=("y", "sum")).reindex(g.index).fillna(0)
            f0 = RLL.f05_vec(base.tp.values, base.acc.values, g.n_gt.values)
            f1 = RLL.f05_vec(base.tp.values - rm.tp.values, base.acc.values - rm.na.values, g.n_gt.values)
            OUT[f"veto_{nm}_p<{pmax}"] = dict(dF=RLL.boot_delta(f1 - f0, n=2000), rm=int(rm.na.sum()), fp=int(rm.na.sum() - rm.tp.sum()))
            print("veto", nm, pmax, OUT[f"veto_{nm}_p<{pmax}"])
    del tr, rec
    te = pickle.load(open(os.path.join(RL, "cache", "test.pkl"), "rb"))
    idx = owner_index(te["s1"]); ES.log("test owner index")
    rec = pd.concat([te["s2"], te["s3"]]).set_index("id", drop=False)
    for c in ["France", "US", "India"]:
        Z = pickle.load(open(os.path.join(HERE, f"E_feat_{c}.pkl"), "rb"))
        for pk in ("p6", "p5"):
            At = feats(Z[pk][Z[pk].acc], rec, idx)
            OUT[f"{c}_{pk}"] = table(At, Z["nS1"], False)
            for r in OUT[f"{c}_{pk}"]: print(c, pk, r)
            if c == "France" and pk == "p6":
                pickle.dump(At[At.own_else][["s", "r", "p", "p_run", "nm", "num", "st", "alt_owner"]], open(os.path.join(HERE, "E_owner_France_p6_flagged.pkl"), "wb"))
        ES.log(c)
    json.dump(OUT, open(os.path.join(HERE, "E_owner.json"), "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
