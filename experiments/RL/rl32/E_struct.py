"""RL-32 / Investigator E -- structural-consistency features the stage 2 never sees (label-free rules, no model fit).

Labelled set L = V1 (p = S006 model p_RRL, trained on T) U T (p = OOF S006-model probs), th .72, max-claimer over L.
Test = rl31 test_scores {US,India,France}: p6 (S006, th .72) and p5 (S005, th .78), max-claimer per country.
For every pair with p >= 0.30 we compute, against the ACCEPTED set of its S1 (excluding the pair itself):
  field relations to S1 : nm (eq / near / far: core token sets equal / Jaccard>=.5 / else; 'na' if a core is empty, e.g. Devanagari),
                          num (eq / diff / na), st (eq: non-digit addr-token Jaccard>=.5 / diff / na: empty record address)
  n_acc, n_o (other accepted), o_eq_F / o_diff_F per field, lonely_F (r differs on F, all >=1 others agree with S1 on F),
  grp2_num (r's number differs from S1's but >=1 other accepted record carries r's number = coherent second address group),
  sib_twin (# other accepted with same (core,num) as r), supp_s (# other accepted agreeing with S1 on name eq & num eq),
  record side: p_run = max p of r over other S1 in the pool (test complete for p>=.01; L sparse, 3.3% of train S1),
  s1_twins = # other S1 (same country, full universe) with identical (core,num,skey) key (complete, label-free).
Writes experiments/RL/rl32/E_feat_{L,US,India,France}.pkl (pandas). READ-ONLY on everything else."""
import os, re, sys, pickle, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__)); RL = os.path.dirname(HERE); EXP = os.path.dirname(RL)
LEGAL = set("llc inc corp corporation ltd limited pvt private llp co company lp pc pllc plc sarl sas sasu eurl sa sci snc selarl scop".split())
ABBR = {"st": "street", "rd": "road", "ave": "avenue", "av": "avenue", "dr": "drive", "ln": "lane", "ct": "court", "blvd": "boulevard",
        "bd": "boulevard", "pl": "place", "r": "rue", "all": "allee", "imp": "impasse", "ch": "chemin", "chem": "chemin", "rte": "route",
        "hwy": "highway", "pkwy": "parkway", "cir": "circle", "sq": "square", "fg": "faubourg"}
T0 = time.time()
def log(*a): print(f"[{time.time()-T0:7.1f}s]", *a, flush=True)


def vfold(s):
    s = pd.Series(s, dtype=object).fillna("").astype(str)
    s = s.str.lower().str.normalize("NFKD").str.encode("ascii", "ignore").str.decode("ascii")
    return s.str.replace(r"[^a-z0-9]+", " ", regex=True).str.strip()


def parse(df):
    """df: id,name,addr,country -> DataFrame indexed by id with core, num, ast (frozenset), skey."""
    fn = vfold(df["name"].fillna("").str.replace(".", "", regex=False).values)
    core = [" ".join(sorted(set(w for w in t.split() if w not in LEGAL))) for t in fn.values]
    fa = vfold(df["addr"].values)
    num = fa.str.extract(r"(\d+)")[0].fillna("").str.lstrip("0").values
    ast = [frozenset(ABBR.get(w, w) for w in t.split() if not w.isdigit()) for t in fa.values]
    sk = [max((w for w in a if len(w) >= 4), key=len, default="") for a in ast]
    return pd.DataFrame({"core": core, "num": num, "ast": ast, "skey": sk, "cty": df["country"].values, "aempty": (fa.values == "")},
                        index=df["id"].values)


def jac(a, b):
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))


def max_claimer(s, r, p, acc):
    """keep an accepted pair only if it is the highest-p accepting S1 of its record."""
    idx = np.flatnonzero(acc)
    d = pd.DataFrame({"i": idx, "r": r[idx], "p": p[idx]}).sort_values(["r", "p"], ascending=[True, False], kind="mergesort")
    keep = d.drop_duplicates("r", keep="first").i.values
    out = np.zeros(len(p), bool); out[keep] = True
    return out


def runner_up(r, p, s):
    """max p of each row's record over OTHER S1 rows (0 if none)."""
    d = pd.DataFrame({"r": r, "p": p})
    g = d.groupby("r")["p"]
    m1 = g.transform("max").values
    # second max: rank within record
    rk = d.groupby("r")["p"].rank(method="first", ascending=False).values
    d2 = d[rk == 2].set_index("r")["p"]
    m2 = pd.Series(r).map(d2).fillna(0.0).values
    is_top = (rk == 1)
    return np.where(is_top, m2, m1)


def build(s, r, p, acc, P, S1P, extra):
    """s,r: id arrays (all pool rows); p prob; acc accepted (after max-claimer); P parsed table for s and r ids; S1P: s1 twin counts."""
    t = time.time()
    prun = runner_up(r, p, s)
    keep = p >= 0.30
    D = pd.DataFrame({"s": s[keep], "r": r[keep], "p": p[keep], "acc": acc[keep], "p_run": prun[keep]})
    for k, v in extra.items(): D[k] = v[keep]
    A = P.loc[D.s.values]; B = P.loc[D.r.values]
    cs, cr = A.core.values, B.core.values
    nm = np.empty(len(D), object)
    for i, (a, b) in enumerate(zip(cs, cr)):
        if not a or not b: nm[i] = "na"
        elif a == b: nm[i] = "eq"
        else:
            sa, sb = set(a.split()), set(b.split())
            nm[i] = "near" if len(sa & sb) / len(sa | sb) >= 0.5 else "far"
    ns, nr = A.num.values, B.num.values
    num = np.where((ns == "") | (nr == ""), "na", np.where(ns == nr, "eq", "diff"))
    st = np.array(["na" if e else ("eq" if jac(a, b) >= 0.5 else "diff") for a, b, e in zip(A.ast.values, B.ast.values, B.aempty.values)], object)
    D["nm"], D["num"], D["st"] = nm, num, st
    D["r_core"], D["r_num"] = cr, nr
    D["s_num"] = ns
    D["n_acc"] = D.groupby("s")["acc"].transform("sum").astype(int).values
    D["n_o"] = D.n_acc - D.acc.astype(int)
    for F, col in (("nm", "nm"), ("num", "num"), ("st", "st")):
        eq = (D[col] == "eq") & D.acc; df_ = (D[col].isin(["diff", "far"] if F != "nm" else ["far", "near"])) & D.acc
        D[f"o_eq_{F}"] = D.assign(z=eq).groupby("s")["z"].transform("sum").values - (eq & D.acc).values
        D[f"o_diff_{F}"] = D.assign(z=df_).groupby("s")["z"].transform("sum").values - (df_ & D.acc).values
    D["lonely_nm"] = D.nm.isin(["near", "far"]) & (D.n_o >= 1) & (D.o_diff_nm == 0) & (D.o_eq_nm >= 1)
    D["lonely_nmfar"] = (D.nm == "far") & (D.n_o >= 1) & (D.o_diff_nm == 0) & (D.o_eq_nm >= 1)
    D["lonely_num"] = (D.num == "diff") & (D.n_o >= 1) & (D.o_diff_num == 0) & (D.o_eq_num >= 1)
    D["lonely_st"] = (D.st == "diff") & (D.n_o >= 1) & (D.o_diff_st == 0) & (D.o_eq_st >= 1)
    # coherent second number group: other accepted records carrying r's (non-empty, != s) number
    acc_num = D[D.acc & (D.r_num != "")].groupby(["s", "r_num"]).size()
    key = pd.MultiIndex.from_arrays([D.s.values, D.r_num.values])
    cnt = pd.Series(acc_num.reindex(key).fillna(0).values, index=D.index) - (D.acc & (D.r_num != "")).astype(int)
    D["grp2_num"] = (D.num == "diff") & (cnt.values >= 1)
    acc_cn = D[D.acc].groupby(["s", "r_core", "r_num"]).size()
    key = pd.MultiIndex.from_arrays([D.s.values, D.r_core.values, D.r_num.values])
    D["sib_twin"] = acc_cn.reindex(key).fillna(0).values - D.acc.astype(int).values
    eqs = (D.nm == "eq") & (D.num == "eq") & D.acc
    D["supp_s"] = D.assign(z=eqs).groupby("s")["z"].transform("sum").values - eqs.values.astype(int)
    D["s1_twins"] = S1P.reindex(D.s.values).fillna(0).values
    log(f"  built {len(D):,} rows in {time.time()-t:.1f}s")
    return D.drop(columns=["r_core"])


def s1_twin_counts(s1df):
    P = parse(s1df)
    k = P.cty + "|" + P.core + "|" + P.num + "|" + P.skey
    bad = (P.core == "") | P.aempty
    vc = k[~bad].value_counts()
    return (k.map(vc).fillna(1) - 1).where(~bad, 0)


def main():
    which = sys.argv[1:] or ["L", "US", "India", "France"]
    if "L" in which:
        tr = pickle.load(open(os.path.join(RL, "cache", "train.pkl"), "rb"))
        log("train loaded")
        sets = []
        for n in ["V1", "T0", "E014", "T2X"]:
            m = np.load(os.path.join(EXP, "E024", f"{n}_a50n10d10a", "meta.npz"))
            sets.append({k: m[k] for k in ["s1_ids", "s1idx", "cand", "y", "n_gt", "country"]} | {"name": n})
        pV = np.load(os.path.join(EXP, "E026_rrL", "cache", "p_RRL_V1_s42.npy")).astype(np.float64)
        pT = np.load(os.path.join(EXP, "E026_rrL", "cache", "p_oof_RRL_s42.npy")).astype(np.float64)
        s = np.concatenate([S["s1_ids"][S["s1idx"]] for S in sets]); r = np.concatenate([S["cand"] for S in sets])
        y = np.concatenate([S["y"] for S in sets]).astype(np.int8); p = np.r_[pV, pT]
        setn = np.concatenate([np.full(len(S["cand"]), S["name"]) for S in sets])
        assert len(p) == len(s)
        acc = max_claimer(s, r, p, p >= 0.72)
        need = set(s[p >= 0.3]) | set(r[p >= 0.3])
        rec = pd.concat([tr["s1"], tr["s2"], tr["s3"]]); P = parse(rec[rec.id.isin(need)]); log("parsed", len(P))
        S1P = s1_twin_counts(tr["s1"]); log("s1 twins")
        D = build(s, r, p, acc, P, S1P, {"y": y, "set": setn})
        gt = pd.concat([pd.DataFrame({"s1": S["s1_ids"], "n_gt": S["n_gt"], "country": S["country"], "set": S["name"]}) for S in sets])
        # GT owner elsewhere (for V1 FP analysis only, never a feature)
        owner = tr["gt"].drop_duplicates("rec").set_index("rec")["s1"]
        D["true_owner"] = pd.Series(D.r.values).map(owner).fillna("").values
        Lcov = set(gt.s1)
        D["owner_in_L"] = pd.Series(D.true_owner.values).isin(Lcov).values
        pickle.dump(dict(D=D, gt=gt, acc_full=None), open(os.path.join(HERE, "E_feat_L.pkl"), "wb"), protocol=4)
        # also per-S1 accepted TP/NA over ALL pool rows (for F computation)
        n_acc_all = pd.DataFrame({"s": s, "acc": acc, "tp": acc & (y == 1)}).groupby("s")[["acc", "tp"]].sum()
        pickle.dump(n_acc_all, open(os.path.join(HERE, "E_L_s1acc.pkl"), "wb"), protocol=4)
        log("L done")
        del tr
    te = None
    for c in [w for w in which if w != "L"]:
        if te is None:
            te = pickle.load(open(os.path.join(RL, "cache", "test.pkl"), "rb")); log("test loaded")
            rec = pd.concat([te["s1"], te["s2"], te["s3"]]).set_index("id", drop=False)
            S1P = s1_twin_counts(te["s1"]); log("test s1 twins")
        z = np.load(os.path.join(RL, "rl31", "test_scores", f"{c}.npz"), allow_pickle=True)
        s, r = z["s1"], z["cand"]
        out = {}
        for pk, th in (("p6", 0.72), ("p5", 0.78)):
            p = z[pk].astype(np.float64)
            acc = max_claimer(s, r, p, p >= th)
            m3 = p >= 0.3
            need = set(s[m3]) | set(r[m3])
            P = parse(rec.loc[list(need)].reset_index(drop=True)); log(c, pk, "parsed", len(P))
            out[pk] = build(s, r, p, acc, P, S1P, {})
            out[pk + "_nacc_all"] = int(acc.sum()); out["nS1"] = len(z["u"])
        pickle.dump(out, open(os.path.join(HERE, f"E_feat_{c}.pkl"), "wb"), protocol=4)
        log(c, "done")


if __name__ == "__main__":
    main()
