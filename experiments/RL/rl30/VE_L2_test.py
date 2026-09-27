"""RL-30 / VE-L (adversarial verification of E's token-role table; lens ALREADY CAPTURED).  LABEL-FREE test-side checks.
Scans the stage-1 test candidate pools (P3 chunks: s1, cand, LF; RL-27 test chunks row-aligned) -- all France chunks, a
systematic sample of US / India chunks -- and for every CORE near-duplicate pool pair (<=1 differing name token per side)
records the role LLR of the differing tokens (E_roles_test.pkl, same smoothing as VE_L1), a typo-robust 'real word' bit
(E_flags2 rule), same-address bit, S005 acceptance (p >= 0.78 before max-claimer) + kept_final, TOK16 IDF and RL-27 columns.
Questions: (1) does NEW's acceptance already fall with the role LLR (in France as in V1, where P(match) and acceptance track
each other)?  (2) for the accepted high-LLR pairs, do existing RL-27 / TOK16 columns already mark them?  (3) how large is the
addressable set in France.   READ-ONLY; writes rl30/VE_L2_test.json and rl30/VE_L2_test_ndpairs.pkl"""
import os, sys, time, json, glob, pickle, collections
import multiprocessing as mp
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from rapidfuzz.distance import Levenshtein
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *
from E_flags import s1_feats, rec_feats

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
RLN = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
TOKN = ["n_s1_only", "n_c_only", "idf_s1_only_sum", "idf_c_only_sum", "idf_c_only_max", "idf_s1_only_max", "frac_idf_s1_only",
        "frac_idf_c_only", "n_s1_only_soft", "n_c_only_soft", "idf_s1_only_soft", "idf_c_only_soft"]
NP = 4
_G = {}


def llr_table(T, min_supp=5, alpha=0.5):
    sd, sf = float(T.D.sum()), float(T.F.sum())
    t = T[T.supp >= min_supp]
    return dict(zip(t.index, np.log(((t.D + alpha) / sd) / ((t.F + alpha) / sf)).astype(float)))


def work(path):
    S1F, RF, L, NV, ACC = _G["S1F"], _G["RF"], _G["L"], _G["NV"], _G["ACC"]
    z = np.load(path)
    s1, cand = z["s1"], z["cand"]
    ci = os.path.basename(path).split("_")[1].split(".")[0]
    rp = os.path.join(os.path.dirname(path).replace(os.path.join("P3", ""), os.path.join("RL", "test_feats", "")), f"rl27_chunk_{ci}.npy")
    rows = []
    n_pool = len(s1)
    for k in range(n_pool):
        a1 = S1F.get(s1[k]); r1 = RF.get(cand[k])
        if a1 is None or r1 is None:
            continue
        A = a1[0]; R = r1[0]
        if not near_dup(A, R, need_core=True):
            continue
        a, b = A - R, R - A; d = a | b
        vals = [L[t] for t in d if t in L]
        real = all(NV.get(t, 0) >= 5 for t in d)
        spelled = bool(a and b) and Levenshtein.normalized_similarity(next(iter(a)), next(iter(b))) >= 0.8
        same_addr = a1[2] is not None and a1[2] == r1[1] and bool(a1[3] & r1[2])
        acc = ACC.get(s1[k] + "|" + cand[k])
        rows.append((k, s1[k], cand[k], max(vals) if vals else np.nan, sum(vals) if vals else 0.0, len(d) - len(vals), real and not spelled,
                     same_addr, " ".join(sorted(a)), " ".join(sorted(b)), -1.0 if acc is None else acc[0], False if acc is None else acc[1]))
    if not rows:
        return dict(n_pool=n_pool, df=None)
    df = pd.DataFrame(rows, columns=["row", "s1", "rec", "llr_max", "llr_sum", "n_unk", "real", "same_addr", "s1_only", "rec_only", "p", "kept"])
    LF = z["LF"]; R27 = np.load(rp, mmap_mode="r")
    ii = df.row.values
    for j, nm in enumerate(TOKN):
        df[nm] = LF[ii, 71 + j]
    for j, nm in enumerate(RLN):
        df[nm] = np.asarray(R27[ii, j])
    df["rank_dense"] = z["rank_dense"][ii]
    return dict(n_pool=n_pool, df=df)


def main():
    t0 = time.time()
    T = load("test", verbose=False)
    RO = pickle.load(open(os.path.join(HERE, "E_roles_test.pkl"), "rb"))
    out = {}; keep = {}
    for c, step in (("France", 1), ("US", 22), ("India", 27)):
        s1c = T["s1"][T["s1"].country == c]
        rc = pd.concat([T["s2"], T["s3"]]); rc = rc[rc.country == c]
        NV = collections.Counter()
        for nm in s1c.name.values:
            NV.update(nset(nm))
        S1F = {s: (nset(nm),) + s1_feats(nm, ad)[1:4] for s, nm, ad in zip(s1c.id.values, s1c.name.values, s1c.addr.values)}
        RF = {}
        for r, nm, ad in zip(rc.id.values, rc.name.values, rc.addr.values):
            f = rec_feats(nm, ad); RF[r] = f[:3]
        a = accepted(f"S005_{c}")
        ACC = dict(zip(a.s1.values + "|" + a.rec.values, zip(a.p.values.astype(float), a.kept_final.values.astype(bool))))
        L = llr_table(RO["roles"][c])
        _G.update(S1F=S1F, RF=RF, L=L, NV=NV, ACC=ACC)
        paths = sorted(glob.glob(PATHS["test_chunks"].format(country=c)))[::step]
        log(f"{c}: tables {len(S1F):,} S1 {len(RF):,} rec, {len(paths)} chunks, accepted {len(a):,} {time.time()-t0:.0f}s")
        dfs = []; n_pool = 0
        with mp.get_context("fork").Pool(NP) as pool:
            for r in pool.imap_unordered(work, paths):
                n_pool += r["n_pool"]
                if r["df"] is not None:
                    dfs.append(r["df"])
        df = pd.concat(dfs, ignore_index=True)
        df["acc"] = df.p >= NEW_TH
        log(f"{c}: pool pairs {n_pool:,}, core near-dup {len(df):,}, accepted near-dup {int(df.acc.sum()):,} {time.time()-t0:.0f}s")
        R = dict(chunks=len(paths), pool_pairs=int(n_pool), nd_pairs=int(len(df)), nd_accepted=int(df.acc.sum()),
                 nd_kept=int(df.kept.sum()), accepted_total_pairs_all_chunks=int(len(a)))
        # (1) acceptance rate by llr_max bin
        bins = [-99, -2, -1, 0, 1, 2, 99]
        tab = {}
        for lab, m0 in (("all_nd", np.ones(len(df), bool)), ("real_word", df.real.values), ("real_word_same_addr", df.real.values & df.same_addr.values),
                        ("real_word_not_same_addr", df.real.values & ~df.same_addr.values)):
            rows = []
            for lo, hi in zip(bins[:-1], bins[1:]):
                m = m0 & (df.llr_max.values >= lo) & (df.llr_max.values < hi)
                rows.append(dict(bin=f"[{lo},{hi})", pairs=int(m.sum()), accepted=int((m & df.acc.values).sum()), kept=int((m & df.kept.values).sum()),
                                 acc_rate=round(float(df.acc.values[m].mean()), 4) if m.any() else None))
            tab[lab] = rows
        R["acc_rate_by_llr_bin"] = tab
        # (1b) token-level: acceptance rate of single-token-difference real-word pairs vs s(t)
        Tt = RO["roles"][c]
        tk = collections.defaultdict(lambda: [0, 0])
        rw = df[df.real]
        for so, ro, ac in zip(rw.s1_only.values, rw.rec_only.values, rw.acc.values):
            for t in set((so + " " + ro).split()):
                if t:
                    tk[t][0] += 1; tk[t][1] += int(ac)
        toks_ = [t for t, v in tk.items() if v[0] >= 50 and t in Tt.index and Tt.supp.loc[t] >= 5]
        sv = np.array([Tt.s.loc[t] for t in toks_]); ar = np.array([tk[t][1] / tk[t][0] for t in toks_])
        R["token_level"] = dict(n_tokens=len(toks_), spearman_s_vs_acc_rate=round(float(spearmanr(sv, ar).correlation), 4) if len(toks_) > 5 else None,
                                top_by_pool_count=[(t, tk[t][0], round(tk[t][1] / tk[t][0], 4), round(float(Tt.s.loc[t]), 3)) for t in sorted(toks_, key=lambda t: -tk[t][0])[:30]],
                                high_s_ge_0_9=[(t, tk[t][0], round(tk[t][1] / tk[t][0], 4), round(float(Tt.s.loc[t]), 3)) for t in sorted([t for t in toks_ if Tt.s.loc[t] >= 0.9], key=lambda t: -tk[t][0])[:20]])
        # (2) accepted pairs: existing columns by llr bin (real word)
        acc_rw = df[df.acc & df.real]
        col = {}
        for lab, m in (("llr<0", acc_rw.llr_max < 0), ("0<=llr<1", (acc_rw.llr_max >= 0) & (acc_rw.llr_max < 1)), ("1<=llr<2", (acc_rw.llr_max >= 1) & (acc_rw.llr_max < 2)),
                       ("llr>=2", acc_rw.llr_max >= 2)):
            x = acc_rw[m]
            if not len(x):
                continue
            col[lab] = dict(n=int(len(x)), kept=int(x.kept.sum()), p_median=round(float(x.p.median()), 5), p_q10=round(float(x.p.quantile(0.1)), 4),
                            share_p_lt_0_95=round(float((x.p < 0.95).mean()), 4), share_same_addr=round(float(x.same_addr.mean()), 4),
                            share_rv_rank_gt1=round(float((x.rv_rank > 1).mean()), 4), share_dupf_gt0=round(float((x.dupf > 0).mean()), 4),
                            share_nf_n_gt0=round(float((x.nf_n > 0).mean()), 4), share_coloc_gt0=round(float((x.coloc > 0).mean()), 4),
                            median_rv_gap=round(float(np.nanmedian(x.rv_gap)), 4), median_idf_s1_only_max=round(float(x.idf_s1_only_max.median()), 3),
                            median_idf_c_only_max=round(float(x.idf_c_only_max.median()), 3))
        R["accepted_real_word_existing_columns_by_llr"] = col
        # (3) addressable set: kept, real-word, llr >= 1 / >= 2; per-S1 counts
        n_s1_c = int((T["s1"].country == c).sum())
        for th in (1, 2):
            m = df.kept & df.real & (df.llr_max >= th)
            R[f"kept_real_llr>={th}"] = dict(pairs=int(m.sum()), s1=int(df.s1[m].nunique()), share_country_s1=round(df.s1[m].nunique() / n_s1_c, 5),
                                             same_addr=int((m & df.same_addr).sum()), rv_rank_eq1=int((m & (df.rv_rank <= 1)).sum()),
                                             rv_rank_eq1_and_dupf0=int((m & (df.rv_rank <= 1) & (df.dupf <= 0)).sum()))
        out[c] = R
        log(c, json.dumps(R, default=str)[:3500])
        keep[c] = df
    pickle.dump(keep, open(os.path.join(HERE, "VE_L2_test_ndpairs.pkl"), "wb"), protocol=pickle.HIGHEST_PROTOCOL)
    json.dump(out, open(os.path.join(HERE, "VE_L2_test.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
