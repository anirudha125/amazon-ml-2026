"""RL-30 / VE-L follow-up (label-free, test France + US/India sample from VE_L2_test_ndpairs.pkl).
(a) veto side: kept high-LLR real-word near-dup pairs -> ESTIMATED per-pair F0.5 gain if FP / loss if TP from the S1's own
    kept-prediction count k (assumes the S1's other kept predictions are correct), break-even FP fraction, France pp bound.
(b) rescue side: same-address real-word near-dup pairs with filler-type differences (llr < 0) that NEW rejected: is the
    record claimed by another S1 (then NEW chose a rival, not a miss), and do existing RL-27 columns (coloc, dupf, rv_rank)
    already describe them?   READ-ONLY; writes rl30/VE_L3_bounds.json"""
import os, sys, json, pickle
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from rl30_lib import *

K = pickle.load(open(os.path.join(HERE, "VE_L2_test_ndpairs.pkl"), "rb"))
N_S1 = dict(France=259452, US=663106, India=809986)
out = {}
for c, df in K.items():
    a = accepted(f"S005_{c}")
    kcount = a[a.kept_final].groupby("s1").size()
    claimed = set(a.rec.values)
    claimed_kept = a[a.kept_final].groupby("rec").s1.first()
    R = {}
    for th in (1, 2):
        m = (df.kept & df.real & (df.llr_max >= th)).values
        x = df[m].copy(); k = x.s1.map(kcount).fillna(1).values.astype(float)
        # if FP: other k-1 correct, n_gt = k-1 ; if TP: all k correct, n_gt = k
        before_fp = np.where(k - 1 > 0, 1.25 * (k - 1) / (0.25 * (k - 1) + k), 0.0); gain = 1.0 - before_fp
        after_tp = np.where(k - 1 > 0, 1.25 * (k - 1) / (0.25 * k + (k - 1)), 0.0); loss = 1.0 - after_tp
        G, L = float(gain.mean()), float(loss.mean())
        R[f"veto_kept_real_llr>={th}"] = dict(pairs=int(m.sum()), s1=int(x.s1.nunique()), mean_gain_if_FP=round(G, 4), mean_loss_if_TP=round(L, 4),
                                             break_even_FP_fraction=round(L / (L + G), 4),
                                             country_pp_if_100pct_FP=round(float(gain.sum()) / N_S1[c] * 100, 4),
                                             country_pp_at_V1_FP_rate_0p34pct=round(float(0.0034 * gain.sum() - 0.9966 * loss.sum()) / N_S1[c] * 100, 4),
                                             subset_rv_rank1_dupf0=int(((x.rv_rank <= 1) & (x.dupf <= 0)).sum()))
    # rescue side
    m = (df.real & df.same_addr & (df.llr_max < 0) & ~df.acc).values
    x = df[m]
    rc = x.rec.isin(claimed).values
    R["rescue_sameaddr_filler_rejected"] = dict(pairs=int(m.sum()), s1=int(x.s1.nunique()), record_claimed_by_some_S1=int(rc.sum()),
                                                record_unclaimed=int((~rc).sum()), unclaimed_s1=int(x.s1[~rc].nunique()),
                                                unclaimed_share_coloc_gt0=round(float((x.coloc[~rc] > 0).mean()), 4) if (~rc).any() else None,
                                                unclaimed_share_dupf_gt0=round(float((x.dupf[~rc] > 0).mean()), 4) if (~rc).any() else None,
                                                unclaimed_share_rv_rank_gt1=round(float((x.rv_rank[~rc] > 1).mean()), 4) if (~rc).any() else None,
                                                unclaimed_share_s1_has_no_kept_pred=round(float((~x.s1[~rc].isin(kcount.index)).mean()), 4) if (~rc).any() else None,
                                                unclaimed_rank_dense_median=float(np.nanmedian(x.rank_dense[~rc])) if (~rc).any() else None,
                                                top_diff_tokens=pd.Series((x.s1_only[~rc] + "->" + x.rec_only[~rc]).values).value_counts().head(15).to_dict())
    ex = x[~rc].sample(min(8, int((~rc).sum())), random_state=0) if (~rc).any() else x.head(0)
    T = load("test", verbose=False) if c == "France" else None
    if T is not None:
        S1 = T["s1"].set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
        R["rescue_examples"] = [dict(s1_name=S1.name.loc[r.s1], s1_addr=S1.addr.loc[r.s1], rec_name=recs.name.loc[r.rec], rec_addr=recs.addr.loc[r.rec],
                                     llr_max=round(float(r.llr_max), 2), coloc=float(r.coloc), dupf=float(r.dupf), rv_rank=float(r.rv_rank))
                                for r in ex.itertuples()]
        hv = df[(df.kept & df.real & (df.llr_max >= 2)).values].sample(8, random_state=0)
        R["veto_examples_kept_llr>=2"] = [dict(s1_name=S1.name.loc[r.s1], s1_addr=S1.addr.loc[r.s1], rec_name=recs.name.loc[r.rec], rec_addr=recs.addr.loc[r.rec],
                                               p=round(float(r.p), 4), llr_max=round(float(r.llr_max), 2), rv_rank=float(r.rv_rank), dupf=float(r.dupf))
                                          for r in hv.itertuples()]
        rj = df[(df.real & (df.llr_max >= 2) & ~df.acc & df.same_addr).values].sample(8, random_state=0)
        R["rejected_same_addr_llr>=2_examples"] = [dict(s1_name=S1.name.loc[r.s1], rec_name=recs.name.loc[r.rec], rec_addr=recs.addr.loc[r.rec],
                                                        llr_max=round(float(r.llr_max), 2), rv_rank=float(r.rv_rank), dupf=float(r.dupf), claimed_by_other=bool(r.rec in claimed))
                                                   for r in rj.itertuples()]
        m2 = (df.real & (df.llr_max >= 2) & ~df.acc & df.same_addr).values
        R["rejected_same_addr_llr>=2"] = dict(pairs=int(m2.sum()), record_claimed_by_other_S1=int(df.rec[m2].isin(claimed).sum()),
                                              share_rv_rank_gt1=round(float((df.rv_rank[m2] > 1).mean()), 4))
    out[c] = R
    print(c, json.dumps(R, default=str, ensure_ascii=False)[:4000], flush=True)
json.dump(out, open(os.path.join(HERE, "VE_L3_bounds.json"), "w"), indent=1, default=str, ensure_ascii=False)
