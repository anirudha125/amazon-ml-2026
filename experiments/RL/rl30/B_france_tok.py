"""RL-30 investigator B -- follow-up to B_france.py (LABEL-FREE on test; V1 part uses stored arrays only).
(a) France kept_final pairs whose name symmetric difference contains a NON-legal top-25 co-located contrast token (from B_france.json):
    count, per-1k S1, p distribution, and how often the same S1 has a co-located S1 carrying the record's differing token.
(b) V1 baseline of RL-27 nf_a / rv_rank==1 on accepted pairs (p_new >= NEW_TH), for comparison with the France kept_final values.
READ-ONLY; writes rl30/B_france_tok.json.
"""
import os, sys, json, time
import numpy as np, pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from rl30_lib import *  # noqa

t0 = time.time()
BF = json.load(open(os.path.join(OUT, "B_france.json")))
CON = [d["tok"] for d in BF["S2_token_role_France"]["top_contrast_tokens"] if d["tok"] not in LEGAL]


def main():
    res = dict(nonlegal_contrast_tokens=CON)
    T = load("test", verbose=False)
    s1 = T["s1"][T["s1"].country == "France"].reset_index(drop=True)
    rec = pd.concat([T["s2"], T["s3"]], ignore_index=True); rec = rec[rec.country == "France"].set_index("id")
    A = accepted("S005_France"); A = A[A.kept_final.astype(bool)].reset_index(drop=True)
    ix = pd.Index(s1.id.values).get_indexer(A.s1.values)
    an = s1.name.values[ix]; rn = rec.name.reindex(A.rec.values).values
    ak = s1.addr.map(akey).values
    grp = pd.Series(np.arange(len(s1))).groupby(ak).apply(list).to_dict()
    cset = set(CON); hit = np.zeros(len(A), bool); coloc_has_tok = np.zeros(len(A), bool); tokdiff = []
    for k in range(len(A)):
        S, R_ = frozenset(toks(an[k])), frozenset(toks(rn[k]))
        d = (S ^ R_) & cset
        if d and (S & R_):
            hit[k] = True; tokdiff.append(sorted(d))
            rec_only = (R_ - S) & cset
            if rec_only and ak[ix[k]]:
                for j in grp.get(ak[ix[k]], []):
                    if j != ix[k] and rec_only & frozenset(toks(s1.name.values[j])):
                        coloc_has_tok[k] = True; break
    p = A.p.values; n_s1 = A.s1.nunique()
    res["France_kept_nonlegal_contrast_diff"] = dict(
        n_pairs=int(hit.sum()), per_1k_s1=round(1000 * hit.sum() / n_s1, 2), share_of_kept=round(float(hit.mean()), 4),
        p_mean=round(float(p[hit].mean()), 4), share_p_ge_0_97=round(float((p[hit] >= 0.97).mean()), 4),
        n_record_token_carried_by_colocated_S1=int(coloc_has_tok.sum()),
        p_mean_record_token_carried_by_colocated_S1=round(float(p[coloc_has_tok].mean()), 4) if coloc_has_tok.any() else None,
        top_diff_tokens=pd.Series([t for d in tokdiff for t in d]).value_counts().head(12).to_dict())
    ex = []
    for k in np.flatnonzero(coloc_has_tok)[:1000:100]:
        mem = [s1.name.values[j] for j in grp.get(ak[ix[k]], []) if j != ix[k]][:4]
        ex.append(dict(s1_name=an[k], rec_name=rn[k], p=round(float(p[k]), 3), colocated=mem, addr=s1.addr.values[ix[k]]))
    res["examples_record_token_carried_by_colocated_S1"] = ex
    # V1 baseline for RL-27 columns on accepted pairs
    R = np.load(PATHS["v1_rl27"]); pv = np.load(PATHS["v1_p_new"]); acc = pv >= NEW_TH
    res["V1_accepted_rl27"] = dict(n=int(acc.sum()), nf_a_eq1=round(float(np.nanmean(R[acc, 1] == 1)), 4),
                                   rv_rank_eq1=round(float(np.mean(R[acc, 6] == 1)), 4), coloc_gt0=round(float(np.mean(R[acc, 4] > 0)), 4))
    res["runtime_s"] = round(time.time() - t0, 1)
    json.dump(res, open(os.path.join(OUT, "B_france_tok.json"), "w"), indent=1, default=str, ensure_ascii=False)
    print(json.dumps(res, indent=1, default=str, ensure_ascii=False)[:4000])


if __name__ == "__main__":
    main()
