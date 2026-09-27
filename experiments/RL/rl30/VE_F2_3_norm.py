"""RL-30 / VE_F2 -- is F2_ge's firing in the France residual (pairs the S005 max-claimer keeps) directional at all?
F2_ge fires on ties (Jaccard(record, sibling) == Jaccard(record, this S1)). Many France ties come from dotted legal forms
("S.A.S." -> tokens s,a,s) that break the token Jaccard. Re-score the flagged pairs after collapsing runs of single-letter
tokens ("s a r l" -> "sarl") and count, among kept / dropped flagged pairs: sibling strictly preferred / tie / this S1 preferred.
Label-free, READ-ONLY.  Output: rl30/VE_F2_3_norm.json"""
import os, sys, json, time, pickle, collections
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from E_common import *

log = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)


def ntoks(s):
    out, run = [], []
    for t in toks(s):
        if len(t) == 1 and t.isalpha():
            run.append(t); continue
        if run:
            out.append("".join(run)) if len(run) >= 2 else out.extend(run); run = []
        out.append(t)
    if run:
        out.append("".join(run)) if len(run) >= 2 else out.extend(run)
    return frozenset(out)


def main():
    t0 = time.time()
    T = load("test", verbose=False)
    s1all = T["s1"]; S1 = s1all.set_index("id"); recs = pd.concat([T["s2"], T["s3"]]).set_index("id")
    RO = pickle.load(open(os.path.join(HERE, "E_roles_test.pkl"), "rb")); SIB = RO["sib"]
    need = set(SIB.keys())
    ak_all = dict(zip(s1all.id.values, (akey(a) for a in s1all.addr.values)))
    grp = collections.defaultdict(list)
    for sid, c in zip(s1all.id.values, s1all.country.values):
        if ak_all[sid]:
            grp[c + "|" + ak_all[sid]].append(sid)
    out = {}; ex = collections.defaultdict(list)
    for c in ("France", "India", "US"):
        a = accepted(f"S005_{c}").reset_index(drop=True)
        cnt = collections.Counter()
        for i in np.flatnonzero(a.s1.isin(need).values):
            s = a.s1[i]; A = nset(S1.name.loc[s]); R = nset(recs.name.loc[a.rec[i]])
            jr = jac(R, A); js = max(jac(R, B) for B in SIB[s])
            if not (js > 0 and js >= jr):
                continue
            sibs = [j for j in grp[c + "|" + ak_all[s]] if j != s and near_dup(A, nset(S1.name.loc[j]), need_core=False)]
            An = ntoks(S1.name.loc[s]); Rn = ntoks(recs.name.loc[a.rec[i]])
            jrn = jac(Rn, An); jsn = max(jac(Rn, ntoks(S1.name.loc[j])) for j in sibs) if sibs else 0.0
            kind = "gt" if js > jr else "tie"
            d = "sib>" if jsn > jrn else ("self>" if jrn > jsn else "tie")
            st = "kept" if a.kept_final[i] else "dropped"
            cnt[f"{st}|{kind}|norm_{d}"] += 1
            if c == "France" and st == "kept" and len(ex[d]) < 12:
                ex[d].append(dict(s1=S1.name.loc[s], rec=recs.name.loc[a.rec[i]], sibs=[S1.name.loc[j] for j in sibs], p=round(float(a.p[i]), 4),
                                  jr=round(jr, 3), js=round(js, 3), jr_norm=round(jrn, 3), js_norm=round(jsn, 3)))
        agg = collections.Counter()
        for k, v in cnt.items():
            st, kind, d = k.split("|")
            agg[f"{st}|{d}"] += v
        out[c] = dict(detail=dict(sorted(cnt.items())), by_state_and_normalized_direction=dict(sorted(agg.items())))
        log(c, out[c])
    out["France_kept_examples"] = dict(ex)
    json.dump(out, open(os.path.join(HERE, "VE_F2_3_norm.json"), "w"), indent=1, default=str)
    log(f"done {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
