"""WD step 6: selection check of D's shape statistic. D's k = S005-final (s1,source) group size; D evaluates only ACCEPTED swaps.
Here the same statistic (k counted as if the record were in the group) is computed for MODEL-FREE same-(number,street) pairs,
accepted AND rejected, by type. Also: slot position of the swapped token (in-place vs appended). Writes rl30/WD_adv_6_results.json."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import OUT, accepted, load, toks
T = load("test", verbose=False)
a = accepted("S005_France"); a = a[a.kept_final]
kk = a.assign(src=a.rec.str[:2]).groupby(["s1", "src"]).size()
ids = T["s1"].id[T["s1"].country == "France"]
allk = pd.concat([kk.xs(s, level="src").reindex(ids, fill_value=0) for s in ("S2", "S3")]).values
pk = np.bincount(allk) / len(allk); kv = np.arange(len(pk)); E = (kv * pk).sum()
REF = dict(REP_p1=float(pk[1] / E), ADD_p1=float(pk[0]))
m = pd.read_pickle(os.path.join(OUT, "WD_adv_4_pairs_test_France.pkl"))
m["k"] = kk.reindex(pd.MultiIndex.from_arrays([m.s1, m.src])).fillna(0).values.astype(int)
m["k_as_in"] = np.where(m.fin, m.k, m.k + 1)
# is the record claimed (final) by ANOTHER S1?
owner = a.groupby("rec").s1.first()
m["other_owner"] = m.rec.map(owner).notna() & ~m.fin


def f(x):
    p1 = float((x.k_as_in == 1).mean())
    return dict(n=int(len(x)), p1=round(p1, 4), f_p1=round((REF["REP_p1"] - p1) / (REF["REP_p1"] - REF["ADD_p1"]), 2),
                se=round(float(np.sqrt(p1 * (1 - p1) / len(x)) / (REF["REP_p1"] - REF["ADD_p1"])), 2),
                other_owner=round(float(x.other_owner.mean()), 3), dupname=round(float((x.n_dupname > 0).mean()), 4))


rows = {}
for (t, e, fi), x in m.groupby(["typ", "exact", "fin"]):
    if len(x) >= 200:
        rows[f"{t}|exact={e}|final={fi}"] = f(x)
# like-with-like: rejected content swaps that are singleton names and not claimed by another S1
cs = m[m.typ == "CSWAP"]
for e in (False, True):
    x = cs[(cs.exact == e) & ~cs.fin & (cs.n_dupname == 0) & ~cs.other_owner]
    rows[f"CSWAP|exact={e}|rejected_singleton_unclaimed"] = f(x)
    x = cs[(cs.exact == e) & cs.fin]
    rows[f"CSWAP|exact={e}|accepted(check)"] = f(x)
tab = pd.DataFrame(rows).T
print("REF", REF); print(tab.to_string())
# slot analysis: relative position of the S1-only token in the S1 name and of the record-only token in the record name
s1n = T["s1"].set_index("id").name; recn = pd.concat([T["s2"], T["s3"]]).set_index("id").name


def pos(name, t):
    tt = toks(name)
    return tt.index(t) / max(len(tt) - 1, 1) if t in tt else np.nan, (tt.index(t) == len(tt) - 1) if t in tt else False


slot = {}
for lab, x in (("CSWAP_final", cs[cs.fin]), ("CSWAP_rejected", cs[~cs.fin]), ("NSWAP_final", m[(m.typ == "NSWAP") & m.fin]),
               ("GSWAP_final", m[(m.typ == "GSWAP") & m.fin])):
    x = x.sample(min(len(x), 20000), random_state=0)
    pa = [pos(s1n.at[s], t) for s, t in zip(x.s1, x.ta)]; pb = [pos(recn.at[r], t) for r, t in zip(x.rec, x.tb)]
    ra = np.array([p[0] for p in pa]); rb = np.array([p[0] for p in pb])
    slot[lab] = dict(n=len(x), mean_relpos_S1tok=round(float(np.nanmean(ra)), 3), mean_relpos_rectok=round(float(np.nanmean(rb)), 3),
                     rec_tok_last=round(float(np.mean([p[1] for p in pb])), 3), same_relpos=round(float(np.nanmean(np.abs(ra - rb) < 0.01)), 3))
print(pd.DataFrame(slot).T.to_string())
json.dump(dict(REF=REF, rows=rows, slot=slot), open(os.path.join(OUT, "WD_adv_6_results.json"), "w"), indent=1)
