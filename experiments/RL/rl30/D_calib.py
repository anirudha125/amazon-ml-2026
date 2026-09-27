"""RL-30 Part 5 (investigator D): calibration of the group-size shape ('additive vs replacing') test.
(a) V1 with labels: for TRUE records (y=1) of type T, shape of the TRUE per-(s1,source) count -> is additivity a generator
    property of true extras (e.g. alias records) rather than a false-positive signature?
(b) US / India test accepted: shape per type (same code as France).
(c) source split (S2/S3) per type in France accepted vs V1 positives.
Writes rl30/D_calib.json (NEW file)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted, PATHS
pd.set_option("display.width", 250); pd.set_option("display.max_rows", 200)
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test"); Tr = load("train")
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl")); ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl"))
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read())


def refs(kk, ids):
    allk = pd.concat([kk.xs(s, level="src").reindex(ids, fill_value=0) if s in kk.index.get_level_values(1) else pd.Series(0, index=ids)
                      for s in ("S2", "S3")]).values
    pk = np.bincount(allk) / len(allk); kv = np.arange(len(pk)); sb = kv * pk / (kv * pk).sum()
    return dict(mean=float((kv * sb).sum()), p1=float(sb[1])), dict(mean=float((kv * pk).sum() + 1), p1=float(pk[0]))


def frac(kk, sel, REP, ADD):
    kv = kk.reindex(pd.MultiIndex.from_frame(sel[["s1", "src"]].drop_duplicates())).fillna(0).values
    if len(kv) == 0:
        return None
    m, p1 = kv.mean(), (kv == 1).mean()
    return dict(n=int(len(kv)), mean=round(float(m), 3), p1=round(float(p1), 4),
                f_mean=round(float((m - REP["mean"]) / (ADD["mean"] - REP["mean"])), 2),
                f_p1=round(float((REP["p1"] - p1) / (REP["p1"] - ADD["p1"])), 2),
                se=round(float(np.sqrt(p1 * (1 - p1) / len(kv)) / (REP["p1"] - ADD["p1"])), 2))


def type_masks(d):
    same_a = d.arel.isin(["exact_addr", "same_num_street"])
    return {"content_swap&same_addr": (d.swapcls == "content_word") & same_a, "to_noise_suffix&same_addr": (d.swapcls == "to_noise_suffix") & same_a,
            "garble&same_addr": (d.swapcls == "garble_or_rare") & same_a, "abbrev&same_addr": (d.swapcls == "abbrev") & same_a,
            "N_TYPO&same_addr": (d.nt == "N_TYPO") & same_a,
            "N_DROP|exact": (d.nt == "N_DROP") & (d.arel == "exact_addr"), "N_DROP|same_num_street": (d.nt == "N_DROP") & (d.arel == "same_num_street"),
            "N_ADD|exact": (d.nt == "N_ADD") & (d.arel == "exact_addr"), "N_ADD|same_num_street": (d.nt == "N_ADD") & (d.arel == "same_num_street"),
            "N_DISJOINT|exact": (d.nt == "N_DISJOINT") & (d.arel == "exact_addr"), "N_DISJOINT|same_num_street": (d.nt == "N_DISJOINT") & (d.arel == "same_num_street"),
            "N_ACRONYM|exact": (d.nt == "N_ACRONYM") & (d.arel == "exact_addr"), "N_ALIAS|exact": (d.nt == "N_ALIAS") & (d.arel == "exact_addr"),
            "N_ALIAS|any": d.nt == "N_ALIAS", "N_HANDLE|same_num_street": (d.nt == "N_HANDLE") & (d.arel == "same_num_street"),
            "N_NONLATIN|any": d.nt == "N_NONLATIN", "N_SAME|num_shift": (d.nt == "N_SAME") & (d.arel == "num_shift_same_street"),
            "N_SAME|street_sub": (d.nt == "N_SAME") & (d.arel == "street_sub_same_num"),
            "any|rec_addr_empty": d.arel == "rec_addr_empty"}


out = {}
# (a) V1 truth
m = np.load(PATHS["v1_meta"])
tv = pd.DataFrame(dict(s1=m["s1_ids"][m["s1idx"]], rec=m["cand"], y=m["y"]))
tv = tv[tv.y == 1]; kk_true = tv.assign(src=tv.rec.str[:2]).groupby(["s1", "src"]).size()
REP, ADD = refs(kk_true, pd.Index(m["s1_ids"]))
L("V1 truth refs", REP, ADD)
vp = v[v.y == 1].assign(src=lambda x: x.rec.str[:2])
out["V1_truth_y1"] = {k: frac(kk_true, vp[np.asarray(mm)], REP, ADD) for k, mm in type_masks(vp).items()}
# decoys (y=0 hard) with +1: expected ~additive by construction; report for completeness on content swaps
# (b) test accepted
for tag, cc, d in (("S005_France", "France", fr), ("S005_US", "US", us), ("S005_India", "India", ind)):
    a = accepted(tag); a = a[a.kept_final]
    kk = a.assign(src=a.rec.str[:2]).groupby(["s1", "src"]).size()
    R_, A_ = refs(kk, T["s1"].id[T["s1"].country == cc])
    dk = d[d.kept_final].assign(src=lambda x: x.rec.str[:2])
    out[f"test_{cc}"] = {k: frac(kk, dk[np.asarray(mm)], R_, A_) for k, mm in type_masks(dk).items()}
tb = pd.DataFrame({k: {t: (f"{x['f_p1']:+.2f}±{x['se']:.2f} (n={x['n']})" if x else "") for t, x in vv.items()} for k, vv in out.items()})
L(tb.to_string())
R["shape_f_p1"] = out
# (c) source split
src = {}
for nm, d in (("France", fr[fr.kept_final]), ("US", us[us.kept_final]), ("India", ind[ind.kept_final]), ("V1_pos", v[v.y == 1])):
    dd = d.assign(src=d.rec.str[:2])
    src[nm] = {k: round(float((dd[np.asarray(mm)].src == "S3").mean()), 3) if np.asarray(mm).sum() else None for k, mm in type_masks(dd).items()}
    src[nm]["ALL"] = round(float((dd.src == "S3").mean()), 3)
L("\nshare from S3 per type:"); L(pd.DataFrame(src).to_string())
R["share_S3"] = src
json.dump(R, open(os.path.join(OUT, "D_calib.json"), "w"), indent=1, default=str)
L("wrote D_calib.json")
