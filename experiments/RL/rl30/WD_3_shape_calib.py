"""WD part 3: is D's group-size 'shape test' f a decoy-fraction estimator? Calibrate it on labelled V1 exactly the way it is
applied to France: groups = (s1, source) of ACCEPTED records (RL-27 NEW p >= 0.78), references from V1's own accepted
k distribution, f_p1 per class; compare with the class's MEASURED false-positive share among accepted rows.
Also: (i) f on accepted y=0 rows vs accepted y=1 rows (the principle), (ii) f of accepted TRUE rows by p bin (model-selection
bias), (iii) France: f of benign classes restricted to S1 whose names contain a business word (confound check).
Writes rl30/WD_3_shapecalib_results.json"""
import os, sys, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, accepted, PATHS, NEW_TH

L = lambda *a: print(*a, flush=True)
R = {}
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl")); ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl"))
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read())   # D's own arel / swapcls definitions (so f is computed exactly as D does)

m = np.load(PATHS["v1_meta"]); p = np.load(PATHS["v1_p_new"])
allv = pd.DataFrame(dict(s1=m["s1_ids"][m["s1idx"]], rec=m["cand"], y=m["y"], p=p))
acc = allv[allv.p >= NEW_TH].assign(src=lambda x: x.rec.str[:2])
kk = acc.groupby(["s1", "src"]).size()
ids = pd.Index(m["s1_ids"])


def refs(kk, ids):
    allk = pd.concat([kk.xs(s, level="src").reindex(ids, fill_value=0) for s in ("S2", "S3")]).values
    pk = np.bincount(allk) / len(allk); kv = np.arange(len(pk)); sb = kv * pk / (kv * pk).sum()
    return dict(p1=float(sb[1]), mean=float((kv * sb).sum())), dict(p1=float(pk[0]), mean=float((kv * pk).sum() + 1))


REP, ADD = refs(kk, ids)
R["V1_acc_refs"] = dict(REP=REP, ADD=ADD)
L("V1 accepted refs", REP, ADD)


def f_of(sel, kk, REP, ADD):
    g = sel.assign(src=sel.rec.str[:2])[["s1", "src"]].drop_duplicates()
    kv = kk.reindex(pd.MultiIndex.from_frame(g)).fillna(0).values
    if len(kv) < 5:
        return dict(n=int(len(kv)))
    p1 = (kv == 1).mean()
    return dict(n=int(len(kv)), p1=round(float(p1), 4), f_p1=round(float((REP["p1"] - p1) / (REP["p1"] - ADD["p1"])), 2),
                se=round(float(np.sqrt(p1 * (1 - p1) / len(kv)) / (REP["p1"] - ADD["p1"])), 2))


va = v[v.p >= NEW_TH].copy()
L("V1 accepted rows in D table:", len(va), "of", len(acc))
same_a = va.arel.isin(["exact_addr", "same_num_street"])
classes = {"ALL": np.ones(len(va), bool),
           "content_swap&same_addr": (va.swapcls == "content_word") & same_a, "to_noise_suffix&same_addr": (va.swapcls == "to_noise_suffix") & same_a,
           "garble&same_addr": (va.swapcls == "garble_or_rare") & same_a, "N_TYPO&same_addr": (va.nt == "N_TYPO") & same_a,
           "N_SAME|exact": (va.nt == "N_SAME") & (va.arel == "exact_addr"), "N_SAME|same_num_street": (va.nt == "N_SAME") & (va.arel == "same_num_street"),
           "N_SAME|num_shift": (va.nt == "N_SAME") & (va.arel == "num_shift_same_street"), "N_SAME|other": (va.nt == "N_SAME") & (va.arel == "other"),
           "N_DROP|exact": (va.nt == "N_DROP") & (va.arel == "exact_addr"), "N_DROP|same_num_street": (va.nt == "N_DROP") & (va.arel == "same_num_street"),
           "N_ADD|exact": (va.nt == "N_ADD") & (va.arel == "exact_addr"), "N_ADD|same_num_street": (va.nt == "N_ADD") & (va.arel == "same_num_street"),
           "N_DISJOINT|exact": (va.nt == "N_DISJOINT") & (va.arel == "exact_addr"), "N_DISJOINT|any": va.nt == "N_DISJOINT",
           "N_ACRONYM|any": va.nt == "N_ACRONYM", "N_ALIAS|any": va.nt == "N_ALIAS", "N_HANDLE|any": va.nt == "N_HANDLE",
           "N_NONLATIN|any": va.nt == "N_NONLATIN", "N_MULTI|any": va.nt == "N_MULTI",
           "rec_addr_empty": va.arel == "rec_addr_empty", "rec_num_missing": va.arel == "rec_num_missing"}
rows = {}
for cn, mk in classes.items():
    s = va[np.asarray(mk)]
    if len(s) < 20:
        continue
    r = f_of(s, kk, REP, ADD); r["n_rows"] = int(len(s)); r["FP_share"] = round(float((s.y == 0).mean()), 4)
    rows[cn] = r
t = pd.DataFrame(rows).T
L(t.to_string())
R["V1_class_f_vs_FP"] = rows
fr_ = t[t.n >= 100]
R["V1_corr_f_FP"] = dict(n_classes=int(len(fr_)), pearson=round(float(np.corrcoef(fr_.f_p1.astype(float), fr_.FP_share.astype(float))[0, 1]), 3),
                         max_abs_err=round(float((fr_.f_p1.astype(float) - fr_.FP_share.astype(float)).abs().max()), 2))
L("corr f vs FP share (classes n>=100):", R["V1_corr_f_FP"])
# (i) principle: accepted FP rows vs accepted TP rows (all types)
R["V1_principle"] = dict(acc_y0=f_of(acc[acc.y == 0], kk, REP, ADD), acc_y1=f_of(acc[acc.y == 1], kk, REP, ADD),
                         n_acc_y0=int((acc.y == 0).sum()), n_acc_y1=int((acc.y == 1).sum()))
L("principle", R["V1_principle"])
# (ii) model-selection: accepted TRUE rows by p bin
bins = [(0.78, 0.9), (0.9, 0.98), (0.98, 0.995), (0.995, 1.01)]
R["V1_true_by_p"] = {f"{a}-{b}": f_of(acc[(acc.y == 1) & (acc.p >= a) & (acc.p < b)], kk, REP, ADD) for a, b in bins}
L("TRUE accepted by p", R["V1_true_by_p"])

# (iii) France confound: benign classes restricted to S1 whose name contains a content word used as a swap source
T = load("test")
a = accepted("S005_France"); a = a[a.kept_final]
kf = a.assign(src=a.rec.str[:2]).groupby(["s1", "src"]).size()
REPf, ADDf = refs(kf, T["s1"].id[T["s1"].country == "France"])
fk = fr[fr.kept_final]
cs = fk[(fk.swapcls == "content_word") & fk.arel.isin(["exact_addr", "same_num_street"])]
src_words = set(cs.sw_a.value_counts().head(30).index)
from rl30_lib import toks
s1n = T["s1"].set_index("id").name
has_bw = fk.s1.map(lambda s: bool(set(toks(s1n[s])) & src_words))
sa_f = fk.arel.isin(["exact_addr", "same_num_street"])
conf = {}
for cn, mk in {"N_SAME|same_num_street": (fk.nt == "N_SAME") & (fk.arel == "same_num_street"),
               "to_noise_suffix&same_addr": (fk.swapcls == "to_noise_suffix") & sa_f,
               "N_TYPO&same_addr": (fk.nt == "N_TYPO") & sa_f,
               "content_swap&same_addr": (fk.swapcls == "content_word") & sa_f}.items():
    conf[cn] = dict(with_bw=f_of(fk[mk & has_bw], kf, REPf, ADDf), without_bw=f_of(fk[mk & ~has_bw], kf, REPf, ADDf))
R["France_confound_business_word_S1"] = conf
R["France_share_S1_with_top30_source_word"] = round(float(T["s1"][T["s1"].country == "France"].name.map(lambda s: bool(set(toks(s)) & src_words)).mean()), 4)
L(json.dumps(conf, indent=0)); L("share of France S1 with a top-30 swap-source word:", R["France_share_S1_with_top30_source_word"])
json.dump(R, open(os.path.join(OUT, "WD_3_shapecalib_results.json"), "w"), indent=1, default=str)
L("wrote WD_3_shapecalib_results.json")
