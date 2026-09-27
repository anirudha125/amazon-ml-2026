"""RL-30 Part 5 (investigator D): label-free count tests.
(1) per-S1 accepted-record count histogram (final, after max-claimer) per test country vs train gt per country.
(2) per-source 'additive vs replacing' test: for S1 that own >=1 accepted record of type T from source S,
    mean #accepted records from S. A decoy of type T adds one record on top of the true ones (mean ~ E[k]+1);
    a noisy copy of a true record replaces one (size-biased mean ~ E[k^2]/E[k]). Calibrated on V1 labels.
Writes rl30/D_counts.json (NEW file)."""
import sys, os, json
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, NEW_TH, accepted, PATHS
L = lambda *a: print(*a, flush=True)
R = {}
T = load("test"); Tr = load("train")

# ---------- (1) count histogram
gt = Tr["gt"].copy(); gt["src"] = gt.rec.str[:2]
c_tr = Tr["s1"].set_index("id").country
H = {}
for cc in ("US", "India"):
    ids = Tr["s1"].id[Tr["s1"].country == cc]
    g = gt[gt.s1.map(c_tr) == cc]
    n = g.groupby("s1").size().reindex(ids, fill_value=0)
    n2 = g[g.src == "S2"].groupby("s1").size().reindex(ids, fill_value=0)
    n3 = g[g.src == "S3"].groupby("s1").size().reindex(ids, fill_value=0)
    H[f"gt_{cc}"] = dict(tot=n.clip(upper=12).value_counts(normalize=True).sort_index().round(5).to_dict(), mean=float(n.mean()),
                        mean_s2=float(n2.mean()), mean_s3=float(n3.mean()), p0_s2=float((n2 == 0).mean()), p0_s3=float((n3 == 0).mean()),
                        E2_s2=float((n2 ** 2).mean()), E2_s3=float((n3 ** 2).mean()))
for tag, cc in (("S005_France", "France"), ("S005_US", "US"), ("S005_India", "India")):
    a = accepted(tag); a = a[a.kept_final]
    ids = T["s1"].id[T["s1"].country == cc]
    n = a.groupby("s1").size().reindex(ids, fill_value=0)
    n2 = a[a.rec.str[:2] == "S2"].groupby("s1").size().reindex(ids, fill_value=0)
    n3 = a[a.rec.str[:2] == "S3"].groupby("s1").size().reindex(ids, fill_value=0)
    H[f"test_{cc}"] = dict(tot=n.clip(upper=12).value_counts(normalize=True).sort_index().round(5).to_dict(), mean=float(n.mean()),
                          mean_s2=float(n2.mean()), mean_s3=float(n3.mean()), p0_s2=float((n2 == 0).mean()), p0_s3=float((n3 == 0).mean()),
                          E2_s2=float((n2 ** 2).mean()), E2_s3=float((n3 ** 2).mean()))
R["count_hist"] = H
tb = pd.DataFrame({k: v["tot"] for k, v in H.items()}).fillna(0)
L(tb.round(4).to_string())
L(pd.DataFrame({k: {m: v[m] for m in ("mean", "mean_s2", "mean_s3", "p0_s2", "p0_s3")} for k, v in H.items()}).round(4).to_string())

# ---------- (2) additive vs replacing test
def src_counts(df_acc):
    """df_acc: accepted rows with s1, rec -> dict (s1,src) -> count"""
    s = df_acc.assign(src=df_acc.rec.str[:2]).groupby(["s1", "src"]).size()
    return s


def type_test(d_acc, typemask, name, counts):
    sel = d_acc[typemask].assign(src=lambda x: x.rec.str[:2])[["s1", "src"]].drop_duplicates()
    k = counts.reindex(pd.MultiIndex.from_frame(sel)).values
    return dict(type=name, n_s1src=int(len(sel)), mean_k=round(float(np.nanmean(k)), 4))


fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
us = pd.read_pickle(os.path.join(OUT, "D_enriched_US5.pkl"))
ind = pd.read_pickle(os.path.join(OUT, "D_enriched_IN5.pkl"))
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
exec(open(os.path.join(OUT, "D_deep_helpers.py")).read())   # addr_rel, swapcls definitions shared with D_deep

rows = []
for nm, d, tag in (("France", fr, "S005_France"), ("US", us, "S005_US"), ("India", ind, "S005_India")):
    a = accepted(tag); a = a[a.kept_final]
    cnt = src_counts(a)
    dk = d[d.kept_final]
    same_a = dk.arel.isin(["exact_addr", "same_num_street"])
    types = {"ALL": np.ones(len(dk), bool), "N_SAME&exact_addr": (dk.nt == "N_SAME") & dk.same_akey,
             "N_TYPO&same_addr": (dk.nt == "N_TYPO") & same_a, "to_noise_suffix&same_addr": (dk.swapcls == "to_noise_suffix") & same_a,
             "garble_swap&same_addr": (dk.swapcls == "garble_or_rare") & same_a, "content_swap&same_addr": (dk.swapcls == "content_word") & same_a,
             "content_swap&other_addr": (dk.swapcls == "content_word") & ~same_a,
             "N_ACRONYM": dk.nt == "N_ACRONYM", "N_DISJOINT": dk.nt == "N_DISJOINT", "N_DROP&same_addr": (dk.nt == "N_DROP") & same_a,
             "N_SAME&num_shift": (dk.nt == "N_SAME") & (dk.arel == "num_shift_same_street"),
             "N_SAME&street_sub": (dk.nt == "N_SAME") & (dk.arel == "street_sub_same_num"),
             "H_D3_10_ODD&N_SAME": (dk.nt == "N_SAME") & (dk.ht == "H_D3_10_ODD")}
    for tn, m in types.items():
        r = type_test(dk, np.asarray(m), tn, cnt); r["country"] = nm; r["pop"] = "test_accepted_final(sample)"
        rows.append(r)
# V1 calibration: counts of TRUE records per (s1,src); for y=1 rows of type T -> 'replacing' reference (size-biased);
# for accepted y=0 / all y=0 hard rows of type T -> k_true + 1 = what an accepted decoy would show
m = np.load(PATHS["v1_meta"])
vv = pd.DataFrame(dict(s1=m["s1_ids"][m["s1idx"]], rec=m["cand"], y=m["y"]))
vv = vv[vv.y == 1]
cnt_true = src_counts(vv)
for cc in ("US", "India"):
    d = v[(v.country == cc)]
    same_a = d.arel.isin(["exact_addr", "same_num_street"])
    for tn, m_ in {"ALL": np.ones(len(d), bool), "to_noise_suffix&same_addr": (d.swapcls == "to_noise_suffix") & same_a,
                   "N_SAME&num_shift": (d.nt == "N_SAME") & (d.arel == "num_shift_same_street"),
                   "content_swap&any": d.swapcls == "content_word", "N_TYPO&same_addr": (d.nt == "N_TYPO") & same_a}.items():
        for lbl in (1, 0):
            dd = d[np.asarray(m_) & (d.y == lbl) & d.hard]
            sel = dd.assign(src=dd.rec.str[:2])[["s1", "src"]].drop_duplicates()
            k = cnt_true.reindex(pd.MultiIndex.from_frame(sel)).fillna(0).values + (1 if lbl == 0 else 0)
            rows.append(dict(type=tn, country=cc, pop=f"V1 y={lbl} (true count{' +1' if lbl == 0 else ''})", n_s1src=int(len(sel)),
                             mean_k=round(float(np.mean(k)), 4) if len(k) else None))
tt = pd.DataFrame(rows)
L(tt.to_string())
R["additive_test"] = rows
# theoretical references from gt per source
ref = {}
for cc in ("US", "India"):
    h = H[f"gt_{cc}"]
    for s in ("s2", "s3"):
        E1, E2 = h[f"mean_{s}"], h[f"E2_{s}"]
        # condition on k>=1 for the 'replacing' (size-biased) reference; decoy reference = E[k]+1 over all S1
        ref[f"{cc}_{s}"] = dict(size_biased=round(E2 / E1, 4), additive=round(E1 + 1, 4))
L("gt references:", ref)
R["gt_refs"] = ref
json.dump(R, open(os.path.join(OUT, "D_counts.json"), "w"), indent=1, default=str)
L("wrote D_counts.json")
