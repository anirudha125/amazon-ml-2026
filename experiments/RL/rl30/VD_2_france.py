"""RL-30 adversarial verification VD, part 2 (France test, label-free): is D's DIST-vs-FILL distinction already present in the
stage-2 input columns (LF 87 + RL-27 10 + dense) of the S005 France pairs, and how does France DIST differ from the V1 DIST
pairs that RL-27 NEW already rejects?  READ-ONLY; writes rl30/VD_2_france.json.
"""
import sys, os, json, glob, collections, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rl30_lib import load, OUT, PATHS, NEW_TH, ROOT, RL
import D_analyze as DA
L = lambda *a: print(time.strftime("%H:%M:%S"), *a, flush=True)
R = {}
TAU = 0.2
T = load("test")
fr = pd.read_pickle(os.path.join(OUT, "D_enriched_FR5.pkl"))
fr = fr[fr.kept_final].reset_index(drop=True)
voc = DA.df_vocab(T["s1"][T["s1"].country == "France"])
add = collections.Counter(" ".join(fr[fr.nt == "N_ADD"]["add"]).split())
sb = collections.Counter(fr[fr.nt == "N_SWAP1"].sw_b)
ratio = lambda t: ((add[t] + sb[t]) / max(1, voc.get(t, 0)), voc.get(t, 0))
same_st = fr.st.isin(["S_SAME", "S_TYPO"])
fr["arel"] = np.select([fr.same_akey, (fr.ht == "H_SAME") & same_st], ["exact_addr", "same_num_street"], "other")
fr["same_addr"] = fr.arel != "other"


def flag_row(nt, a, b):
    ts = a.split() if nt == "N_ADD" else ([b] if nt == "N_SWAP1" else [])
    ts = [t for t in ts if t]
    if not ts:
        return "", np.nan
    rs = [ratio(t) for t in ts]
    words = [(q, d) for q, d in rs if d >= 20]
    if any(q < TAU for q, d in words):
        return "DIST", min(q for q, _ in rs)
    if len(words) == len(rs):
        return "FILL", min(q for q, _ in rs)
    return "OTHER", min(q for q, _ in rs)


ff = [flag_row(*x) for x in zip(fr.nt.values, fr["add"].values, fr.sw_b.values)]
fr["flag"] = [x[0] for x in ff]; fr["dmin"] = [x[1] for x in ff]
R["France_final_counts"] = {f"{sa}|{fl}|{nt}": int(n) for (sa, fl, nt), n in
                            fr[fr.flag != ""].groupby(["same_addr", "flag", "nt"]).size().items()}
L(R["France_final_counts"])
sel = fr[fr.same_addr & fr.flag.isin(["DIST", "FILL"])].copy()
rng = np.random.default_rng(0)
ref = fr[fr.same_addr & (fr.nt == "N_SAME")]
ref = ref.iloc[rng.choice(len(ref), 20000, replace=False)].copy(); ref["flag"] = "REF_N_SAME"
need = pd.concat([sel, ref], ignore_index=True)
need["key"] = need.s1.astype(str) + "|" + need.rec.astype(str)
kidx = dict(zip(need.key.values, range(len(need))))
L(f"need {len(need):,} pairs (DIST/FILL same-addr {len(sel):,} + ref {len(ref):,})")
LFm = np.full((len(need), 87), np.nan, np.float32); RLm = np.full((len(need), 10), np.nan, np.float32)
DN = np.full((len(need), 2), np.nan, np.float32)
chunks = sorted(glob.glob(os.path.join(ROOT, "experiments/P3/France/chunk_*.npz")))
t0 = time.time(); found = 0
for i, c in enumerate(chunks):
    z = np.load(c)
    keys = np.char.add(np.char.add(z["s1"], "|"), z["cand"])
    hit = [(j, kidx[k]) for j, k in enumerate(keys.tolist()) if k in kidx]
    if hit:
        js, ks = np.array([h[0] for h in hit]), np.array([h[1] for h in hit])
        LFm[ks] = z["LF"][js]
        rl = np.load(os.path.join(RL, "test_feats/France", f"rl27_{os.path.basename(c).replace('.npz', '.npy')}"), mmap_mode="r")
        RLm[ks] = rl[js]
        DN[ks, 0] = z["rank_dense"][js]; DN[ks, 1] = z["dcos"][js]
        found += len(hit)
    if i % 40 == 0:
        L(f"chunk {i}/{len(chunks)} found {found:,} {time.time() - t0:.0f}s")
L(f"found {found:,}/{len(need):,}")
LFN = ([f"x22_{i}" for i in range(22)] + [f"B{i}" for i in range(7)] + [f"C{i}" for i in range(6)] + [f"D{i}" for i in range(6)]
       + [f"E{i}" for i in range(3)] + [f"NUM{i}" for i in range(27)] +
       ["n_s1_only", "n_c_only", "idf_s1_only_sum", "idf_c_only_sum", "idf_c_only_max", "idf_s1_only_max", "frac_idf_s1_only",
        "frac_idf_c_only", "n_s1_only_soft", "n_c_only_soft", "idf_s1_only_soft", "idf_c_only_soft", "addr_n_s1_only",
        "addr_n_c_only", "addr_frac_s1_only", "addr_frac_c_only"])
RLN = ["nf_n", "nf_a", "af_n", "af_a", "coloc", "dupf", "rv_rank", "rv_sa", "rv_so", "rv_gap"]
F = pd.DataFrame(np.hstack([LFm, RLm, DN]), columns=LFN + RLN + ["rank_dense", "dcos"])
need = pd.concat([need.reset_index(drop=True), F], axis=1)
need = need[~np.isnan(need.nf_a) | ~np.isnan(need.x22_0)]


def auc(s, y):
    s = np.asarray(s, float); y = np.asarray(y, int); ok = ~np.isnan(s)
    s, y = s[ok], y[ok]
    if len(y) == 0 or y.min() == y.max():
        return None
    r = pd.Series(s).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


cols = ["p"] + LFN + RLN + ["rank_dense", "dcos"]
fs = need[need.flag.isin(["DIST", "FILL"])]
yy = (fs.flag == "DIST").astype(int).values
aucs = {c: auc(fs[c], yy) for c in cols}
sep = {c: round(max(a, 1 - a), 4) for c, a in aucs.items() if a is not None}
top = sorted(sep.items(), key=lambda x: -x[1])[:25]
R["France_DIST_vs_FILL_auc_top25"] = {c: dict(sep=s, auc_raw=round(aucs[c], 4)) for c, s in top}
L("France same-address final: DIST vs FILL, best single existing columns (sep = max(auc,1-auc)):")
for c, s in top:
    L(f"   {c:18s} sep={s:.4f} raw={aucs[c]:.4f}")
# per nt
for nt in ("N_ADD", "N_SWAP1"):
    g = fs[fs.nt == nt]; y2 = (g.flag == "DIST").astype(int).values
    a2 = {c: auc(g[c], y2) for c in cols}
    s2 = sorted(((c, max(a, 1 - a)) for c, a in a2.items() if a is not None), key=lambda x: -x[1])[:10]
    R[f"France_DIST_vs_FILL_auc_top10|{nt}"] = {c: round(s, 4) for c, s in s2}
    L(nt, R[f"France_DIST_vs_FILL_auc_top10|{nt}"], "n DIST", int(y2.sum()), "n FILL", int((1 - y2).sum()))
# medians per group, France vs V1
show = ["p", "x22_0", "x22_3", "x22_4", "n_c_only", "idf_c_only_max", "idf_s1_only_max", "frac_idf_c_only", "C0", "C4",
        "nf_n", "nf_a", "af_n", "coloc", "dupf", "rv_rank", "rv_sa", "rv_gap", "rank_dense", "dcos"]
med = {}
for fl, g in need.groupby("flag"):
    for nt in ("N_ADD", "N_SWAP1", "N_SAME"):
        gg = g[g.nt == nt]
        if len(gg):
            med[f"FR|{fl}|{nt}"] = {c: round(float(np.nanmedian(gg[c])), 3) for c in show} | dict(n=len(gg))
# V1 same-addr flagged (from VD_1 logic)
v = pd.read_pickle(os.path.join(OUT, "D_enriched_V1.pkl"))
st_ = v.st.isin(["S_SAME", "S_TYPO"])
v = v[v.same_akey | ((v.ht == "H_SAME") & st_)].copy()
tabs = {}
for cc, f in (("US", "D_enriched_US5.pkl"), ("India", "D_enriched_IN5.pkl")):
    d = pd.read_pickle(os.path.join(OUT, f)); dk = d[d.kept_final]
    s1c = T["s1"][T["s1"].country == cc]
    tabs[cc] = (DA.df_vocab(s1c), collections.Counter(" ".join(dk[dk.nt == "N_ADD"]["add"]).split()),
                collections.Counter(dk[dk.nt == "N_SWAP1"].sw_b), (len(dk) / len(s1c)) / (len(fr) / (T["s1"].country == "France").sum()))
    del d
def vflag(r):
    vv, a_, b_, sc = tabs[r.country]
    ts = r.add.split() if r.nt == "N_ADD" else ([r.sw_b] if r.nt == "N_SWAP1" else [])
    ts = [t for t in ts if t]
    if not ts:
        return ""
    rs = [((a_[t] + b_[t]) / max(1, vv.get(t, 0)) / sc, vv.get(t, 0)) for t in ts]
    words = [(q, d) for q, d in rs if d >= 20]
    if any(q < TAU for q, d in words):
        return "DIST"
    return "FILL" if len(words) == len(rs) else "OTHER"
v["flag"] = [vflag(r) for r in v[["country", "nt", "add", "sw_b"]].itertuples(index=False)]
LF = np.load(PATHS["v1_LF"], mmap_mode="r"); RL27 = np.load(PATHS["v1_rl27"], mmap_mode="r"); mm = np.load(PATHS["v1_meta"])
rows = v.row.values
VF = pd.DataFrame(np.hstack([np.asarray(LF[rows]), np.asarray(RL27[rows]), mm["rank_dense"][rows][:, None], mm["dcos"][rows][:, None]]),
                  columns=LFN + RLN + ["rank_dense", "dcos"], index=v.index)
v = pd.concat([v, VF], axis=1)
for (fl, nt, y), g in v[v.flag.isin(["DIST", "FILL"])].groupby(["flag", "nt", "y"]):
    med[f"V1|{fl}|{nt}|y={y}"] = {c: round(float(np.nanmedian(g[c])), 3) for c in show} | dict(n=len(g))
tm = pd.DataFrame(med).T
L("\nmedians (France S005 final same-address vs V1 same-address)"); L(tm.to_string())
R["medians"] = med
# V1 DIST vs FILL separation using the same columns (for comparison: is the columns' behaviour the same in V1?)
vs = v[v.flag.isin(["DIST", "FILL"])]
yv = (vs.flag == "DIST").astype(int).values
R["V1_DIST_vs_FILL_auc_for_France_top_cols"] = {c: (None if auc(vs[c], yv) is None else round(auc(vs[c], yv), 4)) for c, _ in top}
L("V1 same-addr DIST vs FILL auc on the same columns:", R["V1_DIST_vs_FILL_auc_for_France_top_cols"])
# France DIST competing-owner indicator from RL-27 (nf_a=0 and nf_n>0 : another S1's core name contains the record's)
for fl in ("DIST", "FILL", "REF_N_SAME"):
    g = need[need.flag == fl]
    R[f"France_{fl}_rl27_rates"] = dict(n=len(g), nf_a0_nfn_pos=round(float(((g.nf_a == 0) & (g.nf_n > 0)).mean()), 4),
                                        nf_a1=round(float((g.nf_a == 1).mean()), 4), rv_rank1=round(float((g.rv_rank == 1).mean()), 4),
                                        rv_rank_gt1=round(float((g.rv_rank > 1).mean()), 4), coloc_pos=round(float((g.coloc > 0).mean()), 4),
                                        p_mean=round(float(g.p.mean()), 4))
    L(fl, R[f"France_{fl}_rl27_rates"])
for fl in ("DIST", "FILL"):
    for y in (0, 1):
        g = v[(v.flag == fl) & (v.y == y)]
        R[f"V1_{fl}_y{y}_rl27_rates"] = dict(n=len(g), nf_a0_nfn_pos=round(float(((g.nf_a == 0) & (g.nf_n > 0)).mean()), 4),
                                             nf_a1=round(float((g.nf_a == 1).mean()), 4), rv_rank1=round(float((g.rv_rank == 1).mean()), 4),
                                             coloc_pos=round(float((g.coloc > 0).mean()), 4), p_mean=round(float(g.p.mean()), 4))
        L(f"V1 {fl} y={y}", R[f"V1_{fl}_y{y}_rl27_rates"])
# France DIST examples with features
s1t = T["s1"].set_index("id"); rect = pd.concat([T["s2"], T["s3"]]).set_index("id")
ex = []
for r in need[need.flag == "DIST"].sample(30, random_state=1).itertuples():
    ex.append(f"p={r.p:.3f} {r.nt} dmin={r.dmin:.3f} nf_n={r.nf_n:.0f} nf_a={r.nf_a:.0f} rv_rank={r.rv_rank:.0f} rv_gap={r.rv_gap:.2f} "
              f"idf_c_only_max={r.idf_c_only_max:.2f} coloc={r.coloc:.0f} | {s1t.at[r.s1, 'name']} | {s1t.at[r.s1, 'addr']} -> "
              f"{rect.at[r.rec, 'name']} | {rect.at[r.rec, 'addr']}")
R["France_DIST_examples"] = ex
L("\nFrance DIST examples:"); [L("  " + e) for e in ex]
need.drop(columns=[c for c in need.columns if c.startswith(("x22_", "B", "D", "E", "NUM")) and c not in ("x22_0", "x22_3", "x22_4")],
          errors="ignore")[["s1", "rec", "p", "nt", "flag", "dmin", "arel", "sw_a", "sw_b", "add"] + RLN + ["idf_c_only_max", "x22_3"]].to_pickle(
    os.path.join(OUT, "VD_2_france_pairs.pkl"))
json.dump(R, open(os.path.join(OUT, "VD_2_france.json"), "w"), indent=1, default=str)
L("wrote VD_2_france.json")
